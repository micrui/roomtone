"""Lock: find our own marker in the mic, line the reference up, learn the room, subtract.

The reference is the exact signal we sent to the speaker (music plus marker). The mic
hears it late, very slightly stretched (two clocks never agree exactly), coloured by
the room, plus whatever people are doing. Subtracting our own sound leaves "the room".
The steps, once per analysis block:

1. Find the marker. Cross-correlate the mic block with the known marker (with FFTs,
   so a 30 s block takes a fraction of a second). The tallest peak is where the
   marker's direct sound landed; its lag is the delay. Peak-to-sidelobe ratio, the
   peak against the tallest correlation value well away from it, says how clearly it
   stood out. Two refinements matter in practice:
   - Clock drift stretches the marker. At 50 ppm a 4 s marker arrives 9 samples
     longer than it left, enough to smear the peak into mush at 7 kHz. So the marker
     template is stretched by the current drift estimate before correlating.
   - The very first time, drift is unknown. We find the marker roughly with just its
     first half second (too short to smear), then time several short pieces of it
     individually: the pieces drift apart at exactly the clock rate, which gives a
     first drift estimate from a single marker.
   The peak is then refined to a fraction of a sample by interpolating the
   correlation, because the room estimate is sensitive to sub-sample shifts.
2. Drift. Each marker gives one (emission time, lag) point; with the mic clock fast
   by 50 ppm the lag grows 50 samples per million, a straight line. A least-squares
   line through recent points gives the drift (its slope) and a smoothed delay.
   Jitter is how far the latest points sit from that line.
3. Align. Resample the reference onto the mic's sample grid (2x oversampling, then
   cubic interpolation between the fine samples) so they line up sample for sample.
   The direct sound is placed `PRE_ROLL` samples into the room response so it has
   some room either side.
4. Room response. Model the mic as the aligned reference passed through a filter, the
   room impulse response (RIR), and pick the filter that minimises the leftover
   energy, plus a small penalty on straying from a starting guess (regularised least
   squares). The ingredients (how the reference correlates with itself, and with the
   mic) come from FFTs over the whole block; the resulting equations have a special
   repeating (Toeplitz) structure that scipy solves exactly in a fraction of a
   second. The starting guess is the previous block's response when it earned trust
   (zero otherwise); the result is then blended half and half with it, so one noisy
   block cannot swing it far.
5. Subtract and score. Residual = mic minus reference passed through the response.
   Echo suppression (ERLE) is the energy ratio of mic to residual in dB, measured on
   the quarter of the block where the residual is quietest, so a person talking does
   not read as failed subtraction. Response change is the normalised distance
   between this block's estimate and the previous response, compared inside the
   marker's band: tonal music only pins the response down at the notes it plays,
   so elsewhere the estimate wanders harmlessly, while the marker band is always
   measured well (that is the marker's job).
6. Decide. Locked means suppression >= 20 dB, jitter <= 2 samples and a stable
   response, for 3 blocks running. A sudden large response change after lock means
   the room changed (LOST): keep the old response this block, start afresh next
   block. Three bad blocks in a row step the marker up its ladder (longer, then
   another band, then louder); three good blocks step it back down. Failing at the
   top of the ladder is SATURATED.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from scipy.fft import irfft, next_fast_len, rfft
from scipy.linalg import solve_toeplitz
from scipy.signal import fftconvolve, resample_poly

from .config import Config, MarkerParams
from .marker import generate
from .types import LockState, LockStatus

log = logging.getLogger(__name__)

PRE_ROLL = 128              # samples of response kept before the direct sound
OVERSAMPLE = 2              # fine grid for alignment interpolation
RIR_REG = 1e-4              # ridge, relative to the reference's energy
ACQUIRE_SEARCH_S = 3.0      # widest delay we look for (Sonos compressed mode is about 2 s)
ACQUIRE_PIECE_S = 0.5       # marker piece used for the first rough search and the drift pieces
DRIFT_PIECES = 8            # pieces timed individually for the single-marker drift estimate
TRACK_SEARCH_S = 0.005      # search half-width once the delay is known
MAX_DRIFT_PPM = 200.0       # largest clock drift we allow for
MIN_PSR_DB = 10.0           # weaker peaks count as "marker not found"
REACQUIRE_JUMP = 16.0       # a marker this far off the drift line means the delay re-settled
STABLE_CHANGE = 0.3         # response change below this counts as stable
LOST_CHANGE = 0.6           # response change above this, after lock, means the room changed
HISTORY_POINTS = 12         # markers kept for the drift fit
JITTER_POINTS = 5           # latest markers the jitter is measured over
ERLE_SEGMENT_S = 0.25       # segment length for speech-robust echo suppression
ERLE_FRACTION = 0.25        # share of segments (quietest residual) ERLE is measured on


# Resampling ------------------------------------------------------------------

def warp(x: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Sample x at fractional positions (zero outside it).

    x is first oversampled 2x with a proper low-pass filter, then read with 4-point
    cubic (Lagrange) interpolation on that fine grid. Error stays below -40 dB up to
    10 kHz at 44.1 kHz, and far lower where music lives.
    """
    fine = resample_poly(np.asarray(x, dtype=np.float64), OVERSAMPLE, 1).astype(np.float32)
    g = np.asarray(positions, dtype=np.float64) * OVERSAMPLE
    i = np.floor(g).astype(np.int64)
    out = np.zeros(len(g), dtype=np.float32)
    ok = (i >= 1) & (i < len(fine) - 2)
    i = i[ok]
    t = (g[ok] - i).astype(np.float32)
    tm1, tp1, tm2 = t - 1.0, t + 1.0, t - 2.0
    out[ok] = (fine[i - 1] * (-t * tm1 * tm2 / 6.0) + fine[i] * (tp1 * tm1 * tm2 / 2.0)
               + fine[i + 1] * (-tp1 * t * tm2 / 2.0) + fine[i + 2] * (tp1 * t * tm1 / 6.0))
    return out.astype(np.float64)


_stretch_cache: dict[tuple[int, int, float], np.ndarray] = {}


def _fingerprint(x: np.ndarray) -> int:
    return hash(np.asarray(x[:64]).tobytes() + np.asarray(x[-64:]).tobytes())


def _stretched(marker: np.ndarray, drift_ppm: float) -> np.ndarray:
    """The marker as a mic running drift_ppm fast would record it (cached, 0.5 ppm steps)."""
    d = round(drift_ppm * 2.0) / 2.0
    if d == 0.0:
        return np.asarray(marker, dtype=np.float64)
    key = (_fingerprint(marker), len(marker), d)
    hit = _stretch_cache.get(key)
    if hit is None:
        r = 1.0 + d * 1e-6
        hit = warp(marker, np.arange(int(len(marker) * r)) / r)
        if len(_stretch_cache) >= 4:
            _stretch_cache.pop(next(iter(_stretch_cache)))
        _stretch_cache[key] = hit
    return hit


# Marker finding --------------------------------------------------------------

def _refine_peak(c: np.ndarray, k: int, half: int = 16, steps: int = 32) -> float:
    """Sub-sample offset of the |c| peak near index k, by windowed-sinc interpolation."""
    lo, hi = max(0, k - half - 1), min(len(c), k + half + 2)
    idx = np.arange(lo, hi) - k
    tau = np.linspace(-1.0, 1.0, 2 * steps + 1)
    d = tau[:, None] - idx[None, :]
    w = np.sinc(d) * np.i0(8.0 * np.sqrt(np.clip(1.0 - (d / (half + 1)) ** 2, 0.0, 1.0))) / np.i0(8.0)
    v = np.abs(w @ c[lo:hi])
    j = int(np.argmax(v))
    off = float(tau[j])
    if 0 < j < len(v) - 1:
        a, b, g = v[j - 1], v[j], v[j + 1]
        den = a - 2 * b + g
        if den < 0:
            off += 0.5 * (a - g) / den * float(tau[1] - tau[0])
    return off


def find_marker(mic: np.ndarray, marker: np.ndarray, expected: float | None, search: int,
                subsample: bool = False, drift_ppm: float = 0.0) -> tuple[float, float]:
    """Locate the marker in the mic block: (lag in samples, peak-to-sidelobe ratio in dB).

    Lags searched: [0, search] when `expected` is None, else expected +- search.
    The sidelobe is the largest |correlation| at any lag outside +-len(marker)/50
    of the peak. Partial overlaps at the block end still count, so a marker running
    past the end is still found. `drift_ppm` stretches the template to match the mic
    clock. With `subsample` the lag is refined to a fraction of a sample (a float);
    otherwise it is an int.
    """
    y = np.asarray(mic, dtype=np.float64)
    m = _stretched(marker, drift_ppm)
    n = len(y)
    nfft = next_fast_len(n + len(m) - 1, real=True)
    c = irfft(rfft(y, nfft) * np.conj(rfft(m, nfft)), nfft)[:n]
    mag = np.abs(c)
    lo, hi = (0.0, float(search)) if expected is None else (expected - search, expected + search)
    lo_i, hi_i = max(0, int(np.floor(lo))), min(n - 1, int(np.ceil(hi)))
    if lo_i > hi_i:
        raise ValueError(f"search window [{lo}, {hi}] lies outside the block")
    lag = lo_i + int(np.argmax(mag[lo_i:hi_i + 1]))
    guard = max(1, len(marker) // 50)
    side = max(mag[:max(0, lag - guard)].max(initial=0.0), mag[lag + guard + 1:].max(initial=0.0))
    psr = float(20.0 * np.log10(max(mag[lag], 1e-300) / max(side, 1e-300)))
    if subsample:
        return lag + _refine_peak(c, lag), psr
    return lag, psr


def marker_drift(mic: np.ndarray, marker: np.ndarray, lag: float, sample_rate: int) -> float:
    """Clock drift in ppm from one marker: time short pieces of it separately and fit a line.

    A piece that starts `a` samples into the marker arrives `a * drift` samples later
    than it would with matched clocks. `lag` only needs to be right to a few samples.
    """
    y = np.asarray(mic, dtype=np.float64)
    m = np.asarray(marker, dtype=np.float64)
    p = min(len(m), int(ACQUIRE_PIECE_S * sample_rate))
    if len(m) < 2 * p:
        return 0.0
    reach = 16 + int(MAX_DRIFT_PPM * 1e-6 * len(m))
    xs, offs = [], []
    for a in np.linspace(0, len(m) - p, DRIFT_PIECES).astype(int):
        s = int(round(lag)) + a - reach
        if s < 0 or s + p + 2 * reach > len(y):
            continue
        c = fftconvolve(y[s:s + p + 2 * reach], m[a:a + p][::-1], mode="valid")
        k = int(np.argmax(np.abs(c)))
        xs.append(float(a))
        offs.append(k + _refine_peak(c, k) - reach)
    return _fit_line(xs, offs)[0] * 1e6 if len(xs) >= 2 else 0.0


def _fit_line(x: Sequence[float], y: Sequence[float]) -> tuple[float, float, float]:
    """Least-squares line through (x, y): (slope, x_mean, y_mean). Centred for conditioning."""
    xa, ya = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    xm, ym = float(xa.mean()), float(ya.mean())
    sxx = float(np.sum((xa - xm) ** 2))
    slope = float(np.sum((xa - xm) * (ya - ym)) / sxx) if sxx > 0 else 0.0
    return slope, xm, ym


def estimate_drift(lags: Sequence[float], emission_starts: Sequence[int]) -> float:
    """Clock drift in ppm: the least-squares slope of lag against emission start."""
    if len(lags) < 2:
        return 0.0
    return _fit_line(emission_starts, lags)[0] * 1e6


def align(reference: np.ndarray, mic: np.ndarray, delay_samples: float, drift_ppm: float) -> np.ndarray:
    """The reference resampled and shifted onto the mic's sample grid (same length as mic).

    Mic sample n heard reference sample (n - delay) / (1 + drift); `delay_samples`
    is where reference sample 0 lands in the mic block.
    """
    n = np.arange(len(mic), dtype=np.float64)
    return warp(reference, (n - delay_samples) / (1.0 + drift_ppm * 1e-6))


# The room response -----------------------------------------------------------

def _fit(n: int, h: np.ndarray | None) -> np.ndarray:
    """h zero-padded or truncated to n samples (zeros for None)."""
    out = np.zeros(n)
    if h is not None:
        k = min(n, len(h))
        out[:k] = np.asarray(h, dtype=np.float64)[:k]
    return out


def _solve_rir(x: np.ndarray, y: np.ndarray, tail: int, reg: float,
               start: np.ndarray | None) -> np.ndarray:
    """Regularised least squares: argmin |y - x*h|^2 + lam |h - start|^2, lam = reg * mean power * N.

    The normal equations R h = p use the reference's autocorrelation R (Toeplitz) and
    its cross-correlation p with the mic, both computed with FFTs.
    """
    hist = len(x) - len(y)
    nfft = next_fast_len(len(x) + tail, real=True)
    X = rfft(x, nfft)
    r = irfft(X.real ** 2 + X.imag ** 2, nfft)[:tail]
    c = irfft(rfft(y, nfft) * np.conj(X), nfft)
    p = c[(np.arange(tail) - hist) % nfft]       # sum_n y[n] x[n + hist - k]
    lam = reg * r[0]
    col = r.copy()
    col[0] += lam
    if start is not None:
        p = p + lam * _fit(tail, start)
    return solve_toeplitz(col, p)


def estimate_rir(reference_aligned: np.ndarray, mic: np.ndarray, tail_samples: int,
                 reg: float = RIR_REG, prior: np.ndarray | None = None) -> np.ndarray:
    """Room impulse response, tail_samples long, from aligned reference to mic.

    `reference_aligned` may be longer than `mic`: extra leading samples are history
    from before mic[0], so reverb carried in from earlier is modelled. With a `prior`
    the estimate is pulled toward it instead of toward zero, then blended 0.5/0.5 with it.

    Why an exact solve rather than dividing averaged segment spectra (the usual
    Welch-style shortcut): with tonal music the shortcut's segment edges and
    truncation bias the answer, and on the simulated room it capped echo
    suppression near 12-15 dB; the exact solve reaches the noise floor (over 30 dB).
    """
    x = np.asarray(reference_aligned, dtype=np.float64)
    y = np.asarray(mic, dtype=np.float64)
    h = _solve_rir(x, y, tail_samples, reg, prior)
    return h if prior is None else 0.5 * h + 0.5 * _fit(tail_samples, prior)


def subtract(reference_aligned: np.ndarray, mic: np.ndarray, rir: np.ndarray) -> np.ndarray:
    """Residual: mic minus the reference passed through the room response.

    As in `estimate_rir`, leading reference samples beyond len(mic) are history.
    """
    x = np.asarray(reference_aligned, dtype=np.float64)
    y = np.asarray(mic, dtype=np.float64)
    hist = len(x) - len(y)
    echo = fftconvolve(x, np.asarray(rir, dtype=np.float64))[hist:hist + len(y)]
    return y - echo


def erle_db(mic: np.ndarray, residual: np.ndarray, segment: int | None = None,
            fraction: float = ERLE_FRACTION) -> float:
    """Echo suppression: mic energy over residual energy, in dB.

    With `segment`, the block is cut into segments of that many samples and the
    ratio is taken over the `fraction` of segments where the residual is quietest
    relative to the mic: the speech-free stretches, when people are talking.
    """
    y = np.asarray(mic, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64)
    if segment is None or segment >= len(y):
        ey, er = float(np.sum(y * y)), float(np.sum(r * r))
    else:
        k = len(y) // segment
        ys = np.sum(y[:k * segment].reshape(k, segment) ** 2, axis=1)
        rs = np.sum(r[:k * segment].reshape(k, segment) ** 2, axis=1)
        pick = np.argsort(rs / np.maximum(ys, 1e-300))[:max(1, int(round(fraction * k)))]
        ey, er = float(ys[pick].sum()), float(rs[pick].sum())
    return float(10.0 * np.log10(max(ey, 1e-300) / max(er, 1e-300)))


def band_limit(h: np.ndarray, lo_hz: float, hi_hz: float, sample_rate: int) -> np.ndarray:
    """The part of a response inside [lo_hz, hi_hz] (FFT mask, zero-padded so nothing wraps)."""
    n = 2 * len(h)
    H = rfft(np.asarray(h, dtype=np.float64), n)
    f = np.arange(len(H)) * sample_rate / n
    H[(f < lo_hz) | (f > hi_hz)] = 0.0
    return irfft(H, n)


def rir_change(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised L2 distance 2|a - b| / (|a| + |b|): 0 when identical, 2 when b = -a."""
    n = max(len(a), len(b))
    a2, b2 = _fit(n, a), _fit(n, b)
    den = float(np.linalg.norm(a2) + np.linalg.norm(b2))
    return float(2.0 * np.linalg.norm(a2 - b2) / den) if den > 0 else 0.0


# Baseline ---------------------------------------------------------------------

def save_baseline(path: str | Path, rir: np.ndarray) -> None:
    """Store a quiet-house room response to start future sessions from."""
    with open(path, "wb") as f:
        np.save(f, np.asarray(rir, dtype=np.float64))


def load_baseline(path: str | Path) -> np.ndarray | None:
    """The stored baseline response, or None if there is none yet."""
    p = Path(path)
    return np.load(p) if p.exists() else None


# The Locker -------------------------------------------------------------------

class Locker:
    """Per-block lock, subtraction and marker ladder. See the module docstring.

    Call `process(reference, mic, emission_start)` once per block. `reference` is
    the output block starting at output sample `emission_start`, where a marker
    starts; `mic` is the mic block with the same nominal sample index in the mic's
    clock. Blocks are expected back to back (a gap clears the carried-over reference
    history). After each call `.params` holds the marker parameters to emit next and
    `.rir` the current room response. A `baseline` response, if given, is the
    starting point.
    """

    def __init__(self, cfg: Config, baseline: np.ndarray | None = None) -> None:
        self.cfg = cfg
        self.sample_rate = int(cfg.sample_rate)
        self.tail = int(round(cfg.rir_tail_seconds * self.sample_rate))
        self.baseline = None if baseline is None else np.asarray(baseline, dtype=np.float64)
        self._context = int(ACQUIRE_SEARCH_S * self.sample_rate) + self.tail + PRE_ROLL + 256
        self.reset(full=True)

    def reset(self, full: bool = False) -> None:
        """Re-acquire timing (after a stream restart). `full` also forgets the room, drift and ladder."""
        self._lags: list[tuple[int, float]] = []
        self._misses = 0
        self._ref_tail = np.zeros(self._context)
        self._expect_start: int | None = None
        self.status = LockStatus.ACQUIRING
        self._ok = self._bad = 0
        self._ladder_ok = self._ladder_bad = 0
        self._from_scratch = False
        self._last_band: tuple[float, float] | None = None
        if full:
            self.params: MarkerParams = self.cfg.marker
            self._rung = 0
            self._drift = 0.0
            self._drift_known = False
            self.rir: np.ndarray | None = None if self.baseline is None else _fit(self.tail, self.baseline)
            self._trust_prior = self.rir is not None

    @property
    def marker(self) -> np.ndarray:
        return generate(self.params, self.sample_rate)

    def _predict(self, start: int) -> float | None:
        if len(self._lags) >= 2:
            s, xm, ym = _fit_line(*zip(*self._lags))
            return ym + s * (start - xm)
        if self._lags:
            e0, l0 = self._lags[-1]
            return l0 + self._drift * 1e-6 * (start - e0)
        return None

    def _jitter(self) -> float | None:
        if len(self._lags) < 3:
            return None
        xs, ys = np.array(self._lags).T
        s, xm, ym = _fit_line(xs, ys)
        return float(np.std((ys - (ym + s * (xs - xm)))[-JITTER_POINTS:]))

    def _locate(self, y: np.ndarray, marker: np.ndarray, start: int) -> tuple[float, float, float | None]:
        """Marker lag and PSR in this block, plus the drift line's prediction (if any)."""
        fs = self.sample_rate
        predicted = self._predict(start)
        if predicted is not None and self._misses < 2:
            search = int(TRACK_SEARCH_S * fs)
            if len(self._lags) < 2 and not self._drift_known:
                search += int(MAX_DRIFT_PPM * 1e-6 * abs(start - self._lags[-1][0]))
            lag, psr = find_marker(y, marker, predicted, search, True, self._drift)
            return lag, psr, predicted
        # acquisition: rough position from the marker's first piece, which drift cannot smear
        piece = marker[:min(len(marker), int(ACQUIRE_PIECE_S * fs))]
        rough, _ = find_marker(y, piece, None, min(int(ACQUIRE_SEARCH_S * fs), len(y) - 1))
        if not self._drift_known:
            self._drift = marker_drift(y, marker, rough, fs)
        search = int(TRACK_SEARCH_S * fs) + int(MAX_DRIFT_PPM * 1e-6 * len(piece))
        lag, psr = find_marker(y, marker, rough, search, True, self._drift)
        return lag, psr, predicted

    def _move(self, new: MarkerParams, step: int) -> None:
        if new == self.params:
            return
        log.info("marker ladder %s: %s -> %s", "up" if step > 0 else "down", self.params, new)
        self.params = new
        self._rung += step
        self._ladder_ok = self._ladder_bad = 0

    def process(self, reference: np.ndarray, mic: np.ndarray, emission_start: int,
                params: MarkerParams | None = None) -> tuple[np.ndarray, LockState]:
        """One block: (residual, same length and float dtype as mic; LockState).

        `params` are the marker parameters actually emitted in this block, when the
        caller knows them (MarkerScheduler.emitted_params); otherwise the Locker
        assumes its own `.params` were used.
        """
        cfg, fs, T = self.cfg, self.sample_rate, self.tail
        used = params or self.params
        x = np.asarray(reference, dtype=np.float64)
        y = np.asarray(mic, dtype=np.float64)
        n = len(y)
        out_dtype = mic.dtype if np.issubdtype(np.asarray(mic).dtype, np.floating) else np.float64
        band = (used.band_lo_hz, used.band_hi_hz)
        band_changed = self._last_band is not None and band != self._last_band
        self._last_band = band
        if self._expect_start is not None and emission_start != self._expect_start:
            self._ref_tail = np.zeros(self._context)
        self._expect_start = emission_start + len(x)

        # 1. find the marker
        lag, psr, predicted = self._locate(y, generate(used, fs), emission_start)
        found = psr >= MIN_PSR_DB
        if found:
            self._misses = 0
            if len(self._lags) >= 2 and predicted is not None and abs(lag - predicted) > REACQUIRE_JUMP:
                log.info("delay moved %.1f samples off the drift line; re-acquiring", lag - predicted)
                self._lags.clear()
            self._lags = (self._lags + [(emission_start, lag)])[-HISTORY_POINTS:]
        else:
            self._misses += 1
            log.debug("marker not found at %d (psr %.1f dB)", emission_start, psr)

        # 2. drift and delay
        if len(self._lags) >= 2:
            self._drift = estimate_drift([l for _, l in self._lags], [e for e, _ in self._lags])
            self._drift_known = True
        delay = self._predict(emission_start)
        jitter = self._jitter()

        ext = np.concatenate([self._ref_tail, x])
        self._ref_tail = ext[len(ext) - self._context:]
        state = LockState(drift_ppm=self._drift, peak_to_sidelobe_db=psr,
                          jitter_samples=jitter or 0.0, marker_rung=self._rung)

        if delay is None:
            residual, erle, change, lost = y.copy(), 0.0, 2.0, False
        else:
            # 3. align: aligned[T + k] lines up with mic[k]; the first T samples are history
            m = np.arange(-T, n, dtype=np.float64)
            aligned = warp(ext, self._context + (m + PRE_ROLL - delay) / (1.0 + self._drift * 1e-6))

            # 4. room response, pulled toward the previous one when that one earned trust
            prior = self.rir if (self._trust_prior and not self._from_scratch) else None
            est = _solve_rir(aligned, y, T, RIR_REG, prior)
            change = (rir_change(band_limit(est, *band, fs), band_limit(self.rir, *band, fs))
                      if self.rir is not None else 2.0)
            lost = self.status == LockStatus.LOCKED and change > LOST_CHANGE and not band_changed
            if lost:
                log.info("room response jumped (change %.2f); keeping the old one this block", change)
                use = self.rir
                self._from_scratch = True
            else:
                use = est if prior is None else 0.5 * est + 0.5 * prior
                self.rir = use
                self._from_scratch = False

            # 5. subtract and score
            residual = subtract(aligned, y, use)
            erle = erle_db(y, residual, segment=int(ERLE_SEGMENT_S * fs))
            self._trust_prior = not lost and erle >= cfg.lock_erle_db / 2
            state.delay_samples = float(delay)

        # 6. decide
        good = (found and erle >= cfg.lock_erle_db and jitter is not None
                and jitter <= cfg.lock_jitter_samples and change < STABLE_CHANGE)
        H = cfg.hysteresis_blocks
        if good:
            self._ok, self._bad = self._ok + 1, 0
            self._ladder_ok, self._ladder_bad = self._ladder_ok + 1, 0
        else:
            self._ok, self._bad = 0, self._bad + 1
            self._ladder_ok, self._ladder_bad = 0, self._ladder_bad + 1

        prev = self.status
        if lost:
            status = LockStatus.LOST
        elif self._ok >= H:
            status = LockStatus.LOCKED
        elif prev in (LockStatus.LOCKED, LockStatus.LOST) and self._bad < H:
            status = prev                       # hold through brief trouble
        else:
            status = LockStatus.ACQUIRING
        if self._ladder_bad >= H:
            if self.params.at_ceiling:
                status = LockStatus.SATURATED
            else:
                self._move(self.params.escalate(), +1)
        elif status == LockStatus.LOCKED and self._ladder_ok >= H and self._rung > 0:
            self._move(self.params.relax(), -1)
        if status != prev:
            log.info("lock %s -> %s (erle %.1f dB, jitter %s, change %.2f)", prev.value, status.value,
                     erle, "n/a" if jitter is None else f"{jitter:.3f}", change)
        self.status = status

        state.status = status
        state.erle_db = erle
        state.rir_change = change
        state.blocks_ok, state.blocks_bad = self._ok, self._bad
        return residual.astype(out_dtype), state
