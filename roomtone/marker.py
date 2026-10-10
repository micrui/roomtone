"""The marker: a quiet, band-limited noise burst mixed into our own output.

Why it exists: to subtract the music from what the mic hears, the Listener has to
know exactly when each output sample reaches the mic, and how the room colours it.
Music is a poor ruler for that (long notes repeat, drones have no edges), so every
analysis period we add a short burst of a known pseudo-random signal underneath it.

How it is made:

1. A maximum length sequence (MLS). This is the output of a linear feedback shift
   register: a string of +1/-1 values that looks like white noise but has one
   special property. Correlating it against a shifted copy of itself gives a single
   sharp spike at zero shift and almost nothing anywhere else, so finding it inside a
   noisy recording is easy, and the longer it is, the more it stands out (the
   "processing gain" is about 10*log10(length in samples) dB).
2. The register's starting state comes from `MarkerParams.seed`, so the Listener can
   regenerate the identical sequence from the parameters alone.
3. It is shaped to a band (4 to 10 kHz by default) by a zero-phase filter: we take
   its spectrum, multiply by a real gain that is 1 in the band, 0 outside it, with
   short raised-cosine ramps just inside the edges, and transform back. A real gain
   delays nothing, so the burst's timing is untouched.
4. Short fades at both ends stop it clicking, and it is normalised to RMS 1.0.
   `scale_to_music` then sets the real playback level relative to the music.

`MarkerScheduler` decides when bursts go out: one every `period_s`, the first at
sample 0, and it remembers where each one started so the Listener knows where to look.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq
from scipy.signal import max_len_seq

from .config import MarkerParams

log = logging.getLogger(__name__)

FADE_S = 0.02          # raised-cosine fade at each end of a burst
EDGE_TAPER = 0.1       # fraction of the bandwidth spent on each in-band edge ramp
FLOOR_DBFS = -60.0     # quietest marker we ever emit, even over silence
_CACHE_MAX = 16

_cache: dict[tuple[float, float, float, int, int], np.ndarray] = {}


def _register_state(seed: int, nbits: int) -> np.ndarray:
    """Fold the seed into an nbits-long nonzero starting state for the shift register."""
    mask = (1 << nbits) - 1
    v, s = 0, int(seed)
    while s:
        v ^= s & mask
        s >>= nbits
    if v == 0:
        v = 1
    return np.array([(v >> i) & 1 for i in range(nbits)], dtype=np.int8)


def _band_gain(n: int, sample_rate: int, lo: float, hi: float) -> np.ndarray:
    """Real, non-negative spectral gain: 1 inside [lo, hi], 0 outside, cosine ramps inside the edges."""
    f = rfftfreq(n, 1.0 / sample_rate)
    ramp = EDGE_TAPER * (hi - lo)
    g = np.zeros_like(f)
    inside = (f >= lo) & (f <= hi)
    g[inside] = 1.0
    up = inside & (f < lo + ramp)
    g[up] = 0.5 - 0.5 * np.cos(np.pi * (f[up] - lo) / ramp)
    down = inside & (f > hi - ramp)
    g[down] = 0.5 - 0.5 * np.cos(np.pi * (hi - f[down]) / ramp)
    return g


def generate(params: MarkerParams, sample_rate: int) -> np.ndarray:
    """The unit-RMS marker for these parameters (float32, read-only, cached).

    Only length, band and seed shape the waveform; period and level do not, so they
    are left out of the cache key.
    """
    key = (float(params.length_s), float(params.band_lo_hz), float(params.band_hi_hz),
           int(params.seed), int(sample_rate))
    hit = _cache.get(key)
    if hit is not None:
        return hit
    if not 0 < params.band_lo_hz < params.band_hi_hz <= sample_rate / 2:
        raise ValueError(f"marker band {params.band_lo_hz}-{params.band_hi_hz} Hz "
                         f"does not fit under Nyquist at {sample_rate} Hz")

    n = int(round(params.length_s * sample_rate))
    nbits = max(2, int(np.ceil(np.log2(n + 1))))       # 2**nbits - 1 >= n
    seq, _ = max_len_seq(nbits, state=_register_state(params.seed, nbits), length=n)
    x = 2.0 * seq.astype(np.float64) - 1.0

    x = irfft(rfft(x) * _band_gain(n, sample_rate, params.band_lo_hz, params.band_hi_hz), n)

    nf = min(int(FADE_S * sample_rate), n // 2)
    if nf > 0:
        fade = 0.5 - 0.5 * np.cos(np.pi * np.arange(nf) / nf)
        x[:nf] *= fade
        x[-nf:] *= fade[::-1]

    x /= np.sqrt(np.mean(x * x))
    out = x.astype(np.float32)
    out.flags.writeable = False
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = out
    log.debug("generated marker %.1f s, %g-%g Hz, nbits %d",
              params.length_s, params.band_lo_hz, params.band_hi_hz, nbits)
    return out


def scale_to_music(marker: np.ndarray, music_rms: float, params: MarkerParams) -> np.ndarray:
    """Scale the marker to `level_db` below the music's RMS, but never below -60 dBFS RMS.

    The floor keeps a faint marker going over silence, so lock survives quiet passages.
    """
    target = max(float(music_rms) * 10.0 ** (params.level_db / 20.0), 10.0 ** (FLOOR_DBFS / 20.0))
    m = np.asarray(marker, dtype=np.float64)
    rms = np.sqrt(np.mean(m * m))
    if rms == 0.0:
        return np.zeros(len(m), dtype=np.float32)
    return (m * (target / rms)).astype(np.float32)


class MarkerScheduler:
    """Hands the mixer the marker samples to add to each output block.

    Call `block(start, frames, music_rms)` with the running output sample index of
    the block. A burst starts every `period_s` seconds from sample 0; its level is
    fixed when it starts, from the `music_rms` given in that block. Bursts that
    straddle block boundaries continue seamlessly. New parameters from `set_params`
    wait for the next burst so no burst is ever cut or changed midway.
    """

    def __init__(self, params: MarkerParams, sample_rate: int) -> None:
        self.params = params
        self.sample_rate = int(sample_rate)
        self.emissions: list[int] = []                # start sample of every burst so far
        self.emitted_params: list[MarkerParams] = []  # parameters of each burst, same order
        self._pending: MarkerParams | None = None
        self._next = 0
        self._active: list[tuple[int, np.ndarray]] = []

    @property
    def period_samples(self) -> int:
        return int(round(self.params.period_s * self.sample_rate))

    def set_params(self, params: MarkerParams) -> None:
        """Use `params` from the next burst on."""
        self._pending = params

    def block(self, start: int, frames: int, music_rms: float = 1.0) -> np.ndarray:
        """Marker samples for output samples [start, start + frames), float32."""
        end = start + frames
        out = np.zeros(frames, dtype=np.float32)
        while self._next < end:
            if self._pending is not None:
                if self._pending != self.params:
                    log.info("marker params now %s from sample %d", self._pending, self._next)
                self.params, self._pending = self._pending, None
            sig = scale_to_music(generate(self.params, self.sample_rate), music_rms, self.params)
            self._active.append((self._next, sig))
            self.emissions.append(self._next)
            self.emitted_params.append(self.params)
            self._next += self.period_samples
        still: list[tuple[int, np.ndarray]] = []
        for s, sig in self._active:
            a, b = max(start, s), min(end, s + len(sig))
            if a < b:
                out[a - start:b - start] += sig[a - s:b - s]
            if s + len(sig) > end:
                still.append((s, sig))
        self._active = still
        return out
