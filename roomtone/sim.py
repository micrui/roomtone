"""A pretend room, so the lock can be tested without a speaker, a mic or a kitchen.

`SimRoom` stands in for everything between the output device and the mic:

1. Delay: the Sonos buffer and the air, a fixed number of output samples.
2. Clock drift: the mic's clock runs `drift_ppm` parts per million faster than the
   output clock, so output sample i arrives at mic sample (i + delay) * (1 + drift).
   This is done with scipy's polyphase resampler at an exact rational rate
   (50 ppm is 20001/20000), a different method from the one the Listener uses to
   undo it, so a test cannot pass just because both sides share the same mistake.
3. The room: convolution with an impulse response (direct sound, a handful of early
   reflections off walls and counters, then a noise-like tail that dies away
   exponentially, reaching -60 dB after RT60 seconds).
4. Whatever the room adds itself (talk, clatter), then the mic's own hiss.

It streams: call `render` with consecutive output blocks and it returns the mic
samples captured over the same span of mic samples, carrying delay, resampler and
reverb state across calls. Swapping `.rir` takes effect on the next call, which is
how tests model furniture being moved.

The helpers make test signals: `pad_like` (slow sine tones, a stand-in for ambient
music), `speech_like` (band-limited noise with syllable-rate bursts and pauses
between phrases), and `clatter` (sparse, sharp, decaying clicks, like cutlery).
"""

from __future__ import annotations

import logging
from fractions import Fraction

import numpy as np
from scipy.signal import fftconvolve, firwin, resample_poly, butter, sosfilt

log = logging.getLogger(__name__)

_PAD = 64                 # extra input samples either side of each resampled chunk
_MAX_DENOMINATOR = 100_000


def _rms_scale(x: np.ndarray, rms_db: float) -> np.ndarray:
    rms = np.sqrt(np.mean(x * x))
    return x * (10.0 ** (rms_db / 20.0) / rms) if rms > 0 else x


def synthetic_rir(sample_rate: int, rng: np.random.Generator, rt60: float = 0.4,
                  length_s: float = 0.3, direct_s: float = 0.002, drr_db: float = 0.0,
                  n_reflections: int = 6) -> np.ndarray:
    """A plausible small-room impulse response, normalised to unit energy.

    `drr_db` is the direct-to-reverberant energy ratio: 0 dB means the direct sound
    and everything after it carry equal energy, typical a couple of metres away.
    """
    n = int(round(length_s * sample_rate))
    decay = 6.91 / rt60                                  # amplitude: exp(-decay*t) is -60 dB at rt60
    h = np.zeros(n)
    d0 = int(round(direct_s * sample_rate))
    h[d0] = 1.0
    for _ in range(n_reflections):
        dt = rng.uniform(0.003, 0.03)
        gain = rng.choice([-1.0, 1.0]) * rng.uniform(0.25, 0.6) * np.exp(-decay * dt)
        h[d0 + int(dt * sample_rate)] += gain
    onset = d0 + int(0.005 * sample_rate)
    t = (np.arange(n) - onset) / sample_rate
    tail = rng.standard_normal(n) * np.exp(-decay * np.maximum(t, 0.0))
    tail[:onset] = 0.0
    early = np.sum(h * h) - 1.0
    target = max(10.0 ** (-drr_db / 10.0) - early, 1e-6)  # reflections count as reverberant
    h += tail * np.sqrt(target / np.sum(tail * tail))
    return h / np.sqrt(np.sum(h * h))


class SimRoom:
    """Streaming speaker-room-mic model. See the module docstring."""

    def __init__(self, sample_rate: int, delay_samples: int = 3300, drift_ppm: float = 50.0,
                 rir: np.ndarray | None = None, noise_db: float = -50.0, seed: int = 0) -> None:
        self.sample_rate = int(sample_rate)
        self.delay_samples = int(delay_samples)
        self.noise_db = float(noise_db)
        self.rng = np.random.default_rng(seed)
        self.rir = (synthetic_rir(sample_rate, self.rng) if rir is None
                    else np.asarray(rir, dtype=np.float64))
        ratio = Fraction(1.0 + drift_ppm * 1e-6).limit_denominator(_MAX_DENOMINATOR)
        self.up, self.down = ratio.numerator, ratio.denominator
        self.ratio = self.up / self.down                  # mic samples per output sample
        self.drift_ppm = (self.ratio - 1.0) * 1e6         # what is actually simulated
        half = 10 * max(self.up, self.down)
        self._fir = firwin(2 * half + 1, 1.0 / max(self.up, self.down), window=("kaiser", 5.0))
        self._out = np.zeros(0)                           # retained output history
        self._out_start = 0                               # output index of _out[0]
        self._n = 0                                       # mic samples rendered so far
        self._dry_tail = np.zeros(0)                      # last dry mic-domain samples, for reverb

    def _output_range(self, s: int, e: int) -> np.ndarray:
        """Output samples [s, e) from history, zeros where there are none."""
        seg = np.zeros(e - s)
        a, b = max(s, self._out_start), min(e, self._out_start + len(self._out))
        if a < b:
            seg[a - s:b - s] = self._out[a - self._out_start:b - self._out_start]
        return seg

    def _warp(self, m0: int, n: int) -> np.ndarray:
        """Delayed, clock-drifted output at mic samples [m0, m0 + n)."""
        # mic sample m hears output position m / ratio - delay. resample_poly gives
        # y[j] = x[s + j / ratio]; that lands on whole mic samples when (s + delay)
        # is a multiple of `down`, so pick such an s just before the span we need.
        q = int(np.floor((m0 / self.ratio - _PAD) / self.down))
        s = q * self.down - self.delay_samples
        j0 = m0 - q * self.up
        e = int(np.ceil((m0 + n) / self.ratio)) - self.delay_samples + _PAD
        y = resample_poly(self._output_range(s, e), self.up, self.down, window=self._fir)
        return y[j0:j0 + n]

    def render(self, output: np.ndarray, room_sound: np.ndarray | None = None) -> np.ndarray:
        """Mic samples (float32) for the next len(output) samples of the stream."""
        out = np.asarray(output, dtype=np.float64)
        n = len(out)
        self._out = np.concatenate([self._out, out])
        dry = self._warp(self._n, n)

        k = len(self.rir) - 1
        tail = self._dry_tail[-k:] if k else np.zeros(0)
        tail = np.concatenate([np.zeros(k - len(tail)), tail])
        buf = np.concatenate([tail, dry])
        mic = fftconvolve(buf, self.rir)[k:k + n]
        self._dry_tail = buf[len(buf) - k:] if k else np.zeros(0)

        if room_sound is not None:
            mic += np.asarray(room_sound, dtype=np.float64)[:n]
        if np.isfinite(self.noise_db):
            mic += self.rng.standard_normal(n) * 10.0 ** (self.noise_db / 20.0)

        self._n += n
        keep_from = int(np.floor(self._n / self.ratio)) - self.delay_samples - self.down - 2 * _PAD
        drop = keep_from - self._out_start
        if drop > 0:
            self._out = self._out[drop:]
            self._out_start += drop
        return mic.astype(np.float32)


def _sin_cos(freq: float, phase: float, t0: float, n: int,
             sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
    """float32 sin and cos of 2*pi*freq*t + phase on the sample grid from t0.

    Angle addition: trig runs once per 4096-sample chunk start (in float64, so the
    phase stays exact over long runs) and once for the in-chunk offsets; everything
    else is float32 multiply-add, several times faster than np.sin over the block.
    """
    j = 4096
    w = 2 * np.pi * freq / sample_rate
    starts = phase + 2 * np.pi * freq * t0 + w * j * np.arange(-(-n // j))
    sj = np.sin(w * np.arange(j)).astype(np.float32)
    cj = np.cos(w * np.arange(j)).astype(np.float32)
    s0 = np.sin(starts).astype(np.float32)[:, None]
    c0 = np.cos(starts).astype(np.float32)[:, None]
    return (s0 * cj + c0 * sj).ravel()[:n], (c0 * cj - s0 * sj).ravel()[:n]


def pad_like(seconds: float, sample_rate: int, t0: float = 0.0, rms_db: float = -20.0) -> np.ndarray:
    """Ambient-pad stand-in: a few sine tones (with two soft harmonics) under slow swells.

    Deterministic and continuous in time: pad_like(2, sr) equals pad_like(1, sr)
    followed by pad_like(1, sr, t0=1). The level is set analytically, so it is the
    same from block to block rather than renormalised per call.
    """
    # (frequency Hz, amplitude, swell period s, swell phase)
    tones = [(110.0, 1.0, 17.0, 0.0), (164.81, 0.8, 11.0, 1.3), (220.0, 0.7, 13.0, 2.1),
             (277.18, 0.5, 19.0, 0.4), (329.63, 0.5, 7.0, 2.8), (493.88, 0.3, 23.0, 1.7),
             (659.26, 0.2, 9.0, 0.9)]
    h2, h3 = 0.3, 0.12                                   # second and third harmonic levels
    n = int(round(seconds * sample_rate))
    x = np.zeros(n, dtype=np.float32)
    power = 0.0
    for f, a, period, ph in tones:
        s, c = _sin_cos(f, ph, t0, n, sample_rate)
        swell = _sin_cos(1.0 / period, ph, t0, n, sample_rate)[0]
        # tone with harmonics: sin u + 2*h2 sin u cos u + h3 sin u (3 - 4 sin^2 u)
        v = s * s
        v *= -4.0 * h3
        c *= 2.0 * h2
        v += c
        v += 1.0 + 3.0 * h3
        v *= s
        swell *= 0.45 * a
        swell += 0.55 * a
        v *= swell
        x += v
        power += a * a * (0.55 ** 2 + 0.45 ** 2 / 2) * (1.0 + h2 * h2 + h3 * h3) / 2
    x *= np.float32(10.0 ** (rms_db / 20.0) / np.sqrt(power))
    return x


def speech_like(seconds: float, sample_rate: int, rng: np.random.Generator,
                rms_db: float = -20.0, syllable_hz: float = 4.0,
                jitter: float = 0.0) -> np.ndarray:
    """Talk stand-in: 300-3000 Hz noise in syllable bursts, grouped into phrases with pauses.

    Syllables come at `syllable_hz` (4 per second by default). With `jitter` > 0
    the rate wanders smoothly by up to that fraction either way (new target every
    0.4 s), as a real talker speeds up and slows down.
    """
    n = int(round(seconds * sample_rate))
    sos = butter(8, [300.0, 3000.0], btype="band", fs=sample_rate, output="sos")
    noise = sosfilt(sos, rng.standard_normal(n + sample_rate))[sample_rate:]
    phase0 = rng.uniform(0, 2 * np.pi)
    if jitter > 0.0:
        knots = rng.uniform(-1.0, 1.0, int(seconds / 0.4) + 2)
        wander = np.interp(np.arange(n) / (0.4 * sample_rate), np.arange(len(knots)), knots)
        angle = phase0 + 2 * np.pi * np.cumsum(syllable_hz * (1.0 + jitter * wander)) / sample_rate
    else:
        angle = phase0 + 2 * np.pi * syllable_hz * np.arange(n) / sample_rate
    syllables = np.maximum(0.0, np.sin(angle)) ** 1.5
    gate = np.zeros(n)
    pos, talking = 0, bool(rng.integers(2))
    while pos < n:
        dur = int(sample_rate * (rng.uniform(0.8, 2.5) if talking else rng.uniform(0.3, 1.0)))
        gate[pos:pos + dur] = 1.0 if talking else 0.0
        pos, talking = pos + dur, not talking
    ramp = int(0.03 * sample_rate)
    gate = np.convolve(gate, np.ones(ramp) / ramp, mode="same")
    return _rms_scale(noise * (0.15 + 0.85 * syllables) * gate, rms_db).astype(np.float32)


_CLICK_TAU_S = (0.002, 0.015)
# mean energy of one click per sample of decay time: E[10 ** (2u)] for u uniform
# on (-1, 0), times the mean decay constant, times the 1/2 from integrating exp(-2t/tau)
_CLICK_ENERGY_S = (1.0 - 0.01) / (2.0 * np.log(10.0)) * np.mean(_CLICK_TAU_S) / 2.0


def clatter(seconds: float, sample_rate: int, rng: np.random.Generator, rate_hz: float,
            rms_db: float = -20.0, normalise: bool = True) -> np.ndarray:
    """Kitchen-clatter stand-in: clicks at random times (about rate_hz per second),
    each a burst of broadband noise dying away in a few milliseconds.

    By default the result is scaled to exactly `rms_db`. With `normalise=False`
    each click keeps a fixed scale chosen so that rms_db is the long-run level
    at rate_hz, which keeps sparse clatter cut into short pieces from making
    one lone click as loud as a whole stretch of them.
    """
    n = int(round(seconds * sample_rate))
    x = np.zeros(n)
    for start in rng.integers(0, n, rng.poisson(rate_hz * seconds)):
        tau = rng.uniform(*_CLICK_TAU_S)
        k = min(int(6 * tau * sample_rate), n - start)
        env = np.exp(-np.arange(k) / (tau * sample_rate))
        x[start:start + k] += 10.0 ** rng.uniform(-1.0, 0.0) * env * rng.standard_normal(k)
    if normalise:
        return _rms_scale(x, rms_db).astype(np.float32)
    gain = 10.0 ** (rms_db / 20.0) / np.sqrt(max(rate_hz, 1e-9) * _CLICK_ENERGY_S)
    return (gain * x).astype(np.float32)
