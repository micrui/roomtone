"""A scripted evening in the simulated kitchen, for `roomtone run --sim`.

`KitchenScript` produces the sound the room itself makes (everything except our
own music), following a list of phases such as
[(150, "empty"), (180, "cooking"), ...] that repeats when it runs out. Modes:

empty
    The noise floor only: white noise at `noise_db`.
cooking
    Clatter (sharp clicks, about 2.5 per second) at -32 dBFS and a faint 60 Hz
    fridge hum with two harmonics at -46 dBFS, over the floor. Clatter levels
    are long-run averages: each click has the same scale whatever the chunk.
dinner
    Three or four overlapping talkers, -28 dBFS together, each at its own
    syllable rate (spread over 3.2 to 5.8 per second, every one wandering by
    15%) so the table never reads as a beat; occasional clatter, 0.3 per second
    at -40 dBFS.
party
    Eight talkers at -18 dBFS together, and a kick drum at 120 BPM: a 60 Hz sine
    dying away with a 150 ms time constant, every 0.5 s, at -22 dBFS.

Sound is made in chunks of at most 5 s that never straddle a phase boundary,
and handed out in whatever sizes `next` is asked for. Random textures (noise,
talk, clatter) are rendered 20 ms long and the overlap is equal-power
crossfaded into the next chunk, so chunk boundaries and phase changes do not
click. The hum and the kick are functions of absolute time, so within a phase
they continue exactly across chunks (their overlap is crossfaded linearly,
which leaves an identical signal unchanged and fades it at a phase change).

`roomtone.sim` (and so scipy) is imported when a script is built, not when this
module is, so the CLI can parse `--script` without it.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

log = logging.getLogger(__name__)

MODES = ("empty", "cooking", "dinner", "party")
DEFAULT_SCRIPT: list[tuple[float, str]] = [
    (150.0, "empty"), (180.0, "cooking"), (180.0, "dinner"), (150.0, "party"), (120.0, "empty")]

CHUNK_SECONDS = 5.0
XFADE_SECONDS = 0.02

COOKING_CLATTER_DB, COOKING_CLATTER_HZ = -32.0, 2.5
HUM_DB, HUM_HZ = -46.0, 60.0
HUM_HARMONICS = (1.0, 0.5, 0.25)               # relative amplitudes at 60, 120, 180 Hz
DINNER_TALK_DB, DINNER_TALKERS = -28.0, (3, 4)
DINNER_CLATTER_DB, DINNER_CLATTER_HZ = -40.0, 0.3
PARTY_TALK_DB, PARTY_TALKERS = -18.0, 8
SYLLABLE_RATES_HZ = (3.2, 5.8)                 # talkers' base rates are spread over this range
SYLLABLE_JITTER = 0.15
KICK_DB, KICK_HZ, KICK_PERIOD_S, KICK_TAU_S = -22.0, 60.0, 0.5, 0.15
KICK_LENGTH_S = 7 * KICK_TAU_S                 # rendered until about -60 dB


def parse_script(text: str) -> list[tuple[float, str]]:
    """'empty:150,cooking:180' -> [(150.0, 'empty'), (180.0, 'cooking')]."""
    phases = []
    for part in text.split(","):
        mode, sep, seconds = part.strip().partition(":")
        mode = mode.strip().lower()
        try:
            length = float(seconds)
        except ValueError:
            length = -1.0
        if not sep or mode not in MODES or length <= 0:
            raise ValueError(f"bad script phase {part.strip()!r}: want MODE:SECONDS with MODE "
                             f"one of {', '.join(MODES)} and SECONDS > 0")
        phases.append((length, mode))
    return phases


def format_script(phases: Sequence[tuple[float, str]]) -> str:
    return ",".join(f"{mode}:{seconds:g}" for seconds, mode in phases)


class KitchenScript:
    """Room sound for a repeating list of (seconds, mode) phases. See the module docstring."""

    def __init__(self, phases: Sequence[tuple[float, str]], sample_rate: int,
                 noise_db: float = -60.0, seed: int = 0) -> None:
        from roomtone.sim import clatter, speech_like  # needs scipy
        self._clatter, self._speech = clatter, speech_like
        if not phases or any(m not in MODES or s <= 0 for s, m in phases):
            raise ValueError(f"bad kitchen script {phases!r}")
        self.phases = [(float(s), m) for s, m in phases]
        self.sample_rate = int(sample_rate)
        self.noise = 10.0 ** (noise_db / 20.0)
        self.rng = np.random.default_rng(seed)
        ends = np.cumsum([round(s * self.sample_rate) for s, _ in self.phases])
        self._ends = [int(e) for e in ends]
        self._period = self._ends[-1]
        lo, hi = SYLLABLE_RATES_HZ
        self._rates = self.rng.permutation(np.linspace(lo, hi, PARTY_TALKERS))
        self._xfade = max(int(round(XFADE_SECONDS * self.sample_rate)), 1)
        self._chunk = int(round(CHUNK_SECONDS * self.sample_rate))
        ramp = (np.arange(self._xfade) + 0.5) / self._xfade
        self._fade_in = ramp
        self._fade_in_power = np.sin(0.5 * np.pi * ramp)
        self._fade_out_power = np.cos(0.5 * np.pi * ramp)
        self._kick_shape = self._make_kick()
        self._rendered = 0                                   # absolute samples rendered
        self._buf = np.zeros(0, dtype=np.float32)            # rendered, not yet handed out
        self._tail_random = np.zeros(self._xfade)            # overlap carried into the next chunk
        self._tail_fixed = np.zeros(self._xfade)
        log.info("kitchen script: %s, repeating every %.0f s",
                 format_script(self.phases), self._period / self.sample_rate)

    # timeline ------------------------------------------------------------------

    def _locate(self, sample: int) -> tuple[str, int, int]:
        """(mode, absolute start, absolute end) of the phase holding `sample`."""
        cycle, offset = divmod(sample, self._period)
        i = next(k for k, e in enumerate(self._ends) if offset < e)
        base = cycle * self._period
        start = base + (self._ends[i - 1] if i else 0)
        return self.phases[i][1], start, base + self._ends[i]

    def phase_at(self, t_seconds: float) -> str:
        """The mode the script is in `t_seconds` after it started."""
        return self._locate(max(int(t_seconds * self.sample_rate), 0))[0]

    # sound ---------------------------------------------------------------------

    def next(self, n_frames: int) -> np.ndarray:
        """The next n_frames of room sound, float32."""
        while len(self._buf) < n_frames:
            self._buf = np.concatenate([self._buf, self._render_chunk()])
        out, self._buf = self._buf[:n_frames], self._buf[n_frames:]
        return out

    def _render_chunk(self) -> np.ndarray:
        mode, phase_start, phase_end = self._locate(self._rendered)
        n = min(self._chunk, phase_end - self._rendered)
        m = n + self._xfade
        random = self._random(mode, m)
        fixed = self._fixed(mode, self._rendered, m, phase_start, phase_end)
        x = self._xfade
        random[:x] = self._tail_random * self._fade_out_power + random[:x] * self._fade_in_power
        fixed[:x] = self._tail_fixed * (1.0 - self._fade_in) + fixed[:x] * self._fade_in
        self._tail_random, self._tail_fixed = random[n:], fixed[n:]
        self._rendered += n
        return (random[:n] + fixed[:n]).astype(np.float32)

    def _random(self, mode: str, m: int) -> np.ndarray:
        sr, rng = self.sample_rate, self.rng
        seconds = m / sr
        x = self.noise * rng.standard_normal(m)
        if mode == "cooking":
            x += self._clatter(seconds, sr, rng, COOKING_CLATTER_HZ, COOKING_CLATTER_DB,
                                normalise=False)[:m]
        elif mode == "dinner":
            x += self._talk(seconds, int(rng.integers(DINNER_TALKERS[0], DINNER_TALKERS[1] + 1)),
                            DINNER_TALK_DB)[:m]
            x += self._clatter(seconds, sr, rng, DINNER_CLATTER_HZ, DINNER_CLATTER_DB,
                                normalise=False)[:m]
        elif mode == "party":
            x += self._talk(seconds, PARTY_TALKERS, PARTY_TALK_DB)[:m]
        return x

    def _talk(self, seconds: float, talkers: int, total_db: float) -> np.ndarray:
        each_db = total_db - 10.0 * np.log10(talkers)
        voices = [self._speech(seconds, self.sample_rate, self.rng, rms_db=each_db,
                               syllable_hz=float(rate), jitter=SYLLABLE_JITTER)
                  for rate in self.rng.choice(self._rates, talkers, replace=False)]
        return np.sum(voices, axis=0, dtype=np.float64)

    def _fixed(self, mode: str, start: int, m: int, phase_start: int,
               phase_end: int) -> np.ndarray:
        """Hum or kick for absolute samples [start, start + m); kicks start inside the phase."""
        sr = self.sample_rate
        if mode == "cooking":
            t = (start + np.arange(m)) / sr
            hum = sum(a * np.sin(2 * np.pi * HUM_HZ * (k + 1) * t)
                      for k, a in enumerate(HUM_HARMONICS))
            power = sum(a * a for a in HUM_HARMONICS) / 2
            return hum * (10.0 ** (HUM_DB / 20.0) / np.sqrt(power))
        out = np.zeros(m)
        if mode == "party":
            period, k = round(KICK_PERIOD_S * sr), len(self._kick_shape)
            first = phase_start + max(-(-(start - k - phase_start) // period), 0) * period
            for hit in range(first, min(start + m, phase_end), period):
                a, b = max(hit, start), min(hit + k, start + m)
                if a < b:
                    out[a - start:b - start] += self._kick_shape[a - hit:b - hit]
        return out

    def _make_kick(self) -> np.ndarray:
        """One kick, scaled so a steady 120 BPM train of them sits at KICK_DB."""
        sr = self.sample_rate
        t = np.arange(int(round(KICK_LENGTH_S * sr))) / sr
        kick = np.sin(2 * np.pi * KICK_HZ * t) * np.exp(-t / KICK_TAU_S)
        period = round(KICK_PERIOD_S * sr)
        train = np.zeros(period * 8 + len(kick))
        for i in range(8):
            train[i * period:i * period + len(kick)] += kick
        settled = -(-len(kick) // period) * period                # whole periods, tails settled
        steady = train[settled:period * 8]
        return kick * (10.0 ** (KICK_DB / 20.0) / np.sqrt(np.mean(steady ** 2)))
