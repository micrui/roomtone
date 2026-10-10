"""Stand-ins for the modules other parts of roomtone provide (marker, lock, features,
mood, sim), with the same interfaces, so the glue can be tested on its own."""

from __future__ import annotations

import sys
import types

import numpy as np

from roomtone.config import MarkerParams
from roomtone.types import Features, LockState, LockStatus, Mood, MoodName


class FakeScheduler:
    """Emits a block of ones `length_s` long at the start of every `period_s`."""

    def __init__(self, params: MarkerParams, sample_rate: int):
        self.params = params
        self.sample_rate = sample_rate
        self.t = 0
        self.emissions: list[int] = []
        self.set_calls: list[MarkerParams] = []

    def block(self, start: int, n: int, music_rms: float = 1.0) -> np.ndarray:
        """Same contract as MarkerScheduler.block: ones during a burst, scaled like fake_scale."""
        assert start == self.t, "blocks must be requested in order"
        period = int(self.params.period_s * self.sample_rate)
        length = int(self.params.length_s * self.sample_rate)
        idx = self.t + np.arange(n)
        phase = idx % period
        for s in idx[phase == 0]:
            self.emissions.append(int(s))
        self.t += n
        return fake_scale((phase < length).astype(np.float32), music_rms, self.params)

    def set_params(self, params: MarkerParams) -> None:
        self.params = params
        self.set_calls.append(params)


def fake_scale(block: np.ndarray, music_rms: float, params: MarkerParams) -> np.ndarray:
    return (block * music_rms * 10 ** (params.level_db / 20.0)).astype(np.float32)


class FakeLocker:
    def __init__(self, cfg=None):
        self.params = MarkerParams()
        self.rir = np.zeros(8)
        self.calls: list[tuple[np.ndarray, np.ndarray, int | None]] = []

    def process(self, reference, mic, emission_start, params=None):
        self.calls.append((reference, mic, emission_start))
        return mic * 0.5, LockState(status=LockStatus.LOCKED, delay_samples=12.0)

    def reset(self):
        self.calls.clear()


def fake_extract(residual, sample_rate, classifier=None) -> Features:
    rms = float(np.sqrt(np.mean(np.asarray(residual, dtype=np.float64) ** 2)))
    return Features(rms_db=20 * np.log10(max(rms, 1e-6)), speech=0.1)


class FakeMoodTracker:
    def __init__(self, cfg=None):
        self.updates = 0

    def update(self, features, lock) -> Mood:
        self.updates += 1
        return Mood(energy=0.5, voices=0.2, rhythm=0.4, name=MoodName.COOKING, confidence=0.5)


def install_fake_modules(monkeypatch) -> None:
    """Put fake roomtone.marker / lock / features / mood into sys.modules."""
    marker = types.ModuleType("roomtone.marker")
    marker.MarkerScheduler = FakeScheduler
    marker.scale_to_music = fake_scale
    lock = types.ModuleType("roomtone.lock")
    lock.Locker = FakeLocker
    lock.load_baseline = lambda path: np.load(path)
    lock.save_baseline = lambda path, rir: np.save(path, rir)
    features = types.ModuleType("roomtone.features")
    features.extract = fake_extract
    mood = types.ModuleType("roomtone.mood")
    mood.MoodTracker = FakeMoodTracker
    for name, mod in [("marker", marker), ("lock", lock), ("features", features), ("mood", mood)]:
        monkeypatch.setitem(sys.modules, f"roomtone.{name}", mod)
        monkeypatch.setattr(sys.modules["roomtone"], name, mod, raising=False)
