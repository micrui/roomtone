"""Feature extraction on small synthetic signals.

Most signals are built here; the crowd tests use roomtone.sim.speech_like for
its jittered talkers, the same ones the simulated kitchen uses.
"""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pytest
from scipy import signal

from roomtone.classify import Classifier, NullClassifier, YamnetClassifier, to_yamnet_input
from roomtone.features import extract
from roomtone.sim import speech_like as sim_talker

SR = 44_100
SECONDS = 10.0


def band_noise(rng: np.random.Generator, seconds: float, lo: float, hi: float) -> np.ndarray:
    """Unit-RMS noise confined to lo..hi Hz."""
    sos = signal.butter(6, [lo, hi], btype="bandpass", fs=SR, output="sos")
    y = signal.sosfilt(sos, rng.standard_normal(int(seconds * SR)))
    return y / np.sqrt(np.mean(y * y))


def syllables(seconds: float, rate_hz: float, phase: float = 0.0) -> np.ndarray:
    """Full on/off envelope at rate_hz: 0.5 * (1 + cos)."""
    t = np.arange(int(seconds * SR)) / SR
    return 0.5 * (1.0 + np.cos(2 * np.pi * rate_hz * t + phase))


def clicks(rng: np.random.Generator, seconds: float, times: np.ndarray) -> np.ndarray:
    """Short decaying noise bursts at the given times over a -60 dBFS noise floor."""
    x = 1e-3 * rng.standard_normal(int(seconds * SR))
    burst = 0.5 * rng.standard_normal(44) * np.exp(-np.arange(44) / 10.0)
    for t in times:
        i = int(t * SR)
        x[i : i + burst.size] += burst[: x.size - i]
    return x


def speech_like(rng: np.random.Generator, seconds: float = SECONDS) -> np.ndarray:
    return 0.05 * band_noise(rng, seconds, 300, 3000) * syllables(seconds, 4.0)


def test_silence():
    f = extract(np.zeros(int(SECONDS * SR)), SR)
    assert f.rms_db == pytest.approx(-90.0)
    assert f.speech == pytest.approx(0.0, abs=1e-6)
    assert f.talkers == 0.0
    assert f.transient_rate == 0.0
    assert f.beat == pytest.approx(0.0, abs=1e-6)
    assert f.labels == {}


def test_speech_like_scores_high(rng):
    f = extract(speech_like(rng), SR)
    assert f.speech > 0.6
    assert 1.0 <= f.talkers < 2.0
    assert 1200 < f.speech_band_centroid_hz < 2000


def test_steady_wide_noise_scores_lower(rng):
    talk = extract(speech_like(rng), SR)
    steady = extract(0.05 * band_noise(rng, SECONDS, 100, 8000), SR)
    assert steady.speech < talk.speech - 0.4
    assert steady.speech < 0.2


def test_many_talkers_smooth_the_envelope(rng):
    one = extract(speech_like(rng), SR)
    crowd = sum(band_noise(rng, SECONDS, 300, 3000)
                * syllables(SECONDS, rng.uniform(3.5, 5.5), rng.uniform(0, 2 * np.pi))
                for _ in range(6))
    many = extract(0.05 * crowd / np.sqrt(6), SR)
    assert many.speech > 0.3          # band stays full
    assert many.talkers > one.talkers + 1.5


def test_click_rate(rng):
    f = extract(clicks(rng, SECONDS, np.arange(0.1, SECONDS, 1 / 3)), SR)
    assert 2.0 <= f.transient_rate <= 4.0


def test_metronome_has_beat(rng):
    f = extract(clicks(rng, SECONDS, np.arange(0.1, SECONDS, 0.5)), SR)  # 120 BPM
    assert f.beat > 0.5
    assert 1.5 <= f.transient_rate <= 2.5


def test_random_clicks_have_no_beat(rng):
    times = np.sort(rng.uniform(0.0, SECONDS - 0.01, 30))
    f = extract(clicks(rng, SECONDS, times), SR)
    assert f.beat < 0.3
    assert f.transient_rate > 1.5


# Crowd and kick, as in the simulated kitchen's party phase ----------------------

TALKER_RATES = np.linspace(3.2, 5.8, 8)    # syllables per second, one per talker
TALKER_JITTER = 0.15


def crowd(rng: np.random.Generator, seconds: float, talkers: int = 8,
          total_db: float = -18.0) -> np.ndarray:
    """`talkers` jittered voices at their own syllable rates, total_db together."""
    each_db = total_db - 10.0 * np.log10(talkers)
    rates = rng.choice(TALKER_RATES, talkers, replace=False)
    return np.sum([sim_talker(seconds, SR, rng, rms_db=each_db, syllable_hz=float(r),
                              jitter=TALKER_JITTER) for r in rates], axis=0, dtype=np.float64)


def kick(seconds: float, period_s: float = 0.5, rms_db: float = -22.0) -> np.ndarray:
    """60 Hz sine dying away with a 150 ms time constant, every period_s, at rms_db."""
    t = np.arange(int(7 * 0.15 * SR)) / SR
    hit = np.sin(2 * np.pi * 60.0 * t) * np.exp(-t / 0.15)
    n, step = int(seconds * SR), int(round(period_s * SR))
    x = np.zeros(n + hit.size)
    for i in range(0, n, step):
        x[i : i + hit.size] += hit
    x = x[:n]
    return x * 10.0 ** (rms_db / 20.0) / np.sqrt(np.mean(x[hit.size :] ** 2))


def test_kick_under_crowd_has_beat(rng):
    f = extract(crowd(rng, SECONDS) + kick(SECONDS), SR)
    assert f.beat > 0.5


def test_crowd_alone_has_no_beat(rng):
    f = extract(crowd(rng, SECONDS), SR)
    assert f.beat < 0.3
    assert f.speech > 0.3


@pytest.mark.parametrize("rate", TALKER_RATES)
def test_jittered_talker_has_no_beat(rng, rate):
    f = extract(sim_talker(SECONDS, SR, rng, rms_db=-25.0, syllable_hz=float(rate),
                           jitter=TALKER_JITTER), SR)
    assert f.beat < 0.3


def test_small_table_has_no_beat(rng):
    assert extract(crowd(rng, SECONDS, talkers=3, total_db=-28.0), SR).beat < 0.3


@pytest.mark.parametrize("bpm", [60, 90, 180])
def test_kick_tempo_range(rng, bpm):
    quiet_crowd = crowd(rng, SECONDS, total_db=-30.0)
    assert extract(quiet_crowd + kick(SECONDS, 60.0 / bpm), SR).beat > 0.5


def test_steady_noise_has_no_onsets(rng):
    f = extract(0.05 * rng.standard_normal(int(SECONDS * SR)), SR)
    assert f.transient_rate == 0.0
    assert f.beat < 0.3


def test_thirty_seconds_is_fast(rng):
    x = speech_like(rng, 30.0) + clicks(rng, 30.0, np.arange(0.1, 30.0, 0.5))
    extract(x[: SR], SR)              # warm up FFT plans
    t0 = time.perf_counter()
    extract(x, SR)
    assert time.perf_counter() - t0 < 2.0   # typically about 0.15 s; loose for slow CI


class FakeClassifier:
    def labels(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        return {"Speech": 0.9, "Silence": 0.1}


def test_classifier_labels_pass_through(rng):
    clf = FakeClassifier()
    assert isinstance(clf, Classifier)
    assert extract(speech_like(rng, 3.0), SR, clf).labels == {"Speech": 0.9, "Silence": 0.1}
    assert extract(speech_like(rng, 3.0), SR, NullClassifier()).labels == {}


def test_yamnet_input_is_16k_mono_float32(rng):
    stereo = 0.1 * rng.standard_normal((SR * 2, 2))
    x = to_yamnet_input(stereo, SR)
    assert x.dtype == np.float32 and x.ndim == 1
    assert x.size == 32_000
    assert to_yamnet_input(np.zeros(100), SR).size == 15_600


@pytest.mark.skipif(importlib.util.find_spec("tflite_runtime") is not None
                    or importlib.util.find_spec("tensorflow") is not None,
                    reason="a TFLite runtime is installed")
def test_yamnet_without_runtime_names_the_extra():
    clf = YamnetClassifier("/nonexistent/yamnet.tflite")   # constructing is free
    with pytest.raises(RuntimeError, match=r"roomtone\[yamnet\]"):
        clf.labels(np.zeros(SR), SR)
