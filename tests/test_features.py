"""Feature extraction on small synthetic signals built here (no roomtone.sim)."""

from __future__ import annotations

import importlib.util
import time

import numpy as np
import pytest
from scipy import signal

from roomtone.classify import Classifier, NullClassifier, YamnetClassifier, to_yamnet_input
from roomtone.features import extract

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
