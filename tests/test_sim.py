import numpy as np
import pytest

from roomtone.sim import SimRoom, clatter, pad_like, speech_like, synthetic_rir

FS = 44_100


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2))))


def band_share(x: np.ndarray, lo: float, hi: float) -> float:
    p = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    return float(p[(f >= lo) & (f <= hi)].sum() / p.sum())


def test_synthetic_rir_shape_and_rt60(rng):
    h = synthetic_rir(FS, rng)
    assert len(h) == round(0.3 * FS)
    assert np.sum(h * h) == pytest.approx(1.0)
    assert int(np.argmax(np.abs(h))) == round(0.002 * FS)       # direct sound is the strongest tap
    # Schroeder backward integration, decay from -5 to -25 dB, extrapolated to -60 dB
    edc = 10 * np.log10(np.cumsum((h * h)[::-1])[::-1] / np.sum(h * h))
    t5, t25 = np.argmax(edc < -5) / FS, np.argmax(edc < -25) / FS
    assert 3 * (t25 - t5) == pytest.approx(0.4, rel=0.25)


def test_delay_and_drift_put_each_sample_where_expected():
    room = SimRoom(FS, delay_samples=3300, drift_ppm=50.0, rir=np.array([1.0]), noise_db=-np.inf)
    seconds = 41
    out = np.zeros(seconds * FS)
    k = np.arange(-30, 31)
    places = [FS, 20 * FS, 40 * FS]
    for p in places:
        out[p + k] = np.exp(-0.5 * (k / 3.0) ** 2)               # smooth pulse, well below Nyquist
    mic = np.concatenate([room.render(out[i:i + FS]) for i in range(0, len(out), FS)]).astype(np.float64)
    for p in places:
        expected = (p + 3300) * room.ratio
        c = int(round(expected))
        w = mic[c - 40:c + 41]
        centroid = c - 40 + np.sum(np.arange(len(w)) * w) / np.sum(w)
        assert centroid == pytest.approx(expected, abs=0.05)


def test_streaming_in_odd_chunks_matches_one_call():
    out = pad_like(3.0, FS)
    one = SimRoom(FS, noise_db=-np.inf, seed=3).render(out)
    room = SimRoom(FS, noise_db=-np.inf, seed=3)
    parts, i = [], 0
    for size in [1024, 777, 30_000, 5, 50_000] * 20:
        if i >= len(out):
            break
        parts.append(room.render(out[i:i + size]))
        i += size
    assert np.allclose(np.concatenate(parts), one, atol=1e-6)


def test_noise_and_room_sound():
    room = SimRoom(FS, rir=np.array([1.0]), noise_db=-50.0)
    assert rms_db(room.render(np.zeros(2 * FS))) == pytest.approx(-50.0, abs=0.2)
    quiet = SimRoom(FS, rir=np.array([1.0]), noise_db=-np.inf)
    s = np.random.default_rng(0).standard_normal(FS).astype(np.float32)
    assert np.array_equal(quiet.render(np.zeros(FS), s), s)


def test_pad_like_level_continuity_and_band():
    x = pad_like(30.0, FS)
    assert x.dtype == np.float32
    assert rms_db(x) == pytest.approx(-20.0, abs=1.0)
    joined = np.concatenate([pad_like(1.0, FS), pad_like(1.0, FS, t0=1.0)])
    assert np.allclose(joined, pad_like(2.0, FS), atol=1e-5)
    assert band_share(x, 0, 2500) > 0.999                       # nothing up in the marker band


def test_speech_like_band_rhythm_and_pauses(rng):
    x = speech_like(20.0, FS, rng, rms_db=-30.0)
    assert rms_db(x) == pytest.approx(-30.0, abs=0.01)
    assert band_share(x, 300, 3000) > 0.95
    frame = FS // 100
    env = np.sqrt(np.mean(x[:len(x) // frame * frame].astype(np.float64).reshape(-1, frame) ** 2, axis=1))
    spec = np.abs(np.fft.rfft(env - env.mean()))
    f = np.fft.rfftfreq(len(env), 0.01)
    assert 3.0 <= f[np.argmax(spec[f > 1.0]) + np.sum(f <= 1.0)] <= 5.0   # syllable rate near 4 Hz
    assert np.mean(env < env.max() * 1e-2) > 0.1                          # real pauses between phrases


def test_clatter_is_sparse_and_broadband(rng):
    x = clatter(20.0, FS, rng, rate_hz=2.0).astype(np.float64)
    assert rms_db(x) == pytest.approx(-20.0, abs=0.01)
    assert band_share(x, 5000, FS / 2) > 0.3
    frame = FS // 100
    env = np.sqrt(np.mean(x[:len(x) // frame * frame].reshape(-1, frame) ** 2, axis=1))
    assert np.mean(env > env.max() * 1e-3) < 0.3                         # mostly silence
