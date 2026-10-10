import numpy as np
import pytest

from roomtone import marker as mk
from roomtone.config import IO_FRAMES, MarkerParams

FS = 44_100


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def out_of_band_db(x: np.ndarray, lo: float, hi: float) -> float:
    p = np.abs(np.fft.rfft(np.asarray(x, dtype=np.float64))) ** 2
    f = np.fft.rfftfreq(len(x), 1.0 / FS)
    return float(10 * np.log10(p[(f < lo) | (f > hi)].sum() / p.sum()))


def test_deterministic_from_params():
    p = MarkerParams()
    a = mk.generate(p, FS).copy()
    mk._cache.clear()
    b = mk.generate(p, FS)
    assert np.array_equal(a, b)
    other = mk.generate(MarkerParams(seed=p.seed + 1), FS)
    assert abs(np.dot(a, other)) / len(a) < 0.02      # a different seed is a different sequence


def test_cached_and_read_only():
    p = MarkerParams()
    a = mk.generate(p, FS)
    assert mk.generate(MarkerParams(level_db=-20.0, period_s=10.0), FS) is a   # level/period do not reshape it
    with pytest.raises(ValueError):
        a[0] = 1.0


@pytest.mark.parametrize("p", [MarkerParams(), MarkerParams(length_s=30.0, band_lo_hz=6000.0, band_hi_hz=12000.0)])
def test_length_rms_dtype_and_band(p):
    x = mk.generate(p, FS)
    assert x.dtype == np.float32
    assert len(x) == round(p.length_s * FS)
    assert rms(x) == pytest.approx(1.0, abs=1e-4)
    assert out_of_band_db(x, p.band_lo_hz, p.band_hi_hz) < -30.0


def test_autocorrelation_is_a_single_sharp_peak():
    x = mk.generate(MarkerParams(), FS).astype(np.float64)
    n = len(x)
    c = np.abs(np.fft.irfft(np.abs(np.fft.rfft(x, 2 * n)) ** 2, 2 * n))
    side = np.concatenate([c[n // 50:n], c[n:2 * n - n // 50]]).max()
    assert 20 * np.log10(c[0] / side) > 30.0


def test_scale_to_music_level_and_floor():
    p = MarkerParams(level_db=-30.0)
    x = mk.generate(p, FS)
    assert rms(mk.scale_to_music(x, 0.1, p)) == pytest.approx(0.1 * 10 ** (-30 / 20), rel=1e-4)
    assert 20 * np.log10(rms(mk.scale_to_music(x, 0.0, p))) == pytest.approx(-60.0, abs=1e-3)


def test_scheduler_emits_on_period_across_any_block_size():
    p = MarkerParams(length_s=1.0, period_s=2.0)
    whole = mk.MarkerScheduler(p, FS).block(0, 7 * FS, 0.1)

    s = mk.MarkerScheduler(p, FS)
    chunks, start = [], 0
    for size in [IO_FRAMES, 333, 5000, 1, 70_000] * 1000:
        if start >= 7 * FS:
            break
        size = min(size, 7 * FS - start)
        chunks.append(s.block(start, size, 0.1))
        start += size
    assert np.array_equal(np.concatenate(chunks), whole)
    assert s.emissions == [0, 2 * FS, 4 * FS, 6 * FS]
    expected = mk.scale_to_music(mk.generate(p, FS), 0.1, p)
    assert np.array_equal(whole[2 * FS:3 * FS], expected)
    assert not whole[FS:2 * FS].any()


def test_set_params_waits_for_next_emission():
    p = MarkerParams(length_s=1.0, period_s=2.0, level_db=-30.0)
    louder = MarkerParams(length_s=1.0, period_s=2.0, level_db=-20.0)
    s = mk.MarkerScheduler(p, FS)
    head = s.block(0, FS // 2, 0.1)
    s.set_params(louder)
    tail = s.block(FS // 2, 4 * FS, 0.1)
    out = np.concatenate([head, tail])
    assert s.emitted_params == [p, louder, louder]
    first = mk.scale_to_music(mk.generate(p, FS), 0.1, p)
    assert np.array_equal(out[:FS], first)                     # the burst in progress is untouched
    assert rms(out[2 * FS:3 * FS]) == pytest.approx(0.1 * 10 ** (-20 / 20), rel=1e-4)
