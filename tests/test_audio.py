import sys

import numpy as np
import pytest

from roomtone.audio import (AudioIO, FakeBackend, PadSynth, RingBuffer, RingRangeError,
                            SimBackend, click_train, measure_click_delays)
from roomtone.config import Config


# RingBuffer ------------------------------------------------------------------

def test_ring_wraps_and_reads_back_exactly(rng):
    ring = RingBuffer(2, 100)
    written = []
    for n in [30, 45, 50, 7, 61]:  # crosses the end of the store several times
        block = rng.standard_normal((n, 2)).astype(np.float32)
        ring.write(block)
        written.append(block)
    allw = np.concatenate(written)
    assert ring.position == len(allw) == 193
    np.testing.assert_array_equal(ring.read(100), allw[-100:])
    for at in [93, 100, 150, 170]:
        np.testing.assert_array_equal(ring.read(20, at=at), allw[at:at + 20])


def test_ring_rejects_unwritten_and_overwritten():
    ring = RingBuffer(1, 10)
    ring.write(np.arange(25, dtype=np.float32))
    with pytest.raises(RingRangeError):
        ring.read(5, at=22)      # runs past what is written
    with pytest.raises(RingRangeError):
        ring.read(5, at=10)      # oldest held is 15
    np.testing.assert_array_equal(ring.read(10, at=15)[:, 0], np.arange(15, 25))


def test_ring_oversized_write_keeps_newest():
    ring = RingBuffer(1, 8)
    ring.write(np.arange(20, dtype=np.float32))
    assert ring.position == 20 and ring.oldest == 12
    np.testing.assert_array_equal(ring.read(8)[:, 0], np.arange(12, 20))


def test_ring_rejects_wrong_channel_count():
    with pytest.raises(ValueError):
        RingBuffer(2, 8).write(np.zeros((4, 3), dtype=np.float32))


# AudioIO ---------------------------------------------------------------------

def small_cfg(**kw) -> Config:
    base = dict(sample_rate=8000, block_seconds=1.0, io_frames=256)
    base.update(kw)
    return Config(**base)


def test_callback_mixes_marker_and_stamps_rings(rng):
    calls = []

    def marker(n, music_rms):
        calls.append((n, music_rms))
        return np.full(n, 0.01, dtype=np.float32)

    backend = FakeBackend()
    io = AudioIO(small_cfg(), marker, backend=backend, m8_channels=(1, 3), mic_channel=0)
    io.start()
    assert backend.running and backend.in_channels == 4
    m8 = (0.1 * rng.standard_normal((256, 2))).astype(np.float32)
    mic = (0.05 * rng.standard_normal(256)).astype(np.float32)
    out = backend.pump(256, m8, mic)

    mono = np.repeat(m8.mean(axis=1, keepdims=True), 2, axis=1)  # folded: one signal, both speakers
    np.testing.assert_allclose(out, mono + 0.01, atol=1e-7)
    assert io.reference_ring.position == io.mic_ring.position == io.position == 256
    np.testing.assert_array_equal(io.reference_ring.read(256, at=0), out)
    np.testing.assert_array_equal(io.mic_ring.read(256, at=0)[:, 0], mic)
    assert calls[0][0] == 256 and calls[0][1] > 0

    for _ in range(5):
        backend.pump(256, m8, mic)
    assert io.reference_ring.position == io.mic_ring.position == 6 * 256


def test_stereo_output_when_mono_fold_is_off(rng):
    backend = FakeBackend()
    cfg = small_cfg()
    cfg.mono_output = False
    io = AudioIO(cfg, lambda n, rms: np.zeros(n, np.float32), backend=backend)
    io.start()
    m8 = (0.1 * rng.standard_normal((256, 2))).astype(np.float32)
    np.testing.assert_array_equal(backend.pump(256, m8), m8)


def test_music_rms_tracks_last_second():
    backend = FakeBackend()
    seen = []
    io = AudioIO(small_cfg(), lambda n, rms: (seen.append(rms), np.zeros(n, np.float32))[1],
                 backend=backend)
    io.open()
    m8 = np.full((256, 2), 0.2, dtype=np.float32)
    for _ in range(40):  # a bit over a second at 8 kHz
        backend.pump(256, m8)
    assert seen[-1] == pytest.approx(0.2, rel=1e-3)
    assert seen[0] < seen[-1]  # ramps up while the one-second window fills


def test_mute_music_and_clip():
    backend = FakeBackend()
    io = AudioIO(small_cfg(), lambda n, rms: np.full(n, 2.0, np.float32), backend=backend,
                 mute_music=True)
    io.open()
    out = backend.pump(256, np.full((256, 2), 0.5, np.float32))
    np.testing.assert_array_equal(out, np.ones((256, 2), np.float32))


def test_callback_without_marker_passes_music_through():
    backend = FakeBackend()
    io = AudioIO(small_cfg(), None, backend=backend)
    io.open()
    m8 = np.full((256, 2), 0.25, np.float32)
    np.testing.assert_array_equal(backend.pump(256, m8), m8)


# SimBackend ------------------------------------------------------------------

def test_sim_backend_fallback_room_delays_output(monkeypatch):
    monkeypatch.setitem(sys.modules, "roomtone.sim", None)  # force the built-in fallback
    cfg = small_cfg()
    backend = SimBackend(cfg, delay_samples=300, realtime=False)
    noise = np.random.default_rng(5)
    io = AudioIO(cfg, lambda n, rms: (0.1 * noise.standard_normal(n)).astype(np.float32),
                 backend=backend, mute_music=True)  # white noise: one sharp correlation peak
    io.open()
    backend.run_frames(8000)
    ref = io.reference_ring.read(6000, at=1000).mean(axis=1)
    mic = io.mic_ring.read(6000, at=1000)[:, 0]
    assert np.abs(ref).max() > 0.01
    corr = np.correlate(mic, ref[1000:5000], mode="valid")
    # room delay plus one callback of input latency
    assert int(np.argmax(corr)) - 1000 == 300 + cfg.io_frames


def test_sim_backend_threaded_runs_and_stops(monkeypatch):
    monkeypatch.setitem(sys.modules, "roomtone.sim", None)
    cfg = small_cfg()
    backend = SimBackend(cfg, realtime=False, max_frames=4096)
    io = AudioIO(cfg, None, backend=backend)
    io.start()
    backend._thread.join(timeout=5)
    io.stop()
    assert io.position >= 4096


def test_sim_backend_plays_the_kitchen_script():
    cfg = small_cfg()
    backend = SimBackend(cfg, realtime=False, script=[(1.0, "empty"), (1.0, "party")])
    assert backend.script.phase_at(1.5) == "party"
    io = AudioIO(cfg, None, backend=backend, mute_music=True)
    io.open()
    backend.run_frames(16000)
    mic = io.mic_ring.read(16000, at=0)[:, 0].astype(np.float64)
    level = [20 * np.log10(np.sqrt(np.mean(m * m))) for m in (mic[1000:7000], mic[9000:15000])]
    assert level[0] < -45 and level[1] > -20


def test_sim_backend_without_sim_module_drops_the_script(monkeypatch):
    monkeypatch.setitem(sys.modules, "roomtone.sim", None)
    backend = SimBackend(small_cfg(), realtime=False, script=[(1.0, "party")])
    assert backend.script is None


def test_pad_is_quiet_and_continuous():
    pad = PadSynth(8000)
    a, b = pad.next(4000), pad.next(4000)
    whole = PadSynth(8000).next(8000)
    np.testing.assert_allclose(np.concatenate([a, b]), whole, atol=1e-6)
    assert -30 < 20 * np.log10(np.sqrt(np.mean(whole.astype(float) ** 2))) < -15


def test_click_delay_measurement():
    sr = 8000
    out, starts = click_train(sr, n_clicks=5, spacing_s=0.4)
    mic = np.zeros_like(out)
    mic[613:] = 0.2 * out[:-613]
    delays = measure_click_delays(mic, starts, sr, window_s=0.2)
    assert delays == [613.0] * 5
