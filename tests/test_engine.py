import json

import numpy as np
import pytest

from roomtone.audio import AudioIO, FakeBackend
from roomtone.config import Config, MarkerParams
from roomtone.control import Controller
from roomtone.engine import Engine, MarkerSource, acquire_baseline, emission_in
from roomtone.log import BlockLog
from roomtone.types import BlockResult, LockState, LockStatus, MoodName
from tests.fakes import FakeLocker, FakeMoodTracker, FakeScheduler, fake_extract, fake_scale

SR, IO, BLOCK_S = 8000, 200, 0.5
BLOCK = int(SR * BLOCK_S)


class CapturePublisher:
    def __init__(self):
        self.results = []

    def publish(self, result):
        self.results.append(result)


class BrokenPublisher:
    def publish(self, result):
        raise ConnectionError("nobody listening")


def rig(tmp_path, publishers=None):
    cfg = Config(sample_rate=SR, block_seconds=BLOCK_S, io_frames=IO)
    params = MarkerParams(length_s=0.05, period_s=BLOCK_S, level_db=-20.0)
    scheduler = FakeScheduler(params, SR)
    source = MarkerSource(scheduler, params)
    backend = FakeBackend()
    io = AudioIO(cfg, source, backend=backend)
    io.open()
    locker = FakeLocker()
    pubs = publishers if publishers is not None else [CapturePublisher()]
    engine = Engine(cfg, io, locker, FakeMoodTracker(), Controller(cfg), pubs,
                    log=BlockLog(tmp_path), extract=fake_extract)
    return cfg, backend, io, locker, scheduler, engine, pubs


def pump(backend, frames, rng):
    for _ in range(frames // IO):
        backend.pump(IO, (0.1 * rng.standard_normal((IO, 2))).astype(np.float32),
                     (0.01 * rng.standard_normal(IO)).astype(np.float32))


def test_step_waits_for_a_whole_block(tmp_path, rng):
    _, backend, _, _, _, engine, pubs = rig(tmp_path)
    assert engine.step() is None
    pump(backend, BLOCK - IO, rng)
    assert engine.step() is None
    assert pubs[0].results == []


def test_step_produces_publishes_and_logs(tmp_path, rng):
    cfg, backend, io, locker, scheduler, engine, pubs = rig(tmp_path)
    pump(backend, BLOCK + 2 * IO, rng)
    result = engine.step()
    assert isinstance(result, BlockResult)
    assert result.index == 0 and result.t_start == 0.0
    assert result.lock.status == LockStatus.LOCKED
    assert result.mood.name == MoodName.COOKING
    assert result.residual_rms_db == pytest.approx(result.raw_rms_db - 6.02, abs=0.01)
    assert pubs[0].results == [result]

    reference, mic, emission_start = locker.calls[0]
    assert reference.shape == mic.shape == (BLOCK,)
    np.testing.assert_allclose(reference, io.reference_ring.read(BLOCK, at=0).mean(axis=1))
    assert emission_start == 0

    lines = engine.log.path.read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["index"] == 0
    assert engine.step() is None  # second block not complete yet


def test_blocks_align_to_marker_bursts(tmp_path, rng):
    _, backend, _, locker, scheduler, engine, _ = rig(tmp_path)
    pump(backend, 3 * BLOCK, rng)
    for _ in range(3):
        engine.step()
    # blocks are aligned to bursts: the Locker gets the absolute start of each
    assert [c[2] for c in locker.calls] == [0, BLOCK, 2 * BLOCK]
    assert scheduler.emissions[:3] == [0, BLOCK, 2 * BLOCK]
    assert emission_in([100, 4100, 8100], 4000, 8000) == 100
    assert emission_in([100], 4000, 8000) is None


def test_escalated_params_reach_the_scheduler(tmp_path, rng):
    _, backend, _, locker, scheduler, engine, _ = rig(tmp_path)
    pump(backend, 2 * BLOCK, rng)
    engine.step()
    assert scheduler.set_calls == [locker.params]  # locker default differs from rig params
    locker.params = locker.params.escalate()
    engine.step()
    assert scheduler.set_calls[-1] == locker.params and len(scheduler.set_calls) == 2


def test_failing_publisher_does_not_stop_the_loop(tmp_path, rng):
    good = CapturePublisher()
    _, backend, _, _, _, engine, _ = rig(tmp_path, [BrokenPublisher(), good])
    pump(backend, BLOCK, rng)
    assert engine.step() is not None and len(good.results) == 1


def test_engine_skips_ahead_when_behind(tmp_path, rng):
    cfg, backend, io, locker, _, engine, _ = rig(tmp_path)
    pump(backend, io.reference_ring.capacity + 3 * BLOCK, rng)
    result = engine.step()
    assert result is not None and result.t_start * SR >= io.reference_ring.oldest - BLOCK


def test_run_forever_in_a_thread(tmp_path, rng):
    _, backend, _, _, _, engine, pubs = rig(tmp_path)
    pump(backend, 2 * BLOCK, rng)
    thread = engine.start(max_blocks=2)
    thread.join(timeout=5)
    assert [r.index for r in pubs[0].results] == [0, 1]
    engine.stop()


def test_acquire_baseline_returns_rir(tmp_path, rng):
    cfg, backend, io, locker, scheduler, _, _ = rig(tmp_path)
    pump(backend, 2 * BLOCK, rng)
    rir = acquire_baseline(cfg, io, locker, io.marker_source, seconds=2 * BLOCK_S)
    assert rir is locker.rir and len(locker.calls) == 2


def test_marker_source_scales_to_music():
    params = MarkerParams(length_s=0.01, period_s=0.1, level_db=-20.0)
    src = MarkerSource(FakeScheduler(params, SR), params)
    block = src(100, 0.5)
    assert block.max() == pytest.approx(0.05)
    fixed = MarkerSource(FakeScheduler(params, SR), params, fixed_rms=1.0)
    assert fixed(100, 0.0).max() == pytest.approx(0.1)


def test_acquire_baseline_aligns_to_bursts_and_passes_their_params(tmp_path, rng):
    """The baseline marker is not the Locker's default; the Locker must be told."""
    cfg, backend, io, _, scheduler, _, _ = rig(tmp_path)

    class RecordingLocker(FakeLocker):
        def process(self, reference, mic, emission_start, params=None):
            self.calls.append((emission_start, params))
            return mic, LockState(status=LockStatus.ACQUIRING)

    locker = RecordingLocker()
    pump(backend, 3 * BLOCK, rng)
    acquire_baseline(cfg, io, locker, io.marker_source, seconds=2 * BLOCK_S)
    starts = [s for s, _ in locker.calls]
    assert starts == scheduler.emissions[:2] and starts[1] - starts[0] == BLOCK
    assert all(p is scheduler.params for _, p in locker.calls)
