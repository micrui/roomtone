from dataclasses import dataclass

import numpy as np
import pytest
from scipy.signal import fftconvolve

from roomtone import lock
from roomtone.config import Config, MarkerParams
from roomtone.marker import MarkerScheduler, generate
from roomtone.sim import SimRoom, pad_like, speech_like, synthetic_rir
from roomtone.types import LockStatus

FS = 44_100
DIRECT = round(0.002 * FS)        # where synthetic_rir puts the direct sound


def energy(x: np.ndarray) -> float:
    return float(np.sum(np.asarray(x, dtype=np.float64) ** 2))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(energy(x) / len(x)))


# Building blocks ---------------------------------------------------------------

def test_find_marker_at_0db_snr_under_music_30db_louder(rng):
    m = generate(MarkerParams(), FS)
    n = 30 * FS
    music = pad_like(30.0, FS).astype(np.float64)
    level = rms(music) * 10 ** (-30 / 20)                 # marker 30 dB under the music
    lag = 123_457
    mic = music.copy()
    mic[lag:lag + len(m)] += level * m
    mic += level * rng.standard_normal(n)                 # white noise as loud as the marker: 0 dB SNR
    found, psr = lock.find_marker(mic.astype(np.float32), m, None, n - 1)
    assert abs(found - lag) <= 1
    assert psr > 10.0
    fine, _ = lock.find_marker(mic, m, lag + 37.5, 100, subsample=True)
    assert abs(fine - lag) < 0.25


def test_estimate_drift_recovers_50ppm_from_6_emissions():
    p = MarkerParams(length_s=1.0, period_s=5.0)
    period = round(p.period_s * FS)
    room = SimRoom(FS, delay_samples=3300, drift_ppm=50.0)
    sched = MarkerScheduler(p, FS)
    m = generate(p, FS)
    lags = []
    for k in range(6):
        music = pad_like(p.period_s, FS, t0=k * p.period_s)
        out = music + sched.block(k * period, period, rms(music))
        lag, _ = lock.find_marker(room.render(out), m, None, FS)
        lags.append(lag)
    assert lock.estimate_drift(lags, sched.emissions) == pytest.approx(50.0, abs=5.0)
    assert lock.estimate_drift([10.0], [0]) == 0.0


def test_marker_drift_from_a_single_marker():
    room = SimRoom(FS, delay_samples=3300, drift_ppm=50.0)
    p = MarkerParams()
    music = pad_like(10.0, FS)
    out = music + MarkerScheduler(p, FS).block(0, 10 * FS, rms(music))
    mic = room.render(out)
    rough = round((3300 + DIRECT) * room.ratio) + 3     # a few samples off is fine
    assert lock.marker_drift(mic, generate(p, FS), rough, FS) == pytest.approx(50.0, abs=2.0)


def test_align_undoes_delay_and_drift():
    room = SimRoom(FS, delay_samples=3300, drift_ppm=50.0, rir=np.array([1.0]), noise_db=-np.inf)
    music = pad_like(10.0, FS)
    out = music + MarkerScheduler(MarkerParams(), FS).block(0, 10 * FS, rms(music))
    mic = room.render(out).astype(np.float64)
    aligned = lock.align(out, mic, 3300 * room.ratio, room.drift_ppm)
    err = mic[4000:] - aligned[4000:]
    assert 10 * np.log10(energy(err) / energy(mic[4000:])) < -50.0


def test_estimate_rir_recovers_a_known_response(rng):
    n, tail = 10 * FS, 2048
    h = rng.standard_normal(1500) * np.exp(-np.arange(1500) / 300.0)
    x = rng.standard_normal(n + tail)                     # leading `tail` samples are history
    y = fftconvolve(x, h)[tail:tail + n] + 1e-3 * rng.standard_normal(n)
    est = lock.estimate_rir(x, y, tail)
    assert np.linalg.norm(est[:1500] - h) / np.linalg.norm(h) < 0.01
    assert lock.erle_db(y, lock.subtract(x, y, est)) > 40.0
    prior = np.zeros(tail)
    blended = lock.estimate_rir(x, y, tail, prior=prior)
    assert np.linalg.norm(blended[:1500] - 0.5 * h) / np.linalg.norm(0.5 * h) < 0.02


def test_rir_change_bounds():
    a = np.array([1.0, 0.5, -0.2])
    assert lock.rir_change(a, a) == 0.0
    assert lock.rir_change(a, -a) == pytest.approx(2.0)
    assert 0.0 < lock.rir_change(a, np.array([1.0, 0.5])) < 2.0


def test_erle_db_whole_and_quiet_stretches(rng):
    mic = rng.standard_normal(FS)
    assert lock.erle_db(mic, 0.1 * mic) == pytest.approx(20.0)
    residual = 0.01 * mic.copy()
    residual[: FS // 2] = mic[: FS // 2]                  # half the block is someone talking
    assert lock.erle_db(mic, residual) < 4.0
    assert lock.erle_db(mic, residual, segment=FS // 20) == pytest.approx(40.0)


def test_baseline_roundtrip_and_prior(tmp_path):
    path = tmp_path / "baseline.npy"
    assert lock.load_baseline(path) is None
    h = np.linspace(1.0, 0.0, 500)
    lock.save_baseline(path, h)
    loaded = lock.load_baseline(path)
    assert np.array_equal(loaded, h)
    locker = lock.Locker(Config(), baseline=loaded)
    assert np.array_equal(locker.rir[:500], h) and not locker.rir[500:].any()


# Full sessions through the simulated room ----------------------------------------

@dataclass
class Block:
    status: LockStatus
    state: object
    mic: np.ndarray
    residual: np.ndarray
    params: MarkerParams


def run_session(cfg: Config, room: SimRoom, blocks: int, room_sound=None, before=None) -> list[Block]:
    """Mixer + room + Listener loop: music plus marker out, mic back, Locker per block."""
    n = round(cfg.block_seconds * FS)
    sched = MarkerScheduler(cfg.marker, FS)
    locker = lock.Locker(cfg)
    log: list[Block] = []
    for k in range(blocks):
        if before:
            before(k, room, log)
        music = pad_like(cfg.block_seconds, FS, t0=k * cfg.block_seconds)
        out = music + sched.block(k * n, n, rms(music))
        mic = room.render(out, room_sound(k, log) if room_sound else None)
        residual, state = locker.process(out, mic, k * n, sched.emitted_params[-1])
        sched.set_params(locker.params)
        assert residual.dtype == np.float32 and len(residual) == n
        log.append(Block(state.status, state, mic, residual, sched.emitted_params[-1]))
    return log


def test_pipeline_locks_subtracts_and_leaves_speech():
    cfg = Config()                                    # 30 s blocks, default marker
    room = SimRoom(FS, delay_samples=3300, drift_ppm=50.0)
    rng = np.random.default_rng(7)
    speech: dict[int, np.ndarray] = {}

    def talk(k, log):
        if not 4 <= k <= 6:
            return None
        echo = rms(log[3].mic)                        # music as the mic hears it, speech-free block
        speech[k] = speech_like(cfg.block_seconds, FS, rng, rms_db=20 * np.log10(echo) - 20.0)
        return speech[k]

    log = run_session(cfg, room, 10, room_sound=talk)
    statuses = [b.status for b in log]
    first = statuses.index(LockStatus.LOCKED)
    assert first <= 5
    assert all(s == LockStatus.LOCKED for s in statuses[first:])      # holds through the talking

    for k in [k for k in range(first, 10) if k not in speech]:
        assert log[k].state.erle_db >= 20.0
        assert lock.erle_db(log[k].mic, log[k].residual) >= 20.0       # whole-block figure too
    for k, s in speech.items():
        assert abs(10 * np.log10(energy(log[k].residual) / energy(s))) <= 3.0

    last = log[-1].state
    assert last.drift_ppm == pytest.approx(room.drift_ppm, abs=0.5)
    assert last.jitter_samples <= cfg.lock_jitter_samples
    assert log[0].state.delay_samples == pytest.approx((3300 + DIRECT) * room.ratio, abs=1.0)
    assert last.marker_rung == 0                                       # never needed to escalate


def test_loud_room_climbs_the_ladder_and_saturates():
    start = MarkerParams(length_s=16.0, level_db=-15.0)               # a few rungs below the top
    cfg = Config(marker=start)
    room = SimRoom(FS, delay_samples=3300, drift_ppm=50.0, noise_db=-10.0)
    log = run_session(cfg, room, 12)
    statuses = [b.status for b in log]
    assert LockStatus.LOCKED not in statuses
    emitted = [b.params for b in log]
    assert emitted[3].length_s == 30.0 and emitted[2].length_s == 16.0      # longer first
    assert emitted[6].band_lo_hz == 6000.0 and emitted[5].band_lo_hz == 4000.0  # then the band
    assert emitted[9].level_db == -10.0 and emitted[8].level_db == -15.0     # then louder
    assert emitted[-1].at_ceiling
    assert statuses[-1] == LockStatus.SATURATED and LockStatus.SATURATED not in statuses[:-1]
    assert log[-1].state.marker_rung == 3


def test_room_change_goes_lost_then_relocks():
    cfg = Config(block_seconds=10.0, marker=MarkerParams(period_s=10.0))
    room = SimRoom(FS, delay_samples=3300, drift_ppm=50.0)
    moved = synthetic_rir(FS, np.random.default_rng(99))

    def furniture(k, room, log):
        if k == 6:
            room.rir = moved

    log = run_session(cfg, room, 12, before=furniture)
    statuses = [b.status for b in log]
    assert statuses[5] == LockStatus.LOCKED
    assert statuses[6] == LockStatus.LOST
    assert log[6].state.rir_change > lock.LOST_CHANGE
    assert LockStatus.LOCKED in statuses[7:]
    relock = 7 + statuses[7:].index(LockStatus.LOCKED)
    assert all(s == LockStatus.LOST for s in statuses[6:relock])
    assert all(s == LockStatus.LOCKED for s in statuses[relock:])
    assert log[-1].state.erle_db >= 20.0
    assert log[-1].state.marker_rung == 0
