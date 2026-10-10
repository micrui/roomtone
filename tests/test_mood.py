"""MoodTracker behaviour on hand-built feature streams."""

from __future__ import annotations

import pytest

from roomtone.config import Config
from roomtone.mood import MAX_STEP, MoodTracker, classify_vector, raw_vector
from roomtone.types import Features, LockState, LockStatus, MoodName

LOCKED = LockState(status=LockStatus.LOCKED)
SILENT = Features(rms_db=-90.0)
TALK = Features(rms_db=-35.0, speech=0.9, talkers=1.2, transient_rate=0.5, beat=0.1)
LOUD = Features(rms_db=-15.0, speech=0.5, talkers=6.0, transient_rate=6.0, beat=0.3)


@pytest.mark.parametrize("energy, voices, rhythm, saturated, expected", [
    (0.0, 0.0, 0.0, False, MoodName.EMPTY),
    (0.19, 0.19, 0.9, False, MoodName.EMPTY),
    (0.5, 0.1, 0.0, False, MoodName.COOKING),
    (0.2, 0.0, 0.0, False, MoodName.COOKING),
    (0.1, 0.3, 0.0, False, MoodName.COOKING),
    (0.5, 0.4, 0.0, False, MoodName.CONVERSATION),
    (0.1, 0.9, 0.0, False, MoodName.CONVERSATION),
    (0.75, 0.9, 0.0, False, MoodName.CONVERSATION),
    (0.76, 0.0, 0.0, False, MoodName.LOUD_BUSY),
    (0.9, 0.9, 0.9, False, MoodName.LOUD_BUSY),
    (0.0, 0.0, 0.0, True, MoodName.LOUD_BUSY),
])
def test_classify_vector_table(energy, voices, rhythm, saturated, expected):
    assert classify_vector(energy, voices, rhythm, saturated) is expected


def test_raw_vector_mapping():
    assert raw_vector(Features(rms_db=-60.0))[0] == pytest.approx(0.0)
    assert raw_vector(Features(rms_db=-40.0))[0] == pytest.approx(0.5)
    assert raw_vector(Features(rms_db=-40.0, transient_rate=100.0))[0] == pytest.approx(0.8)
    assert raw_vector(Features(rms_db=-10.0, transient_rate=100.0))[0] == 1.0
    one, crowd = raw_vector(Features(speech=0.8, talkers=1.0)), raw_vector(Features(speech=0.8, talkers=5))
    assert one[1] == pytest.approx(0.6) and crowd[1] == pytest.approx(0.8)


def test_raw_vector_relative_to_floor():
    assert raw_vector(Features(rms_db=-48.0), floor_db=-48.0)[0] == 0.0
    assert raw_vector(Features(rms_db=-42.0), floor_db=-48.0)[0] == 0.0
    assert raw_vector(Features(rms_db=-27.0), floor_db=-48.0)[0] == pytest.approx(0.5)
    assert raw_vector(Features(rms_db=-19.0), floor_db=-10.0)[0] == 1.0   # loud whatever the floor


def test_steady_hum_is_empty_without_onsets():
    assert classify_vector(0.3, 0.0, 0.0, False, transients=0.0) is MoodName.EMPTY
    assert classify_vector(0.3, 0.0, 0.0, False, transients=0.5) is MoodName.COOKING
    assert classify_vector(0.36, 0.0, 0.0, False, transients=0.0) is MoodName.COOKING


def test_noise_floor_stays_empty_and_clatter_cooks():
    cfg = Config()
    tracker = MoodTracker(cfg)
    floor = Features(rms_db=-48.0, speech=0.05)
    for _ in range(10):
        mood = tracker.update(floor, LOCKED)
        assert mood.name is MoodName.EMPTY
    assert tracker.floor_db == pytest.approx(-48.0)
    assert mood.energy == 0.0
    clatter = Features(rms_db=-48.0, speech=0.05, transient_rate=3.0)   # energy 0.225
    names = [tracker.update(clatter, LOCKED).name for _ in range(cfg.hysteresis_blocks)]
    assert names == [MoodName.EMPTY] * (cfg.hysteresis_blocks - 1) + [MoodName.COOKING]


def test_floor_follows_the_quietest_recent_block():
    tracker = MoodTracker(Config())
    assert tracker.floor_db is None
    tracker.update(Features(rms_db=-40.0), LOCKED)
    tracker.update(Features(rms_db=-50.0), LOCKED)
    tracker.update(Features(rms_db=-30.0), LOCKED)
    assert tracker.floor_db == pytest.approx(-50.0)
    tracker.update(SILENT, LOCKED)
    assert tracker.floor_db == pytest.approx(-60.0)   # digital silence is not a room
    for _ in range(240):
        tracker.update(Features(rms_db=-45.0), LOCKED)
    assert tracker.floor_db == pytest.approx(-45.0)   # old quiet blocks age out


def test_silence_then_speech():
    cfg = Config()
    tracker = MoodTracker(cfg)
    assert tracker.window_blocks == 10

    confidences = []
    for _ in range(10):
        mood = tracker.update(SILENT, LOCKED)
        assert mood.name is MoodName.EMPTY
        confidences.append(mood.confidence)
    assert all(b > a for a, b in zip(confidences, confidences[1:]))
    assert confidences[-1] == pytest.approx(1.0)

    names = [tracker.update(TALK, LOCKED).name for _ in range(cfg.hysteresis_blocks + 2)]
    k = cfg.hysteresis_blocks
    assert names[: k - 1] == [MoodName.EMPTY] * (k - 1)
    assert names[k - 1 :] == [MoodName.CONVERSATION] * 3


def test_slew_limit():
    tracker = MoodTracker(Config())
    prev = tracker.update(SILENT, LOCKED)
    for features in [LOUD] * 8 + [SILENT] * 8 + [TALK, LOUD, SILENT] * 4:
        mood = tracker.update(features, LOCKED)
        for key in ("energy", "voices", "rhythm"):
            assert abs(getattr(mood, key) - getattr(prev, key)) <= MAX_STEP + 1e-9
        prev = mood
    # sustained input still gets there eventually
    for _ in range(20):
        mood = tracker.update(LOUD, LOCKED)
    assert mood.energy == pytest.approx(raw_vector(LOUD)[0])


def test_saturated_lock_forces_loud_busy():
    cfg = Config()
    tracker = MoodTracker(cfg)
    for _ in range(5):
        tracker.update(SILENT, LOCKED)
    saturated = LockState(status=LockStatus.SATURATED)
    names = [tracker.update(SILENT, saturated).name for _ in range(cfg.hysteresis_blocks)]
    assert names[-1] is MoodName.LOUD_BUSY
    assert all(n is MoodName.EMPTY for n in names[:-1])


def test_interrupted_candidate_restarts_count():
    cfg = Config()
    tracker = MoodTracker(cfg)
    stream = [TALK] * (cfg.hysteresis_blocks - 1) + [SILENT] + [TALK] * (cfg.hysteresis_blocks - 1)
    assert all(tracker.update(f, LOCKED).name is MoodName.EMPTY for f in stream)
    assert tracker.update(TALK, LOCKED).name is MoodName.CONVERSATION


def test_unlocked_halves_confidence_and_reset():
    a, b = MoodTracker(Config()), MoodTracker(Config())
    for _ in range(4):
        ma = a.update(SILENT, LOCKED)
        mb = b.update(SILENT, LockState(status=LockStatus.ACQUIRING))
    assert mb.confidence == pytest.approx(0.5 * ma.confidence)
    assert ma.confidence == pytest.approx(0.4)
    a.reset()
    assert a.update(SILENT, LOCKED).confidence == pytest.approx(0.1)


def test_agreement_lowers_confidence_after_a_change():
    cfg = Config()
    tracker = MoodTracker(cfg)
    for _ in range(10):
        tracker.update(SILENT, LOCKED)
    for _ in range(cfg.hysteresis_blocks):
        mood = tracker.update(TALK, LOCKED)
    assert mood.name is MoodName.CONVERSATION
    assert mood.confidence == pytest.approx(cfg.hysteresis_blocks / 10)
