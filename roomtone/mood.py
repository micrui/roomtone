"""Turn a stream of per-block features into a slow, steady room mood.

Each 30 s block of features becomes three raw numbers between 0 and 1:

energy
    Loudness relative to the room's own quiet floor, plus up to 0.3 more for
    transients (0.075 per onset per second, full at 4 per second), so a quiet
    room with clatter in it still reads as active. Clipped to 1.

    The quiet floor is the lowest block level of the last 240 blocks (about
    two hours at 30 s), starting from the first block and never taken below
    -60 dBFS (digital silence is not a room). Loudness is 0 up to 6 dB above
    the floor and rises linearly to 1 at 36 dB above it; anything louder than
    -20 dBFS counts as 1 whatever the floor. So a mic whose own hiss sits at
    -48 dBFS reads an empty room as 0, not as a third of the way to loud.
    Without a floor (raw_vector called on its own), loudness falls back to a
    fixed scale from -60 dBFS (0) to -20 dBFS (1).
voices
    The speech score, scaled from 0.75 of itself for one talker up to all of
    itself at four or more talkers: a crowd counts more than a single voice.
rhythm
    The beat strength as measured.

The published mood vector is a smoothed version of those raw numbers. We keep
the last mood_window_seconds of blocks (10 blocks at the defaults) and take a
weighted mean in which each older block counts 0.75 times as much as the one
after it, so the most recent blocks dominate but one odd block cannot jerk the
music around. The result is then rate-limited: no component moves more than
0.15 per block, so a full swing takes at least seven blocks (about 3.5 min).

The named state is for the log and for coarse layer decisions. Each block's
raw vector, with its onset rate, gets a candidate name from classify_vector:

    LOUD_BUSY     energy > 0.75, or the lock is SATURATED (the room drowns the
                  marker, which only happens when it is very loud)
    CONVERSATION  voices >= 0.4
    EMPTY         energy < 0.2 and voices < 0.2
    COOKING       something is making sound that is not mostly talk and not
                  overwhelming, and there is evidence of activity: at least
                  0.5 onsets per second, or energy >= 0.35 from loudness alone
    EMPTY         everything else: a fridge or a fan with no onsets and no
                  speech is an empty room unless it is loud in its own right

The candidate comes from the block itself rather than the smoothed vector, so
the hysteresis count means what it says: the name changes once the same new
candidate has won cfg.hysteresis_blocks blocks in a row, and stays put
otherwise.

confidence is how full the window is (0 to 1), times the share of blocks in it
whose candidate agrees with the current name, halved whenever the lock is not
LOCKED (the residual may still contain our own music).
"""

from __future__ import annotations

from typing import Sequence

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from roomtone.config import Config
from roomtone.types import Features, LockState, LockStatus, Mood, MoodName

log = logging.getLogger(__name__)

ENERGY_FLOOR_DB = -60.0
ENERGY_CEIL_DB = -20.0
TRANSIENT_BONUS = 0.3
TRANSIENT_FULL_RATE = 4.0       # onsets per second that earn the full bonus
SINGLE_TALKER_VOICES = 0.75     # share of the speech score one talker earns
CROWD_TALKERS = 4.0             # talkers at which the full speech score counts
RECENCY_DECAY = 0.75            # weight ratio between a block and the one after it
MAX_STEP = 0.15                 # largest change per block in any component
UNLOCKED_CONFIDENCE = 0.5

FLOOR_BLOCKS = 240              # blocks of history the quiet floor is the minimum of
FLOOR_MARGIN_DB = 6.0           # loudness starts this far above the floor
RELATIVE_SPAN_DB = 30.0         # and reaches 1 this much further up
FLOOR_LOG_STEP_DB = 1.0         # log the floor when it moves at least this much

LOUD_ENERGY = 0.75
CONVERSATION_VOICES = 0.4
EMPTY_ENERGY = 0.2
EMPTY_VOICES = 0.2
COOKING_TRANSIENTS = 0.5        # onsets per second that count as activity
COOKING_LOUDNESS = 0.35         # or this much energy from loudness alone


def transient_bonus(rate: float) -> float:
    """Energy added for `rate` onsets per second."""
    return TRANSIENT_BONUS * min(max(rate, 0.0) / TRANSIENT_FULL_RATE, 1.0)


def loudness(rms_db: float, floor_db: float | None = None) -> float:
    """0..1 loudness of one block: relative to `floor_db`, or on the fixed scale without one."""
    if floor_db is None:
        loud = (rms_db - ENERGY_FLOOR_DB) / (ENERGY_CEIL_DB - ENERGY_FLOOR_DB)
    elif rms_db > ENERGY_CEIL_DB:
        loud = 1.0
    else:
        loud = (rms_db - floor_db - FLOOR_MARGIN_DB) / RELATIVE_SPAN_DB
    return float(np.clip(loud, 0.0, 1.0))


def classify_vector(energy: float, voices: float, rhythm: float, saturated: bool,
                    transients: float = 1.0) -> MoodName:
    """Named state for one mood vector. rhythm is accepted for future rules but unused.

    `transients` is the block's onset rate; the default assumes there were some.
    """
    if saturated or energy > LOUD_ENERGY:
        return MoodName.LOUD_BUSY
    if voices >= CONVERSATION_VOICES:
        return MoodName.CONVERSATION
    if energy < EMPTY_ENERGY and voices < EMPTY_VOICES:
        return MoodName.EMPTY
    if transients >= COOKING_TRANSIENTS or energy - transient_bonus(transients) >= COOKING_LOUDNESS:
        return MoodName.COOKING
    return MoodName.EMPTY


def raw_vector(f: Features, floor_db: float | None = None) -> tuple[float, float, float]:
    """(energy, voices, rhythm) for one block, before smoothing. See the module docstring."""
    energy = min(loudness(f.rms_db, floor_db) + transient_bonus(f.transient_rate), 1.0)
    crowd = float(np.clip((f.talkers - 1.0) / (CROWD_TALKERS - 1.0), 0.0, 1.0))
    voices = float(np.clip(f.speech, 0.0, 1.0)) * (SINGLE_TALKER_VOICES
                                                    + (1.0 - SINGLE_TALKER_VOICES) * crowd)
    rhythm = float(np.clip(f.beat, 0.0, 1.0))
    return energy, voices, rhythm


@dataclass
class _Block:
    vector: tuple[float, float, float]
    candidate: MoodName


class MoodTracker:
    """Rolling mood estimate with recency weighting, slew limiting and hysteresis."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.window_blocks = max(int(round(cfg.mood_window_seconds / cfg.block_seconds)), 1)
        self.reset()

    def reset(self) -> None:
        self._blocks: deque[_Block] = deque(maxlen=self.window_blocks)
        self._vector = np.zeros(3)
        self._name = MoodName.EMPTY
        self._pending: MoodName | None = None
        self._pending_count = 0
        self._levels: deque[float] = deque(maxlen=FLOOR_BLOCKS)
        self._logged_floor: float | None = None

    @property
    def floor_db(self) -> float | None:
        """The learned quiet floor in dBFS, or None before the first block."""
        return max(min(self._levels), ENERGY_FLOOR_DB) if self._levels else None

    def _track_floor(self, rms_db: float) -> float:
        self._levels.append(rms_db)
        floor = max(min(self._levels), ENERGY_FLOOR_DB)
        if self._logged_floor is None or abs(floor - self._logged_floor) > FLOOR_LOG_STEP_DB:
            log.info("mood: quiet floor %.1f dBFS", floor)
            self._logged_floor = floor
        return floor

    def update(self, features: Features, lock: LockState) -> Mood:
        raw = raw_vector(features, self._track_floor(features.rms_db))
        saturated = lock.status == LockStatus.SATURATED
        candidate = classify_vector(*raw, saturated=saturated,
                                    transients=features.transient_rate)
        self._blocks.append(_Block(raw, candidate))

        values = np.array([b.vector for b in self._blocks])
        weights = RECENCY_DECAY ** np.arange(len(values) - 1, -1, -1, dtype=float)
        target = weights @ values / weights.sum()
        self._vector = self._vector + np.clip(target - self._vector, -MAX_STEP, MAX_STEP)

        self._advance_name(candidate)

        fill = len(self._blocks) / self.window_blocks
        agree = sum(b.candidate == self._name for b in self._blocks) / len(self._blocks)
        confidence = fill * agree * (1.0 if lock.status == LockStatus.LOCKED
                                     else UNLOCKED_CONFIDENCE)

        energy, voices, rhythm = (float(v) for v in self._vector)
        mood = Mood(energy=energy, voices=voices, rhythm=rhythm,
                    name=self._name, confidence=float(confidence))
        log.debug("mood: raw %s candidate %s -> %s", tuple(round(v, 2) for v in raw),
                  candidate.value, mood.as_dict())
        return mood

    def _advance_name(self, candidate: MoodName) -> None:
        if candidate == self._name:
            self._pending, self._pending_count = None, 0
            return
        if candidate == self._pending:
            self._pending_count += 1
        else:
            self._pending, self._pending_count = candidate, 1
        if self._pending_count >= self.cfg.hysteresis_blocks:
            log.info("mood: %s -> %s", self._name.value, candidate.value)
            self._name = candidate
            self._pending, self._pending_count = None, 0


# Audition ----------------------------------------------------------------------

# One representative mood per kitchen script mode (roomtone.kitchen.MODES), chosen so
# the Controller's layer rules light up what that state is meant to sound like:
# cooking brings in pad, figure, phrases and shimmer; dinner is talk (figure drops,
# everything ducks); party reaches the bass and the pulse.
AUDITION_MOODS: dict[str, Mood] = {
    "empty": Mood(0.05, 0.05, 0.05, MoodName.EMPTY, 1.0),
    "cooking": Mood(0.5, 0.2, 0.45, MoodName.COOKING, 1.0),
    "dinner": Mood(0.45, 0.8, 0.2, MoodName.CONVERSATION, 1.0),
    "party": Mood(0.9, 0.7, 0.85, MoodName.LOUD_BUSY, 1.0),
}


class ScriptedMood:
    """A mood tracker that reads a script instead of the room.

    `phases` are (seconds, mode) as roomtone.kitchen.parse_script gives them, and
    the script repeats. Each `update` call is one analysis block of `block_seconds`,
    so the mood follows the script by block count. The Controller's slew still
    applies downstream, so layers arrive as they would in the kitchen.
    """

    def __init__(self, phases: Sequence[tuple[float, str]], block_seconds: float):
        if not phases:
            raise ValueError("the audition script needs at least one phase")
        for _, mode in phases:
            if mode not in AUDITION_MOODS:
                raise ValueError(f"unknown mode {mode!r}; modes: {', '.join(AUDITION_MOODS)}")
        self.phases = list(phases)
        self.block_seconds = float(block_seconds)
        self.total = sum(seconds for seconds, _ in self.phases)
        self.blocks = 0

    def mode_at(self, t_seconds: float) -> str:
        t = t_seconds % self.total if self.total > 0 else 0.0
        for seconds, mode in self.phases:
            if t < seconds:
                return mode
            t -= seconds
        return self.phases[-1][1]

    def update(self, features: Features, lock: LockState) -> Mood:
        mood = AUDITION_MOODS[self.mode_at(self.blocks * self.block_seconds)]
        self.blocks += 1
        return mood
