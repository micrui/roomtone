"""Shared value types passed between the Listener, Controller, TUI and bench tools.

Keep these plain dataclasses with no behaviour beyond trivial helpers, so every
module can import them without pulling in audio or UI dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

import numpy as np


class LockStatus(str, Enum):
    ACQUIRING = "acquiring"   # no marker found yet, or criteria not met
    LOCKED = "locked"         # all three criteria met (section 6.4)
    LOST = "lost"             # had lock, room response jumped; re-estimating
    SATURATED = "saturated"   # marker ladder at ceiling and still failing


@dataclass
class LockState:
    status: LockStatus = LockStatus.ACQUIRING
    delay_samples: float = 0.0          # output -> mic delay at the marker
    drift_ppm: float = 0.0              # mic clock relative to output clock
    peak_to_sidelobe_db: float = 0.0
    jitter_samples: float = 0.0
    erle_db: float = 0.0                # echo suppression achieved this block
    rir_change: float = 0.0             # normalised distance between successive RIR estimates
    marker_rung: int = 0                # position on the escalation ladder
    blocks_ok: int = 0                  # consecutive blocks meeting criteria
    blocks_bad: int = 0
    quiet: bool = False                 # block was quiet enough to update the room response

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


class MoodName(str, Enum):
    EMPTY = "empty"
    COOKING = "cooking"
    CONVERSATION = "conversation"
    LOUD_BUSY = "loud-busy"


@dataclass
class Features:
    """What one analysis block of residual audio looks like."""
    rms_db: float = -90.0
    speech: float = 0.0          # 0..1 probability or energy ratio in the speech band
    talkers: float = 0.0         # rough count, continuous
    transient_rate: float = 0.0  # onsets per second
    beat: float = 0.0            # 0..1 periodicity strength at tempo rates
    speech_band_centroid_hz: float = 0.0
    labels: dict[str, float] = field(default_factory=dict)  # optional YAMNet top labels


@dataclass
class Mood:
    """Continuous mood vector plus the named state derived from it."""
    energy: float = 0.0          # 0..1
    voices: float = 0.0          # 0..1
    rhythm: float = 0.0          # 0..1
    name: MoodName = MoodName.EMPTY
    confidence: float = 0.0      # 0..1, grows with window fill and agreement

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["name"] = self.name.value
        return d


@dataclass
class Controls:
    """What the Controller publishes to Tidal (/ctrl) and the TUI. All 0..1 unless noted."""
    energy: float = 0.0
    voices: float = 0.0
    rhythm: float = 0.0
    bright: float = 0.3
    layers: int = 0b00000001     # bitmask over config.LAYER_ORDER, bit 0 = drone
    tempo_bpm: float = 60.0
    scale: str = "open"          # open | pentatonic | dorian | lydian (any Tidal scale name passes through)
    transpose: int = 0           # semitones, slow key drift

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlockResult:
    """Everything one 30 s analysis block produced. The TUI and the log consume this."""
    index: int
    t_start: float               # seconds since session start
    lock: LockState
    features: Features
    mood: Mood
    controls: Controls
    residual_rms_db: float = -90.0
    raw_rms_db: float = -90.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "index": self.index, "t_start": self.t_start,
            "lock": self.lock.as_dict(), "features": asdict(self.features),
            "mood": self.mood.as_dict(), "controls": self.controls.as_dict(),
            "residual_rms_db": self.residual_rms_db, "raw_rms_db": self.raw_rms_db,
        }


def db(x: float, floor: float = -120.0) -> float:
    """Amplitude ratio or RMS to decibels with a floor."""
    return float(20.0 * np.log10(max(x, 10 ** (floor / 20.0))))
