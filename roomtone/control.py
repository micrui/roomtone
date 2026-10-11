"""From mood to music: the Controller and the two OSC publishers.

The Controller turns the room's mood (energy, voices, rhythm, plus a named
state) into the control values the Tidal patterns read (spec section 8.2).
The mapping in plain words:

- energy, voices and rhythm pass straight through;
- brightness rises with energy (0.2 to 0.8), and drops by 0.2 when people
  are talking, to move the music out of the speech band;
- layers switch on as energy grows: drone always, then pad, figure, phrases
  (only while cooking), shimmer, pulse (also when the room has a beat), bass;
  the figure drops out while people are talking;
- tempo follows rhythm, from 54 to 104 BPM;
- the scale follows the named mood: open (ritusen) when empty, major pentatonic
  when cooking, dorian when people talk, lydian when loud and busy. Dorian rather
  than natural minor for talk: held pad and drone notes from the old scale ring
  on for up to 20 s after a change, and dorian shares every note of the open
  scale and all but one of the pentatonic, where natural minor clashes on two;
- the key drifts: every 40 blocks (20 minutes at 30 s blocks) it moves a fifth
  (5 semitones) or two fifths (2 semitones) up or down, staying within 5
  semitones of the home key. Each Tidal layer picks the new key up at its own
  next note, so for up to 40 s two keys sound at once; keys a fifth apart
  share six of seven notes and two fifths apart five, where a semitone or a
  minor third apart would clash.

Every value moves by at most 0.1 per block, so nothing lurches. Layer changes
happen only here, once per block; the Tidal side lands them on bar lines.

The publishers send the result over OSC: `TidalPublisher` to Tidal's `/ctrl`
listener, `TuiPublisher` to the terminal display.
"""

from __future__ import annotations

import logging
import random
from typing import Any

from pythonosc.udp_client import SimpleUDPClient

from roomtone.config import LAYER_ORDER, Config
from roomtone.log import block_json
from roomtone.types import BlockResult, Controls, LockState, Mood, MoodName

log = logging.getLogger(__name__)

SCALES = {
    MoodName.EMPTY: "open",
    MoodName.COOKING: "pentatonic",
    MoodName.CONVERSATION: "dorian",
    MoodName.LOUD_BUSY: "lydian",
}

TEMPO_MIN_BPM = 54.0
TEMPO_SPAN_BPM = 50.0
TRANSPOSE_EVERY_BLOCKS = 40
TRANSPOSE_LIMIT = 5                      # semitones either side of the home key
TRANSPOSE_STEPS = (-5, -2, 2, 5)         # one or two fifths, either way


def _bit(name: str) -> int:
    return 1 << LAYER_ORDER.index(name)


def layers_for(energy: float, voices: float, rhythm: float, mood: MoodName) -> int:
    """Which layers play, as a bitmask over config.LAYER_ORDER (bit 0 = drone)."""
    mask = _bit("drone")
    if energy > 0.15:
        mask |= _bit("pad")
    if energy > 0.3 and voices <= 0.5:
        mask |= _bit("figure")
    if energy > 0.3 and mood == MoodName.COOKING:
        mask |= _bit("phrases")
    if energy > 0.4:
        mask |= _bit("shimmer")
    if rhythm > 0.3 or energy > 0.5:
        mask |= _bit("pulse")
    if energy > 0.6:
        mask |= _bit("bass")
    return mask  # "spare" is never set in v1


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))


def _slew(current: float, target: float, step: float) -> float:
    return current + max(-step, min(step, target - current))


class Controller:
    """Mood in, Controls out, once per analysis block.

    `max_step` is the most any 0..1 value may move per update. Tempo is derived
    from the slewed rhythm, so it moves at most 5 BPM per update.
    `lock` is accepted for symmetry with the rest of the loop; the v1 mapping
    does not use it (lock trouble reaches the music through the mood instead).
    """

    def __init__(self, cfg: Config | None = None, *, max_step: float = 0.1,
                 transpose_every: int = TRANSPOSE_EVERY_BLOCKS,
                 rng: random.Random | None = None):
        self.cfg = cfg
        self.max_step = max_step
        self.transpose_every = transpose_every
        self.rng = rng if rng is not None else random.Random()
        self.state = Controls()
        self.updates = 0

    def _drift_transpose(self) -> int:
        current = self.state.transpose
        if self.updates > 0 and self.updates % self.transpose_every == 0:
            options = [current + d for d in TRANSPOSE_STEPS if abs(current + d) <= TRANSPOSE_LIMIT]
            return self.rng.choice(options)
        return current

    def update(self, mood: Mood, lock: LockState | None = None) -> Controls:
        s, step = self.state, self.max_step
        energy = _slew(s.energy, _clamp01(mood.energy), step)
        voices = _slew(s.voices, _clamp01(mood.voices), step)
        rhythm = _slew(s.rhythm, _clamp01(mood.rhythm), step)
        bright_target = 0.2 + 0.6 * _clamp01(mood.energy)
        if mood.voices > 0.5:
            bright_target -= 0.2
        bright = _slew(s.bright, _clamp01(bright_target), step)
        self.state = Controls(
            energy=energy, voices=voices, rhythm=rhythm, bright=bright,
            layers=layers_for(energy, voices, rhythm, mood.name),
            tempo_bpm=TEMPO_MIN_BPM + TEMPO_SPAN_BPM * rhythm,
            scale=SCALES.get(mood.name, "open"),
            transpose=self._drift_transpose(),
        )
        self.updates += 1
        return self.state


# Publishers ------------------------------------------------------------------

def ctrl_messages(controls: Controls) -> list[tuple[str, float | str]]:
    """The (key, value) pairs one Controls becomes on Tidal's /ctrl.

    Numbers go as floats (Tidal reads them with cF), the scale as a string, and
    the layer mask both as one float and as one 0/1 `layer_<name>` per layer.
    """
    pairs: list[tuple[str, float | str]] = []
    for key, value in controls.as_dict().items():
        pairs.append((key, value if isinstance(value, str) else float(value)))
    for i, name in enumerate(LAYER_ORDER):
        pairs.append((f"layer_{name}", float((controls.layers >> i) & 1)))
    return pairs


class TidalPublisher:
    """Sends every control value to Tidal's /ctrl listener as `/ctrl <key> <value>`."""

    address = "/ctrl"

    def __init__(self, host: str, port: int, client: Any | None = None):
        self.client = client if client is not None else SimpleUDPClient(host, port)

    def send(self, controls: Controls) -> None:
        for key, value in ctrl_messages(controls):
            self.client.send_message(self.address, [key, value])

    def publish(self, result: BlockResult) -> None:
        self.send(result.controls)


class TuiPublisher:
    """Sends each whole block, as JSON, to the TUI on /roomtone/block."""

    address = "/roomtone/block"

    def __init__(self, host: str, port: int, client: Any | None = None):
        self.client = client if client is not None else SimpleUDPClient(host, port)

    def publish(self, result: BlockResult) -> None:
        self.client.send_message(self.address, block_json(result))
