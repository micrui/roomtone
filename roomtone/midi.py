"""MIDI out to the M8: notes, CCs, clock, and a slow CC sweep for the bench.

Uses mido. mido loads its hardware backend (python-rtmidi) only when a port is
listed or opened, so importing this module never needs MIDI hardware.

Channels here are 1 to 16, as the M8 shows them. mido counts from 0, so every
function converts.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Iterable

import mido

log = logging.getLogger(__name__)

CLOCK_PPQN = 24  # MIDI clock ticks per quarter note; the M8 runs at the same rate


def _channel(channel: int) -> int:
    if not 1 <= channel <= 16:
        raise ValueError(f"MIDI channel must be 1..16, got {channel}")
    return channel - 1


def _clamp7(value: int) -> int:
    return max(0, min(127, int(value)))


def list_ports() -> dict[str, list[str]]:
    """Names of MIDI inputs and outputs. Raises ImportError without a mido backend."""
    return {"inputs": mido.get_input_names(), "outputs": mido.get_output_names()}


def open_out(name_substring: str | None) -> Any:
    """Open the first MIDI output whose name contains `name_substring` (case-insensitive).

    None opens the system default output. Raises LookupError if nothing matches.
    """
    if name_substring is None:
        return mido.open_output()
    names = mido.get_output_names()
    for name in names:
        if name_substring.lower() in name.lower():
            log.info("opening MIDI output %r", name)
            return mido.open_output(name)
    raise LookupError(f"no MIDI output matching {name_substring!r}; available: {names}")


def send_note(port: Any, channel: int, note: int, velocity: int, duration_s: float,
              sleep: Callable[[float], None] | None = None) -> None:
    """Play one note: note on, wait, note off."""
    ch = _channel(channel)
    sleep = sleep or time.sleep
    port.send(mido.Message("note_on", channel=ch, note=_clamp7(note), velocity=_clamp7(velocity)))
    sleep(duration_s)
    port.send(mido.Message("note_off", channel=ch, note=_clamp7(note), velocity=0))


def send_cc(port: Any, channel: int, cc: int, value: int) -> None:
    """Send one control change. The value is clamped to 0..127."""
    port.send(mido.Message("control_change", channel=_channel(channel),
                           control=_clamp7(cc), value=_clamp7(value)))


def _paced(n: int, interval: float, action: Callable[[int], None],
           clock: Callable[[], float] | None, sleep: Callable[[float], None] | None) -> None:
    """Run action(i) n times, `interval` seconds apart, against absolute deadlines so
    small sleep overshoots do not accumulate."""
    clock = clock or time.perf_counter
    sleep = sleep or time.sleep
    t0 = clock()
    for i in range(n):
        wait = t0 + i * interval - clock()
        if wait > 0:
            sleep(wait)
        action(i)


def send_clock(port: Any, bpm: float, seconds: float,
               clock: Callable[[], float] | None = None,
               sleep: Callable[[float], None] | None = None) -> int:
    """Send MIDI clock at `bpm` (24 ticks per beat) for `seconds`. Returns ticks sent."""
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    interval = 60.0 / (bpm * CLOCK_PPQN)
    ticks = int(round(seconds / interval))
    tick = mido.Message("clock")
    _paced(ticks, interval, lambda _i: port.send(tick), clock, sleep)
    return ticks


def cc_sweep(port: Any, channel: int, cc: int, lo: int, hi: int, seconds: float,
             clock: Callable[[], float] | None = None,
             sleep: Callable[[float], None] | None = None) -> list[int]:
    """Step a CC through every value from lo to hi, evenly over `seconds`.

    Slow on purpose: this is how you hear whether the M8 parameter sounds steppy.
    Returns the values sent.
    """
    lo, hi = _clamp7(lo), _clamp7(hi)
    step = 1 if hi >= lo else -1
    values = list(range(lo, hi + step, step))
    interval = seconds / max(1, len(values) - 1)
    _paced(len(values), interval, lambda i: send_cc(port, channel, cc, values[i]), clock, sleep)
    return values


def panic(port: Any, channels: Iterable[int] = range(1, 17)) -> int:
    """Silence every note: all-notes-off and all-sound-off, then an explicit note-off for
    every key, on each channel. Returns the number of messages sent.

    Why the explicit note-offs: a note-off that never arrives (a crashed sender, a MIDI
    port that vanished mid-note) leaves the M8 holding the key, and it then ignores a
    new note-on for that key on that track, so the track falls silent until something
    releases it. The M8 was seen in that state on the bench.
    """
    n = 0
    for channel in channels:
        ch = _channel(channel)
        port.send(mido.Message("control_change", channel=ch, control=123, value=0))
        port.send(mido.Message("control_change", channel=ch, control=120, value=0))
        n += 2
        for note in range(128):
            port.send(mido.Message("note_off", channel=ch, note=note, velocity=0))
            n += 1
    return n
