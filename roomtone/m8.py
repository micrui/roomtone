"""The M8 over its USB serial display protocol: read its screen, press its keys, read settings.

The M8 exposes the same protocol the m8c and web clients use: a serial port (CDC) over
the USB cable. Send `E` and the M8 streams its display as SLIP-framed draw commands
(characters, rectangles, the oscilloscope); send `C` plus a key bitmask to press keys.
That is enough to read any screen as text and to walk the M8's own menus, so the
things the manual says to check by eye (firmware, system settings, the Track MIDI Input
table, the MIDI mappings) can be read from the Mac, and settings could be made from
here too. Nothing in this module edits anything: `roomtone m8 info` presses only
navigation keys (arrows, SHIFT+arrows, and EDIT on menu entries that open pages).

Protocol notes (Model:02, firmware 6.2.2, confirmed on the bench):
- 0xFF system info: model (3 = Model:02), major, minor, patch, font mode.
- 0xFD draw character: char, x (u16), y (u16), fg rgb, bg rgb.
- 0xFE draw rectangle: x, y, w, h (u16 each), rgb; shorter forms reuse the last size
  or colour. A big rectangle is a screen clear.
- 0xFC oscilloscope waveform; ignored here.
- The cursor is a bracket of 12 tiny rectangles (w <= 3, h <= 2) drawn in the
  highlight colour, pulsing; it is erased in black before it moves.
- Rows are 14 px apart in the large font; the title sits on the y = 42 row.
- Keys: EDIT 1, OPTION 2, RIGHT 4, PLAY 8, SHIFT 16, DOWN 32, UP 64, LEFT 128.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

END, ESC, ESC_END, ESC_ESC = 0xC0, 0xDB, 0xDC, 0xDD
MODELS = {0: "headless", 1: "beta M8", 2: "Model:01", 3: "Model:02"}
KEYS = {"edit": 1, "option": 2, "right": 4, "play": 8, "shift": 16, "down": 32, "up": 64, "left": 128}
TITLE_Y = 42
ROW_PX = 14
BAUD = 115200


def decode_slip(buf: bytes) -> tuple[list[bytes], bytes]:
    """Complete SLIP packets in `buf`, and the unterminated tail to keep for next time."""
    packets: list[bytes] = []
    cur = bytearray()
    esc = False
    last_end = -1
    for i, b in enumerate(buf):
        if esc:
            cur.append(END if b == ESC_END else ESC if b == ESC_ESC else b)
            esc = False
        elif b == ESC:
            esc = True
        elif b == END:
            if cur:
                packets.append(bytes(cur))
                cur = bytearray()
            last_end = i
        else:
            cur.append(b)
    return packets, bytes(buf[last_end + 1:])


@dataclass
class Screen:
    """What the M8 has drawn: characters by position, the cursor, the system info."""
    chars: dict[tuple[int, int], str] = field(default_factory=dict)
    cursor: tuple[int, int, int, int] | None = None     # x, y, w, h of the bracket
    info: dict[str, Any] | None = None
    _batch: list[tuple[int, int, int, int, tuple | None]] = field(default_factory=list)
    _last_rect: tuple[int, int, int, int, tuple | None] | None = None

    def feed(self, packet: bytes) -> None:
        cmd = packet[0]
        if cmd == 0xFF and len(packet) >= 5:
            self.info = {"model": MODELS.get(packet[1], str(packet[1])),
                         "firmware": f"{packet[2]}.{packet[3]}.{packet[4]}",
                         "font": packet[5] if len(packet) > 5 else None}
        elif cmd == 0xFD and len(packet) >= 6:
            ch = packet[1]
            x, y = struct.unpack_from("<HH", packet, 2)
            self.chars[(y, x)] = chr(ch) if 32 <= ch < 127 else "?"
        elif cmd == 0xFE:
            self._rect(packet)

    def _rect(self, p: bytes) -> None:
        prev = self._last_rect
        if len(p) >= 12:
            x, y, w, h = struct.unpack_from("<HHHH", p, 1)
            rgb: tuple | None = tuple(p[9:12])
        elif len(p) >= 9:
            x, y, w, h = struct.unpack_from("<HHHH", p, 1)
            rgb = None
        elif len(p) >= 8:
            x, y = struct.unpack_from("<HH", p, 1)
            w, h = (prev[2], prev[3]) if prev else (1, 1)
            rgb = tuple(p[5:8])
        elif len(p) >= 5:
            x, y = struct.unpack_from("<HH", p, 1)
            w, h = (prev[2], prev[3]) if prev else (1, 1)
            rgb = None
        else:
            return
        self._last_rect = (x, y, w, h, rgb if rgb is not None else (prev[4] if prev else None))
        if w > 100 and h > 50:                       # a clear: forget the characters under it
            self.chars = {k: v for k, v in self.chars.items()
                          if not (x <= k[1] < x + w and y <= k[0] < y + h)}
            return
        if w <= 3 and h <= 2:                        # cursor bracket pieces
            if rgb is not None:                      # an explicit colour starts a new batch
                self._close_batch()
            self._batch.append((x, y, w, h, rgb))

    def _close_batch(self) -> None:
        b = self._batch
        if len(b) >= 12 and b[0][4] not in (None, (0, 0, 0)):
            xs = [r[0] for r in b]
            ys = [r[1] for r in b]
            self.cursor = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        self._batch = []

    def settle(self) -> None:
        """Call when the stream goes quiet, so a trailing cursor batch counts."""
        self._close_batch()

    # --- text ---------------------------------------------------------------

    def rows(self) -> list[int]:
        return sorted({y for y, _ in self.chars})

    def row_text(self, y: int) -> str:
        """The row as text, column 0 being the leftmost character column on screen."""
        cells = sorted((x, c) for (yy, x), c in self.chars.items() if yy == y)
        if not cells:
            return ""
        dx, x_min = self._cell_width(), min(x for _, x in self.chars)
        line = [" "] * ((cells[-1][0] - x_min) // dx + 1)
        for x, c in cells:
            line[(x - x_min) // dx] = c
        return "".join(line).rstrip()

    def _cell_width(self) -> int:
        xs = sorted({x for _, x in self.chars})
        gaps = [b - a for a, b in zip(xs, xs[1:]) if b > a]
        return min(gaps) if gaps else 8

    def text(self, main_only: bool = True) -> str:
        """Every row, with the common left margin removed so labels start in column 0.

        With `main_only` the sidebar (the transport "T>" tempo, the track mutes and the
        view map, which share rows with every page) is cut off at the "T>" column.
        """
        lines = [self.row_text(y) for y in self.rows()]
        margin = min((len(l) - len(l.lstrip(" ")) for l in lines if l.strip()), default=0)
        lines = [l[margin:] for l in lines]
        if main_only:
            cut = max((l.find("T>") for l in lines), default=-1)
            if cut >= 20:
                lines = [l[:cut].rstrip() for l in lines]
        return "\n".join(lines)

    def title(self) -> str:
        return self.row_text(TITLE_Y).strip()

    def cursor_field(self) -> tuple[str, str]:
        """(row label, text under the cursor) for the current cursor, or ('', '')."""
        if self.cursor is None or not self.chars:
            return "", ""
        x0, y0, w, _ = self.cursor
        y = min(self.rows(), key=lambda yy: abs(yy - y0))
        line = self.row_text(y)
        dx, x_min = self._cell_width(), min(x for _, x in self.chars)
        margin = len(line) - len(line.lstrip(" "))
        label = line[margin:margin + 14].strip()           # the label column, 13 characters wide
        c0, c1 = max(0, (x0 - x_min) // dx), (x0 + w - x_min) // dx + 1
        under = line[c0:c1].strip()
        return label, under

    def find_row(self, label: str) -> int | None:
        for y in self.rows():
            if self.row_text(y).strip().startswith(label):
                return y
        return None


# --- the link ------------------------------------------------------------------

def find_port() -> str | None:
    """The M8's serial port, by USB product name, or None."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    for p in list_ports.comports():
        if (p.product or "").strip() == "M8" or "M8" in (p.description or ""):
            return p.device
    return None


class M8Link:
    """A serial connection with the display stream enabled. `opener` is for tests."""

    def __init__(self, port: str | None = None, opener: Callable[[str], Any] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.port = port or find_port()
        if self.port is None:
            raise LookupError("no M8 serial port found (is the M8 plugged in over USB?)")
        if opener is None:
            import serial  # optional dependency: pyserial
            opener = lambda p: serial.Serial(p, BAUD, timeout=0.05)  # noqa: E731
        self.ser = opener(self.port)
        self.clock, self.sleep = clock, sleep
        self.screen = Screen()
        self._pending = b""

    def enable(self) -> None:
        self.ser.write(b"E")
        self.ser.write(b"R")
        self.ser.flush()

    def close(self) -> None:
        try:
            self.ser.write(b"D")
            self.ser.flush()
        finally:
            self.ser.close()

    def read(self, seconds: float) -> Screen:
        """Pump the stream for `seconds` and return the screen."""
        t0 = self.clock()
        while self.clock() - t0 < seconds:
            data = self.ser.read(4096)
            if not data:
                continue
            packets, self._pending = decode_slip(self._pending + data)
            for p in packets:
                self.screen.feed(p)
        self.screen.settle()
        return self.screen

    def press(self, *keys: str, hold: float = 0.08, wait: float = 0.5) -> Screen:
        """Press keys together, release, and read what the M8 draws in response."""
        state = sum(KEYS[k] for k in keys)
        self.ser.write(bytes([0x43, state]))
        self.ser.flush()
        self.sleep(hold)
        self.ser.write(bytes([0x43, 0]))
        self.ser.flush()
        return self.read(wait)


# --- walking the menus ---------------------------------------------------------

class Navigator:
    """Moves the cursor around the Project view and opens its pages, checking the screen
    before every EDIT so a wrong keypress can never land on LOAD, NEW or CLEAR."""

    def __init__(self, link: M8Link):
        self.link = link
        self.s = link.screen

    def to_project(self, max_steps: int = 4) -> bool:
        """From the Song view (or Project itself) to the Project view."""
        self.link.read(0.3)
        for _ in range(max_steps):
            if self.s.title() == "PROJECT":
                return True
            if self.s.title() == "SONG":
                self.link.press("shift", "up", wait=0.8)
            else:                                     # some other view: step back towards Song
                self.link.press("shift", "left", wait=0.8)
        return self.s.title() == "PROJECT"

    def goto(self, label: str, field_text: str, max_steps: int = 24) -> bool:
        """Put the cursor on the field starting with `field_text` in the row starting with `label`."""
        if self.s.cursor is None:
            self.link.press("down")
            self.link.press("up")
        for _ in range(max_steps):
            row, under = self.s.cursor_field()
            if row.startswith(label):
                if under.startswith(field_text):
                    return True
                before = under
                self.link.press("right")
                if self.s.cursor_field()[1] == before:
                    return False
                continue
            want = self.s.find_row(label)
            if want is None or self.s.cursor is None:
                return False
            self.link.press("down" if self.s.cursor[1] < want else "up")
        return False

    def open_page(self, label: str, field_text: str) -> str | None:
        """Open the page behind a Project-view entry and return its text; None if not reached."""
        if not self.goto(label, field_text):
            return None
        self.link.press("edit", wait=1.2)
        return self.s.text()

    def back(self) -> str:
        self.link.press("shift", "left", wait=1.0)
        return self.s.title()


def parse_settings(text: str) -> dict[str, str]:
    """'LABEL   VALUE' rows of a settings page as a dict: a label fills at most 13 columns,
    column 13 is blank, the value starts at column 14. Titles and table rows do not match."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        label, value = line[:13].strip(), line[14:].strip()
        if label and value and line[13:14] == " " and not line.startswith(" ") and label.upper() == label:
            out[label] = value
    return out


def parse_track_table(text: str) -> dict[str, list[str]]:
    """The Track MIDI Input table (CHAN. and INST# rows) as lists of 8 values."""
    table: dict[str, list[str]] = {}
    for line in text.splitlines():
        for key in ("CHAN.", "INST#"):
            if line.strip().startswith(key):
                table[key] = line.strip()[len(key):].split()[:8]
        if line.strip().startswith("PG CHANGE"):          # "PG CHANGE ON      MODE LEGATO"
            tokens = line.split()
            if len(tokens) >= 5:
                table["PG CHANGE"] = [tokens[2]]
                table["MODE"] = [tokens[4]]
    return table


def mapping_count(text: str) -> int:
    """Rows of the MIDI MAPPING page that hold a mapping (a channel other than --)."""
    n = 0
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and len(parts[0]) == 2 and all(c in "0123456789ABCDEF" for c in parts[0]) \
                and parts[1].isdigit():
            n += 1
    return n


def read_info(link: M8Link) -> dict[str, Any]:
    """Model, firmware, project name and the three settings pages, ending in the Project view."""
    link.enable()
    link.read(1.5)
    info: dict[str, Any] = dict(link.screen.info or {})
    nav = Navigator(link)
    info["view_at_start"] = link.screen.title()
    if not nav.to_project():
        info["error"] = f"could not reach the Project view (at {link.screen.title()!r})"
        return info
    project = parse_settings(link.screen.text())
    info["project"] = project.get("NAME", "")
    info["tempo"] = project.get("TEMPO", "")
    info["transpose"] = project.get("TRANSPOSE", "")
    for key, (label, field_text) in {"system": ("SYSTEM", "SETTINGS"), "midi": ("MIDI", "SETTINGS"),
                                     "mappings": ("MIDI", "MAPPINGS")}.items():
        page = nav.open_page(label, field_text)
        if page is None:
            info[key] = None
            continue
        if key == "mappings":
            info[key] = mapping_count(page)
        else:
            info[key] = parse_settings(page)
            if key == "midi":
                info["track_midi_input"] = parse_track_table(page)
        nav.back()
    return info
