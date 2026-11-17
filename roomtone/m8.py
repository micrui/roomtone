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
        """The first row below the title whose text starts with `label`."""
        for y in self.rows():
            if y > TITLE_Y and self.row_text(y).strip().startswith(label):
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


# --- editing ---------------------------------------------------------------------

# SHIFT+arrow hops between views, as the sidebar map shows them. Titles carry a
# number ("INST. 00"); `view_name` strips it.
VIEW_GRAPH: dict[str, dict[str, str]] = {
    "PROJECT": {"shift+down": "SONG"},
    "SONG": {"shift+up": "PROJECT", "shift+right": "CHAIN", "shift+down": "MIXER"},
    "CHAIN": {"shift+left": "SONG", "shift+right": "PHRASE"},
    "PHRASE": {"shift+left": "CHAIN", "shift+right": "INST."},
    "INST.": {"shift+left": "PHRASE", "shift+right": "TABLE", "shift+up": "INST. MODS", "shift+down": "MIXER"},
    "INST. MODS": {"shift+down": "INST."},
    "TABLE": {"shift+left": "INST."},
    "MIXER": {},                        # SHIFT+UP returns to whichever row view it was entered from
}


def view_name(title: str) -> str:
    """'INST. MODS 00' -> 'INST. MODS', 'CHAIN 00' -> 'CHAIN', 'SYSTEM SETTINGS' unchanged."""
    for name in sorted(VIEW_GRAPH, key=len, reverse=True):
        if title == name or title.startswith(name + " "):
            return name
    return title


def view_route(start: str, goal: str) -> list[str] | None:
    """The hops (as 'shift+down' strings) from one view to another, or None."""
    seen = {start: []}
    queue = [start]
    while queue:
        here = queue.pop(0)
        if here == goal:
            return seen[here]
        for hop, there in VIEW_GRAPH.get(here, {}).items():
            if there not in seen:
                seen[there] = seen[here] + [hop]
                queue.append(there)
    return None


def row_tokens(screen: Screen, y: int) -> list[tuple[int, str]]:
    """(pixel x, text) of each run of characters on row y, left to right."""
    cells = sorted((x, c) for (yy, x), c in screen.chars.items() if yy == y)
    dx = screen._cell_width()
    tokens: list[tuple[int, str]] = []
    for x, c in cells:
        if tokens and x - tokens[-1][0] == dx * len(tokens[-1][1]) and c != " ":
            tokens[-1] = (tokens[-1][0], tokens[-1][1] + c)
        elif c != " ":
            tokens.append((x, c))
    return tokens


def _hexval(text: str) -> int | None:
    t = text.strip()[:2]
    try:
        return int(t, 16) if len(t) == 2 else None
    except ValueError:
        return None


class Editor(Navigator):
    """Navigation across views, cursor placement on a given cell, and value edits.

    Every edit is verified on the screen; `set_value` returns what the field shows
    afterwards, and raises if it cannot reach the target within its step budget.
    """

    CURSOR_LEFT_PX = 5          # the bracket starts this far left of the field's first character
    confirm_dialogs = True      # answer "LOSE CHANGES?" with OK while stepping values

    def view(self) -> str:
        return view_name(self.s.title())

    def goto_view(self, goal: str, wait: float = 0.8, max_hops: int = 12) -> bool:
        """Hop towards `goal` one SHIFT+arrow at a time, re-routing from wherever each hop
        actually lands: the Mixer's exits lead back to whichever row view it was entered
        from, so the map is only a guide."""
        self.link.read(0.2)
        for _ in range(max_hops):
            here = self.view()
            if here == goal:
                return True
            route = view_route(here, goal)
            if not route:
                # the Mixer, a settings page or a dialog: step back out and look again
                self.link.press("shift", "up" if here == "MIXER" else "left", wait=wait)
                continue
            self.link.press(*route[0].split("+"), wait=wait)
        return self.view() == goal

    def cursor_cell(self) -> tuple[int, int] | None:
        """(y, x of first character) the cursor sits on, or None."""
        if self.s.cursor is None:
            return None
        x0, y0, _, _ = self.s.cursor
        y = min(self.s.rows(), key=lambda yy: abs(yy - y0))
        return y, x0 + self.CURSOR_LEFT_PX

    def goto_cell(self, y: int, x: int, max_steps: int = 40) -> bool:
        """Move the cursor to the field whose first character is at pixel (x, y)."""
        if self.s.cursor is None:
            self.link.press("down", wait=0.25)
            self.link.press("up", wait=0.25)
        for _ in range(max_steps):
            cell = self.cursor_cell()
            if cell is None:
                return False
            cy, cx = cell
            if abs(cy - y) <= 3 and abs(cx - x) <= 3:
                return True
            before = cell
            if abs(cy - y) > 3:
                self.link.press("down" if cy < y else "up", wait=0.25)
            else:
                self.link.press("right" if cx < x else "left", wait=0.25)
            if self.cursor_cell() == before:
                return False               # the cursor cannot go that way
        return False

    def goto_token(self, row_prefix: str, index: int = 0, after: str | None = None) -> bool:
        """Cursor onto token `index` (counting from the first token after `after`, when given)
        of the first row whose text starts with `row_prefix`."""
        y = self.s.find_row(row_prefix)
        if y is None:
            return False
        tokens = row_tokens(self.s, y)
        if after is not None:
            k = next((i for i, (_, t) in enumerate(tokens) if t == after), None)
            if k is None:
                return False
            tokens = tokens[k + 1:]
        if index >= len(tokens):
            return False
        return self.goto_cell(y, tokens[index][0])

    def value(self) -> str:
        return self.s.cursor_field()[1]

    def set_value(self, target: str, max_steps: int = 300) -> str:
        """Step the field under the cursor to `target` with EDIT+arrows and verify.

        Hex fields ('80', 'E0') are stepped by 16 with up/down and by 1 with
        left/right; anything else (ON/OFF, MONO, SAMPLER, 01ADSR ENV) is stepped
        right until the text starts with `target`, wrapping at most once.
        """
        target = target.strip()
        current = self.value()
        if current.startswith(target):
            return current
        tv, cv = _hexval(target), _hexval(current)
        if tv is not None and cv is not None and len(target) == 2:
            steps = 0
            while steps < max_steps:
                cv = _hexval(self.value())
                if cv is None or cv == tv:
                    break
                diff = tv - cv
                if abs(diff) >= 16:
                    self.link.press("edit", "up" if diff > 0 else "down", wait=0.15)
                else:
                    self.link.press("edit", "right" if diff > 0 else "left", wait=0.15)
                steps += 1
            got = self.value()
            if _hexval(got) != tv:
                raise RuntimeError(f"could not set {self.s.cursor_field()[0]!r} to {target}: shows {got!r}")
            return got
        seen = {current}
        for direction in ("right", "left"):       # walk to one end of the list, then to the other
            for _ in range(max_steps):
                before = self.value()
                self.link.press("edit", direction, wait=0.15)
                if self.confirm_dialogs and dialog_text(self.s):
                    self.answer_dialog(ok=True)   # type: ignore[attr-defined]
                v = self.value()
                if v.startswith(target):
                    return v
                seen.add(v)
                if v == before:
                    break
        raise RuntimeError(f"could not set {self.s.cursor_field()[0]!r} to {target!r}: saw {sorted(seen)}")

    def learn(self, send_cc: Callable[[], None], hold: float = 0.4) -> None:
        """MIDI learn for the field under the cursor: hold OPTION while `send_cc` sends the CC."""
        self.link.ser.write(bytes([0x43, KEYS["option"]]))
        self.link.ser.flush()
        self.link.sleep(0.15)
        send_cc()
        self.link.sleep(hold)
        self.link.ser.write(bytes([0x43, 0]))
        self.link.ser.flush()
        self.link.read(0.4)


# --- dialogs ------------------------------------------------------------------------

def dialog_text(screen: Screen) -> str | None:
    """The question of a modal 'OK CANCEL' dialog if one is showing, else None."""
    lines = [l.strip() for l in screen.text().splitlines() if l.strip()]
    if any(l.replace(" ", "") == "OKCANCEL" for l in lines):
        return " ".join(l for l in lines if l.replace(" ", "") != "OKCANCEL")
    return None


def _answer_dialog(self: "Editor", ok: bool) -> str | None:
    """Answer a showing dialog with OK or CANCEL; returns its text, or None if no dialog."""
    text = dialog_text(self.s)
    if text is None:
        return None
    want = "OK" if ok else "CANCEL"
    for _ in range(4):
        if self.value() == want:
            break
        self.link.press("right" if want == "CANCEL" else "left", wait=0.25)
    if self.value() != want:
        self.link.press("edit", "right" if want == "CANCEL" else "left", wait=0.25)
    self.link.press("edit", hold=0.4, wait=1.0)    # a short tap does not confirm OK
    if dialog_text(self.s):
        self.link.press("edit", hold=0.5, wait=1.0)
    return text


Editor.answer_dialog = _answer_dialog  # type: ignore[attr-defined]


# --- applying a song definition --------------------------------------------------------

def _goto_row_token(ed: "Editor", y: int, index_after_label: int) -> bool:
    tokens = row_tokens(ed.s, y)
    if index_after_label >= len(tokens):
        return False
    return ed.goto_cell(y, tokens[index_after_label][0])


def select_instrument(ed: "Editor", number: int) -> bool:
    """Show instrument `number` in the Instrument view, via a phrase step's I column."""
    if not ed.goto_view("PHRASE"):
        return False
    y = ed.s.rows()[2] if len(ed.s.rows()) > 2 else None   # the first step row, under the header
    y = ed.s.find_row("<0") or ed.s.find_row("0 ") or y
    tokens = row_tokens(ed.s, y)
    if len(tokens) < 4:
        return False
    if not ed.goto_cell(y, tokens[3][0]):                  # N V I ...: the I column
        return False
    if ed.value().strip() == "--":                         # an empty step: give it a note first
        ed.goto_cell(y, tokens[1][0])
        ed.link.press("edit", wait=0.3)
        ed.goto_cell(y, tokens[3][0])
    ed.set_value(f"{number:02X}")
    return ed.goto_view("INST.") and ed.s.title() == f"INST. {number:02X}"


def apply_instrument(ed: "Editor", spec: Any, log: Callable[[str], None] = print) -> list[str]:
    """Write one Instrument spec; returns the fields that did not verify."""
    problems: list[str] = []
    if not select_instrument(ed, spec.number):
        return [f"could not open instrument {spec.number:02X} (at {ed.s.title()!r})"]
    ed.goto_token("TYPE", 0, after="TYPE")
    got = ed.set_value(spec.type)
    if dialog_text(ed.s):
        ed.answer_dialog(ok=True)   # type: ignore[attr-defined]
    log(f"  INST {spec.number:02X} {spec.name}: TYPE {got}")
    for row, after, value in spec.inst:
        try:
            if not ed.goto_token(row, 0, after=after):
                problems.append(f"{spec.number:02X} {row}/{after}: field not found")
                continue
            got = ed.set_value(value)
            log(f"    {row:<8}{after:<8} {got}")
        except RuntimeError as exc:
            problems.append(str(exc))
    if spec.mods:
        if not ed.goto_view("INST. MODS"):
            return problems + [f"{spec.number:02X}: MODS page not reached"]
        for mod, fld, value in spec.mods:
            header = f"MOD{1 if mod in (1, 3) else 2}"
            y0 = ed.s.find_row(header)
            if y0 is None:
                problems.append(f"{spec.number:02X} MOD{mod}: row not found")
                continue
            col = 1 if mod in (1, 2) else 3                 # left pair or right pair of the row
            if fld == "TYPE":
                ok = _goto_row_token(ed, y0, col)
            else:
                # the field's row is below the header; find it by label in the right column
                ok = False
                for k in range(1, 8):
                    y = y0 + ROW_PX * k
                    toks = row_tokens(ed.s, y)
                    labels = [t for _, t in toks]
                    if fld in labels:
                        idx = labels.index(fld) if mod in (1, 2) else len(labels) - 1 - labels[::-1].index(fld)
                        ok = ed.goto_cell(y, toks[idx + 1][0]) if idx + 1 < len(toks) else False
                        break
            if not ok:
                problems.append(f"{spec.number:02X} MOD{mod} {fld}: field not found")
                continue
            try:
                got = ed.set_value(value)
                log(f"    MOD{mod} {fld:<5} {got}")
            except RuntimeError as exc:
                problems.append(str(exc))
        ed.goto_view("INST.")
    return problems


# --- MIDI learn --------------------------------------------------------------------------

def _mixer_rows(ed: "Editor") -> tuple[int, int]:
    """(y of the track volume row, y of the MX/DE/RE effect level row) in the Mixer."""
    s = ed.s
    two = {y: [t for _, t in row_tokens(s, y) if len(t) == 2 and _hexval(t) is not None] for y in s.rows()}
    vol = min(y for y, toks in two.items() if len(toks) >= 8)
    fx = min(y for y, toks in two.items() if y > vol and len(toks) >= 3)
    return vol, fx


def goto_target(ed: "Editor", where: tuple) -> bool:
    """Put the cursor on a MIDI-learn target (see roomtone.m8_song.MAPPINGS)."""
    kind = where[0]
    if kind == "project":
        return ed.goto_view("PROJECT") and ed.goto(where[1], "")
    if kind == "inst":
        return select_instrument(ed, where[1]) and ed.goto_token(where[2], 0, after=where[2])
    if kind in ("mixer-track", "mixer-fx"):
        if not ed.goto_view("MIXER"):
            return False
        vol, fx = _mixer_rows(ed)
        if kind == "mixer-track":
            toks = row_tokens(ed.s, vol)
            return ed.goto_cell(vol, toks[where[1] - 1][0])
        index = {"MX": 0, "DE": 1, "RE": 2}[where[1]]
        return ed.goto_cell(fx, row_tokens(ed.s, fx)[index][0])
    if kind == "eq":
        if not ed.goto_view("MIXER"):
            return False
        cell = next(((y, x) for y in ed.s.rows() for x, t in row_tokens(ed.s, y) if t == "EQ"), None)
        if cell is None or not ed.goto_cell(*cell):
            return False
        ed.link.press("edit", wait=1.2)
        if ed.s.title() != "MIX EQ":
            return False
        y = ed.s.find_row("GAIN")
        toks = row_tokens(ed.s, y)
        band = {"LOW": 1, "MID": 2, "HIGH": 3}[where[1]]
        return ed.goto_cell(y, toks[band][0])
    return False


def leave_target(ed: "Editor", where: tuple) -> None:
    if where[0] == "eq":
        ed.link.press("shift", "left", wait=1.0)


def learn_mappings(ed: "Editor", mappings: list[tuple[int, tuple]], send_cc: Callable[[int], None],
                   log: Callable[[str], None] = print) -> list[str]:
    """MIDI-learn each (cc, target): hold OPTION on the field while `send_cc(cc)` sends it."""
    problems: list[str] = []
    for cc, where in mappings:
        if not goto_target(ed, where):
            problems.append(f"CC {cc}: target {where} not reached (at {ed.s.title()!r})")
            continue
        label = ed.s.cursor_field()
        ed.learn(lambda: send_cc(cc))
        log(f"  CC {cc:<3} -> {ed.s.title():<10} {label}")
        leave_target(ed, where)
    return problems


# --- naming and saving --------------------------------------------------------------------

NAME_LENGTH = 12


def set_project_name(ed: "Editor", name: str, max_steps: int = 80) -> str:
    """Type `name` into the Project view's NAME field, one character at a time.

    EDIT on the field enters character mode: the cursor box sits over one character
    (7 px left of it), LEFT/RIGHT move along the name, EDIT+arrows change the
    character. Unused positions are stepped to blank. Returns the name shown afterwards.
    """
    name = name.upper()[:NAME_LENGTH]
    if not (ed.goto_view("PROJECT") and ed.goto("NAME", "")):
        raise RuntimeError("could not reach the NAME field")
    y = ed.s.find_row("NAME")
    dx = ed.s._cell_width()
    x_first = min(x for (yy, x), c in ed.s.chars.items() if yy == y and x >= 14 * dx and c != " ")
    ed.link.press("edit", wait=0.5)                     # character mode
    for _ in range(NAME_LENGTH):                        # to the first character
        before = ed.s.cursor
        ed.link.press("left", wait=0.12)
        if ed.s.cursor == before:
            break
    for i in range(NAME_LENGTH):
        want = name[i] if i < len(name) else " "
        x = x_first + dx * i
        if i > 0:
            before = ed.s.cursor
            ed.link.press("right", wait=0.12)
            if ed.s.cursor == before:
                break                                   # no more positions
        def have() -> str:
            return ed.s.chars.get((y, x), " ")
        if have() == want:
            continue
        for direction in ("right", "left"):
            for _ in range(max_steps):
                before_c = have()
                ed.link.press("edit", direction, wait=0.1)
                if have() == want or have() == before_c:
                    break
            if have() == want:
                break
    ed.link.press("down", wait=0.4)                     # leave character mode
    ed.link.read(0.3)
    return ed.s.row_text(y)[14:].strip()


def save_project(ed: "Editor") -> str | None:
    """PROJECT -> SAVE in the Project view; answers an overwrite dialog with OK.
    Returns the dialog text if one was shown."""
    if not (ed.goto_view("PROJECT") and ed.goto("PROJECT", "SAVE")):
        raise RuntimeError("could not reach PROJECT / SAVE")
    ed.link.press("edit", hold=0.4, wait=1.5)
    text = dialog_text(ed.s)
    if text:
        ed.answer_dialog(ok=True)   # type: ignore[attr-defined]
    ed.link.read(1.0)
    return text
