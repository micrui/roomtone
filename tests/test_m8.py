"""The M8 display protocol without an M8: SLIP framing, draw commands, cursor, pages."""

import struct

import pytest

from roomtone import cli, m8


def char(c, x, y):
    return bytes([0xFD, ord(c)]) + struct.pack("<HH", x, y) + bytes(6)


def rect(x, y, w, h, rgb=None):
    p = bytes([0xFE]) + struct.pack("<HHHH", x, y, w, h)
    return p + bytes(rgb) if rgb is not None else p


def slip(*packets):
    out = bytearray()
    for p in packets:
        for b in p:
            if b == m8.END:
                out += bytes([m8.ESC, m8.ESC_END])
            elif b == m8.ESC:
                out += bytes([m8.ESC, m8.ESC_ESC])
            else:
                out.append(b)
        out.append(m8.END)
    return bytes(out)


def cursor_batch(x, y, w, h, rgb=(0, 252, 248)):
    """The 12 bracket pieces the M8 draws for a cursor, first one with the colour."""
    pieces = [rect(x, y, 3, 1, rgb)]
    for dx, dy in [(w, 0), (0, h), (w, h)]:
        pieces.append(rect(x + dx, y + dy, 3, 1))
    for dx, dy in [(0, 1), (0, h - 2), (1, 1), (1, h - 1), (w + 3, 1), (w + 3, h - 2), (w + 2, 1), (w + 2, h - 1)]:
        pieces.append(rect(x + dx, y + dy, 1, 1))
    return pieces


def test_slip_splits_packets_and_unescapes():
    data = slip(b"\x01\xc0\x02", b"\x03\xdb")
    packets, rest = m8.decode_slip(data + b"\x04\x05")
    assert packets == [b"\x01\xc0\x02", b"\x03\xdb"] and rest == b"\x04\x05"


def test_screen_reads_info_characters_and_cursor():
    s = m8.Screen()
    s.feed(bytes([0xFF, 3, 6, 2, 2, 1]))
    for i, c in enumerate("TEMPO"):
        s.feed(char(c, 12 + 12 * i, 70))
    for i, c in enumerate("120"):
        s.feed(char(c, 180 + 12 * i, 70))
    for i, c in enumerate("PROJECT"):
        s.feed(char(c, 12 + 12 * i, 42))
    for p in cursor_batch(175, 70, 40, 17):
        s.feed(p)
    s.settle()
    assert s.info == {"model": "Model:02", "firmware": "6.2.2", "font": 1}
    assert s.title() == "PROJECT" and s.row_text(70) == "TEMPO         120"
    for i, c in enumerate("T>120"):
        s.feed(char(c, 420 + 12 * i, 70))
    assert s.text().splitlines()[1] == "TEMPO         120"     # the sidebar is cut off
    assert "T>120" in s.text(main_only=False)
    assert s.cursor == (175, 70, 43, 17)
    assert s.cursor_field() == ("TEMPO", "120")
    # a black batch is an erase, not a cursor; a screen clear forgets the text under it
    for p in cursor_batch(175, 70, 40, 17, rgb=(0, 0, 0)):
        s.feed(p)
    s.settle()
    assert s.cursor == (175, 70, 43, 17)
    s.feed(rect(0, 60, 480, 260, (0, 0, 0)))
    assert s.title() == "PROJECT" and s.row_text(70) == ""


def test_parsers_read_settings_tables_and_mapping_rows():
    page = ("SYSTEM SETTINGS\n\nBACKLIGHT     80\nFONT OPTIONS  BOLD UPPERCASE\n"
            "USB AUDIO OUT MIX\n  not a label  x\n")
    assert m8.parse_settings(page) == {"BACKLIGHT": "80", "FONT OPTIONS": "BOLD UPPERCASE",
                                       "USB AUDIO OUT": "MIX"}
    table = ("TRACK MIDI INPUT\n      1  2  3\nCHAN. 01 02 03 04 05 06 07 08\n"
             "INST# 00 01 02 03 04 05 06 07\nPG CHANGE OFF     MODE MONO\n")
    assert m8.parse_track_table(table) == {"CHAN.": ["01", "02", "03", "04", "05", "06", "07", "08"],
                                           "INST#": ["00", "01", "02", "03", "04", "05", "06", "07"],
                                           "PG CHANGE": ["OFF"], "MODE": ["MONO"]}
    mapping = "   CH CTL V  RANGE DEST\n00 10 020 -- 00>7F TEMPO\n01 10 030 -- 00>7F I00 CUT\n02 -- --- -- -->--\n"
    assert m8.mapping_count(mapping) == 2


class FakeM8:
    """A Project page with a movable cursor, driven by the key bytes the link writes."""

    ROWS = [("TEMPO", ["120.00"]), ("MIDI", ["SETTINGS", "MAPPINGS"]), ("PROJECT", ["LOAD", "SAVE", "NEW"]),
            ("SYSTEM", ["SETTINGS"])]

    def __init__(self):
        self.row, self.col, self.view = 0, 0, "PROJECT"
        self.out = b""
        self.edits = []
        self._draw_all()

    def _draw_all(self):
        pk = [rect(0, 0, 320, 320, (0, 0, 0)), bytes([0xFF, 3, 6, 2, 2, 1])]
        pk += [char(c, 12 + 12 * i, 42) for i, c in enumerate(self.view)]
        if self.view == "PROJECT":
            for r, (label, fields) in enumerate(self.ROWS):
                y = 70 + 14 * r
                pk += [char(c, 12 + 12 * i, y) for i, c in enumerate(label)]
                x = 180                                   # values start in column 14, as on the M8
                for f in fields:
                    pk += [char(c, x + 12 * i, y) for i, c in enumerate(f)]
                    x += 12 * (len(f) + 1)
            pk += cursor_batch(*self._cursor_box())
        self.out += slip(*pk)

    def _cursor_box(self):
        label, fields = self.ROWS[self.row]
        x = 180 + sum(12 * (len(f) + 1) for f in fields[:self.col])
        return x - 5, 70 + 14 * self.row, 12 * len(fields[self.col]) - 2, 17   # the bracket sits 5 px left

    def write(self, data):
        if data[:1] == b"C" and len(data) == 2 and data[1]:
            k = data[1]
            if self.view == "PROJECT" and k == m8.KEYS["down"]:
                self.row = min(self.row + 1, len(self.ROWS) - 1)
                self.col = min(self.col, len(self.ROWS[self.row][1]) - 1)
            elif self.view == "PROJECT" and k == m8.KEYS["up"]:
                self.row = max(self.row - 1, 0)
                self.col = min(self.col, len(self.ROWS[self.row][1]) - 1)
            elif self.view == "PROJECT" and k == m8.KEYS["right"]:
                self.col = min(self.col + 1, len(self.ROWS[self.row][1]) - 1)
            elif self.view == "PROJECT" and k == m8.KEYS["edit"]:
                label, fields = self.ROWS[self.row]
                self.edits.append((label, fields[self.col]))
                self.view = f"{label} {fields[self.col]}"
            elif k == m8.KEYS["shift"] | m8.KEYS["left"]:
                self.view = "PROJECT"
            self._draw_all()

    def flush(self):
        pass

    def read(self, n):
        data, self.out = self.out[:n], self.out[n:]
        return data

    def close(self):
        pass


def fake_link():
    fake = FakeM8()
    t = [0.0]

    def clock():
        t[0] += 0.1
        return t[0]

    link = m8.M8Link("fake", opener=lambda p: fake, clock=clock, sleep=lambda s: None)
    return fake, link


def test_navigator_opens_the_right_entry_and_never_edits_elsewhere():
    fake, link = fake_link()
    link.enable()
    link.read(0.5)
    nav = m8.Navigator(link)
    assert nav.to_project()
    assert nav.goto("MIDI", "MAPPINGS") and link.screen.cursor_field() == ("MIDI", "MAPPINGS")
    assert nav.open_page("SYSTEM", "SETTINGS") is not None
    assert fake.edits == [("SYSTEM", "SETTINGS")] and link.screen.title() == "SYSTEM SETTINGS"
    assert nav.back() == "PROJECT"
    assert not nav.goto("NOWHERE", "X") and fake.edits == [("SYSTEM", "SETTINGS")]


def test_cli_m8_explains_missing_pyserial_or_device(monkeypatch, capsys):
    def no_serial(port=None, **kw):
        raise ImportError("No module named 'serial'")

    monkeypatch.setattr(m8, "M8Link", no_serial)
    assert cli.main(["m8", "info"]) == 2
    assert "pip install 'roomtone[m8]'" in capsys.readouterr().out
    monkeypatch.setattr(m8, "M8Link", lambda port=None, **kw: (_ for _ in ()).throw(LookupError("no M8 serial port found")))
    assert cli.main(["m8", "screen"]) == 2
    assert "no M8 serial port" in capsys.readouterr().out
