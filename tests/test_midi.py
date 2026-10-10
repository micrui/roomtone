import pytest

from roomtone import midi


class Port:
    def __init__(self):
        self.sent = []

    def send(self, msg):
        self.sent.append(msg)


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def sleep(self, s):
        self.now += s


def test_channels_are_one_based():
    port = Port()
    midi.send_cc(port, 10, 30, 200)
    midi.send_note(port, 1, 60, 100, 0.0, sleep=lambda s: None)
    cc, on, off = port.sent
    assert (cc.type, cc.channel, cc.control, cc.value) == ("control_change", 9, 30, 127)
    assert (on.type, on.channel, on.note) == ("note_on", 0, 60) and off.type == "note_off"
    with pytest.raises(ValueError):
        midi.send_cc(port, 0, 1, 1)


def test_clock_is_24_ppqn_and_paced(monkeypatch):
    t = FakeTime()
    monkeypatch.setattr(midi, "time", t)
    port = Port()
    ticks = midi.send_clock(port, 60.0, 2.0)
    assert ticks == 48 and all(m.type == "clock" for m in port.sent)
    assert t.now == pytest.approx(47 / 24)


def test_cc_sweep_covers_range_over_duration(monkeypatch):
    t = FakeTime()
    monkeypatch.setattr(midi, "time", t)
    port = Port()
    values = midi.cc_sweep(port, 10, 30, 0, 127, 4.0)
    assert values == list(range(128)) and [m.value for m in port.sent] == values
    assert t.now == pytest.approx(4.0)
    assert midi.cc_sweep(Port(), 10, 30, 5, 2, 0.0) == [5, 4, 3, 2]


def test_open_out_matches_substring(monkeypatch):
    monkeypatch.setattr(midi.mido, "get_output_names", lambda: ["IAC Bus", "M8 MIDI 1"])
    monkeypatch.setattr(midi.mido, "open_output", lambda name=None: f"opened {name}")
    assert midi.open_out("m8") == "opened M8 MIDI 1"
    with pytest.raises(LookupError):
        midi.open_out("nope")
