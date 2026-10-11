import json
import sys

import pytest

from roomtone import cli
from tests.fakes import install_fake_modules

SUBCOMMANDS = [["devices"], ["aggregate"], ["bench"], ["bench", "latency"], ["bench", "marker"],
               ["bench", "midi"], ["baseline"], ["run"]]


@pytest.fixture
def no_sounddevice(monkeypatch):
    monkeypatch.setitem(sys.modules, "sounddevice", None)


@pytest.mark.parametrize("argv", SUBCOMMANDS + [[]])
def test_help_parses(argv, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(argv + ["--help"])
    assert exc.value.code == 0
    assert "usage: roomtone" in capsys.readouterr().out


def test_devices_explains_missing_backends(no_sounddevice, monkeypatch, capsys):
    from roomtone import midi

    def no_backend():
        raise ModuleNotFoundError("No module named 'rtmidi'")

    monkeypatch.setattr(midi, "list_ports", no_backend)
    assert cli.main(["devices"]) == 0
    out = capsys.readouterr().out
    assert "sounddevice is not installed" in out and "python-rtmidi" in out


@pytest.mark.parametrize("argv", [["run"], ["bench", "latency"], ["bench", "marker"],
                                  ["baseline"]])
def test_hardware_commands_explain_without_sounddevice(argv, no_sounddevice, capsys):
    assert cli.main(argv) == 2
    assert "pip install 'roomtone[audio]'" in capsys.readouterr().out


def test_bench_midi_explains_without_rtmidi(monkeypatch, capsys):
    from roomtone import midi

    def no_backend(name):
        raise ImportError("No module named 'rtmidi'")

    monkeypatch.setattr(midi, "open_out", no_backend)
    assert cli.main(["bench", "midi"]) == 2
    assert "python-rtmidi" in capsys.readouterr().out


def test_bench_midi_sends_phrase_sweep_and_clock(monkeypatch, capsys):
    from roomtone import midi

    class Port:
        sent, closed = [], False
        def send(self, m): self.sent.append(m)
        def close(self): Port.closed = True

    class FakeTime:
        now = 0.0
        def perf_counter(self): return self.now
        def sleep(self, s): self.now += s

    monkeypatch.setattr(midi, "open_out", lambda name: Port())
    monkeypatch.setattr(midi, "time", FakeTime())
    assert cli.main(["bench", "midi", "--channel", "2", "--cc", "31"]) == 0
    types = [m.type for m in Port.sent]
    assert types.count("note_on") == 4 and types.count("clock") == 240
    ccs = [m for m in Port.sent if m.type == "control_change"]
    assert len(ccs) == 128 and {m.channel for m in ccs} == {9} and {m.control for m in ccs} == {31}
    assert {m.channel for m in Port.sent if m.type == "note_on"} == {1} and Port.closed
    assert "CONTROL MAP CHANNEL = 10" in capsys.readouterr().out


def test_run_sim_explains_missing_modules(no_sounddevice, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "roomtone.lock", None)
    assert cli.main(["run", "--sim", "--dry-run"]) == 2
    assert "roomtone.lock is not available" in capsys.readouterr().out


def test_run_sim_dry_run(no_sounddevice, monkeypatch, tmp_path, capsys):
    install_fake_modules(monkeypatch)
    assert cli.main(["run", "--sim", "--dry-run", "--log-dir", str(tmp_path)]) == 0
    assert "Dry run" in capsys.readouterr().out


def test_run_sim_fast_blocks(no_sounddevice, monkeypatch, tmp_path, capsys):
    install_fake_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "roomtone.sim", None)  # use the fallback room
    rc = cli.main(["run", "--sim", "--fast", "--no-tidal", "--blocks", "2",
                   "--block-seconds", "0.5", "--log-dir", str(tmp_path)])
    assert rc == 0
    (logfile,) = tmp_path.glob("session-*.jsonl")
    rows = [json.loads(line) for line in logfile.read_text().splitlines()]
    assert [r["index"] for r in rows] == [0, 1]
    assert rows[1]["t_start"] == pytest.approx(0.5)
    out = capsys.readouterr().out
    assert "simulated kitchen" in out and "[   1]" in out


def test_run_sim_prints_the_script_phase(no_sounddevice, monkeypatch, tmp_path, capsys):
    install_fake_modules(monkeypatch)
    rc = cli.main(["run", "--sim", "--fast", "--no-tidal", "--blocks", "2", "--block-seconds",
                   "0.5", "--script", "party:0.5,empty:0.5", "--log-dir", str(tmp_path)])
    assert rc == 0
    lines = [l for l in capsys.readouterr().out.splitlines() if l.startswith("[")]
    assert "phase=party" in lines[0] and "phase=empty" in lines[1]


def test_run_rejects_a_bad_script(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--sim", "--script", "empty:10,disco:5"])
    assert exc.value.code == 2 and "disco" in capsys.readouterr().err


def test_run_sim_tui_missing_falls_back_to_console(no_sounddevice, monkeypatch, tmp_path, capsys):
    install_fake_modules(monkeypatch)
    monkeypatch.setitem(sys.modules, "roomtone.sim", None)
    monkeypatch.setitem(sys.modules, "roomtone.tui", None)
    rc = cli.main(["run", "--sim", "--fast", "--tui", "--no-tidal", "--blocks", "1",
                   "--block-seconds", "0.5", "--log-dir", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "roomtone.tui is not available" in out and "[   0]" in out


# aggregate -------------------------------------------------------------------

class FakeCoreAudio:
    """Stands in for roomtone.coreaudio: a laptop's devices, with create/destroy recorded."""

    def __init__(self, with_aggregate=False):
        from roomtone import coreaudio as ca
        self.ca = ca
        self.devs = [
            ca.Device(1, "m8:in", "M8", "usb", 2, 0, 44100.0, [44100.0]),
            ca.Device(2, "m8:out", "M8", "usb", 0, 2, 44100.0, [44100.0]),
            ca.Device(3, "mic", "MacBook Air Microphone", "built-in", 1, 0, 48000.0, [44100.0, 48000.0]),
            ca.Device(4, "spk", "MacBook Air Speakers", "built-in", 0, 2, 48000.0, [44100.0, 48000.0]),
        ]
        self.created, self.destroyed = [], []
        if with_aggregate:
            self._add_aggregate()

    def _add_aggregate(self):
        self.agg = self.ca.Device(9, "agg", "roomtone", "aggregate", 3, 2, 44100.0, [44100.0])
        self.devs.append(self.agg)

    def devices(self):
        return list(self.devs)

    def find(self, name, **kw):
        return self.ca.find(name, among=self.devs, **kw)

    def aggregate_members(self, agg):
        return ["m8:in", "mic", "spk"], "spk"

    def create_aggregate(self, name, members, clock, sample_rate=None, uid=None):
        self.created.append((name, [(d.uid, drift) for d, drift in members], clock.uid, sample_rate))
        self._add_aggregate()
        return self.agg

    def destroy_aggregate(self, dev):
        self.destroyed.append(dev.uid)
        self.devs = [d for d in self.devs if d.uid != dev.uid]


@pytest.fixture
def fake_coreaudio(monkeypatch):
    def install(**kw):
        fake = FakeCoreAudio(**kw)
        monkeypatch.setattr(cli, "_coreaudio", lambda: fake)
        return fake
    return install


def test_aggregate_create_builds_m8_mic_output_with_output_as_clock(fake_coreaudio, capsys):
    fake = fake_coreaudio()
    assert cli.main(["aggregate", "create"]) == 0
    assert fake.created == [("roomtone", [("m8:in", True), ("mic", True), ("spk", False)], "spk", 44100)]
    out = capsys.readouterr().out
    assert "'M8': input channels 0..1" in out and "'MacBook Air Microphone': input channels 2..2" in out
    assert "'MacBook Air Speakers': output channels 0..1  (clock source)" in out


def test_aggregate_create_refuses_a_duplicate_and_explains_a_missing_device(fake_coreaudio, capsys):
    fake = fake_coreaudio(with_aggregate=True)
    assert cli.main(["aggregate", "create"]) == 1
    assert "already exists" in capsys.readouterr().out and fake.created == []
    fake = fake_coreaudio()
    assert cli.main(["aggregate", "create", "--mic", "UMIK-1"]) == 1
    assert "no input device matching 'UMIK-1'" in capsys.readouterr().out and fake.created == []


def test_aggregate_show_and_destroy(fake_coreaudio, capsys):
    fake = fake_coreaudio()
    assert cli.main(["aggregate", "show"]) == 1
    assert "roomtone aggregate create" in capsys.readouterr().out
    fake = fake_coreaudio(with_aggregate=True)
    assert cli.main(["aggregate", "show"]) == 0
    assert "3 in, 2 out, 44100 Hz" in capsys.readouterr().out
    assert cli.main(["aggregate", "destroy"]) == 0 and fake.destroyed == ["agg"]
    assert cli.main(["aggregate", "destroy"]) == 0
    assert "nothing to do" in capsys.readouterr().out


def test_aggregate_explains_when_coreaudio_is_unreachable(monkeypatch, capsys):
    from roomtone import coreaudio

    def off_mac():
        raise OSError("CoreAudio is only available on macOS")

    monkeypatch.setattr(coreaudio, "devices", off_mac)
    assert cli.main(["aggregate", "show"]) == 2
    assert "Audio MIDI Setup" in capsys.readouterr().out
