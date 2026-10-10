import json
import sys

import pytest

from roomtone import cli
from tests.fakes import install_fake_modules

SUBCOMMANDS = [["devices"], ["bench"], ["bench", "latency"], ["bench", "marker"],
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
