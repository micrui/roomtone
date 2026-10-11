import pytest

from roomtone import cli, tidal

FILE = """-- header comment

setcps (60/60/4)

-- helpers
let energy = cF 0 "energy"
    voices = cF 0 "voices"

-- only a comment here

d1 $ note "0" # s "midi"
"""


def test_blocks_skip_comment_only_blocks_and_keep_comments_inside_code():
    bs = tidal.blocks(FILE)
    assert [b.splitlines()[0] for b in bs] == ["setcps (60/60/4)", "-- helpers", 'd1 $ note "0" # s "midi"']
    assert "voices = cF 0" in bs[1]


def test_ghci_block_wraps_multiline_input():
    assert tidal.ghci_block('d1 $ silence') == "d1 $ silence\n"
    assert tidal.ghci_block("let a = 1\n    b = 2\n") == ":{\nlet a = 1\n    b = 2\n:}\n"


def test_run_evaluates_blocks_then_forwards_stdin(monkeypatch, tmp_path):
    sent, lines = [], []

    class FakeSession:
        def __init__(self, boot, on_line, ghci="ghci"):
            self.boot = boot
        def send(self, code): sent.append(code)
        def alive(self): return True
        def close(self, timeout=10.0): sent.append("<closed>")

    monkeypatch.setattr(tidal, "TidalSession", FakeSession)
    path = tmp_path / "x.tidal"
    path.write_text(FILE)
    rc = tidal.run(path, tmp_path / "Boot.hs", stdin=iter(["d1 $ silence\n", "\n", "hush\n"]),
                   on_line=lines.append, settle=0.0, boot_settle=0.0)
    assert rc == 0
    assert sent == ["setcps (60/60/4)", '-- helpers\nlet energy = cF 0 "energy"\n    voices = cF 0 "voices"',
                    'd1 $ note "0" # s "midi"', "d1 $ silence", "hush", "<closed>"]
    assert any("all blocks evaluated" in l for l in lines)


def test_cli_tidal_explains_missing_ghci(monkeypatch, capsys):
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert cli.main(["tidal"]) == 2
    assert "cabal install tidal --lib" in capsys.readouterr().out


def test_cli_tidal_explains_missing_boot_file(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ghci")
    monkeypatch.setattr(tidal, "find_boot_file", lambda: None)
    f = tmp_path / "a.tidal"; f.write_text("d1 $ silence\n")
    assert cli.main(["tidal", str(f)]) == 2
    assert "BootTidal.hs not found" in capsys.readouterr().out


def test_cli_panic_sends_to_the_configured_port(monkeypatch, capsys):
    from roomtone import midi

    class Port:
        sent, closed = [], False
        def send(self, m): Port.sent.append(m)
        def close(self): Port.closed = True

    monkeypatch.setattr(midi, "open_out", lambda name: Port())
    assert cli.main(["panic"]) == 0
    assert len(Port.sent) == 16 * 130 and Port.closed
    assert "all-notes-off" in capsys.readouterr().out


def test_cli_tidal_panics_on_exit(monkeypatch, tmp_path, capsys):
    calls = []
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/ghci")
    monkeypatch.setattr(tidal, "find_boot_file", lambda: tmp_path / "Boot.hs")
    (tmp_path / "Boot.hs").write_text("")
    monkeypatch.setattr(tidal, "run", lambda *a, **k: 0)
    monkeypatch.setattr(cli, "send_panic", lambda port: calls.append(port) or True)
    f = tmp_path / "a.tidal"; f.write_text("d1 $ silence\n")
    assert cli.main(["tidal", str(f)]) == 0 and calls == ["M8"]


def test_run_hushes_and_closes_on_sigterm(monkeypatch, tmp_path):
    import os, signal, threading
    sent = []

    class FakeSession:
        def __init__(self, boot, on_line, ghci="ghci"): pass
        def send(self, code): sent.append(code)
        def alive(self): return True
        def close(self, timeout=10.0): sent.append("<closed>")

    monkeypatch.setattr(tidal, "TidalSession", FakeSession)
    path = tmp_path / "x.tidal"; path.write_text("d1 $ silence\n")

    def lines():
        yield "d2 $ silence\n"
        os.kill(os.getpid(), signal.SIGTERM)
        yield "\n"
        yield "never\n"

    rc = tidal.run(path, tmp_path / "Boot.hs", stdin=lines(), on_line=lambda l: None, settle=0.0, boot_settle=0.0)
    assert rc == 0 and sent == ["d1 $ silence", "<closed>"]   # close() is what sends hush
