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
