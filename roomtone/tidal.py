"""Run the Tidal file headlessly: `roomtone tidal` is a Tidal editor without the editor.

It starts ghci with Tidal's own BootTidal.hs, evaluates tidal/roomtone.tidal block by
block the way an editor plugin does (blocks are separated by blank lines, multi-line
blocks are wrapped in :{ :}), then keeps ghci alive. Whatever you type afterwards is
sent to Tidal as a block, so `d3 $ silence` or `hush` work from the terminal; an empty
line ends a multi-line block. Ctrl-C sends `hush` and quits.

Needs GHC with the tidal library installed (`cabal install tidal --lib`) and SuperDirt
running (`sclang sc/startup.scd`); without SuperDirt Tidal starts but nothing sounds.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterable

def _say(line: str) -> None:
    """Default line sink: print and flush, so a redirected log shows progress as it happens."""
    print(line, flush=True)


BOOT_SETTLE_S = 6.0      # BootTidal needs a moment to connect to SuperDirt before the first block
BLOCK_SETTLE_S = 1.0     # between blocks, so ghci's output stays readable and ordered


def blocks(text: str) -> list[str]:
    """Code blocks of a .tidal file: runs of non-blank lines that contain something
    other than comments."""
    out: list[str] = []
    cur: list[str] = []
    for line in text.splitlines():
        if line.strip():
            cur.append(line)
        elif cur:
            out.append("\n".join(cur))
            cur = []
    if cur:
        out.append("\n".join(cur))
    return [b for b in out if any(not l.lstrip().startswith("--") for l in b.splitlines())]


def ghci_block(code: str) -> str:
    """One block as ghci accepts it: multi-line input wrapped in :{ :}."""
    code = code.rstrip("\n")
    return code + "\n" if "\n" not in code else ":{\n" + code + "\n:}\n"


def find_boot_file() -> Path | None:
    """BootTidal.hs from the installed tidal package, or None."""
    env = os.environ.get("TIDAL_BOOT")
    if env and Path(env).exists():
        return Path(env)
    ghc_pkg = shutil.which("ghc-pkg")
    if ghc_pkg:
        try:
            out = subprocess.run([ghc_pkg, "field", "tidal", "data-dir", "--simple-output"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
            for line in out.splitlines():
                p = Path(line.strip()) / "BootTidal.hs"
                if p.exists():
                    return p
        except (OSError, subprocess.SubprocessError):
            pass
    home = Path.home()
    for pattern in ("~/.local/state/cabal/store/*/tdl-*/share/BootTidal.hs",
                    "~/.cabal/store/*/tdl-*/share/BootTidal.hs",
                    "~/.cabal/share/*/tidal-*/BootTidal.hs"):
        hits = sorted(glob.glob(str(home / pattern.lstrip("~/"))))
        if hits:
            return Path(hits[-1])
    return None


class TidalSession:
    """A ghci running Tidal. `send` evaluates one block; `close` hushes and quits."""

    def __init__(self, boot: Path, on_line: Callable[[str], None] = _say, ghci: str = "ghci"):
        self.on_line = on_line
        self.proc = subprocess.Popen([ghci, "-ghci-script", str(boot)], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1)
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.on_line(line.rstrip("\n"))

    def send(self, code: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(ghci_block(code))
        self.proc.stdin.flush()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self, timeout: float = 10.0) -> None:
        if self.alive():
            try:
                self.send("hush")
                self.send(":q")
                self.proc.wait(timeout=timeout)
            except (OSError, subprocess.SubprocessError):
                self.proc.kill()


def run(path: Path, boot: Path, keep: bool = True, stdin: Iterable[str] | None = None,
        on_line: Callable[[str], None] = _say, settle: float = BLOCK_SETTLE_S,
        boot_settle: float = BOOT_SETTLE_S, ghci: str = "ghci") -> int:
    """Evaluate every block of `path`, then forward `stdin` lines (blocks end at a blank
    line) until it ends or Ctrl-C. Returns 0, or 1 if ghci died."""
    session = TidalSession(boot, on_line, ghci=ghci)
    try:
        time.sleep(boot_settle)
        for i, block in enumerate(blocks(path.read_text())):
            if not session.alive():
                break
            on_line(f"--- block {i + 1}: {block.splitlines()[0][:72]}")
            session.send(block)
            time.sleep(settle)
        if not session.alive():
            on_line("ghci exited; is the tidal library installed? (cabal install tidal --lib)")
            return 1
        on_line("--- all blocks evaluated; type Tidal code (blank line sends it), Ctrl-C to hush and quit")
        if keep:
            pending: list[str] = []
            for line in (stdin if stdin is not None else sys.stdin):
                line = line.rstrip("\n")
                if line.strip():
                    pending.append(line)
                elif pending:
                    session.send("\n".join(pending))
                    pending = []
            if pending:
                session.send("\n".join(pending))
    except KeyboardInterrupt:
        on_line("\nhush")
    finally:
        session.close()
    return 0
