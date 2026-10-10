"""The session log: one JSON line per analysis block.

Each run writes `<dir>/session-<YYYYmmdd-HHMMSS>.jsonl`. Only features, mood,
lock state and controls are logged; no audio is ever stored (spec section 13).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from roomtone.types import BlockResult


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def block_json(result: BlockResult | dict) -> str:
    """One block as a single-line JSON string (numpy scalars converted)."""
    data = result.as_dict() if isinstance(result, BlockResult) else result
    return json.dumps(data, default=_json_default)


class BlockLog:
    """Appends one JSON line per block to a fresh session file under `dir`."""

    def __init__(self, dir: str | Path):
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.path = self.dir / f"session-{stamp}.jsonl"

    def write(self, result: BlockResult | dict) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(block_json(result) + "\n")
