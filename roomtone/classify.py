"""Optional sound-event labels for a block of room audio.

The core features (roomtone.features) are hand-made measurements. A pretrained
classifier adds names for what is in the room ("Speech", "Dishes, pots, and
pans", "Silence"), which is useful in the log and later for finer decisions.

Anything with a `labels(audio, sample_rate) -> dict[str, float]` method is a
Classifier. Two are provided:

NullClassifier
    Returns no labels. The default, so roomtone runs with numpy and scipy only.

YamnetClassifier
    Google's YAMNet (521 AudioSet classes) as a TFLite model, run locally. The
    model expects 16 kHz mono float32 audio in -1..1 and scores one frame every
    0.48 s, each looking at 0.96 s of audio. We resample the block to 16 kHz,
    run it, average each class's score over all frames, and return the 8 best.

    The TFLite runtime (tflite_runtime, or TensorFlow's bundled copy) is
    imported only when labels are first requested, never when this module is
    imported, and is installed with the `roomtone[yamnet]` extra. The class
    names come from `yamnet_class_map.csv` (columns index, mid, display_name)
    in the same folder as the model file.

    Two TFLite exports exist and both are handled: one takes a waveform of any
    length and returns scores shaped [frames, 521]; the other takes a fixed
    15600-sample window (0.975 s) and returns [1, 521], in which case we slide
    that window along the block with a 0.48 s hop.
"""

from __future__ import annotations

import csv
import logging
import os
from math import gcd
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np
from scipy.signal import resample_poly

log = logging.getLogger(__name__)

YAMNET_RATE = 16_000
YAMNET_WINDOW = 15_600        # 0.975 s, the shortest input that yields one frame
YAMNET_HOP = 7_680            # 0.48 s
YAMNET_CLASSES = 521
TOP_K = 8
MODEL_ENV = "ROOMTONE_YAMNET_MODEL"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "yamnet.tflite"
CLASS_MAP_NAME = "yamnet_class_map.csv"


@runtime_checkable
class Classifier(Protocol):
    def labels(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]: ...


class NullClassifier:
    """No model, no labels."""

    def labels(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        return {}


def _load_interpreter_class() -> Any:
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]
        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf  # type: ignore[import-not-found]
        return tf.lite.Interpreter
    except ImportError as exc:
        raise RuntimeError(
            "YamnetClassifier needs a TFLite runtime (tflite_runtime or tensorflow). "
            "Install it with: pip install 'roomtone[yamnet]'"
        ) from exc


def load_class_names(csv_path: Path) -> list[str]:
    """display_name column of the YAMNet class map, in index order."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: int(r["index"]))
    return [r["display_name"] for r in rows]


def to_yamnet_input(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Mono float32 at 16 kHz, clipped to -1..1, padded to at least one YAMNet window."""
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x = np.ravel(x)
    if sample_rate != YAMNET_RATE:
        g = gcd(int(sample_rate), YAMNET_RATE)
        x = resample_poly(x, YAMNET_RATE // g, int(sample_rate) // g)
    if x.size < YAMNET_WINDOW:
        x = np.pad(x, (0, YAMNET_WINDOW - x.size))
    return np.clip(x, -1.0, 1.0).astype(np.float32)


class YamnetClassifier:
    """YAMNet TFLite labels, loaded on first use. See the module docstring."""

    def __init__(self, model_path: str | None = None, top_k: int = TOP_K) -> None:
        self.model_path = Path(model_path or os.environ.get(MODEL_ENV) or DEFAULT_MODEL)
        self.top_k = top_k
        self._interp: Any = None
        self._names: list[str] = []
        self._in: dict[str, Any] = {}
        self._out_index = -1
        self._fixed_len: int | None = None

    def _ensure_loaded(self) -> None:
        if self._interp is not None:
            return
        interpreter_cls = _load_interpreter_class()
        if not self.model_path.is_file():
            raise RuntimeError(f"YAMNet model not found at {self.model_path} "
                               f"(pass model_path or set {MODEL_ENV})")
        class_map = self.model_path.parent / CLASS_MAP_NAME
        if not class_map.is_file():
            raise RuntimeError(f"YAMNet class map not found at {class_map}")
        self._names = load_class_names(class_map)
        interp = interpreter_cls(model_path=str(self.model_path))
        self._in = interp.get_input_details()[0]
        shape = list(self._in.get("shape_signature", self._in["shape"]))
        self._fixed_len = None if -1 in shape else int(np.prod(self._in["shape"]))
        interp.allocate_tensors()
        outputs = interp.get_output_details()
        scores = [o for o in outputs if int(o["shape"][-1]) == YAMNET_CLASSES]
        self._out_index = (scores[0] if scores else outputs[0])["index"]
        self._interp = interp
        log.info("YAMNet loaded from %s (%s input, %d classes)", self.model_path,
                 "fixed" if self._fixed_len else "variable", len(self._names))

    def _run(self, waveform: np.ndarray) -> np.ndarray:
        """Scores shaped [frames, 521] for one model input."""
        interp = self._interp
        if self._fixed_len is None:
            interp.resize_tensor_input(self._in["index"], [waveform.size], strict=False)
            interp.allocate_tensors()
        else:
            waveform = waveform.reshape(self._in["shape"])
        interp.set_tensor(self._in["index"], waveform)
        interp.invoke()
        return np.asarray(interp.get_tensor(self._out_index)).reshape(-1, YAMNET_CLASSES)

    def scores(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Per-frame class scores for the whole block, shape [frames, 521]."""
        self._ensure_loaded()
        x = to_yamnet_input(audio, sample_rate)
        if self._fixed_len is None:
            return self._run(x)
        n = self._fixed_len
        starts = range(0, max(x.size - n, 0) + 1, YAMNET_HOP)
        return np.concatenate([self._run(x[s : s + n]) for s in starts], axis=0)

    def labels(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        mean = self.scores(audio, sample_rate).mean(axis=0)
        top = np.argsort(mean)[::-1][: self.top_k]
        return {(self._names[i] if i < len(self._names) else f"class {i}"): float(mean[i])
                for i in top}
