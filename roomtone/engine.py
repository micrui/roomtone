"""The analysis loop: every block, read the rings, lock, listen, decide, publish.

Once per block (30 s by default) the Engine:

1. takes the next block of reference (what the speakers played) and mic (what
   the room heard) out of the audio rings, frame for frame the same stretch;
2. finds the marker emission that started inside that block and tells the
   Locker where it is, relative to the block start;
3. runs the Locker (delay, drift, room response, subtraction), then feature
   extraction on what is left, then the mood tracker, then the Controller;
4. bundles it all into a BlockResult, sends it to every publisher (Tidal, TUI)
   and appends it to the session log;
5. hands the Locker's current marker parameters back to the marker scheduler,
   which is how the escalation ladder reaches the speakers.

`MarkerSource` is the small adapter that sits in the audio callback: it pulls
the next stretch of marker from the scheduler and scales it to the music level.
"""

from __future__ import annotations

import bisect
import logging
import threading
from typing import Any, Callable, Iterable

import numpy as np

from roomtone.audio import AudioIO, RingRangeError
from roomtone.config import Config, MarkerParams
from roomtone.log import BlockLog
from roomtone.types import BlockResult, db

logger = logging.getLogger(__name__)

Extract = Callable[..., Any]  # (residual, sample_rate, classifier=None) -> Features


class MarkerSource:
    """Audio-callback adapter: scheduler block, scaled to the music.

    Called as `source(n_frames, music_rms)`. With `fixed_rms` set, the measured
    music level is ignored and the marker is scaled against that fixed RMS
    instead (used for the marker-only baseline, where there is no music).
    """

    def __init__(self, scheduler: Any, params: MarkerParams,
                 fixed_rms: float | None = None):
        self.scheduler = scheduler
        self.params = params
        self.fixed_rms = fixed_rms
        self.position = 0            # output sample index of the next callback block

    def __call__(self, n_frames: int, music_rms: float) -> np.ndarray:
        rms = self.fixed_rms if self.fixed_rms is not None else music_rms
        block = self.scheduler.block(self.position, n_frames, rms)
        self.position += n_frames
        return block

    @property
    def emissions(self) -> list[int]:
        return self.scheduler.emissions

    @property
    def emitted_params(self) -> list[MarkerParams]:
        return list(getattr(self.scheduler, "emitted_params", []))

    def set_params(self, params: MarkerParams) -> None:
        self.scheduler.set_params(params)
        self.params = params


class BlockReader:
    """Hands out consecutive, non-overlapping blocks from the audio rings.

    Each block is (start_frame, reference_mono, mic). The stereo reference is
    folded to mono (mean of left and right): one mic hears both speakers.
    If the reader falls so far behind that its next block has been overwritten,
    it skips forward to the oldest whole block still held and logs a warning.
    """

    def __init__(self, audio: AudioIO, block_frames: int):
        self.audio = audio
        self.block_frames = block_frames
        self.next_start = 0

    def available(self) -> int:
        return min(self.audio.reference_ring.position, self.audio.mic_ring.position)

    def next(self) -> tuple[int, np.ndarray, np.ndarray] | None:
        n = self.block_frames
        avail = self.available()
        oldest = max(self.audio.reference_ring.oldest, self.audio.mic_ring.oldest)
        if self.next_start < oldest:
            skipped = oldest - self.next_start
            self.next_start = oldest
            logger.warning("analysis fell behind; skipped %d frames", skipped)
        if avail < self.next_start + n:
            return None
        start = self.next_start
        try:
            ref = self.audio.reference_ring.read(n, at=start)
            mic = self.audio.mic_ring.read(n, at=start)
        except RingRangeError as exc:  # overwritten mid-read; the next call skips ahead
            logger.warning("block at %d lost: %s", start, exc)
            return None
        self.next_start = start + n
        return start, ref.mean(axis=1), mic[:, 0].copy()


def emission_in(emissions: list[int], start: int, end: int) -> int | None:
    """The first emission start in [start, end), relative to `start`, or None."""
    i = bisect.bisect_left(emissions, start)
    if i < len(emissions) and emissions[i] < end:
        return emissions[i] - start
    return None


def _rms_db(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    return db(float(np.sqrt(np.mean(x * x)))) if x.size else -120.0


class Engine:
    """Runs the per-block chain. Use `.step()` in tests, `.start()` or `.run_forever()` live.

    `marker` is anything with `.emissions` and `.set_params(params)`, normally
    the MarkerSource installed on the AudioIO (the default when omitted).
    `extract` defaults to roomtone.features.extract.
    """

    def __init__(self, cfg: Config, audio: AudioIO, locker: Any, mood_tracker: Any,
                 controller: Any, publishers: Iterable[Any], classifier: Any = None,
                 log: BlockLog | None = None, *, marker: Any = None,
                 extract: Extract | None = None):
        if extract is None:
            from roomtone.features import extract
        self.cfg = cfg
        self.audio = audio
        self.locker = locker
        self.mood_tracker = mood_tracker
        self.controller = controller
        self.publishers = list(publishers)
        self.classifier = classifier
        self.log = log
        self.marker = marker if marker is not None else getattr(audio, "marker_source", None)
        self.extract = extract
        self.reader = BlockReader(audio, int(round(cfg.block_seconds * cfg.sample_rate)))
        self.index = 0
        self.last_result: BlockResult | None = None
        self._pushed_params = getattr(self.marker, "params", None)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _next_emission(self) -> tuple[int, int] | None:
        """Index into the emission list and absolute start of the next block to analyse.

        Blocks are aligned to marker emissions: every analysis block begins on
        the first sample of a burst, which is the convention the Locker expects
        (reference block starts at `emission_start`, marker at sample 0). Without
        a marker source, blocks simply follow each other from the reader's position.
        """
        floor = max(self.reader.next_start,
                    self.audio.reference_ring.oldest, self.audio.mic_ring.oldest)
        if self.marker is None:
            return -1, floor
        emissions = list(self.marker.emissions)
        i = bisect.bisect_left(emissions, floor)
        if i >= len(emissions):
            return None
        return i, emissions[i]

    def step(self) -> BlockResult | None:
        """Process one block if a whole one is waiting in the rings; otherwise return None."""
        nxt = self._next_emission()
        if nxt is None:
            return None
        i, emission_start = nxt
        self.reader.next_start = emission_start
        block = self.reader.next()
        if block is None:
            return None
        start, reference, mic = block
        if start != emission_start:          # the ring overwrote it; try the next burst
            return None
        emitted = self.marker.emitted_params if self.marker is not None else []
        params = emitted[i] if 0 <= i < len(emitted) else None

        residual, lock = self.locker.process(reference, mic, emission_start, params=params)
        features = self.extract(residual, self.cfg.sample_rate, classifier=self.classifier)
        mood = self.mood_tracker.update(features, lock)
        controls = self.controller.update(mood, lock)
        result = BlockResult(
            index=self.index, t_start=start / self.cfg.sample_rate,
            lock=lock, features=features, mood=mood, controls=controls,
            residual_rms_db=_rms_db(residual), raw_rms_db=_rms_db(mic),
        )
        for pub in self.publishers:
            try:
                pub.publish(result)
            except Exception:  # one dead listener must not stop the loop
                logger.exception("publisher %s failed", type(pub).__name__)
        if self.log is not None:
            self.log.write(result)
        self._push_marker_params()
        logger.info("block %d: raw %.1f dB, residual %.1f dB, mood %s, lock %s, emission at %s",
                    self.index, result.raw_rms_db, result.residual_rms_db,
                    mood.name.value, lock.status.value, emission_start)
        self.index += 1
        self.last_result = result
        return result

    def _push_marker_params(self) -> None:
        params = getattr(self.locker, "params", None)
        if self.marker is None or params is None or params == self._pushed_params:
            return
        self.marker.set_params(params)
        self._pushed_params = params
        logger.info("marker params now %s", params)

    def run_forever(self, max_blocks: int | None = None, poll_s: float = 0.05) -> None:
        """Step until `.stop()` (or `max_blocks` blocks), sleeping briefly while waiting for audio."""
        done = 0
        while not self._stop.is_set():
            if max_blocks is not None and done >= max_blocks:
                break
            if self.step() is None:
                self._stop.wait(poll_s)
            else:
                done += 1

    def start(self, max_blocks: int | None = None) -> threading.Thread:
        self._stop.clear()
        self._thread = threading.Thread(target=self.run_forever, kwargs={"max_blocks": max_blocks},
                                        name="roomtone-engine", daemon=True)
        self._thread.start()
        return self._thread

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=10.0)
            self._thread = None


def acquire_baseline(cfg: Config, audio: AudioIO, locker: Any, marker: Any, seconds: float,
                     poll_s: float = 0.05, stop: threading.Event | None = None) -> Any:
    """Feed the Locker marker-only blocks for `seconds` and return its room response.

    The audio stream must already be running. Returns `locker.rir`.
    """
    block_frames = int(round(cfg.block_seconds * cfg.sample_rate))
    reader = BlockReader(audio, block_frames)
    blocks = max(1, int(np.ceil(seconds / cfg.block_seconds)))
    stop = stop or threading.Event()
    done = 0
    while done < blocks and not stop.is_set():
        block = reader.next()
        if block is None:
            stop.wait(poll_s)
            continue
        start, reference, mic = block
        _, lock = locker.process(reference, mic,
                                 emission_in(marker.emissions, start, start + block_frames))
        done += 1
        logger.info("baseline block %d/%d: lock %s, delay %.1f samples",
                 done, blocks, lock.status.value, lock.delay_samples)
    return locker.rir
