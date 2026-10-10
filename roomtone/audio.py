"""Audio in and out: one stream, two ring buffers, and stand-ins for the hardware.

The Mac runs a single duplex stream on the aggregate device. Each time the
sound card asks for a block, the callback here:

1. takes the M8's stereo pair from the input channels,
2. asks the marker source for this block's marker (zeros most of the time),
3. adds the marker to both output channels and sends the mix to the speakers,
4. writes that exact mix (the "reference", spec section 4.3) and the mic
   channel into two ring buffers, stamped with the same frame position.

The analysis engine later reads matching stretches of reference and mic out of
the rings. Nothing in the callback allocates memory: the scratch arrays are made
once up front.

Everything that touches the sound card goes through a small backend object, so
the same code runs against real hardware (`SoundDeviceBackend`), a test double
you pump by hand (`FakeBackend`), or a simulated kitchen (`SimBackend`, used by
`roomtone run --sim`).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from roomtone.config import Config

log = logging.getLogger(__name__)

# A marker source turns (n_frames, music_rms) into n_frames of float32 marker signal.
MarkerSource = Callable[[int, float], np.ndarray]
# The sounddevice callback signature: (indata, outdata, frames, time_info, status).
StreamCallback = Callable[[np.ndarray, np.ndarray, int, Any, Any], None]


def list_devices() -> list[dict]:
    """Every audio device sounddevice can see, as plain dicts.

    Raises ImportError when sounddevice is not installed; callers explain that.
    """
    import sounddevice as sd  # optional dependency, imported on demand

    hostapis = sd.query_hostapis()
    devices = []
    for index, dev in enumerate(sd.query_devices()):
        devices.append({
            "index": index,
            "name": dev["name"],
            "hostapi": hostapis[dev["hostapi"]]["name"],
            "max_input_channels": dev["max_input_channels"],
            "max_output_channels": dev["max_output_channels"],
            "default_samplerate": dev["default_samplerate"],
        })
    return devices


# Ring buffer -----------------------------------------------------------------

class RingRangeError(LookupError):
    """The requested frames are not in the ring: not written yet, or already overwritten."""


class RingBuffer:
    """Fixed-size circular store of multichannel audio, addressed by absolute frame index.

    Safe for one writer (the audio callback) and one reader (the engine) without a
    lock: the writer copies the data first and only then advances `position`, and
    the reader re-checks `position` after copying to detect an overwrite mid-read.
    """

    def __init__(self, channels: int, capacity: int):
        if channels < 1 or capacity < 1:
            raise ValueError("channels and capacity must be positive")
        self.channels = channels
        self.capacity = capacity
        self._data = np.zeros((capacity, channels), dtype=np.float32)
        self._position = 0  # total frames ever written

    @property
    def position(self) -> int:
        """Total frames written since creation (the index of the next frame)."""
        return self._position

    @property
    def oldest(self) -> int:
        """Absolute index of the oldest frame still held."""
        return max(0, self._position - self.capacity)

    def write(self, block: np.ndarray) -> None:
        """Append frames. A 1-D block is accepted for a one-channel ring."""
        if block.ndim == 1:
            block = block.reshape(-1, 1)
        n = block.shape[0]
        if block.shape[1] != self.channels:
            raise ValueError(f"block has {block.shape[1]} channels, ring has {self.channels}")
        pos = self._position
        if n >= self.capacity:  # only the newest `capacity` frames survive
            block = block[n - self.capacity:]
            pos += n - self.capacity
            n = self.capacity
        start = pos % self.capacity
        first = min(n, self.capacity - start)
        self._data[start:start + first] = block[:first]
        if first < n:
            self._data[:n - first] = block[first:]
        self._position = pos + n

    def read(self, n: int, at: int | None = None) -> np.ndarray:
        """Copy out `n` frames starting at absolute frame `at` (default: the newest n).

        Returns an array of shape (n, channels). Raises RingRangeError if any of
        the frames is not available.
        """
        pos = self._position
        if at is None:
            at = pos - n
        if n < 0 or n > self.capacity:
            raise RingRangeError(f"cannot read {n} frames from a ring of {self.capacity}")
        if at < 0 or at + n > pos:
            raise RingRangeError(f"frames [{at}, {at + n}) not written yet (position {pos})")
        if at < pos - self.capacity:
            raise RingRangeError(f"frames from {at} already overwritten (oldest {pos - self.capacity})")
        out = np.empty((n, self.channels), dtype=np.float32)
        start = at % self.capacity
        first = min(n, self.capacity - start)
        out[:first] = self._data[start:start + first]
        if first < n:
            out[first:] = self._data[:n - first]
        if at < self._position - self.capacity:  # writer lapped us while copying
            raise RingRangeError(f"frames from {at} overwritten during read")
        return out


# Backends --------------------------------------------------------------------

class Backend(Protocol):
    def open(self, callback: StreamCallback, *, sample_rate: int, frames: int,
             in_channels: int, out_channels: int,
             m8_channels: tuple[int, int], mic_channel: int) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...


class SoundDeviceBackend:
    """The real thing: one sounddevice.Stream on the aggregate device."""

    def __init__(self, cfg: Config, device: str | int | None = None):
        self.device = device if device is not None else cfg.aggregate_device
        self._stream = None

    def open(self, callback, *, sample_rate, frames, in_channels, out_channels,
             m8_channels, mic_channel) -> None:
        import sounddevice as sd  # optional dependency

        self._stream = sd.Stream(
            device=self.device, samplerate=sample_rate, blocksize=frames,
            channels=(in_channels, out_channels), dtype="float32", callback=callback,
        )

    def start(self) -> None:
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None


class FakeBackend:
    """Test double. Nothing runs by itself; call `pump` to run one callback."""

    def __init__(self) -> None:
        self.callback: StreamCallback | None = None
        self.running = False
        self.in_channels = 0
        self.out_channels = 0
        self.m8_channels = (0, 1)
        self.mic_channel = 2

    def open(self, callback, *, sample_rate, frames, in_channels, out_channels,
             m8_channels, mic_channel) -> None:
        self.callback = callback
        self.in_channels, self.out_channels = in_channels, out_channels
        self.m8_channels, self.mic_channel = m8_channels, mic_channel

    def start(self) -> None:
        self.running = True

    def stop(self) -> None:
        self.running = False

    def pump(self, n_frames: int, m8_block: np.ndarray | None = None,
             mic_block: np.ndarray | None = None) -> np.ndarray:
        """Run the callback once with the given M8 (n, 2) and mic (n,) input; return the output."""
        if self.callback is None:
            raise RuntimeError("backend not opened")
        indata = np.zeros((n_frames, self.in_channels), dtype=np.float32)
        if m8_block is not None:
            indata[:, self.m8_channels[0]] = m8_block[:, 0]
            indata[:, self.m8_channels[1]] = m8_block[:, 1]
        if mic_block is not None:
            indata[:, self.mic_channel] = np.asarray(mic_block).reshape(-1)
        outdata = np.zeros((n_frames, self.out_channels), dtype=np.float32)
        self.callback(indata, outdata, n_frames, None, None)
        return outdata


class PadSynth:
    """A slow, quiet stereo pad: a minor-seventh chord of soft sines that swell and fade.

    Stands in for the M8 in the simulator and in the marker audibility bench.
    """

    CHORD_HZ = (110.0, 164.81, 220.0, 261.63, 329.63)
    SWELL_S = (17.8, 21.3, 25.1, 29.7, 13.9)  # incommensurate, like the real patterns

    def __init__(self, sample_rate: int, level_db: float = -20.0):
        self.sample_rate = sample_rate
        self.level = 10 ** (level_db / 20.0)
        self._t = 0  # frames rendered so far

    def next(self, n: int) -> np.ndarray:
        t = (self._t + np.arange(n)) / self.sample_rate
        self._t += n
        left = np.zeros(n)
        right = np.zeros(n)
        for i, (f, swell) in enumerate(zip(self.CHORD_HZ, self.SWELL_S)):
            env = 0.6 + 0.4 * np.sin(2 * np.pi * t / swell + 1.3 * i)  # never fully silent
            left += env * np.sin(2 * np.pi * f * t)
            right += env * np.sin(2 * np.pi * f * 1.002 * t + 0.7)
        # each voice averages about 0.47 RMS; scale the five to the target level
        gain = self.level / (0.47 * math.sqrt(len(self.CHORD_HZ)))
        return (np.stack([left, right], axis=1) * gain).astype(np.float32)


class _DelayRoom:
    """Fallback room when roomtone.sim is not available: a delayed, quieter copy plus noise.

    Streaming: successive render() calls continue one signal. Clock drift is ignored.
    """

    def __init__(self, sample_rate: int, delay_samples: int, drift_ppm: float = 0.0,
                 gain: float = 0.3, noise_db: float = -65.0, seed: int = 0):
        self.delay = int(delay_samples)
        self.gain = gain
        self.noise = 10 ** (noise_db / 20.0)
        self._line = np.zeros(self.delay, dtype=np.float32)
        self._rng = np.random.default_rng(seed)

    def render(self, output: np.ndarray, room_sound: np.ndarray | None = None) -> np.ndarray:
        x = np.concatenate([self._line, np.asarray(output, dtype=np.float32)])
        n = len(output)
        mic = self.gain * x[:n]
        self._line = x[n:]
        mic = mic + self.noise * self._rng.standard_normal(n)
        if room_sound is not None:
            mic = mic + room_sound
        return mic.astype(np.float32)


class SimBackend:
    """A pretend kitchen. Feeds a synthetic pad as the M8 input and plays the output
    through a simulated room to make the mic channel.

    Uses roomtone.sim.SimRoom when present, otherwise a delayed copy plus noise.
    The room's own sound is white noise at `room_noise_db`, or, given `script`
    (a list of (seconds, mode) phases, see roomtone.kitchen), a scripted evening
    over that same floor; `.script` is then the KitchenScript, None otherwise.
    The mic for each callback is the room's response to the previous callback's
    output, which adds one I/O block of latency, as real input buffering does.
    `realtime=False` runs as fast as the CPU allows (for tests and quick looks).
    """

    def __init__(self, cfg: Config, delay_samples: int | None = None, drift_ppm: float = 20.0,
                 realtime: bool = True, room_noise_db: float = -60.0, seed: int = 0,
                 max_frames: int | None = None,
                 script: Sequence[tuple[float, str]] | None = None):
        self.sample_rate = cfg.sample_rate
        self.delay_samples = (int(round(0.075 * cfg.sample_rate)) + 200
                              if delay_samples is None else int(delay_samples))
        self.drift_ppm = drift_ppm
        self.realtime = realtime
        self.max_frames = max_frames
        self.pad = PadSynth(cfg.sample_rate)
        self.room_noise = 10 ** (room_noise_db / 20.0)
        self._rng = np.random.default_rng(seed)
        try:
            from roomtone.sim import SimRoom
            self.room = SimRoom(cfg.sample_rate, self.delay_samples, drift_ppm)
        except ImportError:
            log.info("roomtone.sim not available; simulating the room as a delayed copy plus noise")
            self.room = _DelayRoom(cfg.sample_rate, self.delay_samples, drift_ppm, seed=seed)
        self.script: Any = None
        if script is not None:
            try:
                from roomtone.kitchen import KitchenScript
                self.script = KitchenScript(script, cfg.sample_rate, room_noise_db, seed=seed)
            except ImportError:
                log.info("roomtone.sim not available; the kitchen script is off, room sound is noise")
        self.frames_run = 0
        self._callback: StreamCallback | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pending_mic: np.ndarray | None = None

    def open(self, callback, *, sample_rate, frames, in_channels, out_channels,
             m8_channels, mic_channel) -> None:
        self._callback = callback
        self.frames = frames
        self.m8_channels, self.mic_channel = m8_channels, mic_channel
        self._indata = np.zeros((frames, in_channels), dtype=np.float32)
        self._outdata = np.zeros((frames, out_channels), dtype=np.float32)
        self._pending_mic = np.zeros(frames, dtype=np.float32)

    def cycle(self) -> None:
        """Run one callback's worth of simulated time."""
        n = self.frames
        pad = self.pad.next(n)
        self._indata[:] = 0.0
        self._indata[:, self.m8_channels[0]] = pad[:, 0]
        self._indata[:, self.m8_channels[1]] = pad[:, 1]
        self._indata[:, self.mic_channel] = self._pending_mic
        self._callback(self._indata, self._outdata, n, None, None)
        room_sound = (self.script.next(n) if self.script is not None else
                      (self.room_noise * self._rng.standard_normal(n)).astype(np.float32))
        self._pending_mic = np.asarray(
            self.room.render(self._outdata.mean(axis=1), room_sound), dtype=np.float32).reshape(-1)
        self.frames_run += n

    def run_frames(self, total: int) -> None:
        """Run synchronously until at least `total` more frames have been simulated."""
        target = self.frames_run + total
        while self.frames_run < target:
            self.cycle()

    def _loop(self) -> None:
        t0 = time.perf_counter()
        start_frames = self.frames_run
        while not self._stop.is_set():
            if self.max_frames is not None and self.frames_run >= self.max_frames:
                break
            self.cycle()
            if self.realtime:
                due = t0 + (self.frames_run - start_frames) / self.sample_rate
                wait = due - time.perf_counter()
                if wait > 0:
                    self._stop.wait(wait)

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="roomtone-sim", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None


# The stream ------------------------------------------------------------------

class AudioIO:
    """The duplex stream: M8 in plus marker out, reference and mic into rings.

    `on_output_block` is the marker source: a callable taking (n_frames,
    music_rms) and returning n_frames of float32 marker signal. It is called
    once per callback, so a marker scheduler behind it counts frames in step
    with the reference ring (both start at zero when the stream starts).
    `music_rms` is the RMS of the M8 input over roughly the last second.
    """

    def __init__(self, cfg: Config, on_output_block: MarkerSource | None = None, *,
                 backend: Backend | None = None,
                 m8_channels: tuple[int, int] = (0, 1), mic_channel: int = 2,
                 ring_seconds: float | None = None, mute_music: bool = False):
        self.cfg = cfg
        self.marker_source = on_output_block
        self.backend: Backend = backend if backend is not None else SoundDeviceBackend(cfg)
        self.m8_channels = tuple(m8_channels)
        self.mic_channel = mic_channel
        self.in_channels = max(*self.m8_channels, mic_channel) + 1
        self.out_channels = 2
        self.mute_music = mute_music
        seconds = ring_seconds if ring_seconds is not None else max(3 * cfg.block_seconds, 10.0)
        capacity = int(seconds * cfg.sample_rate)
        self.reference_ring = RingBuffer(2, capacity)
        self.mic_ring = RingBuffer(1, capacity)
        self.music_rms = 0.0
        self.status_flags = 0  # callbacks that reported an over/underrun
        self._alloc(cfg.io_frames)
        hist = max(1, round(cfg.sample_rate / cfg.io_frames))  # about one second of callbacks
        self._ms_hist = np.zeros(hist, dtype=np.float64)
        self._ms_index = 0
        self._opened = False

    def _alloc(self, frames: int) -> None:
        self._mix = np.zeros((frames, 2), dtype=np.float32)

    @property
    def position(self) -> int:
        """Frames delivered so far; reference and mic rings agree on it."""
        return self.reference_ring.position

    def _callback(self, indata: np.ndarray, outdata: np.ndarray, frames: int,
                  time_info: Any, status: Any) -> None:
        if status:
            self.status_flags += 1
        if frames > self._mix.shape[0]:  # only if the host changes block size on us
            self._alloc(frames)
        mix = self._mix[:frames]
        left = indata[:frames, self.m8_channels[0]]
        right = indata[:frames, self.m8_channels[1]]
        if self.mute_music:
            mix.fill(0.0)
        else:
            mix[:, 0] = left
            mix[:, 1] = right
        # music level over the last second, from the M8 input itself
        self._ms_hist[self._ms_index] = (float(np.dot(left, left)) + float(np.dot(right, right))) / (2 * frames)
        self._ms_index = (self._ms_index + 1) % self._ms_hist.shape[0]
        self.music_rms = math.sqrt(float(self._ms_hist.mean()))
        if self.marker_source is not None:
            marker = self.marker_source(frames, self.music_rms)
            mix[:, 0] += marker
            mix[:, 1] += marker
        np.clip(mix, -1.0, 1.0, out=mix)
        outdata[:frames] = mix
        # same callback, same frame stamp for both rings
        self.mic_ring.write(indata[:frames, self.mic_channel:self.mic_channel + 1])
        self.reference_ring.write(mix)

    def open(self) -> None:
        if not self._opened:
            self.backend.open(self._callback, sample_rate=self.cfg.sample_rate,
                              frames=self.cfg.io_frames, in_channels=self.in_channels,
                              out_channels=self.out_channels, m8_channels=self.m8_channels,
                              mic_channel=self.mic_channel)
            self._opened = True

    def start(self) -> None:
        self.open()
        self.backend.start()
        log.info("audio started: %d in, %d out, %d Hz, %d-frame callbacks",
                 self.in_channels, self.out_channels, self.cfg.sample_rate, self.cfg.io_frames)

    def stop(self) -> None:
        self.backend.stop()
        log.info("audio stopped after %d frames (%d callbacks flagged a glitch)",
                 self.position, self.status_flags)


# Bench helpers -----------------------------------------------------------------

def click_train(sample_rate: int, n_clicks: int = 10, spacing_s: float = 0.6,
                lead_s: float = 0.5, tail_s: float = 1.0,
                level: float = 0.5) -> tuple[np.ndarray, list[int]]:
    """A mono signal of short clicks; returns (signal, click start indices)."""
    click = level * np.hanning(32)
    spacing = int(spacing_s * sample_rate)
    lead = int(lead_s * sample_rate)
    total = lead + spacing * (n_clicks - 1) + int(tail_s * sample_rate) + len(click)
    out = np.zeros(total, dtype=np.float32)
    starts = [lead + i * spacing for i in range(n_clicks)]
    for s in starts:
        out[s:s + len(click)] += click
    return out, starts


def measure_click_delays(mic: np.ndarray, starts: list[int], sample_rate: int,
                         window_s: float = 0.5) -> list[float]:
    """For each click, the delay in samples until its loudest echo in the mic.

    Searches `window_s` after each click by cross-correlating with the click shape.
    """
    click = np.hanning(32)
    window = int(window_s * sample_rate)
    mic = np.asarray(mic, dtype=np.float64).reshape(-1)
    delays = []
    for s in starts:
        seg = mic[s:s + window + len(click)]
        if len(seg) <= len(click):
            continue
        corr = np.correlate(seg, click, mode="valid")
        delays.append(float(np.argmax(np.abs(corr))))
    return delays
