"""Terminal UI for roomtone, laid out like an M8 tracker screen.

Every panel is a single ``Static`` whose content is a Rich ``Text`` built by a
pure ``render_*`` function, so the whole screen can be unit-tested without a
terminal. Data arrives as ``BlockResult`` objects, either in-process through
``RoomtoneApp.feed`` or over OSC (``/roomtone/block`` with one JSON string
argument in the ``BlockResult.as_dict()`` shape). ``--demo`` replays a
synthetic two-hour session at one block every half second.

Display conventions: values the M8 would show in hex are hex (unit-interval
quantities as 00..7F, the layer mask as two digits, block counters); values a
person reads in units stay decimal (dB, ppm, Hz, BPM, samples).
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from collections import deque
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import fields
from typing import Any

import numpy as np
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Static

from roomtone import config
from roomtone.config import Config, MarkerParams
from roomtone.types import (
    BlockResult, Controls, Features, LockState, LockStatus, Mood, MoodName,
)

log = logging.getLogger(__name__)

OSC_ADDRESS = "/roomtone/block"
DEMO_INTERVAL_S = 0.5


# Theme -----------------------------------------------------------------------

Theme = Mapping[str, str]

THEMES: dict[str, dict[str, str]] = {
    # The M8 default palette: white on black, teal titles, pink selection.
    "m8": {
        "bg": "#000000", "fg": "#f0f0f0", "dim": "#5c5c5c",
        "title": "#00d2d2", "select": "#ff2e9a",
        "ok": "#2fe36a", "warn": "#ffd21e", "hot": "#ff3434",
    },
    "amber": {
        "bg": "#0c0700", "fg": "#ffb000", "dim": "#6b4a00",
        "title": "#ffd27a", "select": "#ff6a00",
        "ok": "#ffc840", "warn": "#ff9a00", "hot": "#ff4000",
    },
}
DEFAULT_THEME = "m8"
THEME: dict[str, str] = THEMES[DEFAULT_THEME]

STATUS_ROLE: dict[LockStatus, str] = {
    LockStatus.ACQUIRING: "warn",
    LockStatus.LOCKED: "ok",
    LockStatus.LOST: "select",
    LockStatus.SATURATED: "hot",
}

MOOD_LETTER: dict[MoodName, str] = {
    MoodName.EMPTY: "E", MoodName.COOKING: "C",
    MoodName.CONVERSATION: "V", MoodName.LOUD_BUSY: "L",
}
MOOD_ROLE: dict[MoodName, str] = {
    MoodName.EMPTY: "dim", MoodName.COOKING: "warn",
    MoodName.CONVERSATION: "title", MoodName.LOUD_BUSY: "hot",
}

BAR_WIDE = 16
BAR_COMPACT = 10
HISTORY_LEN = 16
FULL, EMPTY_CELL = "█", "·"     # full block, middle dot
ON_CELL, OFF_CELL = "■", "□"    # filled and empty square
LABEL_W = 8


# Small formatting helpers ----------------------------------------------------

def cc(x: float) -> int:
    """Unit interval to a MIDI CC value 0..127."""
    return int(round(127 * min(max(x, 0.0), 1.0)))


def hex2(n: int) -> str:
    """Two-digit uppercase hex, wider when the value needs it."""
    return f"{max(int(n), 0):02X}"


def clock(seconds: float) -> str:
    s = max(int(seconds), 0)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def meter(frac: float, width: int, theme: Theme = THEME, role: str | None = None,
          zones: tuple[float, float] = (0.6, 0.85)) -> Text:
    """A fixed-width bar. Without ``role`` each lit cell takes the M8 meter
    colour for its position: green, then yellow, then red near the top."""
    frac = min(max(frac, 0.0), 1.0)
    lit = int(round(frac * width))
    t = Text()
    for i in range(width):
        if i < lit:
            pos = (i + 1) / width
            r = role or ("ok" if pos <= zones[0] else "warn" if pos <= zones[1] else "hot")
            t.append(FULL, style=theme[r])
        else:
            t.append(EMPTY_CELL, style=theme["dim"])
    return t


def _title(name: str, theme: Theme, extra: str = "") -> Text:
    t = Text().append(name, style=f"bold {theme['title']}")
    if extra:
        t.append(f"  {extra}", style=theme["dim"])
    return t


def _row(t: Text, label: str, value: str, theme: Theme, value_style: str | None = None,
         value_w: int = 10) -> Text:
    t.append("\n")
    t.append(f"{label:<{LABEL_W}}", style=theme["dim"])
    t.append(f"{value:>{value_w}}", style=value_style or theme["fg"])
    return t


# Marker ladder ---------------------------------------------------------------

def marker_ladder(base: MarkerParams = config.DEFAULT_MARKER) -> list[MarkerParams]:
    """Every rung from the tracking default up to the ceiling."""
    rungs = [base]
    while not rungs[-1].at_ceiling:
        rungs.append(rungs[-1].escalate())
    return rungs


LADDER = marker_ladder()


def marker_label(p: MarkerParams) -> str:
    lo, hi = int(p.band_lo_hz // 1000), int(p.band_hi_hz // 1000)
    return f"LEN {int(p.length_s):02d}s BAND {lo}K-{hi}K LVL {int(p.level_db):+03d}"


# Pure render functions -------------------------------------------------------

def render_header(result: BlockResult | None, theme: Theme = THEME,
                  source: str = "") -> Text:
    t = Text().append("ROOMTONE", style=f"bold {theme['title']}")
    if result is None:
        t.append("  --:--:--  BLK --  ", style=theme["dim"])
        t.append("NO DATA", style=theme["dim"])
    else:
        t.append(f"  {clock(result.t_start)}  ", style=theme["fg"])
        t.append("BLK ", style=theme["dim"])
        t.append(hex2(result.index), style=theme["fg"])
        t.append("  ")
        status = result.lock.status
        t.append(status.name, style=f"bold {theme[STATUS_ROLE[status]]}")
    if source:
        t.append(f"  {source}", style=theme["dim"])
    return t


def render_lock(lock: LockState, theme: Theme = THEME, bar: int = BAR_WIDE,
                sample_rate: int = config.SAMPLE_RATE,
                erle_target_db: float = config.LOCK_ERLE_DB,
                jitter_limit: float = config.LOCK_JITTER_SAMPLES) -> Text:
    t = _title("LOCK", theme)
    ms = 1000.0 * lock.delay_samples / sample_rate
    _row(t, "DELAY", f"{lock.delay_samples:.1f}", theme).append(f" smp {ms:6.2f}ms", style=theme["dim"])
    _row(t, "DRIFT", f"{lock.drift_ppm:+.2f}", theme).append(" ppm", style=theme["dim"])
    _row(t, "P/SL", f"{lock.peak_to_sidelobe_db:.1f}", theme).append(" dB", style=theme["dim"])
    jit_ok = lock.jitter_samples < jitter_limit
    _row(t, "JIT", f"{lock.jitter_samples:.2f}", theme,
         theme["ok"] if jit_ok else theme["hot"]).append(" smp", style=theme["dim"])
    erle_ok = lock.erle_db >= erle_target_db
    _row(t, "ERLE", f"{lock.erle_db:.1f}", theme, theme["ok"] if erle_ok else theme["warn"])
    t.append(" dB ", style=theme["dim"])
    span = erle_target_db * 1.5
    m = meter(lock.erle_db / span, bar, theme, role="ok" if erle_ok else "warn")
    tick = min(int(round(erle_target_db / span * bar)), bar - 1)
    if m.plain[tick] == EMPTY_CELL:
        m = m[:tick] + Text("|", style=theme["title"]) + m[tick + 1:]
    t.append_text(m)
    _row(t, "RIR CHG", f"{lock.rir_change:.3f}", theme)
    rung = min(max(lock.marker_rung, 0), len(LADDER) - 1)
    t.append("\n")
    t.append(f"{'MARKER':<{LABEL_W}}", style=theme["dim"])
    t.append(marker_label(LADDER[rung]), style=theme["fg"])
    t.append("\n")
    t.append(f"{'RUNG':<{LABEL_W}}", style=theme["dim"])
    top = len(LADDER) - 1
    for i in range(len(LADDER)):
        if i <= rung:
            t.append(ON_CELL, style=theme["hot"] if rung == top else theme["warn"])
        else:
            t.append(OFF_CELL, style=theme["dim"])
    t.append(f" {hex2(rung)}/{hex2(top)}", style=theme["fg"])
    t.append("\n")
    t.append(f"{'OK':<{LABEL_W}}", style=theme["dim"])
    t.append(hex2(lock.blocks_ok), style=theme["ok"])
    t.append("  BAD ", style=theme["dim"])
    t.append(hex2(lock.blocks_bad), style=theme["hot"] if lock.blocks_bad else theme["fg"])
    return t


FEATURE_SCALE = {
    "rms_db": (-90.0, 0.0), "talkers": (0.0, 6.0),
    "transient_rate": (0.0, 8.0), "speech_band_centroid_hz": (0.0, 4000.0),
}


def _scaled(value: float, key: str) -> float:
    lo, hi = FEATURE_SCALE[key]
    return (value - lo) / (hi - lo)


def render_room(features: Features, theme: Theme = THEME, bar: int = BAR_WIDE,
                n_labels: int = 3) -> Text:
    t = _title("ROOM", theme)
    rows: list[tuple[str, str, float]] = [
        ("RMS", f"{features.rms_db:.1f}dB", _scaled(features.rms_db, "rms_db")),
        ("SPEECH", hex2(cc(features.speech)), features.speech),
        ("TALKERS", f"{features.talkers:.1f}", _scaled(features.talkers, "talkers")),
        ("TRANS/s", f"{features.transient_rate:.1f}", _scaled(features.transient_rate, "transient_rate")),
        ("BEAT", hex2(cc(features.beat)), features.beat),
        ("CENTROID", f"{features.speech_band_centroid_hz:.0f}Hz",
         _scaled(features.speech_band_centroid_hz, "speech_band_centroid_hz")),
    ]
    for label, value, frac in rows:
        t.append("\n")
        t.append(f"{label:<9}", style=theme["dim"])
        t.append(f"{value:>8} ", style=theme["fg"])
        t.append_text(meter(frac, bar, theme))
    top = sorted(features.labels.items(), key=lambda kv: kv[1], reverse=True)[:n_labels]
    for name, score in top:
        t.append("\n")
        t.append(f"{name.upper()[:14]:<15}", style=theme["title"])
        t.append(f"{hex2(cc(score)):>2} ", style=theme["fg"])
        t.append_text(meter(score, bar, theme, role="title"))
    return t


def history_strip(history: Sequence[MoodName], theme: Theme = THEME) -> Text:
    """Last 16 mood names as letters, oldest left, grouped in fours like a phrase."""
    last = list(history)[-HISTORY_LEN:]
    t = Text()
    for i in range(HISTORY_LEN):
        if i and i % 4 == 0:
            t.append(" ")
        if i < len(last):
            name = last[i]
            role = "select" if i == len(last) - 1 else MOOD_ROLE[name]
            t.append(MOOD_LETTER[name], style=theme[role])
        else:
            t.append("-", style=theme["dim"])
    return t


def render_mood(mood: Mood, history: Sequence[MoodName] = (), theme: Theme = THEME,
                bar: int = BAR_WIDE) -> Text:
    t = _title("MOOD", theme)
    for label, v in (("ENERGY", mood.energy), ("VOICES", mood.voices), ("RHYTHM", mood.rhythm)):
        _row(t, label, hex2(cc(v)), theme, value_w=2).append(" ")
        t.append_text(meter(v, bar, theme))
    t.append("\n")
    t.append(f"{'STATE':<{LABEL_W}}", style=theme["dim"])
    t.append(mood.name.value.upper().replace("-", " "), style=f"bold {theme['title']}")
    _row(t, "CONF", hex2(cc(mood.confidence)), theme, value_w=2).append(" ")
    t.append_text(meter(mood.confidence, bar, theme, role="title"))
    t.append("\n")
    t.append(" " * LABEL_W)
    t.append("0123 4567 89AB CDEF", style=theme["dim"])
    t.append("\n")
    t.append(f"{'HIST':<{LABEL_W}}", style=theme["dim"])
    t.append_text(history_strip(history, theme))
    return t


LAYER_VOL_FLOOR = 0.4   # display model: an active layer sits at 40 % plus energy


def layer_vol(controls: Controls, bit: int) -> int | None:
    if not controls.layers >> bit & 1:
        return None
    return cc(LAYER_VOL_FLOOR + (1.0 - LAYER_VOL_FLOOR) * controls.energy)


def layer_cut(controls: Controls, layer: str) -> int | None:
    return cc(controls.bright) if f"{layer}_cutoff" in config.CC else None


def render_layers(controls: Controls, playhead: int = 0, theme: Theme = THEME) -> Text:
    t = _title("LAYERS", theme, f"MASK {hex2(controls.layers & 0xFF)}")
    t.append("\n")
    t.append(" TRK CH LAYER    ON VOL CUT", style=theme["title"])
    for bit, layer in enumerate(config.LAYER_ORDER):
        on = bool(controls.layers >> bit & 1)
        vol, cut = layer_vol(controls, bit), layer_cut(controls, layer)
        sel = bit == playhead % len(config.LAYER_ORDER)
        base = f"bold {theme['select']}" if sel else theme["fg"]
        t.append("\n")
        t.append(">" if sel else " ", style=theme["select"])
        t.append(f"{bit + 1:>3} {config.LAYER_CHANNELS[layer]:02d} {layer.upper():<8} ", style=base)
        t.append(f"{ON_CELL if on else OFF_CELL} ",
                 style=base if sel else theme["ok"] if on else theme["dim"])
        t.append(f"{'--' if vol is None else hex2(vol):>3} ", style=base if vol is not None else theme["dim"])
        t.append(f"{'--' if cut is None else hex2(cut):>3}", style=base if cut is not None else theme["dim"])
    return t


def render_footer(controls: Controls, theme: Theme = THEME, demo: bool = False) -> Text:
    t = Text()
    for label, value in (("TEMPO", f"{int(round(controls.tempo_bpm)):03d}"),
                         ("SCALE", controls.scale.upper()),
                         ("TRN", f"{controls.transpose:+03d}")):
        if t:
            t.append(" ")
        t.append(f"{label} ", style=theme["dim"])
        t.append(value, style=theme["fg"])
    t.append("\n")
    for key, what in (("q", "quit"), ("d", "demo"), ("l", "log")):
        t.append(key, style=theme["select"])
        t.append(f" {what}  ", style=theme["dim"])
    if demo:
        t.append("DEMO", style=f"bold {theme['warn']}")
    return t


def render_log(lines: Sequence[str], theme: Theme = THEME, rows: int = 9) -> Text:
    t = _title("LOG", theme)
    for line in list(lines)[-rows:]:
        t.append("\n")
        t.append(line, style=theme["fg"])
    return t


# Decoding --------------------------------------------------------------------

def _build(cls: type, data: Mapping[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def block_from_dict(d: Mapping[str, Any]) -> BlockResult:
    """Inverse of ``BlockResult.as_dict``; unknown keys are ignored."""
    lock = dict(d.get("lock", {}))
    lock["status"] = LockStatus(lock.get("status", LockStatus.ACQUIRING.value))
    mood = dict(d.get("mood", {}))
    mood["name"] = MoodName(mood.get("name", MoodName.EMPTY.value))
    return BlockResult(
        index=int(d["index"]), t_start=float(d.get("t_start", 0.0)),
        lock=_build(LockState, lock), features=_build(Features, d.get("features", {})),
        mood=_build(Mood, mood), controls=_build(Controls, d.get("controls", {})),
        residual_rms_db=float(d.get("residual_rms_db", -90.0)),
        raw_rms_db=float(d.get("raw_rms_db", -90.0)),
    )


# Demo session ----------------------------------------------------------------

# (segment, blocks): 240 blocks of 30 s is two hours.
DEMO_SCRIPT: list[tuple[str, int]] = [
    ("acquire", 6), ("empty", 114), ("cooking", 40),
    ("conversation", 40), ("loud", 30), ("empty", 10),
]

# Per segment: room features, mood target (energy, voices, rhythm), base ERLE dB.
_ROOM: dict[str, dict[str, Any]] = {
    "acquire": dict(rms=-63.0, speech=0.02, talkers=0.0, trans=0.1, beat=0.02, cent=500.0,
                    labels={"Silence": 0.85, "Hum": 0.20}, mood=(0.05, 0.0, 0.0), erle=24.0),
    "empty": dict(rms=-62.0, speech=0.03, talkers=0.0, trans=0.1, beat=0.02, cent=550.0,
                  labels={"Silence": 0.80, "Hum": 0.25, "Refrigerator": 0.10},
                  mood=(0.06, 0.02, 0.02), erle=26.0),
    "cooking": dict(rms=-46.0, speech=0.15, talkers=0.6, trans=2.8, beat=0.12, cent=1250.0,
                    labels={"Dishes, pots, and pans": 0.62, "Chopping (food)": 0.41, "Speech": 0.18},
                    mood=(0.45, 0.12, 0.55), erle=22.5),
    "conversation": dict(rms=-41.0, speech=0.78, talkers=3.6, trans=0.8, beat=0.05, cent=1820.0,
                         labels={"Speech": 0.84, "Conversation": 0.57, "Laughter": 0.21},
                         mood=(0.50, 0.85, 0.10), erle=21.8),
    "loud": dict(rms=-27.0, speech=0.62, talkers=6.5, trans=5.5, beat=0.48, cent=2250.0,
                 labels={"Crowd": 0.71, "Speech": 0.52, "Music": 0.33},
                 mood=(0.92, 0.70, 0.55), erle=8.0),
}

_PROTO: dict[MoodName, tuple[float, float, float]] = {
    MoodName.EMPTY: (0.05, 0.0, 0.0), MoodName.COOKING: (0.45, 0.12, 0.55),
    MoodName.CONVERSATION: (0.50, 0.85, 0.10), MoodName.LOUD_BUSY: (0.92, 0.70, 0.55),
}
_LAYERS: dict[MoodName, int] = {
    MoodName.EMPTY: 0b00000001, MoodName.COOKING: 0b00100011,
    MoodName.CONVERSATION: 0b00000111, MoodName.LOUD_BUSY: 0b01110011,
}
_TEMPO = {MoodName.EMPTY: 60.0, MoodName.COOKING: 72.0,
          MoodName.CONVERSATION: 66.0, MoodName.LOUD_BUSY: 84.0}
_SCALE = {MoodName.EMPTY: "open", MoodName.COOKING: "pentatonic",
          MoodName.CONVERSATION: "lydian", MoodName.LOUD_BUSY: "minor"}
_TRANSPOSE = (0, 2, -1, 3)


def demo_session(cfg: Config | None = None, seed: int = 0x4D38) -> Iterator[BlockResult]:
    """A plausible two-hour evening: acquire, lock, an empty hour, cooking (with
    a door that jumps the room response), dinner conversation, a loud stretch
    that climbs the marker ladder to saturation, then empty again."""
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    segments = [name for name, n in DEMO_SCRIPT for _ in range(n)]
    hyst = cfg.hysteresis_blocks
    top = len(LADDER) - 1

    status, rung = LockStatus.ACQUIRING, 3   # acquisition starts on the long marker
    ok_run = bad_run = ladder_run = 0
    drift = 23.6
    delay = 3307.0                          # Sonos buffer plus air, after drift correction
    m = np.zeros(3)
    name, candidate, cand_run, since_change = MoodName.EMPTY, MoodName.EMPTY, 0, 0
    tempo = 60.0
    cook_start = segments.index("cooking")

    def n(sigma: float) -> float:
        return float(rng.normal(0.0, sigma))

    for index, seg in enumerate(segments):
        room = _ROOM[seg]

        # Lock ---------------------------------------------------------------
        erle = room["erle"] + 1.0 * rung * (seg == "loud") + n(0.8)
        rir = abs(0.02 + n(0.01))
        found = not (seg == "acquire" and index < 3)
        if index == cook_start + 12:
            rir = 0.46                          # the back door opens
        if not found:
            erle, psl, jit = 3.0 + n(1.0), 6.0 + n(1.5), 9.0 + abs(n(3.0))
        else:
            psl = 30.0 + erle * 0.5 + n(1.0)
            jit = abs(0.5 + n(0.2)) if erle >= cfg.lock_erle_db else 2.4 + abs(n(1.2))
        ok = found and erle >= cfg.lock_erle_db and jit < cfg.lock_jitter_samples and rir < 0.1
        ok_run, bad_run = (ok_run + 1, 0) if ok else (0, bad_run + 1)
        # signed run since the last rung change: +n good blocks or -n bad ones
        if ok:
            ladder_run = ladder_run + 1 if ladder_run > 0 else 1
        else:
            ladder_run = ladder_run - 1 if ladder_run < 0 else -1
        if ok and ladder_run >= hyst and rung > 0 and status is LockStatus.LOCKED:
            rung, ladder_run = rung - 1, 0
        elif not ok and -ladder_run >= hyst and rung < top and status is not LockStatus.ACQUIRING:
            rung, ladder_run = rung + 1, 0
        if status is LockStatus.LOCKED and (rir > 0.3 or bad_run >= hyst):
            status = LockStatus.LOST
        elif status is not LockStatus.LOCKED and ok_run >= hyst:
            status = LockStatus.LOCKED
        elif status is LockStatus.LOST and rung == top and bad_run >= hyst:
            status = LockStatus.SATURATED
        lock = LockState(status=status, delay_samples=delay + n(0.3), drift_ppm=drift + n(0.15),
                         peak_to_sidelobe_db=psl, jitter_samples=jit, erle_db=erle,
                         rir_change=rir, marker_rung=rung, blocks_ok=ok_run, blocks_bad=bad_run)

        # Room and mood --------------------------------------------------------
        labels = {k: float(np.clip(v + n(0.04), 0.0, 1.0)) for k, v in room["labels"].items()}
        feats = Features(
            rms_db=room["rms"] + n(1.5), speech=float(np.clip(room["speech"] + n(0.04), 0, 1)),
            talkers=max(room["talkers"] + n(0.3), 0.0),
            transient_rate=max(room["trans"] + n(0.3), 0.0),
            beat=float(np.clip(room["beat"] + n(0.03), 0, 1)),
            speech_band_centroid_hz=max(room["cent"] + n(60.0), 0.0), labels=labels)
        m += (np.asarray(room["mood"]) - m) * 0.25
        nearest = min(_PROTO, key=lambda k: float(np.sum((np.asarray(_PROTO[k]) - m) ** 2)))
        if status is LockStatus.SATURATED:
            nearest = MoodName.LOUD_BUSY
        cand_run = cand_run + 1 if nearest == candidate else 1
        candidate = nearest
        since_change += 1
        if candidate != name and cand_run >= hyst:
            name, since_change = candidate, 0
        agree = float(np.exp(-8.0 * np.sum((np.asarray(_PROTO[name]) - m) ** 2)))
        mood = Mood(energy=float(m[0]), voices=float(m[1]), rhythm=float(m[2]), name=name,
                    confidence=min(1.0, (since_change + 1) / 10.0) * agree)

        tempo += (_TEMPO[name] - tempo) * 0.2
        controls = Controls(
            energy=float(m[0]), voices=float(m[1]), rhythm=float(m[2]),
            bright=float(np.clip(0.25 + 0.5 * m[0] - 0.2 * m[1], 0.0, 1.0)),
            layers=_LAYERS[name], tempo_bpm=tempo, scale=_SCALE[name],
            transpose=_TRANSPOSE[index // 60 % len(_TRANSPOSE)])
        yield BlockResult(index=index, t_start=index * cfg.block_seconds, lock=lock,
                          features=feats, mood=mood, controls=controls,
                          residual_rms_db=feats.rms_db, raw_rms_db=-36.0 + n(0.8))


# Feeds -------------------------------------------------------------------------

class OscFeed:
    """UDP OSC server in a daemon thread; each ``/roomtone/block`` goes to the app."""

    def __init__(self, app: "RoomtoneApp", host: str, port: int) -> None:
        from pythonosc.dispatcher import Dispatcher
        from pythonosc.osc_server import BlockingOSCUDPServer

        self.app = app
        dispatcher = Dispatcher()
        dispatcher.map(OSC_ADDRESS, self._on_block)
        self.server = BlockingOSCUDPServer((host, port), dispatcher)
        self.port: int = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.1},
                                       name="roomtone-osc", daemon=True)

    def start(self) -> None:
        self.thread.start()
        log.info("OSC feed listening on %s:%d", *self.server.server_address)

    def stop(self) -> None:
        # shutdown() blocks until serve_forever returns, and a handler may be
        # waiting on the UI loop inside call_from_thread, so never block the loop.
        def close() -> None:
            self.server.shutdown()
            self.server.server_close()
        threading.Thread(target=close, name="roomtone-osc-close", daemon=True).start()

    def _on_block(self, address: str, *args: Any) -> None:
        try:
            result = block_from_dict(json.loads(args[0]))
        except (IndexError, TypeError, ValueError, KeyError) as exc:
            log.warning("bad %s payload: %s", address, exc)
            return
        self.app.feed(result)


class DemoFeed:
    """Replays ``demo_session`` at one block per ``interval`` seconds, looping."""

    def __init__(self, app: "RoomtoneApp", cfg: Config, interval: float = DEMO_INTERVAL_S) -> None:
        self.app, self.cfg, self.interval = app, cfg, interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="roomtone-demo", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            for result in demo_session(self.cfg):
                if self.stop_event.wait(self.interval):
                    return
                self.app.feed(result)


# App ---------------------------------------------------------------------------

CSS = """
Screen {
    background: $rt-bg;
    color: $rt-fg;
    layout: vertical;
    padding: 0 1;
}
#header { height: 1; }
.row { height: 10; margin-top: 1; }
.panel { width: 1fr; height: 10; padding-right: 1; background: $rt-bg; }
#log { height: 10; margin-top: 1; display: none; }
#footer { height: 2; dock: bottom; }
RoomtoneApp.compact .row, RoomtoneApp.compact #log { margin-top: 0; }
RoomtoneApp.show-log #bottom { display: none; }
RoomtoneApp.show-log #log { display: block; }
"""


class RoomtoneApp(App[None]):
    """M8-styled monitor for the Listener and Controller."""

    CSS = CSS
    TITLE = "roomtone"
    BINDINGS = [
        Binding("q", "quit", "quit"),
        Binding("d", "toggle_demo", "demo"),
        Binding("l", "toggle_log", "log"),
    ]

    result: reactive[BlockResult | None] = reactive(None, always_update=True, init=False)
    playhead: reactive[int] = reactive(0, init=False)
    compact: reactive[bool] = reactive(False, init=False)
    show_log: reactive[bool] = reactive(False, init=False)

    def __init__(self, cfg: Config | None = None, demo: bool = False,
                 osc_port: int | None = -1, theme_name: str = DEFAULT_THEME,
                 playhead_interval: float = 1.0) -> None:
        self.cfg = cfg or Config()
        self.palette: dict[str, str] = dict(THEMES[theme_name])
        super().__init__()
        self._want_demo = demo
        self._osc_port = self.cfg.tui_port if osc_port == -1 else osc_port
        self._playhead_interval = playhead_interval
        self.osc: OscFeed | None = None
        self.demo: DemoFeed | None = None
        self.history: deque[MoodName] = deque(maxlen=HISTORY_LEN)
        self.events: deque[str] = deque(maxlen=64)
        self.received = 0
        self._ui_thread: int | None = None
        self._live = False

    # theme ----------------------------------------------------------------------

    def get_css_variables(self) -> dict[str, str]:
        variables = super().get_css_variables()
        variables.update({f"rt-{k}": v for k, v in self.palette.items()})
        return variables

    # lifecycle ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static(id="header")
        with Horizontal(classes="row", id="top"):
            yield Static(id="lock", classes="panel")
            yield Static(id="room", classes="panel")
        with Horizontal(classes="row", id="bottom"):
            yield Static(id="mood", classes="panel")
            yield Static(id="layers", classes="panel")
        yield Static(id="log")
        yield Static(id="footer")

    def on_mount(self) -> None:
        self._ui_thread = threading.get_ident()
        self._live = True
        self._apply_size(self.size.width, self.size.height)
        if self._osc_port is not None:
            try:
                self.osc = OscFeed(self, self.cfg.tui_host, self._osc_port)
                self.osc.start()
            except OSError as exc:
                log.error("OSC feed unavailable on port %s: %s", self._osc_port, exc)
                self._event(f"OSC PORT {self._osc_port} UNAVAILABLE")
                self.osc = None
        if self._want_demo:
            self.action_toggle_demo()
        self.set_interval(self._playhead_interval, self._advance_playhead)
        self._paint()

    def on_unmount(self) -> None:
        if self.demo:
            self.demo.stop()
        if self.osc:
            self.osc.stop()

    def on_resize(self, event: events.Resize) -> None:
        self._apply_size(event.size.width, event.size.height)

    def _apply_size(self, width: int, height: int) -> None:
        self.compact = width < 100 or height < 28

    # data in --------------------------------------------------------------------

    def feed(self, result: BlockResult) -> None:
        """Hand the app one block. Safe from any thread."""
        if self._ui_thread is not None and threading.get_ident() != self._ui_thread:
            try:
                self.call_from_thread(self.update, result)
            except RuntimeError as exc:
                log.debug("dropped block %d: %s", result.index, exc)
            return
        self.update(result)

    def update(self, result: BlockResult) -> None:
        """The single entry point for new state; everything repaints from here."""
        prev = self.result
        self.received += 1
        self.history.append(result.mood.name)
        stamp = f"{clock(result.t_start)} BLK {hex2(result.index)}"
        if prev is None or prev.lock.status is not result.lock.status:
            self._event(f"{stamp} LOCK {result.lock.status.name}")
        if prev is None or prev.mood.name is not result.mood.name:
            self._event(f"{stamp} MOOD {result.mood.name.value.upper()}")
        if prev is not None and prev.lock.marker_rung != result.lock.marker_rung:
            self._event(f"{stamp} RUNG {hex2(result.lock.marker_rung)}")
        self.result = result

    def _event(self, line: str) -> None:
        self.events.append(line)
        log.info(line)

    # rendering ------------------------------------------------------------------

    @property
    def source(self) -> str:
        parts = []
        if self.demo:
            parts.append("DEMO")
        if self.osc:
            parts.append(f"OSC {self.osc.port}")
        return " ".join(parts)

    @property
    def bar(self) -> int:
        return BAR_COMPACT if self.compact else BAR_WIDE

    def watch_result(self) -> None:
        self._paint()

    def watch_playhead(self) -> None:
        if self._live:
            self._paint_layers()

    def watch_compact(self, compact: bool) -> None:
        self.set_class(compact, "compact")
        self._paint()

    def watch_show_log(self, show: bool) -> None:
        self.set_class(show, "show-log")
        self._paint()

    def _paint_layers(self) -> None:
        controls = self.result.controls if self.result else Controls()
        self.query_one("#layers", Static).update(render_layers(controls, self.playhead, self.palette))

    def _paint(self) -> None:
        if not self._live:
            return
        r, th = self.result, self.palette

        def q(wid: str) -> Static:
            return self.query_one(wid, Static)

        q("#header").update(render_header(r, th, self.source))
        q("#lock").update(render_lock(r.lock if r else LockState(), th, self.bar,
                                      self.cfg.sample_rate, self.cfg.lock_erle_db,
                                      self.cfg.lock_jitter_samples))
        q("#room").update(render_room(r.features if r else Features(), th, self.bar))
        q("#mood").update(render_mood(r.mood if r else Mood(), self.history, th, self.bar))
        self._paint_layers()
        q("#log").update(render_log(self.events, th))
        q("#footer").update(render_footer(r.controls if r else Controls(), th, bool(self.demo)))

    def _advance_playhead(self) -> None:
        self.playhead = (self.playhead + 1) % len(config.LAYER_ORDER)

    # actions --------------------------------------------------------------------

    def action_toggle_demo(self) -> None:
        if self.demo:
            self.demo.stop()
            self.demo = None
            self._event("DEMO OFF")
        else:
            self.demo = DemoFeed(self, self.cfg)
            self.demo.start()
            self._event("DEMO ON")
        self._paint()

    def action_toggle_log(self) -> None:
        self.show_log = not self.show_log


# Entry points ----------------------------------------------------------------

def run(cfg: Config | None = None, demo: bool = False, theme_name: str = DEFAULT_THEME) -> None:
    if not logging.getLogger().handlers:
        from textual.logging import TextualHandler
        logging.basicConfig(level=logging.INFO, handlers=[TextualHandler()])
    RoomtoneApp(cfg or Config(), demo=demo, theme_name=theme_name).run()


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="roomtone.tui", description="M8-styled roomtone monitor")
    ap.add_argument("--demo", action="store_true", help="replay a synthetic two-hour session")
    ap.add_argument("--port", type=int, default=config.TUI_PORT, help="OSC listen port")
    ap.add_argument("--theme", choices=sorted(THEMES), default=DEFAULT_THEME)
    args = ap.parse_args(argv)
    cfg = Config(tui_port=args.port)
    run(cfg, demo=args.demo, theme_name=args.theme)


if __name__ == "__main__":
    main()
