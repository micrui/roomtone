"""TUI tests: pure renderers, the mounted app under a pilot, and the OSC feed."""

from __future__ import annotations

import asyncio
import json
from itertools import islice

from pythonosc.udp_client import SimpleUDPClient
from rich.text import Text

from roomtone import config
from roomtone.config import Config
from roomtone.tui import (
    EMPTY_CELL, FULL, LADDER, OFF_CELL, ON_CELL, THEME, RoomtoneApp, Static,
    block_from_dict, cc, demo_session, history_strip, marker_label, meter,
    render_footer, render_header, render_layers, render_lock, render_mood, render_room,
)
from roomtone.types import (
    BlockResult, Controls, Features, LockState, LockStatus, Mood, MoodName,
)


def sample() -> BlockResult:
    return BlockResult(
        index=10, t_start=3723.0,
        lock=LockState(status=LockStatus.LOCKED, delay_samples=4410.0, drift_ppm=23.5,
                       peak_to_sidelobe_db=41.2, jitter_samples=0.5, erle_db=24.0,
                       rir_change=0.012, marker_rung=2, blocks_ok=31, blocks_bad=0),
        features=Features(rms_db=-45.0, speech=0.5, talkers=3.0, transient_rate=2.0,
                          beat=1.0, speech_band_centroid_hz=2000.0,
                          labels={"Speech": 1.0, "Dishes": 0.25, "Silence": 0.1, "Hum": 0.05}),
        mood=Mood(energy=0.5, voices=1.0, rhythm=0.0, name=MoodName.CONVERSATION, confidence=0.75),
        controls=Controls(energy=0.5, bright=0.5, layers=0b00100011, tempo_bpm=66.4,
                          scale="lydian", transpose=-2),
    )


def line(text: Text, prefix: str) -> str:
    return next(l for l in text.plain.splitlines() if l.startswith(prefix))


def styles_of(text: Text, needle: str) -> set[str]:
    start = text.plain.index(needle)
    end = start + len(needle)
    return {str(s.style) for s in text.spans if s.start <= start and s.end >= end}


# Pure renderers ------------------------------------------------------------

def test_helpers():
    assert cc(0.0) == 0 and cc(1.0) == 127 and cc(0.5) == 64 and cc(2.0) == 127
    m = meter(0.5, 10)
    assert m.plain == FULL * 5 + EMPTY_CELL * 5
    assert len(meter(1.0, 16).plain) == 16 and meter(0.0, 7).plain == EMPTY_CELL * 7
    full = meter(1.0, 10)
    assert full.spans[0].style == THEME["ok"] and full.spans[-1].style == THEME["hot"]
    assert marker_label(LADDER[0]) == "LEN 04s BAND 4K-10K LVL -30"
    assert marker_label(LADDER[-1]) == "LEN 30s BAND 6K-12K LVL -10"
    assert len(LADDER) == 9


def test_header():
    r = sample()
    h = render_header(r)
    assert h.plain.startswith("ROOMTONE  01:02:03  BLK 0A  LOCKED")
    assert f"bold {THEME['ok']}" in styles_of(h, "LOCKED")
    assert f"bold {THEME['title']}" in styles_of(h, "ROOMTONE")
    for status, role in ((LockStatus.ACQUIRING, "warn"), (LockStatus.LOST, "select"),
                         (LockStatus.SATURATED, "hot")):
        r.lock.status = status
        assert f"bold {THEME[role]}" in styles_of(render_header(r), status.name)
    assert "BLK --" in render_header(None).plain


def test_lock_panel():
    t = render_lock(sample().lock, bar=15)
    assert line(t, "DELAY").split() == ["DELAY", "4410.0", "smp", "100.00ms"]
    assert line(t, "DRIFT").split() == ["DRIFT", "+23.50", "ppm"]
    assert line(t, "ERLE").startswith("ERLE          24.0 dB ")
    assert line(t, "ERLE").endswith(FULL * 12 + EMPTY_CELL * 3)       # 24 of 30 dB
    assert line(t, "MARKER") == "MARKER  LEN 16s BAND 4K-10K LVL -30"
    assert line(t, "RUNG") == "RUNG    " + ON_CELL * 3 + OFF_CELL * 6 + " 02/08"
    assert line(t, "OK") == "OK      1F  BAD 00"
    weak = render_lock(LockState(erle_db=10.0, jitter_samples=3.0), bar=15)
    assert "|" in line(weak, "ERLE")                                   # target tick at 20 dB
    assert THEME["hot"] in styles_of(weak, "3.00")


def test_room_panel():
    t = render_room(sample().features, bar=10)
    assert line(t, "RMS").split()[1] == "-45.0dB"
    assert line(t, "SPEECH ").split()[1] == "40"                       # 0.5 -> 0x40
    assert line(t, "BEAT").split()[1:] == ["7F", FULL * 10]
    assert line(t, "CENTROID").split()[1:] == ["2000Hz", FULL * 5 + EMPTY_CELL * 5]
    labels = t.plain.splitlines()[7:]
    assert [l.split()[0] for l in labels] == ["SPEECH", "DISHES", "SILENCE"]
    assert labels[1].split()[1] == "20"


def test_mood_panel():
    hist = [MoodName.EMPTY] * 3 + [MoodName.COOKING, MoodName.CONVERSATION]
    t = render_mood(sample().mood, hist, bar=8)
    assert line(t, "ENERGY") == "ENERGY  40 " + FULL * 4 + EMPTY_CELL * 4
    assert line(t, "VOICES").split()[1] == "7F"
    assert line(t, "STATE") == "STATE   CONVERSATION"
    assert f"bold {THEME['title']}" in styles_of(t, "CONVERSATION")
    assert line(t, "HIST") == "HIST    EEEC V--- ---- ----"
    strip = history_strip([MoodName.LOUD_BUSY] * 20)
    assert strip.plain == "LLLL LLLL LLLL LLLL"
    assert strip.spans[-1].style == THEME["select"]


def test_layers_panel():
    t = render_layers(sample().controls, playhead=1)
    rows = t.plain.splitlines()
    assert rows[0].startswith("LAYERS  MASK 23")
    assert rows[1] == " TRK CH LAYER    ON VOL CUT"
    assert len(rows) == 2 + len(config.LAYER_ORDER)
    assert rows[2].split() == ["1", "01", "DRONE", ON_CELL, "59", "40"]
    assert rows[3].startswith(">") and rows[3].split()[2:] == ["02", "PAD", ON_CELL, "59", "40"]
    assert rows[5].split() == ["4", "04", "PHRASES", OFF_CELL, "--", "--"]
    assert rows[7].split() == ["6", "06", "PULSE", ON_CELL, "59", "--"]
    assert f"bold {THEME['select']}" in styles_of(t, "PAD")


def test_footer():
    t = render_footer(sample().controls)
    assert t.plain.splitlines()[0] == "TEMPO 066 SCALE LYDIAN TRN -02"
    assert t.plain.splitlines()[1].startswith("q quit  d demo  l log")
    assert render_footer(Controls()).plain.startswith("TEMPO 060 SCALE OPEN TRN +00")


def test_round_trip_and_demo_shape():
    r = sample()
    assert block_from_dict(json.loads(json.dumps(r.as_dict()))) == r
    session = list(demo_session())
    assert len(session) == 240 and session[-1].t_start == 239 * config.BLOCK_SECONDS
    seen = {b.lock.status for b in session}
    assert seen == set(LockStatus)
    assert {b.mood.name for b in session} == set(MoodName)
    assert session[0].lock.status is LockStatus.ACQUIRING


# Mounted app -----------------------------------------------------------------

def header_plain(app: RoomtoneApp) -> str:
    return str(app.query_one("#header", Static).content)


def test_app_feed_demo_results():
    async def scenario() -> None:
        app = RoomtoneApp(Config(), osc_port=None)
        async with app.run_test(size=(100, 32)) as pilot:
            for result in islice(demo_session(), 3):
                app.feed(result)
            await pilot.pause()
            text = header_plain(app)
            assert "BLK 02" in text and "ACQUIRING" in text
            assert not app.compact
            assert len(app.history) == 3
            await pilot.press("l")
            assert app.show_log and app.has_class("show-log")
            await pilot.press("q")

    asyncio.run(scenario())


def test_app_compact_at_80x24():
    async def scenario() -> None:
        app = RoomtoneApp(Config(), osc_port=None)
        async with app.run_test(size=(80, 24)) as pilot:
            app.feed(sample())
            await pilot.pause()
            assert app.compact and app.has_class("compact")
            assert "BLK 0A" in header_plain(app)
            assert app.query_one("#footer").region.bottom <= 24

    asyncio.run(scenario())


def test_osc_feed():
    async def scenario() -> None:
        app = RoomtoneApp(Config(), osc_port=0)
        async with app.run_test(size=(100, 32)) as pilot:
            assert app.osc is not None
            client = SimpleUDPClient("127.0.0.1", app.osc.port)
            payload = sample()
            payload.index = 0x2C
            client.send_message("/roomtone/block", json.dumps(payload.as_dict()))
            for _ in range(50):
                await pilot.pause(0.05)
                if app.received:
                    break
            assert app.received == 1
            assert app.result is not None and app.result.index == 0x2C
            assert "BLK 2C" in header_plain(app) and "LOCKED" in header_plain(app)

    asyncio.run(scenario())
