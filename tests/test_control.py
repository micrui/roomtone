import json
import random
import threading

import pytest
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from roomtone.config import LAYER_ORDER
from roomtone.control import (Controller, TidalPublisher, TuiPublisher, ctrl_messages,
                              layers_for)
from roomtone.types import (BlockResult, Controls, Features, LockState, Mood, MoodName)


def names(mask: int) -> set[str]:
    return {n for i, n in enumerate(LAYER_ORDER) if mask >> i & 1}


@pytest.mark.parametrize("energy, voices, rhythm, mood, expected", [
    (0.0, 0.0, 0.0, MoodName.EMPTY, {"drone"}),
    (0.15, 0.0, 0.0, MoodName.EMPTY, {"drone"}),
    (0.16, 0.0, 0.0, MoodName.EMPTY, {"drone", "pad"}),
    (0.31, 0.0, 0.0, MoodName.EMPTY, {"drone", "pad", "figure"}),
    (0.31, 0.0, 0.0, MoodName.COOKING, {"drone", "pad", "figure", "phrases"}),
    (0.41, 0.0, 0.0, MoodName.EMPTY, {"drone", "pad", "figure", "shimmer"}),
    (0.0, 0.0, 0.31, MoodName.EMPTY, {"drone", "pulse"}),
    (0.51, 0.0, 0.0, MoodName.EMPTY, {"drone", "pad", "figure", "shimmer", "pulse"}),
    (0.61, 0.0, 0.0, MoodName.COOKING,
     {"drone", "pad", "figure", "phrases", "shimmer", "pulse", "bass"}),
    (0.61, 0.6, 0.0, MoodName.CONVERSATION, {"drone", "pad", "shimmer", "pulse", "bass"}),
    (1.0, 1.0, 1.0, MoodName.LOUD_BUSY, {"drone", "pad", "shimmer", "pulse", "bass"}),
])
def test_layer_thresholds(energy, voices, rhythm, mood, expected):
    assert names(layers_for(energy, voices, rhythm, mood)) == expected
    assert "spare" not in names(layers_for(1, 0, 1, MoodName.COOKING))


def test_controller_mapping_without_slew():
    c = Controller(max_step=1.0)
    out = c.update(Mood(energy=0.5, voices=0.2, rhythm=0.6, name=MoodName.COOKING), LockState())
    assert (out.energy, out.voices, out.rhythm) == (0.5, 0.2, 0.6)
    assert out.bright == pytest.approx(0.5)
    assert out.tempo_bpm == pytest.approx(84.0)
    assert out.scale == "pentatonic"
    assert "figure" in names(out.layers)


def test_voices_drop_figure_and_darken():
    c = Controller(max_step=1.0)
    out = c.update(Mood(energy=0.5, voices=0.7, name=MoodName.CONVERSATION))
    assert out.bright == pytest.approx(0.3)
    assert "figure" not in names(out.layers)


@pytest.mark.parametrize("mood, scale", [
    (MoodName.EMPTY, "open"), (MoodName.COOKING, "pentatonic"),
    (MoodName.CONVERSATION, "dorian"), (MoodName.LOUD_BUSY, "lydian")])
def test_scale_by_mood(mood, scale):
    assert Controller().update(Mood(name=mood)).scale == scale


def test_slew_limits_every_float():
    c = Controller()
    prev = c.state
    for _ in range(12):
        out = c.update(Mood(energy=1.0, voices=1.0, rhythm=1.0, name=MoodName.LOUD_BUSY))
        for key in ("energy", "voices", "rhythm", "bright"):
            assert abs(getattr(out, key) - getattr(prev, key)) <= 0.1 + 1e-9
        assert abs(out.tempo_bpm - (54 + 50 * prev.rhythm)) <= 5.0 + 1e-9
        prev = out
    assert out.energy == pytest.approx(1.0) and out.tempo_bpm == pytest.approx(104.0)
    assert out.bright == pytest.approx(0.6)


def test_transpose_drifts_every_40_blocks():
    c = Controller(rng=random.Random(7))
    seen = [c.update(Mood()).transpose for _ in range(200)]
    changes = [i for i in range(1, 200) if seen[i] != seen[i - 1]]
    assert changes == [40, 80, 120, 160]
    assert all(-5 <= v <= 5 for v in seen) and seen[0] == 0
    assert all(abs(seen[i] - seen[i - 1]) in (2, 5) for i in changes)   # by fifths, never a semitone


def test_transpose_drift_covers_both_directions_and_stays_bounded():
    from roomtone.control import TRANSPOSE_LIMIT
    c = Controller(rng=random.Random(1), transpose_every=1)
    seen = [c.update(Mood()).transpose for _ in range(400)]
    assert min(seen) < 0 < max(seen) and all(abs(v) <= TRANSPOSE_LIMIT for v in seen)


class CaptureClient:
    def __init__(self):
        self.sent = []

    def send_message(self, address, value):
        self.sent.append((address, value))


def test_ctrl_messages_shape():
    msgs = dict(ctrl_messages(Controls(layers=0b101, transpose=-2, scale="minor")))
    assert msgs["layers"] == 5.0 and msgs["transpose"] == -2.0 and msgs["scale"] == "minor"
    assert msgs["layer_drone"] == 1.0 and msgs["layer_pad"] == 0.0 and msgs["layer_figure"] == 1.0
    assert all(isinstance(v, float) for k, v in msgs.items() if k != "scale")
    assert len(msgs) == len(Controls().as_dict()) + len(LAYER_ORDER)


def test_tidal_publisher_with_fake_client():
    client = CaptureClient()
    TidalPublisher("127.0.0.1", 1, client=client).send(Controls(energy=0.25))
    assert all(addr == "/ctrl" and len(args) == 2 for addr, args in client.sent)
    assert ["energy", 0.25] in [args for _, args in client.sent]


def test_tidal_publisher_over_udp():
    got = []
    done = threading.Event()
    disp = Dispatcher()

    def on_ctrl(addr, key, value):
        got.append((key, value))
        if len(got) == len(Controls().as_dict()) + len(LAYER_ORDER):
            done.set()

    disp.map("/ctrl", on_ctrl)
    server = ThreadingOSCUDPServer(("127.0.0.1", 0), disp)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        port = server.server_address[1]
        TidalPublisher("127.0.0.1", port).send(Controls(energy=0.5, scale="lydian", layers=3))
        assert done.wait(5)
    finally:
        server.shutdown()
        server.server_close()
    d = dict(got)
    assert d["energy"] == pytest.approx(0.5) and d["scale"] == "lydian"
    assert d["layers"] == pytest.approx(3.0) and d["layer_pad"] == pytest.approx(1.0)


def test_tui_publisher_sends_json():
    client = CaptureClient()
    r = BlockResult(index=3, t_start=90.0, lock=LockState(), features=Features(),
                    mood=Mood(name=MoodName.COOKING), controls=Controls())
    TuiPublisher("127.0.0.1", 1, client=client).publish(r)
    (addr, payload), = client.sent
    assert addr == "/roomtone/block"
    data = json.loads(payload)
    assert data["index"] == 3 and data["mood"]["name"] == "cooking"
    assert data["lock"]["status"] == "acquiring"
