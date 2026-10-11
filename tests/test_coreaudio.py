"""The parts of roomtone.coreaudio that need no HAL: codes, errors, device matching.

The HAL calls themselves (devices, create/destroy) only run on macOS and change
system state, so they are exercised by hand with `roomtone aggregate`, not here.
"""

import struct

import pytest

from roomtone import coreaudio as ca


def dev(name, uid=None, inputs=0, outputs=0, transport="usb", rate=44100.0, rates=(44100.0,)):
    return ca.Device(id=hash(name) & 0xFFFF, uid=uid or f"uid:{name}", name=name, transport=transport,
                     inputs=inputs, outputs=outputs, sample_rate=rate, rates=list(rates))


LAPTOP = [
    dev("M8", "m8:in", inputs=2),
    dev("M8", "m8:out", outputs=2),
    dev("MacBook Air Microphone", inputs=1, transport="built-in", rate=48000.0,
        rates=(44100.0, 48000.0, 96000.0)),
    dev("MacBook Air Speakers", outputs=2, transport="built-in", rate=48000.0,
        rates=(44100.0, 48000.0, 96000.0)),
    dev("Michael's AirPods Pro", inputs=1, transport="bluetooth", rate=24000.0, rates=(24000.0,)),
]


def test_fourcc_is_big_endian_ascii():
    assert ca._fourcc("dev#") == struct.unpack(">I", b"dev#")[0]
    assert ca.PROP_UID == ca._fourcc("uid ") and ca.SCOPE_INPUT == ca._fourcc("inpt")


def test_error_shows_four_char_codes_and_plain_numbers():
    e = ca.CoreAudioError("opening", struct.unpack(">i", b"!dev")[0])
    assert "!dev" in str(e) and "opening" in str(e)
    assert "-50" in str(ca.CoreAudioError("x", -50))


def test_find_separates_the_two_m8_devices():
    assert ca.find("M8", inputs=True, among=LAPTOP).uid == "m8:in"
    assert ca.find("M8", outputs=True, among=LAPTOP).uid == "m8:out"


def test_find_prefers_exact_then_substring_case_insensitively():
    assert ca.find("macbook air micro", inputs=True, among=LAPTOP).name == "MacBook Air Microphone"
    assert ca.find("MacBook Air Speakers", among=LAPTOP).outputs == 2


def test_find_lists_what_is_there_when_nothing_matches():
    with pytest.raises(LookupError) as exc:
        ca.find("UMIK-1", inputs=True, among=LAPTOP)
    msg = str(exc.value)
    assert "no input device matching 'UMIK-1'" in msg
    assert "'MacBook Air Microphone'" in msg and "Speakers" not in msg


def test_device_str_mentions_rates_and_uid():
    text = str(LAPTOP[2])
    assert "44.1k/48k/96k" in text and "uid:MacBook Air Microphone" in text and "built-in" in text
    assert not LAPTOP[2].is_aggregate
    assert dev("roomtone", transport="aggregate").is_aggregate


def test_create_aggregate_rejects_a_clock_outside_the_members_and_bad_rates(monkeypatch):
    monkeypatch.setattr(ca, "_load", lambda: (None, None))
    m8, mic, spk, pods = LAPTOP[0], LAPTOP[2], LAPTOP[3], LAPTOP[4]
    with pytest.raises(ValueError, match="must be one of the members"):
        ca.create_aggregate("x", [(m8, True), (mic, True)], clock=spk)
    with pytest.raises(ValueError, match="does not support 44100 Hz"):
        ca.create_aggregate("x", [(pods, True), (spk, False)], clock=spk, sample_rate=44100)


def test_destroy_refuses_a_plain_device(monkeypatch):
    monkeypatch.setattr(ca, "_load", lambda: (None, None))
    with pytest.raises(ValueError, match="not an aggregate"):
        ca.destroy_aggregate(LAPTOP[0])
