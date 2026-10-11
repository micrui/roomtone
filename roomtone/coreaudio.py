"""The "roomtone" aggregate device, built from the command line instead of Audio MIDI Setup.

docs/BENCH.md step 3 builds an aggregate device by hand: tick the M8, the mic and the
output in Audio MIDI Setup, name it "roomtone", make the output the clock source, turn on
drift correction for the others. This module does the same through CoreAudio's HAL with
ctypes, so the setup is one repeatable command (`roomtone aggregate create`) and the
channel order is known: the members are added in the order given, so with M8 first and
the mic second the M8 is input channels 0 and 1 and the mic is channel 2, which is what
`roomtone.audio.AudioIO` assumes.

An aggregate made this way is public (not private to this process), so it persists like
one made in Audio MIDI Setup and shows up there; `roomtone aggregate destroy` removes it.
Nothing here needs microphone permission: it only describes devices, it never opens one.

macOS only. Importing works anywhere; the first call into CoreAudio raises OSError
elsewhere.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import struct
from dataclasses import dataclass
from typing import Iterable

# --- CoreAudio constants ------------------------------------------------------

def _fourcc(code: str) -> int:
    return struct.unpack(">I", code.encode("ascii"))[0]


SYSTEM_OBJECT = 1                                   # kAudioObjectSystemObject
PROP_DEVICES = _fourcc("dev#")                      # kAudioHardwarePropertyDevices
PROP_UID = _fourcc("uid ")                          # kAudioDevicePropertyDeviceUID
PROP_NAME = _fourcc("lnam")                         # kAudioObjectPropertyName
PROP_TRANSPORT = _fourcc("tran")                    # kAudioDevicePropertyTransportType
PROP_NOMINAL_RATE = _fourcc("nsrt")                 # kAudioDevicePropertyNominalSampleRate
PROP_AVAILABLE_RATES = _fourcc("nsr#")              # kAudioDevicePropertyAvailableNominalSampleRates
PROP_STREAM_CONFIG = _fourcc("slay")                # kAudioDevicePropertyStreamConfiguration
PROP_SUBDEVICES = _fourcc("grup")                   # kAudioAggregateDevicePropertyFullSubDeviceList
PROP_MAIN_SUBDEVICE = _fourcc("amst")               # kAudioAggregateDevicePropertyMainSubDevice
SCOPE_GLOBAL = _fourcc("glob")
SCOPE_INPUT = _fourcc("inpt")
SCOPE_OUTPUT = _fourcc("outp")
ELEMENT_MAIN = 0
TRANSPORT_AGGREGATE = _fourcc("grup")
TRANSPORT_NAMES = {
    _fourcc("bltn"): "built-in", _fourcc("usb "): "usb", _fourcc("grup"): "aggregate",
    _fourcc("blue"): "bluetooth", _fourcc("virt"): "virtual", _fourcc("thnd"): "thunderbolt",
    _fourcc("hdmi"): "hdmi", _fourcc("dprt"): "displayport", _fourcc("airp"): "airplay",
    _fourcc("cont"): "continuity", _fourcc("pci "): "pci", _fourcc("fwir"): "firewire",
}

CF_STRING_UTF8 = 0x08000100
CF_NUMBER_SINT32 = 3


class _PropertyAddress(ctypes.Structure):
    _fields_ = [("selector", ctypes.c_uint32), ("scope", ctypes.c_uint32), ("element", ctypes.c_uint32)]


def _address(selector: int, scope: int = SCOPE_GLOBAL) -> _PropertyAddress:
    return _PropertyAddress(selector, scope, ELEMENT_MAIN)


class CoreAudioError(OSError):
    """A HAL call returned a non-zero OSStatus."""

    def __init__(self, what: str, status: int):
        code = struct.pack(">i", status)
        text = code.decode("ascii") if all(32 <= b < 127 for b in code) else str(status)
        super().__init__(f"{what}: CoreAudio status {text}")
        self.status = status


_libs: tuple[ctypes.CDLL, ctypes.CDLL] | None = None


def _load() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    """CoreFoundation and CoreAudio, with the signatures this module uses."""
    global _libs
    if _libs is not None:
        return _libs
    cf_path = ctypes.util.find_library("CoreFoundation")
    ca_path = ctypes.util.find_library("CoreAudio")
    if not cf_path or not ca_path:
        raise OSError("CoreAudio is only available on macOS")
    cf, ca = ctypes.CDLL(cf_path), ctypes.CDLL(ca_path)
    p, i, u = ctypes.c_void_p, ctypes.c_long, ctypes.c_uint32

    cf.CFStringCreateWithCString.restype = p
    cf.CFStringCreateWithCString.argtypes = [p, ctypes.c_char_p, u]
    cf.CFStringGetCString.restype = ctypes.c_bool
    cf.CFStringGetCString.argtypes = [p, ctypes.c_char_p, i, u]
    cf.CFDictionaryCreateMutable.restype = p
    cf.CFDictionaryCreateMutable.argtypes = [p, i, p, p]
    cf.CFDictionarySetValue.restype = None
    cf.CFDictionarySetValue.argtypes = [p, p, p]
    cf.CFArrayCreateMutable.restype = p
    cf.CFArrayCreateMutable.argtypes = [p, i, p]
    cf.CFArrayAppendValue.restype = None
    cf.CFArrayAppendValue.argtypes = [p, p]
    cf.CFArrayGetCount.restype = i
    cf.CFArrayGetCount.argtypes = [p]
    cf.CFArrayGetValueAtIndex.restype = p
    cf.CFArrayGetValueAtIndex.argtypes = [p, i]
    cf.CFNumberCreate.restype = p
    cf.CFNumberCreate.argtypes = [p, i, p]
    cf.CFRelease.restype = None
    cf.CFRelease.argtypes = [p]

    addr = ctypes.POINTER(_PropertyAddress)
    ca.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
    ca.AudioObjectGetPropertyDataSize.argtypes = [u, addr, u, p, ctypes.POINTER(u)]
    ca.AudioObjectGetPropertyData.restype = ctypes.c_int32
    ca.AudioObjectGetPropertyData.argtypes = [u, addr, u, p, ctypes.POINTER(u), p]
    ca.AudioObjectSetPropertyData.restype = ctypes.c_int32
    ca.AudioObjectSetPropertyData.argtypes = [u, addr, u, p, u, p]
    ca.AudioObjectHasProperty.restype = ctypes.c_bool
    ca.AudioObjectHasProperty.argtypes = [u, addr]
    ca.AudioHardwareCreateAggregateDevice.restype = ctypes.c_int32
    ca.AudioHardwareCreateAggregateDevice.argtypes = [p, ctypes.POINTER(u)]
    ca.AudioHardwareDestroyAggregateDevice.restype = ctypes.c_int32
    ca.AudioHardwareDestroyAggregateDevice.argtypes = [u]
    _libs = (cf, ca)
    return _libs


# --- CF helpers ---------------------------------------------------------------

def _cfstr(text: str) -> int:
    cf, _ = _load()
    return cf.CFStringCreateWithCString(None, text.encode("utf-8"), CF_STRING_UTF8)


def _cfstr_to_py(ref: int, release: bool = False) -> str:
    cf, _ = _load()
    if not ref:
        return ""
    buf = ctypes.create_string_buffer(1024)
    ok = cf.CFStringGetCString(ref, buf, len(buf), CF_STRING_UTF8)
    if release:
        cf.CFRelease(ref)
    return buf.value.decode("utf-8") if ok else ""


def _cfnum(value: int) -> int:
    cf, _ = _load()
    v = ctypes.c_int32(value)
    return cf.CFNumberCreate(None, CF_NUMBER_SINT32, ctypes.byref(v))


def _callbacks(cf: ctypes.CDLL, name: str) -> int:
    return ctypes.addressof(ctypes.c_char.in_dll(cf, name))


# --- property reads -----------------------------------------------------------

def _get_raw(obj: int, selector: int, scope: int = SCOPE_GLOBAL) -> bytes:
    _, ca = _load()
    addr = _address(selector, scope)
    size = ctypes.c_uint32(0)
    st = ca.AudioObjectGetPropertyDataSize(obj, ctypes.byref(addr), 0, None, ctypes.byref(size))
    if st != 0:
        raise CoreAudioError(f"size of property {selector:#x} on object {obj}", st)
    buf = ctypes.create_string_buffer(size.value)
    st = ca.AudioObjectGetPropertyData(obj, ctypes.byref(addr), 0, None, ctypes.byref(size), buf)
    if st != 0:
        raise CoreAudioError(f"property {selector:#x} on object {obj}", st)
    return buf.raw[:size.value]


def _get_string(obj: int, selector: int) -> str:
    raw = _get_raw(obj, selector)
    (ref,) = struct.unpack("P", raw[:8])
    return _cfstr_to_py(ref, release=True)


def _get_string_list(obj: int, selector: int) -> list[str]:
    cf, _ = _load()
    (arr,) = struct.unpack("P", _get_raw(obj, selector)[:8])
    if not arr:
        return []
    out = [_cfstr_to_py(cf.CFArrayGetValueAtIndex(arr, k)) for k in range(cf.CFArrayGetCount(arr))]
    cf.CFRelease(arr)
    return out


def _channels(obj: int, scope: int) -> int:
    """Total channels across the device's streams in this scope (an AudioBufferList)."""
    _, ca = _load()
    if not ca.AudioObjectHasProperty(obj, ctypes.byref(_address(PROP_STREAM_CONFIG, scope))):
        return 0
    raw = _get_raw(obj, PROP_STREAM_CONFIG, scope)
    (n_buffers,) = struct.unpack_from("I", raw, 0)
    total = 0
    for k in range(n_buffers):              # AudioBuffer: UInt32 channels, UInt32 bytes, void* data
        (chans,) = struct.unpack_from("I", raw, 8 + 16 * k)
        total += chans
    return total


def _float64(obj: int, selector: int) -> float:
    (v,) = struct.unpack("d", _get_raw(obj, selector)[:8])
    return v


def _set_float64(obj: int, selector: int, value: float) -> None:
    _, ca = _load()
    addr = _address(selector)
    v = ctypes.c_double(value)
    st = ca.AudioObjectSetPropertyData(obj, ctypes.byref(addr), 0, None, 8, ctypes.byref(v))
    if st != 0:
        raise CoreAudioError(f"setting property {selector:#x} on object {obj}", st)


# --- the public API -----------------------------------------------------------

@dataclass
class Device:
    id: int
    uid: str
    name: str
    transport: str
    inputs: int
    outputs: int
    sample_rate: float
    rates: list[float]

    @property
    def is_aggregate(self) -> bool:
        return self.transport == "aggregate"

    def __str__(self) -> str:
        rates = "/".join(f"{r / 1000:g}k" for r in self.rates)
        return (f"[{self.id:3d}] {self.name!r}  {self.transport}; {self.inputs} in, "
                f"{self.outputs} out, {self.sample_rate:.0f} Hz (supports {rates})  uid {self.uid}")


def devices() -> list[Device]:
    """Every audio device the HAL knows, aggregates included."""
    raw = _get_raw(SYSTEM_OBJECT, PROP_DEVICES)
    ids = struct.unpack(f"{len(raw) // 4}I", raw)
    out = []
    for dev in ids:
        (transport,) = struct.unpack("I", _get_raw(dev, PROP_TRANSPORT)[:4])
        ranges = _get_raw(dev, PROP_AVAILABLE_RATES)
        rates = sorted({struct.unpack_from("dd", ranges, 16 * k)[0] for k in range(len(ranges) // 16)})
        out.append(Device(
            id=dev, uid=_get_string(dev, PROP_UID), name=_get_string(dev, PROP_NAME),
            transport=TRANSPORT_NAMES.get(transport, struct.pack(">I", transport).decode("ascii", "replace")),
            inputs=_channels(dev, SCOPE_INPUT), outputs=_channels(dev, SCOPE_OUTPUT),
            sample_rate=_float64(dev, PROP_NOMINAL_RATE), rates=rates,
        ))
    return out


def find(name: str, *, inputs: bool = False, outputs: bool = False,
         among: Iterable[Device] | None = None) -> Device:
    """The device whose name contains `name` (case-insensitive; an exact match wins).

    With `inputs` or `outputs`, only devices with channels of that kind count. That
    matters for the M8, which shows up as two devices of the same name, one each way.
    Raises LookupError, listing what is there, when nothing matches.
    """
    pool = [d for d in (among if among is not None else devices())
            if (d.inputs > 0 or not inputs) and (d.outputs > 0 or not outputs)]
    exact = [d for d in pool if d.name == name]
    loose = [d for d in pool if name.lower() in d.name.lower()]
    for hits in (exact, loose):
        if hits:
            return hits[0]
    kind = "input" if inputs else "output" if outputs else "audio"
    raise LookupError(f"no {kind} device matching {name!r}; have: "
                      + ", ".join(repr(d.name) for d in pool))


def aggregate_members(dev: Device) -> tuple[list[str], str]:
    """(member device UIDs in channel order, clock-source UID) of an aggregate device."""
    return _get_string_list(dev.id, PROP_SUBDEVICES), _get_string(dev.id, PROP_MAIN_SUBDEVICE)


def create_aggregate(name: str, members: list[tuple[Device, bool]], clock: Device,
                     sample_rate: float | None = None, uid: str | None = None) -> Device:
    """Make a public aggregate device and return it.

    `members` are (device, drift_correction) in the order their channels should
    appear. `clock` is the member whose clock the aggregate follows (drift
    correction is pointless on it and is forced off). All members are switched
    to `sample_rate` first when one is given; an aggregate runs at one rate.
    """
    cf, ca = _load()
    if clock.uid not in {d.uid for d, _ in members}:
        raise ValueError(f"clock device {clock.name!r} must be one of the members")
    if sample_rate is not None:
        for dev, _ in members:
            if sample_rate not in dev.rates:
                raise ValueError(f"{dev.name!r} does not support {sample_rate:g} Hz "
                                 f"(supports {', '.join(f'{r:g}' for r in dev.rates)})")
        for dev, _ in members:
            if dev.sample_rate != sample_rate:
                _set_float64(dev.id, PROP_NOMINAL_RATE, sample_rate)

    uid = uid or f"com.roomtone.aggregate.{name}"
    keep: list[int] = []            # CF objects to release afterwards

    def s(text: str) -> int:
        ref = _cfstr(text)
        keep.append(ref)
        return ref

    def n(value: int) -> int:
        ref = _cfnum(value)
        keep.append(ref)
        return ref

    subs = cf.CFArrayCreateMutable(None, 0, _callbacks(cf, "kCFTypeArrayCallBacks"))
    keep.append(subs)
    for dev, drift in members:
        sub = cf.CFDictionaryCreateMutable(None, 0, _callbacks(cf, "kCFTypeDictionaryKeyCallBacks"),
                                           _callbacks(cf, "kCFTypeDictionaryValueCallBacks"))
        keep.append(sub)
        cf.CFDictionarySetValue(sub, s("uid"), s(dev.uid))
        cf.CFDictionarySetValue(sub, s("drift"), n(1 if (drift and dev.uid != clock.uid) else 0))
        cf.CFArrayAppendValue(subs, sub)

    desc = cf.CFDictionaryCreateMutable(None, 0, _callbacks(cf, "kCFTypeDictionaryKeyCallBacks"),
                                        _callbacks(cf, "kCFTypeDictionaryValueCallBacks"))
    keep.append(desc)
    cf.CFDictionarySetValue(desc, s("name"), s(name))
    cf.CFDictionarySetValue(desc, s("uid"), s(uid))
    cf.CFDictionarySetValue(desc, s("subdevices"), subs)
    cf.CFDictionarySetValue(desc, s("master"), s(clock.uid))
    cf.CFDictionarySetValue(desc, s("private"), n(0))
    cf.CFDictionarySetValue(desc, s("stacked"), n(0))

    new_id = ctypes.c_uint32(0)
    try:
        st = ca.AudioHardwareCreateAggregateDevice(desc, ctypes.byref(new_id))
    finally:
        for ref in reversed(keep):
            cf.CFRelease(ref)
    if st != 0:
        raise CoreAudioError(f"creating aggregate device {name!r}", st)
    if sample_rate is not None:
        _set_float64(new_id.value, PROP_NOMINAL_RATE, sample_rate)
    for dev in devices():
        if dev.id == new_id.value:
            return dev
    raise LookupError(f"aggregate {name!r} was created (id {new_id.value}) but is not listed")


def destroy_aggregate(dev: Device) -> None:
    _, ca = _load()
    if not dev.is_aggregate:
        raise ValueError(f"{dev.name!r} is not an aggregate device")
    st = ca.AudioHardwareDestroyAggregateDevice(dev.id)
    if st != 0:
        raise CoreAudioError(f"destroying aggregate device {dev.name!r}", st)
