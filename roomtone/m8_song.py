"""The roomtone song on the M8, as data: eight instruments, the mixer, the MIDI setup.

`roomtone m8 setup` applies this through the M8's own menus over USB serial
(roomtone.m8.Editor). Every value is written the way the M8 shows it, so the
applier can verify each one on the screen. Hex values are two characters;
list values are matched by prefix against the text the M8 displays.

The palette is synthwave (Kavinsky, Crystal Castles): detuned saw drones and
chord pads with chorus and long reverb, a square-wave chip lead with dotted
echo, a filtered saw bass with a pluck envelope, an 80s kick for the pulse.
The held layers (drone, pad, bass, spare) carry an ADSR volume envelope with
full sustain, so a note lasts exactly as long as Tidal holds it; the short
layers decay on their own. Cutoffs and volumes are left where the mood CCs
will put them (m8/SETUP.md section 5).

Field addressing, for the applier: `inst` entries are (row prefix, token after
which the value sits, value). On the Instrument view most rows show a left
parameter and a right one ("TIMBRE  80        AMP 00"), hence the `after` key.
`mods` entries are (mod number 1-4, field, value); the MODS page shows MOD1 and
MOD3 side by side in the top half, MOD2 and MOD4 in the bottom half.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Instrument:
    number: int
    name: str
    type: str                                      # WAVSYNTH, MACROSYN, SAMPLER, FMSYNTH, HYPERSYN
    inst: list[tuple[str, str, str]] = field(default_factory=list)
    mods: list[tuple[int, str, str]] = field(default_factory=list)


def adsr(mod: int, dest: str, atk: str, dec: str, sus: str, rel: str, amt: str = "FF") -> list[tuple[int, str, str]]:
    return [(mod, "TYPE", "01ADSR"), (mod, "DEST", dest), (mod, "AMT", amt),
            (mod, "ATK", atk), (mod, "DEC", dec), (mod, "SUS", sus), (mod, "REL", rel)]


def ahd(mod: int, dest: str, atk: str, hold: str, dec: str, amt: str = "FF") -> list[tuple[int, str, str]]:
    return [(mod, "TYPE", "00AHD"), (mod, "DEST", dest), (mod, "AMT", amt),
            (mod, "ATK", atk), (mod, "HOLD", hold), (mod, "DEC", dec)]


def lfo(mod: int, dest: str, amt: str, freq: str, osc: str = "00TRI") -> list[tuple[int, str, str]]:
    return [(mod, "TYPE", "03LFO"), (mod, "DEST", dest), (mod, "AMT", amt), (mod, "OSC", osc),
            (mod, "FREQ", freq)]


def sends(dry: str, mfx: str, delay: str, rev: str, amp: str = "00") -> list[tuple[str, str, str]]:
    """The right-hand column of the Instrument view: AMP, DRY, MFX (chorus), DEL, REV."""
    return [("TIMBRE", "AMP", amp), ("REDUX", "DRY", dry), ("FILTER", "MFX", mfx),
            ("CUTOFF", "DEL", delay), ("RES", "REV", rev)]


def filt(kind: str, cutoff: str, res: str) -> list[tuple[str, str, str]]:
    return [("FILTER", "FILTER", kind), ("CUTOFF", "CUTOFF", cutoff), ("RES", "RES", res)]


INSTRUMENTS: list[Instrument] = [
    Instrument(0, "DRONE", "MACROSYN",
               inst=[("SHAPE", "SHAPE", "0ESAW SWARM"), ("TIMBRE", "TIMBRE", "70"), ("COLOR", "COLOR", "60")]
               + filt("01LOWPASS", "50", "10") + sends(dry="A0", mfx="60", delay="20", rev="B0"),
               mods=adsr(1, "01VOLUME", atk="60", dec="40", sus="FF", rel="80")
               + lfo(2, "07CUTOFF", amt="30", freq="02")),
    Instrument(1, "PAD", "HYPERSYN",
               inst=[("SHIFT", "SHIFT", "80"), ("SWARM", "SWARM", "40"), ("WIDTH", "WIDTH", "90"), ("SUBOSC", "SUBOSC", "60")]
               + filt("01LOWPASS", "60", "10") + [("SHIFT", "AMP", "00"), ("SUBOSC", "DRY", "90"),
                                                   ("FILTER", "MFX", "90"), ("CUTOFF", "DEL", "30"), ("RES", "REV", "C0")],
               mods=adsr(1, "01VOLUME", atk="80", dec="40", sus="E0", rel="A0")
               + lfo(2, "07CUTOFF", amt="20", freq="03")),
    Instrument(2, "FIGURE", "WAVSYNTH",
               inst=[("SHAPE", "SHAPE", "01PULSE 25%"), ("SIZE", "SIZE", "20")]
               + filt("01LOWPASS", "A0", "00") + [("SIZE", "AMP", "00"), ("SCAN", "DRY", "B0"),
                                                   ("FILTER", "MFX", "20"), ("CUTOFF", "DEL", "70"), ("RES", "REV", "60")],
               mods=ahd(1, "01VOLUME", atk="00", hold="10", dec="40")),
    Instrument(3, "PHRASES", "MACROSYN",                    # no samples chosen yet: a toy chip pluck stands in
               inst=[("SHAPE", "SHAPE", "10TOY"), ("TIMBRE", "TIMBRE", "40"), ("COLOR", "COLOR", "80")]
               + filt("01LOWPASS", "90", "00") + sends(dry="A0", mfx="30", delay="60", rev="80"),
               mods=ahd(1, "01VOLUME", atk="00", hold="20", dec="60")),
    Instrument(4, "SHIMMER", "FMSYNTH",
               inst=filt("00OFF", "FF", "00") + [("MOD1", "AMP", "00"), ("MOD4", "DRY", "A0"),
                                                  ("FILTER", "MFX", "40"), ("CUTOFF", "DEL", "90"), ("RES", "REV", "C0")],
               mods=ahd(1, "01VOLUME", atk="00", hold="00", dec="60")),
    Instrument(5, "PULSE", "MACROSYN",
               inst=[("SHAPE", "SHAPE", "22KICK"), ("TIMBRE", "TIMBRE", "40"), ("COLOR", "COLOR", "80")]
               + filt("00OFF", "FF", "00") + sends(dry="C0", mfx="00", delay="00", rev="50", amp="10"),
               mods=[]),
    Instrument(6, "BASS", "MACROSYN",
               inst=[("SHAPE", "SHAPE", "00CSAW"), ("TIMBRE", "TIMBRE", "60"), ("COLOR", "COLOR", "80")]
               + filt("01LOWPASS", "38", "30") + sends(dry="C0", mfx="20", delay="00", rev="10"),
               mods=adsr(1, "01VOLUME", atk="05", dec="60", sus="C0", rel="30")
               + ahd(2, "07CUTOFF", atk="00", hold="00", dec="50", amt="40")),
    Instrument(7, "SPARE", "MACROSYN",
               inst=[("SHAPE", "SHAPE", "0CTRIPLE SIN"), ("TIMBRE", "TIMBRE", "80"), ("COLOR", "COLOR", "80")]
               + filt("01LOWPASS", "80", "00") + sends(dry="80", mfx="30", delay="40", rev="80"),
               mods=adsr(1, "01VOLUME", atk="40", dec="40", sus="FF", rel="60")),
]

TRACK_VOLUME = "E0"          # Mixer track volumes 1-8; the mood CCs take over once mapped
SONG_NAME = "ROOMTONE"


# MIDI learn targets, in the order they are learned. Each is (CC number, where):
#   ("project", LABEL)        a Project view field (TEMPO, TRANSPOSE)
#   ("inst", N, LABEL)        a field of instrument N in the Instrument view
#   ("mixer-track", I)        track I (1-8) volume in the Mixer
#   ("mixer-fx", NAME)        an effect level on the Mixer's MX/DE/RE row
#   ("eq", BAND)              a band's GAIN on the main EQ page (LOW, MID, HIGH)
# The CC numbers are roomtone.config.CC; all arrive on CONTROL_MAP_CHANNEL (10).
MAPPINGS: list[tuple[int, tuple]] = [
    (20, ("project", "TEMPO")),
    (21, ("project", "TRANSPOSE")),
    (30, ("inst", 0, "CUTOFF")),
    (31, ("inst", 1, "CUTOFF")),
    (32, ("inst", 2, "CUTOFF")),
    (33, ("inst", 4, "CUTOFF")),
] + [(40 + i, ("mixer-track", i + 1)) for i in range(8)] + [
    (50, ("mixer-fx", "RE")),
    (51, ("mixer-fx", "DE")),
    (52, ("eq", "MID")),
]
