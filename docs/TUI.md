# roomtone TUI

`roomtone/tui.py` is a monitor for the Listener and Controller, laid out like an M8
screen: one monospaced grid on near-black, two-letter and short column headers, hex
where the M8 would use hex, small bar meters, no borders. It is read-only; it shows what
each 30 s analysis block produced and never sends anything to the M8.

## Screen

Demo mode at 100x32, block A9 (1 h 24 min into the session, dinner conversation):

```
ROOMTONE  01:24:30  BLK A9  LOCKED

LOCK                                             ROOM
DELAY       3307.2 smp  74.99ms                  RMS       -39.8dB █████████·······
DRIFT       +23.60 ppm                           SPEECH         5E ████████████····
P/SL          40.2 dB                            TALKERS       3.5 █████████·······
JIT           0.71 smp                           TRANS/s       0.7 █···············
ERLE          22.2 dB ████████████····           BEAT           02 ················
RIR CHG      0.014                               CENTROID   1810Hz ███████·········
MARKER  LEN 04s BAND 4K-10K LVL -30              SPEECH         6A █████████████···
RUNG    ■□□□□□□□□ 00/08                          CONVERSATION   46 █████████·······
OK      25  BAD 00                               LAUGHTER       1B ███·············

MOOD                                             LAYERS  MASK 07
ENERGY  3F ████████········                       TRK CH LAYER    ON VOL CUT
VOICES  67 █████████████···                         1 01 DRONE    ■  59  2B
RHYTHM  10 ██··············                         2 02 PAD      ■  59  2B
STATE   CONVERSATION                                3 03 FIGURE   ■  59  2B
CONF    4B █████████·······                      >  4 04 PHRASES  □  --  --
        0123 4567 89AB CDEF                         5 05 SHIMMER  □  --  2B
HIST    CCCC CCCC CCVV VVVV                         6 06 PULSE    □  --  --
                                                    7 07 BASS     □  --  --
                                                    8 08 SPARE    □  --  --

TEMPO 068 SCALE LYDIAN TRN -01
q quit  d demo  l log
```

The footer sits on the bottom row. Below 100 columns or 28 rows the screen goes compact:
meters shrink from 16 to 10 cells and the blank rows between panels go, so the whole
grid fits 80x24.

## Panels

| Panel | Rows |
|---|---|
| Header | title, session clock (block start time), block index in hex, lock status word, active sources (`DEMO`, `OSC <port>`) |
| LOCK | `DELAY` samples and ms, `DRIFT` ppm, `P/SL` peak-to-sidelobe dB, `JIT` marker timing jitter in samples, `ERLE` echo suppression dB with a meter scaled to 1.5 times the 20 dB target (the `\|` tick marks the target until the bar passes it), `RIR CHG` change between successive room response estimates, `MARKER` parameters of the current ladder rung, `RUNG` the ladder as nine cells (tracking default to ceiling), `OK`/`BAD` consecutive block counters in hex |
| ROOM | `RMS` dB (meter spans -90..0), `SPEECH` 00..7F, `TALKERS` (meter spans 0..6), `TRANS/s` onsets per second (0..8), `BEAT` 00..7F, `CENTROID` speech-band centroid Hz (0..4000), then the top three YAMNet labels with 00..7F scores when the Listener sends them |
| MOOD | `ENERGY` `VOICES` `RHYTHM` as 00..7F with meters, `STATE` the named mood, `CONF` confidence, `HIST` the last 16 blocks as letters under a 0..F step ruler, oldest left |
| LAYERS | the eight layers in `config.LAYER_ORDER`: `TRK` M8 track, `CH` MIDI channel, `ON` from the layers bitmask (title shows the mask as two hex digits), `VOL` 00..7F, `CUT` 00..7F, `--` where a layer is off or has no cutoff CC |
| Footer | `TEMPO` BPM, `SCALE`, `TRN` transpose in semitones, key help |

History letters: `E` empty, `C` cooking, `V` conversation (voices), `L` loud-busy.

`VOL` and `CUT` are a display model, not the Controller's exact CC output: an active
layer shows `40 % + 60 % x energy`, and `CUT` shows `bright` for layers that have a
`*_cutoff` entry in `config.CC` (drone, pad, figure, shimmer). The `>` row is a cosmetic
playhead that steps one track per second.

Number formats: unit-interval quantities (speech, beat, mood axes, confidence, label
scores, CC values) are hex 00..7F, as the M8 shows CC values; the layer mask and block
counters are hex; dB, ppm, Hz, samples, ms and BPM stay decimal.

## Colour

| Name | Default | Meaning |
|---|---|---|
| `bg` | `#000000` | background |
| `fg` | `#f0f0f0` | values |
| `dim` | `#5c5c5c` | labels, units, empty meter cells, `--` |
| `title` | `#00d2d2` | panel titles, column headers, mood state, YAMNet labels, the ERLE target tick |
| `select` | `#ff2e9a` | playhead row, newest history step, key letters, `LOST` |
| `ok` | `#2fe36a` | `LOCKED`, criteria met (JIT under 2 samples, ERLE at or over 20 dB), active layer cells, the low part of a level meter |
| `warn` | `#ffd21e` | `ACQUIRING`, ERLE under target, lit ladder rungs, the upper part of a level meter |
| `hot` | `#ff3434` | `SATURATED`, JIT over limit, nonzero `BAD`, the ladder at its ceiling, the top of a level meter |

Lock status words: `ACQUIRING` yellow, `LOCKED` green, `LOST` magenta, `SATURATED` red.

Level meters colour each lit cell by position (green to 60 %, yellow to 85 %, red above),
like the M8 mixer. Palettes live in `THEMES` in `roomtone/tui.py` and reach the Textual
CSS as `$rt-<name>` variables; `--theme amber` selects the second built-in palette, and a
new M8 theme is one more dict with the same eight keys.

## Keys

| Key | Action |
|---|---|
| `q` | quit |
| `d` | start or stop the demo generator (alongside any OSC feed) |
| `l` | swap the MOOD and LAYERS row for an event log: lock status changes, mood changes, marker rung changes, demo on and off |

## Running

```
python3 -m roomtone.tui --demo              # synthetic session, one block per 0.5 s
python3 -m roomtone.tui                     # wait for the Listener on OSC port 6020
python3 -m roomtone.tui --port 6021 --theme amber
```

From code: `roomtone.tui.run(cfg, demo=True)`.

Live data arrives as OSC `/roomtone/block` on `cfg.tui_host:cfg.tui_port` with one string
argument, the JSON of `BlockResult.as_dict()`. In-process callers use
`RoomtoneApp.feed(result)`, which is safe from any thread.

Demo mode replays a two-hour evening (240 blocks) in two minutes, then loops: five
blocks of acquisition on the long marker, lock and the ladder relaxing to the tracking
default, an empty hour, cooking (with one jump in the room response that drops the lock
to `LOST` for three blocks), dinner conversation, a loud stretch that climbs the marker
ladder rung by rung to `SATURATED`, and back to empty.
