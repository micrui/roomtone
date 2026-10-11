# M8 setup for roomtone v1

This is the M8 Model:02 side of roomtone, as a checklist. Do it once, save the result as a
song, and set that song as the template so every new song starts from it. The MIDI plan
(channels, CC numbers) is the contract in `roomtone/config.py`; this file and
`tidal/roomtone.tidal` both follow it.

Sources: the official M8 Operation Manual v6.5.2 and the community guide, as summarised in
[docs/SPEC.md](../docs/SPEC.md) sections 8.1, 8.3 and 10. Key names use the manual's notation:
[SHIFT], [PLAY], [OPTION], [EDIT] and the arrows [UP] [DOWN] [LEFT] [RIGHT]. Move between
views with [SHIFT] plus an arrow (the view map is in the manual's navigation chapter). Change
a value by holding [EDIT] and pressing arrows. Field names below are as the manual and guide
give them; if your firmware words one differently, the manual's chapter for that view is the
reference.

The M8 sequencer stays stopped the whole time. Incoming MIDI notes play each track's
instrument whether or not the sequencer runs, so the Mac is the only sequencer. Do not press
[PLAY] while roomtone is running.

## What you need before starting

1. A USB-C cable with data lines (the manual notes some cables carry power only) and a
   powered hub or a direct port. The manual says the M8 needs 500 mA.
2. The Mac side running far enough to send test MIDI: `sc/startup.scd` evaluated in
   SuperCollider, and the `setcps` and `let` blocks of `tidal/roomtone.tidal` evaluated in
   your Tidal editor. Do not start d1..d10 yet; you send single messages with `once`.
3. A note of the M8's firmware version (step 1.1). The settings below are documented in the
   6.5.2 manual. Update first if yours is older.
4. The M8's built-in speaker silent whenever the mic is live: plug headphones (or nothing
   that plays) into its jack, or turn the speaker off in System Settings if your firmware
   has that option. roomtone subtracts only what the Mac plays; whatever the M8's own
   speaker adds to the room comes out the other side as "room" sound and counts against
   the mood.

## 1. System Settings

1. Open System Settings (from Project View). Write down the firmware version shown.
2. Set USB AUDIO MODE to **STEREO OUT**. The Mac receives one 24-bit stereo mix.
   (MULTICHANNEL, 12 stereo channels, is the v1.5 upgrade; leave it for now.)
3. Set USB MAIN OUT to **MIX**. The USB stereo stream then carries the same mix as the
   analog jack.
4. Leave SONG TEMPLATE for now. You set it in section 9, after the song is saved.

## 2. MIDI Settings View

1. Open MIDI Settings (from Project View).
2. Set SYNC IN to **CLOCK**. The M8 runs at 24 ticks per quarter note, the same rate as MIDI
   clock. In v1 roomtone sends tempo as CC 20 (section 5) and does not send clock by
   default, so this setting matters only if you enable the clock line in
   `roomtone.tidal`. Bench check: with SYNC IN = CLOCK and no clock arriving, confirm in
   section 5 that CC 20 still moves TEMPO. If it does not, set SYNC IN to OFF (the guide
   lists OFF as an option) and keep the CC path.
3. Set CONTROL MAP CHANNEL to **10**. Every mood CC from roomtone arrives on channel 10.
4. Leave Rec Note Channel at its default, **9**. roomtone never sends on channel 9, so no
   note is ever recorded into a phrase by accident.
5. Fill in the Track MIDI Input table exactly like this:

   | Track | Layer | CHAN | INST# | PG CHANGE | MODE |
   |---|---|---|---|---|---|
   | 1 | drone | 1 | 00 | OFF | MONO |
   | 2 | pad | 2 | 01 | OFF | MONO |
   | 3 | figure | 3 | 02 | OFF | MONO |
   | 4 | phrases | 4 | 03 | OFF | MONO |
   | 5 | shimmer | 5 | 04 | OFF | MONO |
   | 6 | pulse | 6 | 05 | OFF | MONO |
   | 7 | bass | 7 | 06 | OFF | MONO |
   | 8 | spare | 8 | 07 | OFF | MONO |

   Track N on channel N is the M8's default, so CHAN may already be right. PG CHANGE stays
   OFF in v1: when on, a program change swaps the instrument on every track using that
   channel, which is a later feature. MONO means one note at a time per track; the pad
   still gives chords because Hypersynth builds a chord from one incoming note.
6. Make sure no track is set to channel 9 or 10.

## 3. The eight instruments

Create instruments 00 to 07 (one per track, matching INST# above) and set each TYPE in
Instrument View. Types follow docs/SPEC.md section 8.1. The sound hints are starting points,
not requirements. Every instrument here has a filter; the cutoff of instruments 00, 01, 02
and 04 is driven by CC in section 5, so set the others by ear.

Envelopes matter more than the sound. Tidal holds the drone, pad and bass notes for most of
their loops (17.8, 21.3 and 37.9 s) and sends the note-off at the end, so those three need
an **ADSR ENV** on the amplitude with sustain up: the note then holds until the note-off.
An AHD envelope decays while the note is still held, and the first audition on the bench
showed every drone note gone within 10 s of its 17.8 s loop, leaving the room silent
between notes. Figure, shimmer, pulse and phrases are short by nature and an AHD or DRUM
envelope suits them.

1. **00 drone: Macrosynth** (the Braids port, over 40 models). Pick a model that moves
   slowly on a held note; SAW SWARM, WAVE MAP or HARMONICS are good first tries (the M8
   may abbreviate model names). Low-pass filter, cutoff fairly low, generous reverb send.
   Tidal holds each drone note for most of its 17.8 s loop.
2. **01 pad: Hypersynth** (6-note chord synth). Set an open chord: stacked fifths
   (0, 7, 14 semitones) are the most forgiving, since a fixed chord shape moves in
   parallel and some shapes leave the scale on some roots. Slow attack, low-pass filter,
   reverb send up.
3. **02 figure: FM**, bells. Fast attack, long decay, and one modulator ratio set to a
   non-whole number for a bell-like tone. Keep it quieter than the pad; it is the most
   noticeable layer.
4. **03 phrases: Sampler**. Load your recorded phrases (short melodic or textural clips in
   the song's key). Tidal triggers note 0 (MIDI note 60) twice per 31.7 s loop, shifted by
   transpose. Send that note once (section 4) and confirm the sample plays at its recorded
   pitch; if it is off, adjust the instrument's tuning, not the Tidal file.
5. **04 shimmer: FM**, high register. Sine-like operators, short decay, a lot of delay and
   reverb send. Tidal plays it two octaves above the figure.
6. **05 pulse: Sampler**, soft percussion: a brushed hit, a muted wood tap, a low click.
   Short. Tidal sends note 0 only; this is the one layer that keeps the bar.
7. **06 bass: Macrosynth**. A plain, round model such as SINE TRIANGLE or TRIPLE SINE,
   low-pass filtered, little or no reverb.
8. **07 spare: Sampler**. Load anything quiet for now. This track is reserved for the
   musical marker later (docs/SPEC.md section 5).

## 4. Test each track from the Mac

With the M8 sequencer stopped, evaluate one line at a time in your Tidal editor. Each should
sound exactly one track (midichan is 0-based, so channel N is `midichan (N-1)`):

```haskell
once $ note 0 # s "midi" # midichan 0   -- track 1, drone
once $ note 0 # s "midi" # midichan 1   -- track 2, pad (a chord)
-- ... and so on up to
once $ note 0 # s "midi" # midichan 7   -- track 8, spare
```

1. Confirm each of the 8 lines plays its own track and no other.
2. Velocity check: send `once $ note 0 # amp 0.2 # s "midi" # midichan 0`, then the same
   with `amp 0.9`. Write down whether track 1 got louder. If it did not, roomtone still
   controls loudness through the volume CCs in section 5; velocity changes only add detail.

## 5. MIDI learn, one CC at a time

All 17 CCs below arrive on channel 10. Each mapping is made by putting the cursor on the
parameter, holding [OPTION], and sending the CC from the Mac. Mappings are stored in the
song (up to 128 per song; this plan uses 17).

Before you start, evaluate `hush` in Tidal. If d9 or d10 were running, the M8 would receive
all 16 running CCs at once and learn whichever arrived first.

The learn procedure, repeated for every row of the table:

1. Navigate to the view and parameter in the table and put the cursor on the value.
2. Hold [OPTION].
3. In Tidal, evaluate the row's line: `once $ ccv 64 # ccn <CC> # s "midi" # midichan 9`
   (replace `<CC>` with the row's CC number).
4. Release [OPTION]. Open MIDI Mapping View and confirm a new mapping: channel 10, the
   right CC number, the right destination.
5. Set the mapping's range (MIN and MAX) as the table's last column says. The M8 shows
   MIN and MAX in the parameter's own units. CC values are 0 to 127, and slow sweeps can
   sound steppy, so a narrow range spends the 128 steps on a small span.

| CC | config.CC key | View | Parameter | Range (MIN / MAX) |
|---|---|---|---|---|
| 20 | tempo | Project View | TEMPO | Calibrate: CC 0 = 40 BPM, CC 127 = 167 BPM (see below) |
| 21 | transpose | Project View | TRANSPOSE | Calibrate: CC 64 = no transpose, one step = one semitone (see below) |
| 30 | drone_cutoff | Instrument View, instrument 00 | filter cutoff | Darkest to brightest you want for the drone, kept narrow |
| 31 | pad_cutoff | Instrument View, instrument 01 | filter cutoff | Same idea for the pad |
| 32 | figure_cutoff | Instrument View, instrument 02 | filter cutoff | Same idea for the bells |
| 33 | shimmer_cutoff | Instrument View, instrument 04 | filter cutoff | Same idea for the shimmer |
| 40 | drone_vol | Mixer View | track 1 volume | MIN quiet but audible, MAX your loudest drone |
| 41 | pad_vol | Mixer View | track 2 volume | As above |
| 42 | figure_vol | Mixer View | track 3 volume | As above |
| 43 | phrases_vol | Mixer View | track 4 volume | As above |
| 44 | shimmer_vol | Mixer View | track 5 volume | As above |
| 45 | pulse_vol | Mixer View | track 6 volume | As above |
| 46 | bass_vol | Mixer View | track 7 volume | As above |
| 47 | spare_vol | Mixer View | track 8 volume | As above |
| 50 | reverb_send | Mixer View | reverb effect level | Dry-ish to the wettest you like |
| 51 | delay_send | Mixer View | delay effect level | None to a clear echo |
| 52 | main_eq_mid_gain | EQ View (the main EQ, reached from Mixer View) | mid band gain | Symmetric around flat, so CC 64 is no change |

Notes on the table:

- Volumes never need to reach silence. A layer that is off is silenced by the gate in
  Tidal (it stops sending notes), not by its volume.
- One CC may drive several parameters (a macro). If you want the reverb to change per
  instrument instead of at the effect, map CC 50 again onto each instrument's reverb send
  in Instrument View. One parameter can only take one CC.
- Tempo calibration. With the cursor anywhere, send
  `once $ ccv 0 # ccn 20 # s "midi" # midichan 9` and read TEMPO in Project View. Adjust
  the mapping's MIN until CC 0 gives 40. Send `ccv 127` and adjust MAX until it gives 167.
  Then send `ccv 20`: TEMPO should read 60, the default `tempo_bpm`. If your M8 cannot be
  set to exactly this span, write down what CC 0 and CC 127 give and change the `40` and
  `127` in d9 of `roomtone.tidal` to the low end and the span.
- Transpose calibration, the same way: CC 64 should give TRANSPOSE at its no-change value,
  and CC 65 one semitone up. `roomtone.tidal` transposes in Tidal by default and does not
  send CC 21; the mapping is there for the alternative path described in the file. Use
  one path, never both.
- Bench check (docs/SPEC.md section 11, step 6): on CC 30, do a slow sweep by hand with a
  few `once` lines (`ccv 40`, `ccv 41`, `ccv 42` ...) and note whether the cutoff sounds
  steppy. If it does, narrow its range further.

## 6. MIDI Mapping View: review

1. Count the mappings: 17, all on channel 10.
2. Check each CC number against the table once more. A wrong number here fails silently:
   the music plays, but that parameter never follows the room.

## 7. Mixer View levels

1. Track volumes 1 to 8 are driven by CC 40 to 47 once d10 runs, so their manual position
   does not matter after that point.
2. Set the input level (analog in) and the USB input level to their minimum. v1 sends no
   audio into the M8, and this keeps anything the Mac might play from coming back
   through it.
3. Chorus is unused in v1; set its level as you like.
4. Main output: play the loudest case. In `roomtone.tidal`, temporarily change every
   `layerGate 0` to `layerGate 1` and the `energy` default from `cF 0` to `cF 1`,
   re-evaluate, and set the main level so the M8's meters never reach the top. The Mac
   records this stream; clipping here cannot be undone later. Change the defaults back
   afterwards.

## 8. Final listening check

1. Evaluate d1 to d10 in `roomtone.tidal`. With no Controller running, only the drone plays
   (its gate defaults on; the others default off) and the CCs settle at their defaults.
2. Confirm in Project View that TEMPO reads 60.
3. Confirm the drone's cutoff moves slowly over a few minutes. That is the macro cycle,
   one full sine over 64 cycles (256 s at 60 BPM).

## 9. Save as the template

1. Save the song from Project View under a name you will recognise, for example
   `ROOMTONE`.
2. Open System Settings and set SONG TEMPLATE to that song. New songs then start with these
   instruments, the Track MIDI Input table and all 17 mappings, since mappings are stored
   in the song file.
3. Write the song name, firmware version, and the calibrated tempo and transpose ranges into
   your bench notes (docs/BENCH.md).
