# Day-one bench procedure

This is [SPEC.md section 11](SPEC.md#11-day-one-bench-checks-no-music-yet) expanded into
commands, settings and pass criteria. It ends milestone 1: the bench checks pass and the
two open decisions are written down (aggregate device or separate mic, section 4.2; how
steppy CC sweeps are, risk 3).

Every `roomtone` command below takes `--help`, which lists its options. Run each step in
order. Write down what each step asks for in the bench sheet at the end before moving on;
later steps and later sessions depend on those numbers.

## 0. Before you start

- Hardware from SPEC section 2: MacBook, M8 Model:02, USB measurement mic, 3.5 mm TRS cable
  from the Mac's headphone jack to the Five's line-in, Sonos Five, powered USB-C hub.
- The M8 cable must carry data (the manual notes some cables are power only), and the M8
  needs 500 mA, so use a powered hub or a direct port.
- Software installed per the [README](../README.md): `pip install -e '.[audio,dev]'`,
  SuperCollider with SuperDirt, Tidal 1.9 and an editor.
- The kitchen quiet for steps 5 and 7: nobody cooking or talking, no dishwasher.

## 1. The M8 shows up

1. Plug the M8 into the Mac.
2. Run:

   ```
   roomtone devices
   ```

   It lists the audio devices (with channel counts and sample rates) and the MIDI ports.
3. Open Audio MIDI Setup (Applications > Utilities), Window > Show Audio Devices, and
   select the M8. Read its Format field.
4. On the M8, open System Settings and read the firmware version.

Write down: the M8's audio device name, sample rate, bit depth, input channel count, MIDI
port name, firmware version.

Pass: the M8 appears both as an audio input and as a MIDI port. The manual gives 24-bit
stereo for STEREO OUT mode. The sample rate is not in the manual; 44.1 kHz is expected
because the M8 renders at 44.1 kHz (SPEC section 10, "still not confirmed"). What you read
here settles it. If the MIDI port is not called `M8`, change `midi_out_port` in
`roomtone/config.py` and the name in `sc/startup.scd` to match. If the firmware is older
than 6.5, update it (directly, not through a hub, per the manual's troubleshooting notes).

## 2. The mic shows up

1. Plug in the measurement mic and run `roomtone devices` again.
2. In Audio MIDI Setup, select the mic and open its Format menu: it lists every sample rate
   the mic supports. Check the mic's spec sheet as well; some measurement mics (the
   miniDSP UMIK-1 for one) are fixed at 48 kHz.

Write down: the mic's device name and every rate it supports.

Decision (SPEC section 4.2), write it down:

- The mic supports the M8's rate: put both in one aggregate device (step 3 as written).
- It does not: leave the mic out of the aggregate. Set `mic_input_device` in
  `roomtone/config.py` to the mic's name and let the marker lock absorb the drift, which it
  must do for the Sonos anyway.

## 3. The aggregate device, and a 10-minute run

### 3a. Cable the jack first

Plug the 3.5 mm cable into the Mac's headphone jack now, before building the aggregate.
On recent MacBooks the headphone jack can show up as its own output device only while
something is plugged in, and the aggregate has to contain the device that actually feeds
the Five. The Five's settings come in step 4; it can sit silent until then.

### 3b. Build "roomtone" in Audio MIDI Setup

1. Open Audio MIDI Setup, Window > Show Audio Devices.
2. Set the M8, the mic (if it is joining) and the built-in output to the same sample rate in
   each device's Format menu. An aggregate device runs at one rate.
3. Click **+** at the bottom left and choose **Create Aggregate Device**.
4. Double-click the new device's name and rename it **roomtone**. This exact name is
   `aggregate_device` in `roomtone/config.py`.
5. In the device list on the right, tick **Use** for the M8, then the mic, then the built-in
   output (the headphone output). The order of the ticks sets the channel order.
6. Set **Clock Source** to the built-in output. It is the reference clock: the reference
   signal is whatever roomtone writes to this device (SPEC section 4.3).
7. Tick **Drift Correction** for the M8 and for the mic. Leave it off for the built-in
   output, since it is the clock source.
8. Check the aggregate's own sample rate matches the members.
9. Keep the macOS default output (System Settings > Sound) on another device, so
   SuperCollider, which boots on the default device, never opens "roomtone".
10. Run `roomtone devices` and find "roomtone" in the list.

Write down: the aggregate's sample rate, its input channel count, which input channels are
the M8 and which are the mic.

### 3c. The 10-minute run

1. Start SuperCollider (`sc/startup.scd`) and, in Tidal, evaluate the `setcps` and `let`
   blocks and d1 of `tidal/roomtone.tidal`. With no Controller running only the drone
   plays, which gives the M8 input a steady signal.
2. Run:

   ```
   roomtone bench latency
   ```

   It plays a click out of the aggregate's output at a fixed period, finds each click in the
   mic input, and prints the delay. Let it run for 10 minutes. The click goes out of the
   headphone jack to the Five, so if the Five is not yet playing its line-in, do step 4a
   first. This step is about the clocks, not the room, so the level does not matter yet.
3. Watch for dropouts on both inputs: gaps, clicks, or reported over- or underruns.

Write down: the delay at the start and at 10 minutes (in samples), the count of dropouts on
each input.

Pass: no dropouts on either input over 10 minutes, and the delay changes along a straight
line (constant drift). Drift itself is not a failure; the markers measure it (SPEC section
6.2 expects tens of ppm for a device pair, constant during a session). To turn the change
into ppm: (delay at 10 min minus delay at start, in samples) divided by the samples in
10 minutes (26,460,000 at 44.1 kHz), times 1,000,000. If the aggregate glitches with drift
correction on (risk 4), fall back to separate devices: set `aggregate_device` to `None` and
name the three devices in `roomtone/config.py`.

## 4. The Sonos Five, and the round trip

### 4a. Five settings (SPEC section 9)

In the Sonos app, open the Five's room settings and find Line-In:

1. **Audio Compression: Uncompressed.** Community reports put the line-in delay at 75 ms in
   this mode against about 2 s in Compressed.
2. **Autoplay: on**, with the Five as the autoplay room, so the line-in plays whenever there
   is signal.
3. **Source level:** start in the middle. You set it properly in 4b.
4. The Five is the group host. If other Sonos rooms join its line-in later, they add their
   own larger delay; the markers measure that as part of the room.

Set the Mac's output volume for the headphone jack to a fixed value and leave it there for
every later session. Write it down.

### 4b. Level and round trip

1. With the drone playing (step 3c), raise the Five's source level and the Mac's volume until
   the music sits well above the room's noise floor without clipping. If the Mac's jack
   cannot get loud enough even at full volume, that is risk 5: switch to a small USB DAC.
2. Run `roomtone bench latency` again, mic in its kitchen position.

Write down: Five source level, Mac output volume, the delay in ms and in samples, the
distance from the Five to the mic.

Pass: the delay is roughly 75 ms plus the acoustic path (the Five-to-mic distance divided by
the speed of sound). Much more than that usually means Compressed mode is still on.

## 5. Marker audibility, with ears

This is a listening test, not a calculation (SPEC section 5).

1. Keep only a quiet pad playing: in Tidal, evaluate d2 with its gate default changed from
   `layerGate 0 "pad"` to `layerGate 1 "pad"`, and stop d1 with `d1 $ silence`.
2. Run:

   ```
   roomtone bench marker
   ```

   It plays the marker at the tracking settings: a 4 s maximum length sequence every 30 s,
   band 4 to 10 kHz, 30 dB below the music's RMS (`DEFAULT_MARKER` in
   `roomtone/config.py`). Stand where people stand in the kitchen and listen through
   several periods.
3. Evaluate `hush` in Tidal and listen again with the marker alone.
4. Step the marker level up (the options are in `roomtone bench marker --help`) until you
   can hear it in the empty kitchen.

Write down: heard or not under the pad at tracking level; heard or not alone at tracking
level; the lowest level you can hear in the empty kitchen.

Pass: inaudible under the quiet pad at tracking level (the target in SPEC section 5). Hearing
it alone in a silent kitchen is expected and is not a failure. If it is audible under the
pad, write it down: that result moves the marker's starting level and is risk 2.

## 6. MIDI to the M8

Do sections 2, 4 and 5 of [m8/SETUP.md](../m8/SETUP.md) first: Track MIDI Input table,
MONO, SYNC IN = CLOCK, CONTROL MAP CHANNEL = 10, and the MIDI learn for every CC. Then:

1. Run:

   ```
   roomtone bench midi
   ```

   It sends a test note on each channel 1 to 8 and test CCs on channel 10, through the MIDI
   port named in `roomtone/config.py`. Stop SuperCollider and Tidal's d1..d10 first if you
   want only this command on the port.
2. Confirm, with the M8 sequencer stopped:
   - each channel 1 to 8 plays its own track, and only that track;
   - CC 30 moves instrument 00's filter cutoff; a slow sweep sounds smooth or steppy (write
     down which);
   - CC 40 moves track 1's volume in Mixer View;
   - CC 20 moves TEMPO in Project View, and CC 20 at value 20 reads 60 BPM after the
     calibration in m8/SETUP.md;
   - whether note velocity changes loudness (m8/SETUP.md section 4, step 2).
3. Clock check (SPEC section 11, step 6). In Tidal:

   ```haskell
   d12 $ midicmd "midiClock*96" # s "midi"
   ```

   Set an M8 LFO to a tempo-synced rate and confirm it follows. Change `setcps` and confirm
   it follows again. Then `d12 $ silence`, and confirm CC 20 still sets TEMPO with no clock
   arriving. If it does not, set SYNC IN to OFF (m8/SETUP.md section 2).
4. In System Settings: USB AUDIO MODE = STEREO OUT and USB MAIN OUT = MIX (m8/SETUP.md
   section 1).

Write down: smooth or steppy on CC 30 (the risk 3 decision), velocity yes or no, clock
follow yes or no, TEMPO by CC with clock absent yes or no.

Pass: all eight tracks answer on their own channel with the sequencer stopped, and CCs 30,
40 and 20 each move their parameter.

## 7. Empty-house baseline

1. Quiet kitchen, nobody home or everyone out of the room, appliances off where you can.
2. Start the Tidal session as in step 3c (drone only).
3. Run:

   ```
   roomtone baseline
   ```

   It measures the room response from the output to the mic (SPEC section 6.3; starting
   tail 200 ms) and stores it as the starting point for every session.
4. Run it a second time without moving anything.

Write down: where the baseline was saved, the delay and echo suppression the command
reports, and whether the two runs agree.

Pass: the two runs agree (the "response stability" criterion in SPEC section 6.4). If they
do not, something in the room changed between them, or the level is too low; raise the
Five's source level a little and repeat.

## 8. The control path without the room: `roomtone run --sim`

1. SuperCollider running, all blocks of `tidal/roomtone.tidal` evaluated (d1 to d10).
2. Run:

   ```
   roomtone run --sim
   ```

   This runs the Controller on simulated room readings and publishes the control values to
   Tidal's `/ctrl` listener on port 6010, with no mic involved.
3. Listen and watch the M8: layers enter and leave, volumes and cutoffs move, and TEMPO in
   Project View follows `tempo_bpm` whenever the simulation changes it.

Pass: every layer can be heard to come in and go out at least once, and no layer plays when
its `layer_*` value is 0. If nothing changes, Tidal is not receiving `/ctrl`: check that
nothing else holds port 6010.

## 9. First live run: `roomtone run --tui`

```
roomtone run --tui
```

The terminal UI shows each 30 s block: lock state (acquiring, locked, lost, saturated),
delay, drift, echo suppression, the mood vector and named state, and the control values
sent to Tidal. Let it run for at least 10 blocks (5 minutes, the start of the mood window).

Write down: how many blocks until the first "locked", the delay and drift it reports
(compare with steps 3 and 4), the echo suppression.

The lock targets (SPEC section 6.4) belong to milestone 2, not day one, but note how close
the first run gets:

| Metric | Target |
|---|---|
| Marker peak-to-sidelobe ratio | High and stable; timing jitter under 2 samples |
| Echo suppression | Plateaued, 20 dB or more |
| Response stability | Consecutive estimates agree |

## Bench sheet

Copy this into your notes and fill it in.

| Step | Item | Value |
|---|---|---|
| 1 | M8 firmware version | |
| 1 | M8 audio device name, rate, bit depth, input channels | |
| 1 | M8 MIDI port name | |
| 2 | Mic name, supported rates | |
| 2 | Decision: mic in aggregate, or separate | |
| 3 | Aggregate rate, channel map (M8 inputs, mic inputs) | |
| 3 | 10-min run: delay at start, at 10 min (samples), drift (ppm), dropouts | |
| 4 | Five: compression, autoplay, source level | |
| 4 | Mac output volume | |
| 4 | Round-trip delay (ms, samples), Five-to-mic distance | |
| 5 | Marker at tracking level: heard under pad? heard alone? | |
| 5 | Lowest audible marker level, empty kitchen | |
| 6 | All 8 tracks on own channel (yes/no) | |
| 6 | CC 30 sweep: smooth or steppy | |
| 6 | Velocity changes loudness (yes/no) | |
| 6 | Clock follow (yes/no); TEMPO by CC with no clock (yes/no) | |
| 6 | M8 song name and calibrated tempo/transpose ranges (m8/SETUP.md) | |
| 7 | Baseline file, two runs agree (yes/no) | |
| 8 | `--sim`: all layers in and out (yes/no) | |
| 9 | Blocks to first lock; delay; drift; echo suppression | |
