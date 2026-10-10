# Room-responsive ambient music: v1 architecture (MacBook + M8 + Sonos Five)

Status: architecture and vetting only. Nothing built. Dated 2026-10-10.
Revision 2: M8 MIDI and USB audio facts vetted against the community M8 guide
("Interacting with other devices" chapter), section 10 updated accordingly.
Revision 3: vetted against the official M8 Operation Manual v6.5.2 (2026-04-21):
USB spec, bit depth, channel counts, MIDI mapping scope, sync options, track polyphony.
Revision 4: built. Mixer moved into the Python process (8.3); analysis blocks are
aligned to marker bursts; see README.md and docs/BENCH.md for the running system.

## 1. Goal and scope

A system that listens to the kitchen every few minutes, estimates the room's mood from
acoustic features (not words), and steers live-generated ambient music to match. The
music is played by a Dirtywave M8 Model:02, composed in real time by a MacBook, and
delivered to the house through a Sonos Five line-in.

In v1:

- One room (kitchen), one mic, one Sonos Five as the audio host.
- Mood range: empty / calm, cooking, conversation, loud-busy.
- Marker-based lock on the room (delay, clock drift, room response) so the system can
  subtract its own music from what the mic hears.
- Live-coding surface so a human can jam alongside the room controller.

Deferred past v1:

- Party-mode handoff to a Sonos playlist.
- Generative drums and bass as a real dance layer.
- HomePods and AirPlay delivery.
- Editing the M8's own patterns over its serial protocol (m8c-style key injection).
- Multi-room sensing.

## 2. Bill of materials

| Item | Role | Notes |
|---|---|---|
| MacBook (macOS, current) | Composer, mixer, listener, controller | Needs 3 USB ports via hub: M8, mic, optional DAC |
| Dirtywave M8 Model:02 | Sound engine | USB-C: MIDI both ways plus USB audio to the Mac |
| USB measurement mic | Room sensor | Flat response, class compliant. Check its supported sample rates (see 4.2) |
| 3.5 mm TRS cable | Mac headphone jack to Five line-in | Or a small USB DAC if the jack's level is marginal |
| Sonos Five | Room playback, group host | Line-in set to Uncompressed |
| USB-C hub | Ports | Powered. The manual says the M8 needs 500 mA; a bus-powered hub shared with a mic is marginal. Use a cable with data lines (the manual notes some cables are power only) |

No cloud services. No accounts beyond the Sonos app.

## 3. Signal flow

```
                 MIDI notes, CC, clock (USB)
   +---------------------------------------------->  M8 Model:02
   |                                                   |
   |  Mac                                              | stereo audio (USB capture)
   |  +-------------------------------------------+    |
   |  | Composer (TidalCycles / SuperCollider)    |<---+
   |  |   patterns read mood values over OSC      |
   |  +-------------------------------------------+
   |  | Mixer + marker (SuperCollider)            |----> headphone jack ----> Sonos Five line-in
   |  |   M8 in + marker -> out; logs reference   |                              |
   |  +-------------------------------------------+                              v
   |  | Listener (Python)                         |<--- USB mic <---------- room (+ Five)
   |  |   lock, subtract, classify, mood state    |
   |  +-------------------------------------------+
   |  | Controller                                |----> OSC /ctrl to Composer
   |  |   hysteresis, marker escalation, logging  |----> marker params to Mixer
   |  +-------------------------------------------+
   +--------------------------------------------------
```

Four processes on the Mac, loosely coupled over OSC on localhost, each restartable alone.

## 4. Clock domains and alignment

This is the part most likely to bite, so it gets its own section.

### 4.1 The clocks

1. M8 USB audio clock (capture into the Mac).
2. Mic USB clock (capture into the Mac).
3. Mac output DAC clock (the headphone jack). This is the reference domain, since the
   reference signal is whatever the mixer writes to this device.
4. Sonos Five internal clock, downstream of the analog line-in, invisible to the Mac.

### 4.2 Policy

- Build a macOS Aggregate Device containing M8 input, mic input, and the built-in output.
  Apple's guidance for devices without a shared hardware clock: pick one clock source,
  enable Drift Correction on every other device, and run all members at the same sample
  rate. Clock source: built-in output. Drift correction on: M8 and mic.
- Sample rate conflict to check on day one. The manual does not state the USB sample
  rate, but the M8's own renders are 16-bit 44.1 kHz and the engine is widely reported at
  44.1 kHz, so expect 44.1 kHz over USB at 24-bit. Some measurement mics (the miniDSP
  UMIK-1 for one) are fixed at 48 kHz. An aggregate device
  requires one rate. Options, in order: a mic that supports 44.1 kHz; or leave the mic
  out of the aggregate, open it as a separate device, and let the marker lock absorb the
  drift (it must handle Sonos drift anyway). Either works. Decide after reading the mic
  spec sheet.
- Drift between the Mac output and what the mic hears is measured by the markers, not
  assumed. Community reports put the Five's line-in buffer at 75 ms in Uncompressed mode,
  about 2 s in Compressed. Treat 75 ms as the expected value and measure it.

### 4.3 What the reference is

The reference is the exact buffer the mixer sends to the output device, after the marker
is added, written to a ring buffer with sample indices. Not the M8's capture stream. This
keeps the reference in the output clock domain and includes the marker.

## 5. Marker specification

Purpose: make alignment and room-response estimation easy regardless of the music.

- Signal: a maximum length sequence (pseudo-random noise with a sharp autocorrelation),
  band-shaped by a filter. An exponential sine sweep is the alternative; start with MLS.
- Default (tracking) parameters: length 4 s, period 30 s, band 4 to 10 kHz, level
  about 30 dB below the music's RMS. Target: inaudible under music, still a clean
  correlation peak thanks to processing gain (roughly 10 log10 of the sequence length in
  samples, about 52 dB for 4 s at 44.1 kHz).
- Acquisition parameters: length up to 30 s, period 30 s, level up to 15 dB below music,
  used until lock criteria (section 6.4) are met.
- Escalation ladder when lock is failing, applied in this order: longer sequence, then
  move band into a quieter part of the current room spectrum, then raise level. Each
  step requires 3 consecutive failed blocks; each step down requires 3 consecutive
  successes. The ladder tops out at audible, and failing at the top is itself a reading
  (loud-busy).
- Musical variant (later): envelope-shaped noise bursts or chime-spectrum sequences so
  an audible marker reads as a motif rather than a test signal.
- Audibility test is a day-one bench check with real ears, not a calculation.

## 6. Lock and subtraction (Listener, offline per block)

### 6.1 Block structure

Analysis block: 30 s, one marker per block. Mood window: 5 to 10 min (10 to 20 blocks),
re-evaluated every block. Nothing waits for the full window.

### 6.2 Alignment

1. Cross-correlate the known marker against the mic block. Peak position gives delay.
2. Successive marker peaks give the clock rate ratio (drift). Expect tens of ppm, constant
   for a given device pair during a session.
3. Resample the reference to the mic clock using the measured ratio.
4. Re-acquire after any stream restart, since the Sonos buffer delay re-settles.

### 6.3 Room response and residual

- Estimate the room impulse response by regularized frequency-domain least squares over
  the aligned block (reference in, mic out). Tail length: start at 200 ms (about 8800
  taps at 44.1 kHz), extend if the kitchen's reverberation needs it.
- Residual = mic minus (reference convolved with estimate). This is "the room."
- Carry the previous estimate into the next block as the starting point. Store a quiet
  empty-house baseline response and start every session from it.

### 6.4 Lock criteria (all three)

| Metric | Pass |
|---|---|
| Marker peak-to-sidelobe ratio | High and stable; timing jitter under 2 samples |
| Echo suppression (mic energy before vs after subtraction, speech-free stretch) | Plateaued, target 20 dB or more |
| Response stability | Consecutive estimates agree; a jump means the room changed, keep old estimate and re-estimate |

Lock state (acquiring / locked / lost / saturated) is logged next to every mood reading.

## 7. Room classification (Listener)

Inputs: the residual signal per block, plus lock state.

Features:

- Loudness (RMS, slow envelope).
- Speech presence and approximate number of talkers (pretrained audio event model such
  as YAMNet; labels like speech, crowd, dishes, silence). Runs locally on the Mac.
- Transient rate (clatter, chopping, doors).
- Beat presence (any periodic energy at tempo rates not coming from our own output).
- Spectral shape of the residual, used later to carve the music away from the speech band.

Mood representation: a small continuous vector (energy, voices, rhythm), plus a named
state derived from it for logging and for coarse layer decisions:

| State | Typical evidence |
|---|---|
| empty | low energy, no speech, no transients |
| cooking | transients, little speech, moderate energy |
| conversation | speech present, moderate energy |
| loud-busy | high energy, lock saturated or failing |

Hysteresis: a state change requires 3 consecutive blocks of agreement. All continuous
parameters slew over 30 to 120 s. No audio is stored; features only.

## 8. Music engine (M8 + Composer)

### 8.1 M8 track plan

| Track | Layer | Instrument type | MIDI ch |
|---|---|---|---|
| 1 | drone | Macrosynth (Braids port) | 1 |
| 2 | pad | Hypersynth (6-note chord synth, one note in gives a chord) | 2 |
| 3 | slow figure | FM (bells) | 3 |
| 4 | phrases | Sampler, recorded phrases on incommensurate loops | 4 |
| 5 | shimmer | FM or Macrosynth, high register | 5 |
| 6 | pulse | Sampler, soft percussion | 6 |
| 7 | bass | Macrosynth | 7 |
| 8 | spare / marker (later) | Sampler | 8 |

Every M8 track is monophonic (8 tracks, 8 voices). Chords come from Hypersynth on one
track, or from POLY mode spreading overlapping notes across several tracks on one channel.
The layer plan above assumes one voice per layer except the pad.

M8-side configuration, from the guide and the manual:

- MIDI Settings View, Track MIDI Input table: one MIDI channel and one instrument per
  track. The default is track N on channel N, which matches the table above. Incoming
  notes play that track's instrument whether or not the M8 sequencer is running, so the
  M8 sequencer stays stopped and the Mac is the only sequencer.
- MODE per that table: MONO for the layers above. POLY spreads overlapping notes across
  several tracks on one channel; useful later for a chordal pad spanning spare tracks.
- Keep channel 9 free. It is the default Rec Note Channel and would insert notes into
  phrases if the cursor is in a note column.
- SYNC IN set to CLOCK (options: CLOCK, TRANSPORT, CLOCK+TRANSP, TRANSPORT+SPP,
  CLK+TRANSP+SPP). The M8 runs at 24 ticks per quarter note, the same rate as MIDI clock.
  Since the M8 sequencer stays stopped in v1, clock matters only for tempo-synced
  modulators (LFOs, envelopes) and tempo-based delay times.
- Song TEMPO and TRANSPOSE in Project View are themselves MIDI-mappable (OPTION plus an
  incoming CC). That gives the Controller a direct tempo knob and a global key-drift knob
  without touching note data.
- CONTROL MAP CHANNEL in MIDI Settings selects which incoming channel carries mapping
  CCs (or ALL). Use a dedicated channel, say 10, for all mood CCs, separate from the
  note channels 1 to 8 and from Rec Note Channel 9.
- Mood-carrying parameters (filter cutoff, timbre, reverb send, delay send, volume) are
  reached by MIDI learn: hold OPTION on the parameter, send the CC from the Mac. Mappings
  live in MIDI Mapping View, 128 per song, each editable for channel, CC number, and
  value range. One CC may drive several parameters (a macro), one parameter takes one CC.
  The manual lists the gesture in Instrument, Mixer, EQ, and Project views, so per-track
  mixer volume, effect send levels, the main EQ, tempo and transpose are all reachable.
  Mappings are stored in the song file; set SONG TEMPLATE in System Settings so new
  songs inherit them.
- Program change, if enabled in MIDI Settings, swaps the instrument on every track using
  that channel. That gives a cheap, bar-aligned timbre change per layer for mood shifts
  that a CC sweep would not reach.
- CC values are 0 to 127 and the guide notes that sweeps can sound steppy on pitch and
  sometimes on filter cutoff. Mitigations: narrow the mapped range in MIDI Mappings View
  so 128 steps cover a small span, slew slowly, and keep fast or wide sweeps for Mac-side
  processing (see 8.3).

### 8.2 Composer

TidalCycles on SuperCollider, sending MIDI to the M8 (SuperDirt MIDI). Each layer is one
pattern, gated and shaped by control values the Controller pushes over OSC to Tidal's
`/ctrl` listener (default localhost:6010), read in patterns with `cF` and friends.

Controls the Controller publishes:

| Key | Range | Meaning |
|---|---|---|
| energy | 0..1 | overall level and density |
| voices | 0..1 | speech present; lowers volume, moves notes out of the speech band |
| rhythm | 0..1 | pulse presence and tempo scaling |
| layers | bitmask | which tracks are audible |
| bright | 0..1 | filter cutoff and timbre |
| scale | enum | pentatonic / open / minor etc |

Structure rules, all of which the patterns enforce without the Controller knowing:

- One tonal center at a time; slow key drift every 20 min or so, independent of mood.
- Loop lengths long and incommensurate (17.8, 21.3, 25.1 s and so on).
- Layer changes land on bar boundaries; parameter changes slew.
- A slow macro cycle so the piece breathes when the room does not change.

The human live-codes in the same Tidal session. The room controller only touches the
control values, so a human edit and a room edit never collide.

### 8.3 Mixer

Revision 4 (build): the mixer lives in the Python process, not in SuperCollider. The
`roomtone` audio callback reads the M8 pair and the mic from the aggregate device, adds
the marker, writes the output, and stamps the reference and the mic into ring buffers
with one shared frame counter. That keeps the reference and the mic in one clock domain
in one process, which is what the lock needs. SuperCollider only runs SuperDirt for
Tidal's MIDI output and needs no audio device of its own.

USB audio, per the manual (USB Audio Class 1 and USB MIDI compliant, 24-bit):

- USB AUDIO MODE = STEREO OUT with USB MAIN OUT = MIX: the Mac receives the same stereo
  mix the analog jack carries, 24-bit 2 channel. Start here. Nothing to route.
- USB AUDIO MODE = MULTICHANNEL: 24-bit, 12 stereo channels: all 8 tracks and the
  effects individually. The Mac then has every layer as its own stem and can do per-layer
  volume, speech-band carving, and effects itself. This is the v1.5 upgrade once the
  loop works. It also makes the CC steppiness issue moot for anything the Mac can do on
  the stems.
- Other USB MAIN OUT options exist (PRE:TRACK/EFFECT, LINE IN, MIX AND MONITOR, POST:MIX)
  and are not needed in v1.

The M8 also accepts 24-bit stereo USB audio in, with a mixer level under "USB" and
routing to the three effect sends. Not used in v1, but it means a later variant could
send the marker into the M8 and run M8 analog out to the Five, if the Mac ever leaves the
audio path.

The Model:02 has a built-in MEMS microphone. It is the M8's sampling mic, and the manual
does not say it can be routed to USB out, so it does not replace the measurement mic.

## 9. Sonos Five settings

- Line-in audio compression: Uncompressed.
- Autoplay line-in: on.
- Source level: set so the Mac's mix sits well above the noise floor without clipping.
- Five is the group host; other Sonos rooms join its line-in with shared, larger delay,
  which the markers measure as part of the combined room response.

## 10. Vetting: what is confirmed, what is not

Confirmed from the official manual (v6.5.2):

- USB Audio Class 1 and USB MIDI compliant. Audio in: 24-bit stereo. Audio out: 24-bit
  stereo, or 24-bit 12 stereo channels in multichannel mode. Works with any host that
  supplies 500 mA.
- 8 monophonic tracks; Hypersynth is a 6-note chord synth; Macrosynth is the Braids port
  with over 40 models; FM, Wavsynth, Sampler as assumed.
- MIDI mapping: up to 128 per song, one mapping per destination parameter, one control to
  many parameters, with editable range; the assign gesture exists in Instrument, Mixer,
  EQ, and Project views (tempo and transpose included). CONTROL MAP CHANNEL selects the
  incoming channel for mapping CCs.
- Track MIDI Input: CHAN, INST#, PG CHANGE, MODE (MONO / LEGATO / POLY) per track; the
  manual describes this as the M8 acting "like a sound module" under an external sequencer.
- SYNC IN options including CLOCK and transport; USB MIDI behaves the same as TRS MIDI.
- Renders are 16-bit 44.1 kHz, which implies the engine rate.
- Troubleshooting notes: use a cable with data lines; avoid hubs when flashing firmware.

Confirmed from the community guide:

- The M8 is a USB peripheral (not a host) carrying both audio and MIDI over its USB-C
  port. The Mac is the host, so the topology in section 3 is valid.
- Per-track MIDI input (channel and instrument per track, default track N = channel N),
  MONO / LEGATO / POLY modes, external notes playing internal instruments with the
  sequencer stopped: all stated in the community guide's MIDI chapter.
- MIDI learn to any parameter, 128 mappings per project, editable channel, CC, and range;
  one CC to many parameters allowed. Stated.
- SYNC IN options OFF / CLOCK / TRANSPORT / CLOCK+TRANSPORT; M8 tick rate equals MIDI
  clock rate. Stated.
- Program change can swap the instrument for tracks on a channel. Stated.
- USB audio out: MIX mirrors the analog out; firmware 6.0 adds single-track and
  effect-send routing; firmware 6.5 adds a 24-channel MULTICHANNEL mode. USB audio in
  (stereo) is mixable with sends. Stated.
- The M8 presents to a computer as a USB audio capture device named "M8" (the m8c client
  routes it, and its audio guide shows it enumerating on Linux and macOS). m8c talks to
  the M8 over USB serial (SLIP framing) and can send key presses; m8.run does the same
  over WebUSB in Chromium browsers. Relevant only for the deferred pattern-editing path.
- Dirtywave's published spec (shown for the original model) lists USB MIDI and audio as
  class compliant; a community guide states the Model:02 shares the original's
  capabilities apart from physical differences. Treat Model:02 class compliance as very
  likely, confirm by plugging in.
- Sonos community reports consistently give 75 ms as the line-in floor in Uncompressed
  mode and about 2 s in Compressed, with no user option to go lower.
- Apple's aggregate-device guidance (one clock source, drift correction on the rest,
  matched sample rates) is official.
- Tidal's external control path (`/ctrl` OSC on port 6010, `cF` in patterns) is
  documented; MIDI output goes through SuperDirt.

Still not confirmed:

- The USB sample rate. The manual states 24-bit and channel counts but not the rate;
  44.1 kHz is strongly implied by the render format. Read it off Audio MIDI Setup on day
  one; it decides the aggregate-device question in 4.2.
- Installed firmware version on your M8. MULTICHANNEL and the USB MAIN OUT options are
  documented in the 6.5.2 manual; update if older.
- Whether the Model:02's built-in mic can reach USB out. Not needed; the measurement mic
  is the sensor either way.

Risks, ranked:

1. Sample-rate mismatch between M8 and mic in one aggregate device (section 4.2).
2. Marker audibility at the level needed for lock in a lively kitchen; mitigated by the
   escalation ladder and the "saturated is a reading" rule.
3. CC steppiness on slow filter and timbre sweeps (128 steps, noted in the guide).
   Mitigations: narrow mapped ranges, slow slew, program change for larger timbre moves,
   and MULTICHANNEL stems so wide sweeps happen on the Mac instead.
4. Aggregate device glitches on some Macs with drift correction enabled; fallback is
   separate devices and software resampling.
5. Mac headphone jack level too low for the Five's line-in; fallback is a USB DAC.

## 11. Day-one bench checks (no music yet)

1. Plug in the M8. Confirm it appears as a USB audio input and a MIDI device in Audio
   MIDI Setup. Note sample rate and bit depth.
2. Plug in the mic. Note its supported rates. Decide aggregate vs separate (4.2).
3. Build the aggregate device; record 10 min from both inputs while playing a test tone
   out; check for dropouts and measure residual drift between the inputs.
4. Cable the headphone jack to the Five. Set Uncompressed, autoplay, level. Play a
   click; measure round-trip delay at the mic. Expect roughly 75 ms plus acoustic path.
5. Play the marker alone at tracking level under a quiet pad; listen for it. Then under
   nothing; listen again. Record what level is audible in an empty kitchen.
6. In MIDI Settings View: check the Track MIDI Input table (track N on channel N), set
   MODE to MONO, SYNC IN to CLOCK, CONTROL MAP CHANNEL to 10. Send notes from the Mac
   on channel 1; confirm track 1 plays its instrument with the sequencer stopped.
   MIDI-learn a CC on channel 10 to filter cutoff (hold OPTION on the parameter, send
   the CC); confirm it moves and note whether it sounds steppy on a slow sweep. Map a
   second CC to track 1 volume in Mixer View and a third to TEMPO in Project View;
   confirm both. Send MIDI clock; confirm a tempo-synced LFO follows. In System Settings
   note the firmware version, set USB AUDIO MODE to STEREO OUT and USB MAIN OUT to MIX.
7. Record the empty-house baseline room response.

## 12. Milestones

1. Bench checks pass (section 11). Decisions recorded for 4.2 and risk 3.
2. Lock works on a static room: marker in, delay and drift measured, 20 dB suppression
   on a recorded test session.
3. Classifier produces a stable state on recorded kitchen sessions (empty, cooking, dinner).
4. Composer plays a fixed ambient piece on the M8 from Tidal with all 8 layers and the
   control keys wired but hand-driven.
5. Close the loop: Controller drives the control keys from live mood; hysteresis and slew
   tuned by living with it for a week.
6. Marker escalation and musical-marker variant.
7. Then the deferred list.

## 13. Privacy posture

All audio stays on the Mac. Each block is reduced to features and the audio is dropped.
No speech recognition. Everyone who lives in or visits the house should know there is a
microphone in the kitchen and what it does.
