# Build log

How roomtone v0.1.0 was built, what each agent was told, what broke at integration, and
what was decided. Dated 2026-10-10 and 2026-10-11. Read this before continuing the build.

## Method

One integrating session wrote the skeleton (pyproject.toml, roomtone/config.py,
roomtone/types.py, docs/SPEC.md), then handed five briefs to five agents running in
parallel. Each agent saw only the skeleton, the spec and its brief, and was told the
interfaces of the modules it depended on but could not see. Agents never ran git. The
integrating session merged, ran the suite and an end-to-end simulated run, fixed the
seams, and sent two modules back for a second round with the findings from that run.

Rules every brief carried: tight code, type hints, plain-English module docstrings for
a hobbyist to read later, logging not print, no em-dashes, no new dependencies, tests
that run with no hardware.

## The five briefs (condensed)

### 1. Marker and lock (marker.py, lock.py, sim.py)

- Marker: band-shaped maximum length sequence, deterministic from a seed, RMS 1, scaled
  to the music at `level_db`; a scheduler emitting one burst per period from sample 0.
- Lock: FFT cross-correlation to find the marker; drift by least squares over
  successive bursts; align by resampling; room response by regularised least squares;
  subtract; echo suppression (ERLE); response change; a `Locker` applying the spec's
  three lock criteria with 3-block hysteresis, escalating the marker ladder on failure.
- Room simulator with delay, clock drift, synthetic impulse response, noise, and
  helpers for speech-like, clatter and pad-like signals.
- Required results: lag within 1 sample at -30 dB marker level under music; 50 ppm
  drift within 5 ppm; full pipeline locks by about block 4 with 20 dB suppression.

### 2. Features and mood (features.py, classify.py, mood.py)

- Features from the residual: RMS, speech (band share times syllabic modulation),
  talkers, transient rate (spectral flux onsets), beat (onset autocorrelation in 60 to
  180 BPM), speech-band centroid, optional YAMNet labels.
- Classifier protocol with a null and a lazy YAMNet implementation.
- MoodTracker: rolling window, recency weighting, 0.15 per block slew, named state with
  3-block hysteresis, confidence from window fill and agreement.

### 3. Audio engine, control, MIDI, CLI (audio.py, control.py, midi.py, engine.py,
log.py, cli.py)

- One sounddevice stream on the aggregate device; callback mixes the marker into the
  output and stamps reference and mic into ring buffers with one frame counter; backends
  for real hardware, tests, and simulation.
- Controller mapping mood to layers, brightness, tempo, scale, transpose, with slew;
  OSC publishers for Tidal's /ctrl and the TUI.
- Bench MIDI helpers; the `roomtone` CLI (devices, bench latency, bench marker, bench
  midi, baseline, run with --sim and --tui).

### 4. Tidal, SuperCollider, M8 docs (tidal/roomtone.tidal, sc/startup.scd,
m8/SETUP.md, docs/BENCH.md, README.md)

- Eight Tidal patterns, one per layer, gated and shaped by the room controls; tempo and
  the mood CCs on channel 10; SuperDirt MIDI startup; the M8 checklist view by view; the
  day-one bench procedure; the README.

### 5. TUI (tui.py, docs/TUI.md)

- Textual app styled as an M8 screen: hex values, bar meters, cyan titles, magenta
  playhead, lock, room, mood and layers panels, a phrase-view history strip; OSC feed and
  a scripted two-hour demo.

## What broke at integration, and the fix

1. Block convention. The engine sliced blocks on its own grid and passed the marker's
   offset within the block; the Locker expected each block to start on a burst and
   receive the absolute index. Fix: the engine aligns every analysis block to a burst
   and the marker period is tied to the block length (engine.py, cli.py).
2. Scheduler API. The engine called a `next_block(n)` that did not exist and rescaled
   the marker; the real scheduler is `block(start, n, music_rms)` and fixes each burst's
   level when it starts. Fix: `MarkerSource` tracks its own position and calls `block`.
3. Energy was absolute. A -48 dBFS mic floor read as energy 0.29 and the remainder rule
   made an empty room "cooking". Fix: energy is relative to a learned quiet floor
   (running minimum over 240 blocks, never below -60 dBFS); cooking needs onsets or
   strong loudness.
4. The simulated kitchen was silent, so the mood never changed. Fix: kitchen.py, a
   scripted room (empty, cooking, dinner, party, empty) with `--script` on `run`.
5. The lock dropped out whenever the room was loud. This was the real finding of the
   first full run. Fix (second round on lock.py): double-talk gating. A block is "quiet"
   when the marker is clear (peak-to-sidelobe at least 10 dB), the candidate response
   suppresses the echo by 20 dB over the quietest quarter, and the marker band's
   leftover is within 3 dB of the predicted echo. Only quiet blocks update the response;
   loud blocks subtract with the frozen one and touch no counters. Lock holds while the
   marker is found within 8 samples of the drift line at 6 dB or better. LOST needs a
   quiet block with a response change over 0.6 and a 6 dB worse fit from the old
   response. The ladder climbs on a weak marker, never on low ERLE. A baseline-primed
   session locks on block 2.
6. The party kick was invisible under a crowd. Fix (second round on features.py): a
   second onset envelope from 40 to 200 Hz, beat taken as the max of both, and beat now
   requires alignment at the second beat as well, which keeps jittered speech under 0.3.

## The simulated evening (final run)

28 blocks of 30 s through the scripted kitchen: lock at block 4, held through every
phase, never lost. Mood: empty, cooking from block 7, conversation from 13, loud-busy
from 19, empty from 24, each on the third block of its phase. Echo suppression about
30 dB in quiet phases, 22 to 24 dB in cooking, 10 dB at dinner, 2 dB at the party (the
room is simply louder than the music there; lock is a matter of geometry and holds).

## Constants worth knowing

- Marker: 4 s, period equals block (30 s), 4 to 10 kHz, -30 dB relative to music;
  ladder: length to 30 s, then band 6 to 12 kHz, then level up to -10 dB.
- Lock: ERLE target 20 dB, jitter 2 samples, hysteresis 3 blocks, RIR tail 0.2 s.
- Mood: floor is a 240-block running minimum, never below -60 dBFS; loudness 0 at
  floor plus 6 dB, 1 at floor plus 36 dB or at -20 dBFS absolute; slew 0.15 per block.
- Controller: tempo 54 plus 50 times rhythm BPM, at most 5 BPM per block; transpose
  changes every 40 blocks; layer thresholds in control.py.

## Open items

- Everything on hardware: docs/BENCH.md, then live with it for a week.
- Tidal file and SuperCollider startup have never been evaluated.
- Known limit: if the music is far quieter than the mic floor, every block reads loud
  and the response never adapts. Raise the marker or the music.
- A `--monitor` flag so `run --sim` can play the simulated room through the speakers.
- Multichannel USB from the M8 (12 stereo stems) for Mac-side per-layer processing.
- The musical marker (an audible burst composed as a motif) for the top of the ladder.
- Party handoff to a Sonos playlist; HomePods over AirPlay.
