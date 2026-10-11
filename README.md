# roomtone

roomtone is ambient music that follows the room it plays in. A MacBook listens to a kitchen
through a measurement mic, works out every 30 seconds whether the room is empty, someone is
cooking, people are talking, or it is loud and busy, and steers live-generated music to
match. A Dirtywave M8 Model:02 plays the music, TidalCycles on the Mac composes it in real
time, and a Sonos Five plays it into the house through its line-in. Talking turns the music
down and thins it out; an empty room gets a slow drone; a busy kitchen gets more layers and
a soft pulse.

The idea comes from Brian Eno's ambient records. *Music for Airports* was built from tape
loops of different lengths that drift in and out of phase, so the same few sounds keep
forming new combinations, and its liner notes say ambient music must be "as ignorable as it
is interesting." roomtone keeps both parts. Each layer runs on its own long loop (17.8 s,
21.3 s, 25.1 s and so on) so the layers rarely line up the same way twice, and a slow cycle
keeps the piece moving when the room is still. The new part is that the room, not a fixed
score, decides which layers play and how loud and bright they are.

To hear the room, the Mac has to ignore its own music, which the mic also picks up. It does
that by locking onto its playback: every 30 s it buries a short burst of shaped noise (a
maximum length sequence, a pseudo-random signal with a sharp correlation peak) about 30 dB
under the music. Finding that burst in the mic signal gives the delay through the Sonos, the
drift between clocks, and the room's acoustic response, so the Mac can subtract what it
played and keep what is left: the room. The M8, a handheld tracker, works here as a sound
module: it takes MIDI notes from the Mac on 8 channels with its own sequencer stopped,
takes mood changes as MIDI CCs mapped onto filter cutoffs, volumes, effect sends, tempo and
EQ, and sends its stereo mix back to the Mac over the same USB cable.

## Signal flow

```
                   MIDI notes ch 1-8, mood CCs ch 10 (USB)
   +-------------------------------------------------------------> M8 Model:02
   |                                                                  |
   |  Mac                                                             | stereo mix
   |  +----------------------------------------------+                | (USB audio)
   +--| Composer: TidalCycles + SuperDirt            |                |
      |   patterns read control values from /ctrl    |                |
      +----------------------------------------------+                |
      | Mixer + marker (roomtone, Python)            |<---------------+
      |   M8 in + marker -> out; logs the reference  |----> headphone jack ----> Sonos Five line-in
      +----------------------------------------------+                                |
      | Listener (roomtone, Python)                  |                                v
      |   lock, subtract, classify, mood state       |<---- USB mic <---------- room (+ Five)
      +----------------------------------------------+
      | Controller (roomtone, Python)                |----> OSC /ctrl (port 6010) to Composer
      |   hysteresis, marker escalation, logging     |----> marker params to Mixer
      +----------------------------------------------+
```

This follows [docs/SPEC.md](docs/SPEC.md) section 3 with one change: the mixer and marker run
in the roomtone Python process, which owns the "roomtone" aggregate audio device (M8 in, mic
in, built-in output). SuperCollider only carries SuperDirt's MIDI output to the M8 and plays
no audio in v1. The processes talk over OSC on localhost, and each can be restarted alone.
The mixer folds the M8 to mono before it goes out (`mono_output` in `roomtone/config.py`):
the Listener models one path from output to mic, and the M8's stereo effects make its two
channels nearly independent, so two speakers playing different signals would leave the
difference in the residual. On the laptop rig that was the difference between 2 dB and
20 dB of echo suppression.

## Install

macOS, Python 3.11 or newer.

1. roomtone itself, from the repo root:

   ```
   pip install -e '.[audio,dev]'
   ```

   `audio` adds sounddevice and python-rtmidi; `dev` adds pytest. The optional `yamnet`
   extra adds the local audio event model used for speech and dish detection.
2. SuperCollider and the SuperDirt quark. With Homebrew:

   ```
   brew install --cask supercollider
   /Applications/SuperCollider.app/Contents/MacOS/sclang sc/install-superdirt.scd
   ```

   (or install SuperCollider from [supercollider.github.io](https://supercollider.github.io)
   and evaluate `Quarks.install("SuperDirt");` in the IDE, then recompile).
3. TidalCycles. Tidal is a Haskell library, so it needs GHC; ghcup is the easy way, and
   GHC 9.8 is the newest release Tidal 1.10 is tested with:

   ```
   brew install ghcup
   ghcup install ghc 9.8.4 --set
   ghcup install cabal 3.16.1.0 --set
   cabal update
   cabal install tidal --lib
   ```

   Put `~/.ghcup/bin` on your PATH. An editor is optional: Pulsar with the TidalCycles
   package or VS Code with the TidalCycles extension gives you live coding, and
   `roomtone tidal` runs the file from a terminal with no editor (below). The default
   BootTidal.hs is enough; `tidal/roomtone.tidal` needs nothing extra from it.
4. The M8 configured per [m8/SETUP.md](m8/SETUP.md), and the bench checks in
   [docs/BENCH.md](docs/BENCH.md) done once (they build the aggregate device with
   `roomtone aggregate create` and record the empty-house baseline).

## Run order

1. Plug in the M8, the mic and the headphone cable to the Five. The aggregate device
   "roomtone" picks them up (`roomtone aggregate show` lists its members; docs/BENCH.md
   step 3 builds it).
2. Start SuperDirt with MIDI out to the M8:

   ```
   sclang sc/startup.scd
   ```

   (or open the file in the SuperCollider IDE and evaluate it). Wait for the line
   `roomtone: SuperDirt is up`.
3. Start Tidal. Either open `tidal/roomtone.tidal` in your Tidal editor and evaluate the
   blocks top to bottom (`setcps`, the `let` block, d1 to d8, d9, d10), or run

   ```
   roomtone tidal
   ```

   which evaluates the same blocks in a headless ghci and stays up; anything you type
   there (ended by a blank line) goes to Tidal, so `hush` and `d3 $ silence` work.
   With nothing else running you hear the drone alone.
4. Start the listener and controller with the terminal UI:

   ```
   roomtone run --tui
   ```

   It locks onto the room over the first blocks and starts publishing control values.
   `roomtone run --sim` does the same with simulated room readings, for working on the
   control path without the mic. To hear what each state of the room sounds like without
   waiting for the room, `roomtone run --audition empty:60,cooking:120,dinner:90,party:120`
   keeps the real audio running and feeds the Controller that mood script instead of the
   mic (the Controller's slew still applies, so layers arrive over a minute or two).

To stop: Ctrl-C in the roomtone terminal, `hush` in Tidal. You can keep live-coding in the
Tidal file the whole time; the room only changes control values, never the patterns, so your
edits and the room's never collide.

## Privacy

All audio stays on the Mac. Each 30 s block is reduced to features (loudness, speech
presence, transient rate, beat presence, spectral shape) and the audio is dropped. There is
no speech recognition, and nothing is sent anywhere: no cloud services, no accounts beyond
the Sonos app. Everyone who lives in or visits the house should know there is a microphone in
the kitchen and what it does.

## Documents

- [docs/SPEC.md](docs/SPEC.md): the v1 architecture, the marker and lock design, and what is
  confirmed about the M8 and the Sonos.
- [docs/BENCH.md](docs/BENCH.md): the day-one bench procedure, with pass criteria and a sheet
  to fill in.
- [m8/SETUP.md](m8/SETUP.md): the M8 checklist, view by view.
- [tidal/roomtone.tidal](tidal/roomtone.tidal) and [sc/startup.scd](sc/startup.scd): the
  music and the MIDI bridge.

## Why this is fun

A pocket-sized tracker becomes the voice of a kitchen. The house plays a piece that never
repeats, that gets out of the way when people start talking and fills back in when they
leave, and the only thing listening is a mic and a laptop on the counter. The trick that
makes it work is a signal nobody hears, 30 dB under the music, that lets the machine tell its
own playing apart from the room. And the score stays open: you can sit down with the Tidal
file at any moment and play along with the room, and neither of you overwrites the other.

## Known limits of this build

- Everything above was verified in simulation only. No M8, mic, or Sonos has touched
  this code yet; docs/BENCH.md is the first thing to run on real hardware.
- The lock decides a block is "quiet" partly by comparing leftover energy in the marker
  band with the echo it predicts. If the music is far quieter than the mic noise floor,
  the marker is too, every block reads as loud, and the room response never adapts.
  Raise the marker level or the music level if the TUI never shows a quiet block.
- Tidal patterns and the SuperCollider startup were written carefully but never
  evaluated here. Expect to fix a name or two on first evaluation.
- YAMNet labels are wired but need the model file and the `roomtone[yamnet]` extra.
