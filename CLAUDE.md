# CLAUDE.md: roomtone

Room-responsive ambient music. A Mac listens to a kitchen through a measurement mic,
locks onto its own playback with pseudo-random markers buried under the music, subtracts
itself, classifies what is left, and steers a Dirtywave M8 Model:02 through TidalCycles.
Playback goes out the Mac's headphone jack into a Sonos Five line-in.

Read in this order when starting a session: README.md, docs/SPEC.md (architecture, the
contract everything was built to), docs/BUILD-LOG.md (how it was built and what was
decided), docs/BENCH.md (the day-one hardware checks), m8/SETUP.md (the M8 side).

## State of the build

v0.1.0 is complete in simulation and untested on hardware. `python3 -m pytest -q` runs
166 tests in about 70 s and must stay green. `roomtone run --sim --fast --blocks 28
--no-tidal` plays a scripted kitchen evening through the whole chain with no hardware.
The next real step is docs/BENCH.md on the laptop with the M8, the mic and the Five
plugged in. Nothing in this repo has heard a real room yet.

## Working rules (non-negotiable)

- No em-dashes anywhere: code, comments, docs, commit messages. Use commas, colons,
  parentheses or a new sentence.
- Plain English. Module docstrings explain the method so a hobbyist can read them a year
  later. No marketing adjectives in docs.
- Every constant that shapes behaviour lives in roomtone/config.py, or is named at the
  top of its module with a comment saying why it has that value.
- Shared value types live in roomtone/types.py and stay plain dataclasses.
- Library code logs; only roomtone/cli.py prints.
- Dependencies: numpy, scipy, python-osc, mido, textual. Hardware extras (sounddevice,
  python-rtmidi) and YAMNet are optional and imported lazily inside functions. Never
  import them at module level.
- Hardware touches go behind a seam that tests can fake (see AudioIO backends and
  tests/fakes.py). Every new feature ships with tests that run with no hardware.
- Keep the test suite under about 90 s. The lock tests are the slow ones; shrink block
  lengths in tests where it does not weaken the claim.
- Every number in a doc traces to a measurement, the spec, or the M8 manual.

## How the build is organised

| Module | Owns | Contract |
|---|---|---|
| marker.py | pseudo-random marker bursts | `generate`, `scale_to_music`, `MarkerScheduler.block(start, n, music_rms)` |
| lock.py | find marker, drift, room response, subtraction, lock state, ladder | `Locker.process(reference, mic, emission_start, params) -> (residual, LockState)` |
| features.py | room features from the residual | `extract(residual, sr, classifier) -> Features` |
| classify.py | optional YAMNet labels | `Classifier` protocol |
| mood.py | mood vector, named state, hysteresis, learned quiet floor | `MoodTracker.update(features, lock) -> Mood` |
| control.py | mood to music controls, OSC to Tidal and the TUI | `Controller.update(mood, lock) -> Controls` |
| audio.py | device I/O, ring buffers, backends (real, fake, simulated) | `AudioIO` |
| kitchen.py | scripted simulated room sound | `KitchenScript` |
| engine.py | the per-block loop; blocks are aligned to marker bursts | `Engine.step()` |
| midi.py | bench MIDI to the M8 | functions |
| tui.py | M8-styled Textual dashboard | `run(cfg, demo)` |
| cli.py | `roomtone` subcommands | argparse |
| sim.py | room simulator and synthetic signals for tests | `SimRoom`, helpers |

Conventions that cross modules and must not drift:

- Analysis blocks start on the first sample of a marker burst. The Locker receives the
  absolute start index of the block, and the marker period equals the block length.
- The reference is the mixed output buffer (music plus marker) as written to the output
  device, folded to mono. Not the M8 capture stream.
- MIDI channels 1 to 8 are the layers in config.LAYER_ORDER; channel 9 is never used;
  channel 10 carries every mood CC. CC numbers are in config.CC and must match
  m8/SETUP.md and tidal/roomtone.tidal.
- Tidal control keys are the field names of `Controls` plus `layer_<name>`.

## How to fan work out

The modules above were built in parallel by separate agents from written briefs, each
agent seeing only config.py, types.py, docs/SPEC.md and its brief (see docs/BUILD-LOG.md
for the briefs). That worked because the interfaces were fixed first. Keep doing it that
way: write the interface into config.py or types.py, write the brief, then build. Agents
never run git; the integrating session commits. Integration is where the seams break, so
budget for an end-to-end `run --sim` after every merge and read the per-block lines.

## Privacy

All audio stays on the Mac. Blocks are reduced to features and the audio is dropped. No
speech recognition. The house should know there is a microphone in the kitchen.
