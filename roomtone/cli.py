"""The `roomtone` command.

    roomtone devices                 audio devices and MIDI ports
    roomtone bench latency           click round trip through the speakers
    roomtone bench marker            listen for the marker under a pad, then alone
    roomtone bench midi              notes, a CC sweep and clock to the M8
    roomtone baseline                record the empty-house room response
    roomtone run [--sim]             the whole loop (--sim needs no hardware;
                                     --script sets what the simulated kitchen does)

The bench commands follow the day-one checks in docs/SPEC.md section 11.
Anything that needs hardware or an optional package says so and exits,
instead of failing with a traceback.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import logging
import sys
from types import ModuleType
from typing import Any, Callable, Sequence

import numpy as np

from roomtone import audio as audio_mod
from roomtone import kitchen
from roomtone.config import CONTROL_MAP_CHANNEL, Config, MarkerParams
from roomtone.types import BlockResult, db

AUDIO_HINT = ("sounddevice is not installed, so roomtone cannot reach audio hardware.\n"
              "Install the audio extras with:  pip install 'roomtone[audio]'\n"
              "To try the loop with no hardware at all:  roomtone run --sim")
MIDI_HINT = ("No MIDI backend: mido needs python-rtmidi to reach real ports.\n"
             "Install the audio extras with:  pip install 'roomtone[audio]'")
NOMINAL_MUSIC_RMS = 10 ** (-20.0 / 20.0)  # stand-in music level when there is no music


def _sounddevice() -> ModuleType | None:
    try:
        import sounddevice
        return sounddevice
    except (ImportError, OSError):  # OSError: the package is there but PortAudio is not
        print(AUDIO_HINT)
        return None


def _module(name: str, purpose: str) -> ModuleType | None:
    """Import a roomtone module another part of the project provides, or explain its absence."""
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        print(f"{name} is not available ({exc}); it is needed for {purpose}.")
        return None


def _config(args: argparse.Namespace) -> Config:
    cfg = Config()
    if getattr(args, "block_seconds", None):
        cfg.block_seconds = args.block_seconds
    # one marker burst per analysis block: the engine aligns blocks to bursts
    if cfg.marker.period_s != cfg.block_seconds:
        cfg.marker = replace(cfg.marker, period_s=cfg.block_seconds)
    if getattr(args, "log_dir", None):
        cfg.log_dir = args.log_dir
    return cfg


# devices ---------------------------------------------------------------------

def cmd_devices(args: argparse.Namespace) -> int:
    print("Audio devices")
    try:
        for d in audio_mod.list_devices():
            print(f"  [{d['index']:2d}] {d['name']}  ({d['hostapi']}; "
                  f"{d['max_input_channels']} in, {d['max_output_channels']} out, "
                  f"{d['default_samplerate']:.0f} Hz)")
    except (ImportError, OSError):
        print("  " + AUDIO_HINT.replace("\n", "\n  "))
    print("MIDI ports")
    try:
        from roomtone import midi
        ports = midi.list_ports()
        for kind in ("outputs", "inputs"):
            names = ports[kind] or ["(none)"]
            print(f"  {kind}: " + ", ".join(names))
    except ImportError:
        print("  " + MIDI_HINT.replace("\n", "\n  "))
    return 0


# bench -----------------------------------------------------------------------

def cmd_bench_latency(args: argparse.Namespace) -> int:
    sd = _sounddevice()
    if sd is None:
        return 2
    cfg = _config(args)
    sr = cfg.sample_rate
    signal, starts = audio_mod.click_train(sr, n_clicks=10)
    out = np.repeat(signal[:, None], 2, axis=1)
    print(f"Playing 10 clicks through {cfg.aggregate_device!r}, recording mic channel {args.mic_channel}...")
    rec = sd.playrec(out, samplerate=sr, channels=args.mic_channel + 1,
                     device=cfg.aggregate_device, dtype="float32", blocking=True)
    delays = np.array(audio_mod.measure_click_delays(rec[:, args.mic_channel], starts, sr))
    if delays.size == 0:
        print("No clicks found in the recording. Check the mic channel and the speaker level.")
        return 1
    ms = delays / sr * 1000.0
    print(f"Round trip: median {np.median(ms):.1f} ms, spread {ms.max() - ms.min():.1f} ms "
          f"(min {ms.min():.1f}, max {ms.max():.1f}, n={ms.size})")
    print("Expected: about 75 ms (Sonos line-in, Uncompressed) plus the acoustic path.")
    return 0


def cmd_bench_marker(args: argparse.Namespace) -> int:
    sd = _sounddevice()
    if sd is None:
        return 2
    marker = _module("roomtone.marker", "generating the marker")
    if marker is None:
        return 2
    cfg = _config(args)
    sr = cfg.sample_rate
    params = MarkerParams(length_s=args.length, period_s=max(args.length + 2.0, 5.0),
                          level_db=args.level)
    under_s, alone_s = 20.0, 10.0
    n_under, n_total = int(under_s * sr), int((under_s + alone_s) * sr)
    pad = audio_mod.PadSynth(sr).next(n_under)
    pad_rms = float(np.sqrt(np.mean(pad.astype(np.float64) ** 2)))
    scheduler = marker.MarkerScheduler(params, sr)
    chunks, done = [], 0
    while done < n_total:
        n = min(cfg.io_frames, n_total - done)
        # the same music reference throughout, so the marker's absolute level never changes
        chunks.append(scheduler.block(done, n, pad_rms))
        done += n
    mk = np.concatenate(chunks).astype(np.float32)
    out = np.repeat(mk[:, None], 2, axis=1)
    out[:n_under] += pad
    on = mk[np.abs(mk) > 0]
    mk_db = db(float(np.sqrt(np.mean(on.astype(np.float64) ** 2)))) if on.size else -120.0
    print(f"Marker: {params.length_s:g} s long every {params.period_s:g} s, band "
          f"{params.band_lo_hz:.0f} to {params.band_hi_hz:.0f} Hz, {params.level_db:g} dB re music.")
    print(f"Pad RMS {db(pad_rms):.1f} dBFS; marker RMS during emissions {mk_db:.1f} dBFS.")
    print(f"0 to {under_s:.0f} s: marker under the pad. {under_s:.0f} to {under_s + alone_s:.0f} s: marker alone.")
    sd.play(out, samplerate=sr, device=cfg.aggregate_device, blocking=True)
    print(f"Done. Write down whether you heard it at {params.level_db:g} dB, under the pad and alone.")
    return 0


MIDI_CHECKLIST = """\
What to check on the M8 (spec section 11, check 6):
  - MIDI Settings: Track MIDI Input has track N on channel N, MODE MONO,
    SYNC IN = CLOCK, CONTROL MAP CHANNEL = {map_ch}.
  - The 4-note phrase plays track {ch}'s instrument with the sequencer stopped.
  - Before the sweep, MIDI-learn CC {cc} on channel {map_ch} to a filter cutoff
    (hold OPTION on the parameter, then send the CC). During the slow 0 to 127
    sweep, note whether the cutoff sounds steppy.
  - During the 10 s of clock at 60 BPM, a tempo-synced LFO should follow.
  - Also confirm by hand: a second CC mapped to track 1 volume in Mixer View, a
    third to TEMPO in Project View; firmware version; USB AUDIO MODE = STEREO OUT,
    USB MAIN OUT = MIX."""


def cmd_bench_midi(args: argparse.Namespace) -> int:
    from roomtone import midi
    cfg = _config(args)
    try:
        port = midi.open_out(cfg.midi_out_port)
    except ImportError:
        print(MIDI_HINT)
        return 2
    except LookupError as exc:
        print(exc)
        return 2
    print(MIDI_CHECKLIST.format(map_ch=CONTROL_MAP_CHANNEL, ch=args.channel, cc=args.cc))
    try:
        print(f"\nPhrase on channel {args.channel}...")
        for note in (57, 60, 64, 67):
            midi.send_note(port, args.channel, note, 96, 0.5)
        print(f"CC {args.cc} sweep 0 to 127 on channel {CONTROL_MAP_CHANNEL} over {args.sweep_seconds:g} s...")
        midi.cc_sweep(port, CONTROL_MAP_CHANNEL, args.cc, 0, 127, args.sweep_seconds)
        print("Clock at 60 BPM for 10 s...")
        ticks = midi.send_clock(port, 60.0, 10.0)
        print(f"Sent {ticks} clock ticks. Done.")
    finally:
        port.close()
    return 0


# baseline --------------------------------------------------------------------

def cmd_baseline(args: argparse.Namespace) -> int:
    sd = _sounddevice()
    if sd is None:
        return 2
    marker = _module("roomtone.marker", "generating the marker")
    lock = _module("roomtone.lock", "estimating the room response")
    if marker is None or lock is None:
        return 2
    from roomtone.engine import MarkerSource, acquire_baseline
    cfg = _config(args)
    params = MarkerParams(length_s=args.length, period_s=cfg.block_seconds, level_db=args.level)
    source = MarkerSource(marker.MarkerScheduler(params, cfg.sample_rate), params,
                          fixed_rms=NOMINAL_MUSIC_RMS)
    io = audio_mod.AudioIO(cfg, source, mute_music=True)
    locker = lock.Locker(cfg)
    print(f"Marker-only acquisition for {args.seconds:g} s. Keep the house quiet.")
    io.start()
    try:
        rir = acquire_baseline(cfg, io, locker, source, args.seconds)
    except KeyboardInterrupt:
        print("Interrupted; nothing saved.")
        return 1
    finally:
        io.stop()
    lock.save_baseline(args.out, rir)
    print(f"Saved the room response to {args.out}.")
    return 0


# run -------------------------------------------------------------------------

class _PrintPublisher:
    """One console line per block, for runs without the TUI.

    With `phase_at` (the simulated kitchen's script), each line also says which
    phase the script was in when the block started.
    """

    def __init__(self, phase_at: Callable[[float], str] | None = None) -> None:
        self.phase_at = phase_at

    def publish(self, r: BlockResult) -> None:
        c = r.controls
        phase = f"phase={self.phase_at(r.t_start):<8} " if self.phase_at else ""
        print(f"[{r.index:4d}] t={r.t_start:7.1f}s  {phase}lock={r.lock.status.value:<10} "
              f"mood={r.mood.name.value:<12} energy={c.energy:.2f} voices={c.voices:.2f} "
              f"rhythm={c.rhythm:.2f} layers={c.layers:08b} {c.scale} {c.tempo_bpm:.0f} BPM  "
              f"raw {r.raw_rms_db:.1f} dB, residual {r.residual_rms_db:.1f} dB", flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    if not args.sim and _sounddevice() is None:
        return 2
    mods = {name: _module(f"roomtone.{name}", purpose) for name, purpose in [
        ("marker", "the marker"), ("lock", "locking onto the room"),
        ("features", "listening to the room"), ("mood", "the mood estimate")]}
    if any(m is None for m in mods.values()):
        return 2
    from roomtone.control import Controller, TidalPublisher, TuiPublisher
    from roomtone.engine import Engine, MarkerSource
    from roomtone.log import BlockLog

    cfg = _config(args)
    marker, lock = mods["marker"], mods["lock"]
    source = MarkerSource(marker.MarkerScheduler(cfg.marker, cfg.sample_rate), cfg.marker)
    backend: Any = (audio_mod.SimBackend(cfg, realtime=not args.fast,
                                         script=args.script or kitchen.DEFAULT_SCRIPT)
                    if args.sim else audio_mod.SoundDeviceBackend(cfg))
    script = getattr(backend, "script", None)
    io = audio_mod.AudioIO(cfg, source, backend=backend)
    locker = lock.Locker(cfg)
    if args.baseline:
        locker.rir = lock.load_baseline(args.baseline)
    publishers: list[Any] = [TuiPublisher(cfg.tui_host, cfg.tui_port)]
    if not args.no_tidal:
        publishers.append(TidalPublisher(cfg.tidal_host, cfg.tidal_port))
    tui = _module("roomtone.tui", "the terminal display") if args.tui else None
    if tui is None:
        publishers.append(_PrintPublisher(script.phase_at if script is not None else None))
    log = BlockLog(cfg.log_dir)
    engine = Engine(cfg, io, locker, mods["mood"].MoodTracker(cfg), Controller(cfg), publishers,
                    log=log, marker=source, extract=mods["features"].extract)

    where = (f"audio device {cfg.aggregate_device!r}" if not args.sim else
             "the simulated kitchen" + (f" (script {kitchen.format_script(script.phases)})"
                                        if script is not None else ""))
    print(f"roomtone: {where}, {cfg.block_seconds:g} s blocks, "
          f"Tidal {'off' if args.no_tidal else f'{cfg.tidal_host}:{cfg.tidal_port}'}, "
          f"TUI feed {cfg.tui_host}:{cfg.tui_port}, log {log.path}")
    if args.dry_run:
        print("Dry run: everything is wired; not starting audio.")
        return 0

    try:
        if args.sim and args.fast:  # lockstep: simulate a block, analyse it, repeat
            io.open()
            done = 0
            while args.blocks is None or done < args.blocks:
                if engine.step() is None:
                    backend.run_frames(cfg.io_frames)
                else:
                    done += 1
            return 0
        io.start()
        if tui is not None:
            engine.start(max_blocks=args.blocks)
            tui.run(cfg)
        else:
            engine.run_forever(max_blocks=args.blocks)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        engine.stop()
        io.stop()
    return 0


# parser ----------------------------------------------------------------------

def _script_arg(text: str) -> list[tuple[float, str]]:
    try:
        return kitchen.parse_script(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="roomtone", description="Room-responsive ambient music.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="more logging (-vv for debug)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("devices", help="list audio devices and MIDI ports").set_defaults(func=cmd_devices)

    bench = sub.add_parser("bench", help="day-one bench checks")
    bsub = bench.add_subparsers(dest="bench", required=True)
    b = bsub.add_parser("latency", help="click round trip through the speakers to the mic")
    b.add_argument("--mic-channel", type=int, default=2, help="mic input channel (default 2)")
    b.set_defaults(func=cmd_bench_latency)
    b = bsub.add_parser("marker", help="play the marker under a pad for 20 s, then alone for 10 s")
    b.add_argument("--level", type=float, default=-30.0, help="dB relative to the music (default -30)")
    b.add_argument("--length", type=float, default=4.0, help="marker length in seconds (default 4)")
    b.set_defaults(func=cmd_bench_marker)
    b = bsub.add_parser("midi", help="phrase, CC sweep and clock to the M8")
    b.add_argument("--channel", type=int, default=1, help="note channel, 1 to 16 (default 1)")
    b.add_argument("--cc", type=int, default=30, help=f"CC to sweep on channel {CONTROL_MAP_CHANNEL} (default 30)")
    b.add_argument("--sweep-seconds", type=float, default=8.0, help="sweep duration (default 8)")
    b.set_defaults(func=cmd_bench_midi)

    b = sub.add_parser("baseline", help="record the empty-house room response")
    b.add_argument("--seconds", type=float, default=60.0, help="acquisition time (default 60)")
    b.add_argument("--out", default="baseline.npy", help="where to save it (default baseline.npy)")
    b.add_argument("--level", type=float, default=-15.0,
                   help="marker level in dB re a nominal -20 dBFS music level (default -15)")
    b.add_argument("--length", type=float, default=8.0, help="marker length in seconds (default 8)")
    b.add_argument("--block-seconds", type=float, help="analysis block length (default from config)")
    b.set_defaults(func=cmd_baseline)

    b = sub.add_parser("run", help="run the whole loop")
    b.add_argument("--sim", action="store_true", help="simulated M8 and kitchen; no hardware needed")
    b.add_argument("--baseline", help="start from this saved room response (.npy)")
    b.add_argument("--no-tidal", action="store_true", help="do not send /ctrl to Tidal")
    b.add_argument("--tui", action="store_true", help="show the terminal display")
    b.add_argument("--fast", action="store_true", help="with --sim: run faster than real time")
    b.add_argument("--script", type=_script_arg, metavar="MODE:SECONDS,...",
                   help="with --sim: what the kitchen does, phase by phase, repeating; modes "
                        f"{', '.join(kitchen.MODES)} (default "
                        f"{kitchen.format_script(kitchen.DEFAULT_SCRIPT)})")
    b.add_argument("--blocks", type=int, help="stop after this many blocks")
    b.add_argument("--block-seconds", type=float, help="analysis block length (default from config)")
    b.add_argument("--log-dir", help="where session logs go (default from config)")
    b.add_argument("--dry-run", action="store_true", help="wire everything, then exit without starting")
    b.set_defaults(func=cmd_run)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    level = [logging.WARNING, logging.INFO, logging.DEBUG][min(args.verbose, 2)]
    logging.basicConfig(level=level, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
