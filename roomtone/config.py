"""Project-wide constants and the runtime configuration object.

Every module reads from here instead of carrying its own magic numbers, so the
sample rate, block length, OSC endpoints and MIDI channel plan live in exactly
one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Audio ----------------------------------------------------------------------

SAMPLE_RATE = 44_100          # M8 engine rate (renders are 44.1 kHz); confirm on the bench
BLOCK_SECONDS = 30.0          # one analysis block, one marker per block
MOOD_WINDOW_SECONDS = 300.0   # rolling window the mood estimate integrates over
IO_FRAMES = 1024              # audio callback size


# Marker (section 5 of the spec) ----------------------------------------------

@dataclass
class MarkerParams:
    """Everything that defines one marker emission."""
    length_s: float = 4.0        # MLS length in seconds
    period_s: float = 30.0       # one marker per analysis block
    band_lo_hz: float = 4000.0
    band_hi_hz: float = 10000.0
    level_db: float = -30.0      # relative to recent music RMS
    seed: int = 0x4D38           # "M8"; fixed so the Listener regenerates the same sequence

    def escalate(self) -> "MarkerParams":
        """One rung up the ladder: longer first, then band, then level."""
        if self.length_s < 30.0:
            return MarkerParams(min(self.length_s * 2, 30.0), self.period_s,
                                self.band_lo_hz, self.band_hi_hz, self.level_db, self.seed)
        if self.band_lo_hz < 6000.0:
            return MarkerParams(self.length_s, self.period_s, 6000.0, 12000.0,
                                self.level_db, self.seed)
        return MarkerParams(self.length_s, self.period_s, self.band_lo_hz, self.band_hi_hz,
                            min(self.level_db + 5.0, -10.0), self.seed)

    def relax(self) -> "MarkerParams":
        """One rung down."""
        if self.level_db > -30.0:
            return MarkerParams(self.length_s, self.period_s, self.band_lo_hz, self.band_hi_hz,
                                max(self.level_db - 5.0, -30.0), self.seed)
        if self.band_lo_hz > 4000.0:
            return MarkerParams(self.length_s, self.period_s, 4000.0, 10000.0,
                                self.level_db, self.seed)
        return MarkerParams(max(self.length_s / 2, 4.0), self.period_s,
                            self.band_lo_hz, self.band_hi_hz, self.level_db, self.seed)

    @property
    def at_ceiling(self) -> bool:
        return self.length_s >= 30.0 and self.band_lo_hz >= 6000.0 and self.level_db >= -10.0


DEFAULT_MARKER = MarkerParams()


# Lock (section 6) ------------------------------------------------------------

RIR_TAIL_SECONDS = 0.2         # initial room impulse response length
LOCK_ERLE_DB = 20.0            # echo suppression that counts as locked
LOCK_JITTER_SAMPLES = 2.0      # marker timing jitter that counts as locked
HYSTERESIS_BLOCKS = 3          # consecutive blocks before any state change


# MIDI plan (section 8.1) -----------------------------------------------------

LAYER_CHANNELS = {
    "drone": 1, "pad": 2, "figure": 3, "phrases": 4,
    "shimmer": 5, "pulse": 6, "bass": 7, "spare": 8,
}
LAYER_ORDER = list(LAYER_CHANNELS)
REC_NOTE_CHANNEL = 9           # M8 default; never used for layers
CONTROL_MAP_CHANNEL = 10       # all mood CCs go here (M8 CONTROL MAP CHANNEL setting)

# CC numbers on CONTROL_MAP_CHANNEL. Learn each one on the M8 with OPTION + CC.
CC = {
    "tempo": 20, "transpose": 21,
    "drone_cutoff": 30, "pad_cutoff": 31, "figure_cutoff": 32, "shimmer_cutoff": 33,
    "drone_vol": 40, "pad_vol": 41, "figure_vol": 42, "phrases_vol": 43,
    "shimmer_vol": 44, "pulse_vol": 45, "bass_vol": 46, "spare_vol": 47,
    "reverb_send": 50, "delay_send": 51, "main_eq_mid_gain": 52,
}


# OSC endpoints ---------------------------------------------------------------

TIDAL_CTRL_HOST = "127.0.0.1"
TIDAL_CTRL_PORT = 6010         # Tidal's default /ctrl listener
TUI_HOST = "127.0.0.1"
TUI_PORT = 6020                # Listener -> TUI state feed


@dataclass
class Config:
    sample_rate: int = SAMPLE_RATE
    block_seconds: float = BLOCK_SECONDS
    mood_window_seconds: float = MOOD_WINDOW_SECONDS
    io_frames: int = IO_FRAMES
    marker: MarkerParams = field(default_factory=MarkerParams)
    rir_tail_seconds: float = RIR_TAIL_SECONDS
    lock_erle_db: float = LOCK_ERLE_DB
    lock_jitter_samples: float = LOCK_JITTER_SAMPLES
    hysteresis_blocks: int = HYSTERESIS_BLOCKS
    tidal_host: str = TIDAL_CTRL_HOST
    tidal_port: int = TIDAL_CTRL_PORT
    tui_host: str = TUI_HOST
    tui_port: int = TUI_PORT
    # Device names as CoreAudio and sounddevice report them (case-insensitive
    # substring, exact name preferred). The three members build `aggregate_device`
    # with `roomtone aggregate create`; the stream itself opens the aggregate.
    # Defaults are the laptop rig: M8 over USB, the MacBook's own mic and
    # speakers. For the kitchen rig name the measurement mic and the headphone
    # output here instead (docs/BENCH.md steps 2 and 3).
    m8_input_device: str = "M8"
    mic_input_device: str = "MacBook Air Microphone"
    output_device: str = "MacBook Air Speakers"
    aggregate_device: str | None = "roomtone"
    # Fold the M8 to mono before it goes out. The Listener models one path from
    # "what we played" to the mic; two speakers playing different signals (the
    # M8's stereo effects make left and right nearly uncorrelated) cannot be
    # subtracted with one response. Measured on the laptop rig: 2 dB of echo
    # suppression in stereo, 20 dB in mono.
    mono_output: bool = True
    midi_out_port: str | None = "M8"
    log_dir: str = "logs"
