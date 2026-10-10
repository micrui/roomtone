"""Per-block features of the room sound, measured on the residual.

The residual is what the mic hears after roomtone has subtracted its own music
(spec section 6). Everything here is plain signal processing on one block of it,
usually 30 s, with numpy and scipy only. Each measurement is cheap (filters,
FFTs, a median filter), so a 30 s block at 44.1 kHz takes a small fraction of a
second.

What each feature means and how it is measured:

rms_db
    Loudness of the whole block: RMS in dB relative to full scale, floored at
    -90 dBFS so silence has a finite value.

speech (0..1)
    Two pieces of evidence, multiplied together and softened:

    1. Band share: the fraction of the block's energy that falls between 300 Hz
       and 3000 Hz, where voices carry most of their intelligible energy. A
       share near 1 means "everything is in the voice band"; broadband noise
       (fans, traffic, white noise) puts only a small share there.
    2. Syllable rhythm: speech switches on and off a few times per second as
       syllables come and go. We take the loudness envelope of the voice band
       (RMS every 10 ms), look at its spectrum, and measure how much it
       fluctuates at 3 to 8 Hz. That is reported as a modulation depth: 1.0
       means the envelope swings fully on and off at syllable rate, 0 means
       steady. A single clear talker sits around 0.5 to 1.

    The band share goes through a smooth S-curve between 0.25 and 0.75. The
    syllable depth goes through a saturating curve, 1 - exp(-depth / 0.3).
    speech = band_curve * (0.35 + 0.65 * syllable_curve), so steady noise
    confined to the voice band still scores up to 0.35 (that is what a crowd
    sounds like: many voices smear the syllable rhythm while the band stays
    full), and a clear modulated voice scores close to 1. Finally the score
    fades to 0 for very quiet blocks (below -75 dBFS, fully on above
    -60 dBFS), so a near-silent noise floor never reads as talk.

talkers (continuous, rough)
    One voice has deep syllable modulation. Add independent voices and their
    syllables overlap, so the combined envelope gets smoother: the depth falls
    roughly as 1/sqrt(number of voices) while the band energy stays high. We
    combine two depth measures (syllable-rate depth, and overall envelope
    variation from 0.5 to 12 Hz), take their geometric mean, and invert:
    talkers = (0.7 / depth) ** 2, clamped to 1..8, where 0.7 is the depth we
    treat as one typical talker. Reported as 0 when speech < 0.2.

transient_rate (onsets per second)
    Spectral-flux onset detector. Short-time spectra (1024-sample frames, a
    new frame every 256 samples), log-compressed; for each frame, sum how much
    every frequency got louder than the frame before (increases only). Sharp
    sounds (a knife on a board, a dropped lid, a door) make big jumps. A frame
    counts as an onset when its flux is a local peak, exceeds 1.5 times the
    median flux of the surrounding half second (so steady noise never
    triggers), and is at least 50 ms after the previous onset.

beat (0..1)
    The flux curve from the onset detector, lightly smoothed and with its
    average removed, is correlated with delayed copies of itself. If sounds
    repeat at a steady tempo, the copy delayed by one beat lines up with the
    original. beat is the best alignment found for delays between 1/3 s and
    1 s (180 down to 60 BPM), relative to the zero-delay alignment, clipped to
    0..1. A metronome scores high; random clatter scores near 0.

speech_band_centroid_hz
    The energy-weighted average frequency inside 300 to 3000 Hz: where in the
    voice band the room's sound is centred. The Composer uses it to move the
    music's notes away from the voices.

labels
    If a sound-event classifier (see roomtone.classify) is supplied, its top
    labels for the block, unchanged.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sfft
from scipy import signal
from scipy.ndimage import median_filter

from roomtone.types import Features, db

if TYPE_CHECKING:
    from roomtone.classify import Classifier

log = logging.getLogger(__name__)

RMS_FLOOR_DB = -90.0

# speech band and syllable rhythm
SPEECH_LO_HZ = 300.0
SPEECH_HI_HZ = 3000.0
ENV_RATE_HZ = 100                  # voice-band envelope sample rate (10 ms RMS frames)
SYLLABLE_LO_HZ, SYLLABLE_HI_HZ = 3.0, 8.0
BROAD_LO_HZ, BROAD_HI_HZ = 0.5, 12.0
MIN_ENV_SECONDS = 2.0              # below this the modulation spectrum is too coarse
BAND_SHARE_LO, BAND_SHARE_HI = 0.25, 0.75
SYLLABLE_DEPTH_SCALE = 0.3
STEADY_BAND_WEIGHT = 0.35          # speech score of unmodulated voice-band sound
SPEECH_GATE_DB = (-75.0, -60.0)
ONE_TALKER_DEPTH = 0.7
MAX_TALKERS = 8.0
TALKERS_MIN_SPEECH = 0.2

# onsets and beat
N_FFT = 1024
HOP = 256
LOG_GAIN = 100.0                   # log1p(LOG_GAIN * |X|) compression
MEDIAN_SECONDS = 0.5
THRESHOLD_RATIO = 1.5
THRESHOLD_FLOOR_PER_BIN = 0.01     # absolute floor so a silent block never triggers
MIN_ONSET_GAP_S = 0.05
BEAT_BPM_LO, BEAT_BPM_HI = 60.0, 180.0
BEAT_SMOOTH_FRAMES = 5


def smoothstep(lo: float, hi: float, x: float) -> float:
    """0 below lo, 1 above hi, a smooth S-curve in between."""
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    return np.ravel(x)


def _band_share_and_centroid(mag: np.ndarray, sr: int) -> tuple[float, float]:
    """Fraction of energy in the voice band, and the energy-weighted centroid inside it.

    Uses the average power spectrum of the STFT frames already computed for the
    onset detector (43 Hz resolution, plenty for a 300..3000 Hz band).
    """
    if mag.size == 0:
        return 0.0, 0.0
    power = np.mean(mag.astype(np.float64) ** 2, axis=0)
    f = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    total = float(power.sum())
    band = (f >= SPEECH_LO_HZ) & (f <= SPEECH_HI_HZ)
    band_energy = float(power[band].sum())
    if total <= 0.0 or band_energy <= 0.0:
        return 0.0, 0.0
    centroid = float((f[band] * power[band]).sum() / band_energy)
    return band_energy / total, centroid


def _modulation_depths(x: np.ndarray, sr: int) -> tuple[float, float]:
    """Voice-band envelope modulation depth at syllable rate and over 0.5..12 Hz.

    Depth is the amplitude of the envelope fluctuation in the band divided by
    the envelope mean, so a sinusoidal on/off swing (1 + d*cos) reads as d.
    """
    hop = max(sr // ENV_RATE_HZ, 1)
    n = len(x) // hop
    if n < MIN_ENV_SECONDS * ENV_RATE_HZ or SPEECH_HI_HZ >= sr / 2:
        return 0.0, 0.0
    sos = signal.butter(4, [SPEECH_LO_HZ, SPEECH_HI_HZ], btype="bandpass", fs=sr, output="sos")
    y = signal.sosfilt(sos, x)
    env = np.sqrt(np.mean(y[: n * hop].reshape(n, hop) ** 2, axis=1))
    mean = float(env.mean())
    if mean <= 0.0:
        return 0.0, 0.0
    rate = sr / hop
    w = np.hanning(n)
    spec = np.abs(np.fft.rfft((env - mean) * w)) ** 2
    fm = np.fft.rfftfreq(n, 1.0 / rate)
    # one-sided band power, normalised so a cosine of amplitude a gives a**2 / 2
    scale = 2.0 / (n * float((w * w).sum()))

    def depth(lo: float, hi: float) -> float:
        power = scale * float(spec[(fm >= lo) & (fm <= hi)].sum())
        return float(np.sqrt(2.0 * power)) / mean

    return depth(SYLLABLE_LO_HZ, SYLLABLE_HI_HZ), depth(BROAD_LO_HZ, BROAD_HI_HZ)


def stft_magnitude(x: np.ndarray) -> np.ndarray:
    """|STFT| with a 1024-sample Hann window and a 256-sample hop, shape (frames, 513)."""
    if len(x) < N_FFT:
        return np.zeros((0, N_FFT // 2 + 1), dtype=np.float32)
    frames = sliding_window_view(x.astype(np.float32), N_FFT)[::HOP]
    window = np.hanning(N_FFT).astype(np.float32)
    return np.abs(sfft.rfft(frames * window, axis=1))


def onset_envelope(mag: np.ndarray) -> np.ndarray:
    """Half-wave rectified spectral flux of log magnitudes, one value per hop (sr / 256 per s)."""
    if mag.shape[0] < 2:
        return np.zeros(0)
    logmag = np.log1p(LOG_GAIN * mag)
    return np.maximum(np.diff(logmag, axis=0), 0.0).sum(axis=1).astype(np.float64)


def detect_onsets(flux: np.ndarray, sample_rate: int) -> np.ndarray:
    """Frame indices of onsets: local flux peaks above an adaptive median threshold."""
    if flux.size < 3:
        return np.zeros(0, dtype=int)
    fps = sample_rate / HOP
    size = max(int(round(MEDIAN_SECONDS * fps)) | 1, 3)
    floor = THRESHOLD_FLOOR_PER_BIN * (N_FFT // 2 + 1)
    threshold = THRESHOLD_RATIO * median_filter(flux, size=size, mode="nearest") + floor
    gap = max(int(np.ceil(MIN_ONSET_GAP_S * fps)), 1)
    peaks, _ = signal.find_peaks(flux, height=threshold, distance=gap)
    return peaks


def beat_strength(flux: np.ndarray, sample_rate: int) -> float:
    """Peak normalised autocorrelation of the onset envelope at 60..180 BPM lags."""
    fps = sample_rate / HOP
    lag_lo = int(np.floor(fps * 60.0 / BEAT_BPM_HI))
    lag_hi = int(np.ceil(fps * 60.0 / BEAT_BPM_LO))
    if flux.size <= lag_hi + 1:
        return 0.0
    kernel = np.hanning(BEAT_SMOOTH_FRAMES + 2)[1:-1]
    env = np.convolve(flux, kernel / kernel.sum(), mode="same")
    env = env - env.mean()
    n = env.size
    spec = np.fft.rfft(env, 2 * n)
    acf = np.fft.irfft(spec.real ** 2 + spec.imag ** 2)[:n]
    if acf[0] <= 1e-12:
        return 0.0
    return float(np.clip(acf[lag_lo : lag_hi + 1].max() / acf[0], 0.0, 1.0))


def extract(residual: np.ndarray, sample_rate: int,
            classifier: "Classifier | None" = None) -> Features:
    """Measure one block of residual room sound. See the module docstring."""
    x = _mono(residual)
    if x.size == 0:
        return Features()
    rms_db = db(float(np.sqrt(np.mean(x * x))), floor=RMS_FLOOR_DB)

    mag = stft_magnitude(x)
    share, centroid = _band_share_and_centroid(mag, sample_rate)
    syllable_depth, broad_depth = _modulation_depths(x, sample_rate)
    syllable = 1.0 - float(np.exp(-syllable_depth / SYLLABLE_DEPTH_SCALE))
    speech = (smoothstep(*SPEECH_GATE_DB, rms_db)
              * smoothstep(BAND_SHARE_LO, BAND_SHARE_HI, share)
              * (STEADY_BAND_WEIGHT + (1.0 - STEADY_BAND_WEIGHT) * syllable))

    talkers = 0.0
    if speech >= TALKERS_MIN_SPEECH:
        depth = max(float(np.sqrt(syllable_depth * broad_depth)), 1e-3)
        talkers = float(np.clip((ONE_TALKER_DEPTH / depth) ** 2, 1.0, MAX_TALKERS))

    flux = onset_envelope(mag)
    onsets = detect_onsets(flux, sample_rate)
    transient_rate = onsets.size / (x.size / sample_rate)
    beat = beat_strength(flux, sample_rate)

    labels = classifier.labels(x, sample_rate) if classifier is not None else {}

    feats = Features(rms_db=rms_db, speech=float(speech), talkers=talkers,
                     transient_rate=float(transient_rate), beat=beat,
                     speech_band_centroid_hz=centroid, labels=dict(labels))
    log.debug("features: rms %.1f dB speech %.2f (share %.2f depth %.2f/%.2f) talkers %.1f "
              "onsets %.2f/s beat %.2f centroid %.0f Hz", rms_db, speech, share,
              syllable_depth, broad_depth, talkers, transient_rate, beat, centroid)
    return feats
