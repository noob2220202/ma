"""Audio front end producing the mel features the Wav2Lip checkpoints were trained on."""

from __future__ import annotations

import math
import subprocess
from pathlib import Path

import numpy as np
import torch

from . import LipsyncError

SAMPLE_RATE = 16000
N_FFT = 800
HOP_LENGTH = 200
N_MELS = 80
FMIN, FMAX = 55.0, 7600.0
PREEMPHASIS = 0.97
REF_LEVEL_DB = 20.0
MIN_LEVEL_DB = -100.0
MAX_ABS_VALUE = 4.0
MEL_WINDOW = 16
MEL_FRAMES_PER_SECOND = SAMPLE_RATE / HOP_LENGTH
WINDOW_SECONDS = MEL_WINDOW / MEL_FRAMES_PER_SECOND
SPEECH_LEVEL = 0.07  # RMS, about -23 dBFS: the middle of the range where sync quality is flat
_MAX_LEVEL_GAIN = 10.0  # 20 dB, so room tone is not amplified into speech

_MEL_BREAK_HZ = 1000.0
_MEL_LINEAR_HZ = 200.0 / 3
_MEL_LOG_STEP = math.log(6.4) / 27


def ffmpeg_path(path: Path) -> str:
    """The path as ffmpeg must be given it, so that a colon or a leading dash in a file name is not read as syntax."""
    return f"file:{path}"


def decode(path: Path) -> np.ndarray:
    """Any ffmpeg-readable audio as mono float32 samples at SAMPLE_RATE, with sample 0 at the file's time zero."""
    # Raw output drops timestamps, so a track that starts after its file does (as a player would present
    # it, and as muxing it preserves) is padded with that much silence first.
    on_timeline = "aresample=async=1:first_pts=0"
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", ffmpeg_path(path), "-vn", "-af", on_timeline, "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"]
    result = subprocess.run(command, capture_output=True)
    if result.returncode != 0:
        raise LipsyncError(f"cannot decode audio {path}: {result.stderr.decode(errors='replace').strip()}")
    wav = np.frombuffer(result.stdout, dtype=np.float32)
    if wav.size == 0:
        raise LipsyncError(f"{path} contains no audio")
    return wav


def at_speech_level(wav: np.ndarray) -> np.ndarray:
    """`wav` scaled so that its loud stretches sit at SPEECH_LEVEL.

    The models read absolute level: for a recording 20 dB quieter than the speech they were trained on
    the generated mouth moves a quarter less, and at 30 dB about half.
    """
    windows = wav[: len(wav) // N_FFT * N_FFT].reshape(-1, N_FFT)
    loud = np.quantile(np.sqrt(np.square(windows).mean(axis=1)), 0.95) if len(windows) else 0.0
    if loud == 0:
        return wav
    return wav * np.float32(np.clip(SPEECH_LEVEL / loud, 1 / _MAX_LEVEL_GAIN, _MAX_LEVEL_GAIN))


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    log_region = _MEL_BREAK_HZ / _MEL_LINEAR_HZ + np.log(np.maximum(hz, _MEL_BREAK_HZ) / _MEL_BREAK_HZ) / _MEL_LOG_STEP
    return np.where(hz >= _MEL_BREAK_HZ, log_region, hz / _MEL_LINEAR_HZ)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    break_mel = _MEL_BREAK_HZ / _MEL_LINEAR_HZ
    log_region = _MEL_BREAK_HZ * np.exp(_MEL_LOG_STEP * (mel - break_mel))
    return np.where(mel >= break_mel, log_region, mel * _MEL_LINEAR_HZ)


def mel_filterbank() -> np.ndarray:
    """(N_MELS, N_FFT // 2 + 1) Slaney-scale, area-normalised triangular filters, as librosa.filters.mel builds them."""
    fft_hz = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    edges_hz = _mel_to_hz(np.linspace(_hz_to_mel(np.float64(FMIN)), _hz_to_mel(np.float64(FMAX)), N_MELS + 2))
    widths = np.diff(edges_hz)
    ramps = edges_hz[:, None] - fft_hz[None, :]
    rising = -ramps[:-2] / widths[:-1, None]
    falling = ramps[2:] / widths[1:, None]
    filters = np.maximum(0, np.minimum(rising, falling))
    filters *= (2 / (edges_hz[2:] - edges_hz[:-2]))[:, None]
    return filters.astype(np.float32)


def melspectrogram(wav: np.ndarray) -> np.ndarray:
    """(N_MELS, 1 + len(wav) // HOP_LENGTH) log-mel spectrogram scaled to [-MAX_ABS_VALUE, MAX_ABS_VALUE]."""
    emphasized = np.append(wav[:1], wav[1:] - PREEMPHASIS * wav[:-1]).astype(np.float32)
    spectrum = torch.stft(
        torch.from_numpy(emphasized),
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=torch.hann_window(N_FFT),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    mel = mel_filterbank() @ spectrum.abs().numpy()
    level_db = 20 * np.log10(np.maximum(10 ** (MIN_LEVEL_DB / 20), mel)) - REF_LEVEL_DB
    scaled = 2 * MAX_ABS_VALUE * (level_db - MIN_LEVEL_DB) / -MIN_LEVEL_DB - MAX_ABS_VALUE
    return np.clip(scaled, -MAX_ABS_VALUE, MAX_ABS_VALUE).astype(np.float32)


def frame_count(wav: np.ndarray, fps: float) -> int:
    """Video frames needed to cover the whole recording."""
    return math.ceil(len(wav) / SAMPLE_RATE * fps)


def mel_windows(wav: np.ndarray, starts: np.ndarray) -> np.ndarray:
    """(len(starts), N_MELS, MEL_WINDOW): the mel slice beginning at each time in seconds, silent beyond the recording."""
    first = np.floor(np.asarray(starts) * MEL_FRAMES_PER_SECOND + 1e-6).astype(int)
    lead = max(0, -int(first.min()))
    tail = max(0, int(first.max()) + MEL_WINDOW - len(wav) // HOP_LENGTH)
    padded = np.concatenate([np.zeros(lead * HOP_LENGTH, wav.dtype), wav, np.zeros(tail * HOP_LENGTH, wav.dtype)])
    mel = melspectrogram(padded)
    return np.stack([mel[:, start : start + MEL_WINDOW] for start in first + lead])


def frame_windows(wav: np.ndarray, fps: float) -> np.ndarray:
    """(frames, N_MELS, MEL_WINDOW) generator input: the mel slice centred on the middle of each video frame."""
    # The generator draws the mouth for the middle of its window. Wav2Lip's own inference script starts
    # each window at the frame's timestamp instead, which its sync expert scores as lips running two
    # frames (80 ms) ahead of the audio; centred windows score as in sync.
    middles = (np.arange(frame_count(wav, fps)) + 0.5) / fps
    return mel_windows(wav, middles - WINDOW_SECONDS / 2)
