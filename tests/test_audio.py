import subprocess

import numpy as np
import pytest

from lipsync import audio


def _speechlike(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(int(seconds * audio.SAMPLE_RATE)) / audio.SAMPLE_RATE
    voiced = 0.3 * np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t))
    return (voiced + 0.05 * rng.standard_normal(t.size)).astype(np.float32)


def test_decode_keeps_a_late_starting_track_on_the_file_timeline(tmp_path):
    clip = tmp_path / "late_audio.mp4"
    command = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=d=1:s=64x48:r=25",
               "-itsoffset", "0.5", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5", str(clip)]
    subprocess.run(command, check=True)
    wav = audio.decode(clip)
    assert abs(len(wav) / audio.SAMPLE_RATE - 1.0) < 0.1
    assert np.abs(wav[: int(0.4 * audio.SAMPLE_RATE)]).max() < 1e-3
    assert np.abs(wav[int(0.6 * audio.SAMPLE_RATE) : int(0.9 * audio.SAMPLE_RATE)]).max() > 0.05


def test_filterbank_matches_librosa():
    librosa = pytest.importorskip("librosa")
    expected = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600)
    np.testing.assert_allclose(audio.mel_filterbank(), expected, atol=1e-7)


def test_melspectrogram_matches_wav2lip_training_features():
    librosa = pytest.importorskip("librosa")
    signal = pytest.importorskip("scipy.signal")
    wav = _speechlike(2.0)
    # Wav2Lip's audio.py, spelled out against the current librosa API (reflect padding was its default then).
    stft = librosa.stft(y=signal.lfilter([1, -0.97], [1], wav), n_fft=800, hop_length=200, win_length=800, pad_mode="reflect")
    mel = librosa.filters.mel(sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600) @ np.abs(stft)
    level_db = 20 * np.log10(np.maximum(1e-5, mel)) - 20
    expected = np.clip(8 * (level_db + 100) / 100 - 4, -4, 4)
    np.testing.assert_allclose(audio.melspectrogram(wav), expected, atol=1e-3)


def test_mel_windows_begin_at_the_requested_times():
    wav = _speechlike(2.0)
    windows = audio.mel_windows(wav, np.array([0.0, 0.4, 1.0]))
    mel = audio.melspectrogram(wav)
    assert windows.shape == (3, audio.N_MELS, audio.MEL_WINDOW)
    for window, start in zip(windows, (0, 32, 80)):
        np.testing.assert_allclose(window, mel[:, start : start + 16], atol=1e-5)


def test_mel_windows_are_silent_beyond_the_recording():
    windows = audio.mel_windows(_speechlike(1.0), np.array([-1.0, 0.5, 2.0]))
    assert (windows[[0, 2]] == -audio.MAX_ABS_VALUE).all()
    assert windows[1].max() > 0


def test_frame_windows_cover_the_whole_recording():
    windows = audio.frame_windows(_speechlike(1.01), fps=25)
    assert windows.shape == (26, audio.N_MELS, audio.MEL_WINDOW)


def test_frame_window_is_centred_on_its_frame():
    wav = _speechlike(2.0)
    windows = audio.frame_windows(wav, fps=25)
    # Frame 10 spans 0.40-0.44 s; the 0.2 s around its middle starts at 0.32 s, which is mel frame 25.
    np.testing.assert_allclose(windows[10], audio.melspectrogram(wav)[:, 25:41], atol=1e-5)
