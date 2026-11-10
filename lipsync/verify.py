"""Check a rendered video against its inputs: container, untouched background, and lip/audio timing."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from . import LipsyncError, audio, weights
from .pipeline import RenderInfo, load_image
from .wav2lip import FACE_SIZE, SYNC_FRAMES, SyncNet, load_sync_expert

_BACKGROUND_PSNR_DB = 35.0
_BACKGROUND_SAMPLE_EVERY = 25
_AUDIO_DURATION_TOLERANCE_S = 0.1
_EXPERT_FPS = 25.0
_SYNC_SHIFTS = 6
# One-second stretches of an in-sync render peak 0.4 or more above the median; near-silent clips, whose
# best offset is noise, stay under 0.2.
_MIN_SYNC_CONFIDENCE = 0.3


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool | None  # None: could not be assessed
    detail: str


def _probe(path: Path) -> dict:
    command = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", "-i", str(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise LipsyncError(f"cannot probe {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _render_info(probe: dict, path: Path) -> RenderInfo:
    try:
        return RenderInfo(**json.loads(probe["format"]["tags"]["comment"]))
    except (KeyError, TypeError, ValueError):
        raise LipsyncError(f"{path} carries no render metadata; was it produced by lipsync render?") from None


def _decoded_frames(path: Path, width: int, height: int) -> Iterator[np.ndarray]:
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        size = width * height * 3
        while len(buffer := process.stdout.read(size)) == size:
            yield np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)


def _embed(encode, inputs: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
    return torch.cat([encode(inputs[start : start + batch_size].flatten(1, -3)) for start in range(0, len(inputs), batch_size)])


@torch.inference_mode()
def _sync_check(lower_faces: np.ndarray, wav: np.ndarray, fps: float, expert: SyncNet, device: str) -> Check:
    """Which audio offset does Wav2Lip's sync expert find the mouths best matched to?

    `lower_faces` is (frames, 48, 96, 3): the lower half of each frame's face box at the expert's
    resolution. Every run of SYNC_FRAMES frames is scored against the audio starting from -_SYNC_SHIFTS
    to +_SYNC_SHIFTS frames away; real, in-sync speech peaks at zero.
    """
    name = "lip/audio sync"
    windows = len(lower_faces) - SYNC_FRAMES + 1
    if fps != _EXPERT_FPS:
        return Check(name, None, f"the sync expert only judges {_EXPERT_FPS:g} fps video")
    if windows < 1:
        return Check(name, None, f"needs at least {SYNC_FRAMES} frames")

    frames = torch.from_numpy(lower_faces).permute(0, 3, 1, 2).float().div(255).to(device)
    stacks = frames.unfold(0, SYNC_FRAMES, 1).permute(0, 4, 1, 2, 3)
    starts = np.arange(-_SYNC_SHIFTS, windows + _SYNC_SHIFTS) / fps
    mels = torch.from_numpy(audio.mel_windows(wav, starts)).unsqueeze(1).to(device)
    faces, sounds = _embed(expert.embed_faces, stacks), _embed(expert.embed_audio, mels)

    shifts = range(-_SYNC_SHIFTS, _SYNC_SHIFTS + 1)
    similarity = {shift: float((faces * sounds[shift + _SYNC_SHIFTS :][:windows]).sum(dim=1).mean()) for shift in shifts}
    best = max(similarity, key=similarity.get)
    confidence = similarity[best] - float(np.median(list(similarity.values())))
    frames_off = f"{abs(best)} frame{'s' if abs(best) > 1 else ''}"
    timing = "in sync" if best == 0 else f"lips {frames_off} {'ahead of' if best > 0 else 'behind'} the audio"
    detail = f"{timing}; similarity {similarity[best]:.2f}, {confidence:.2f} above the median over ±{_SYNC_SHIFTS} frames"
    if confidence < _MIN_SYNC_CONFIDENCE:
        return Check(name, None, f"no clear best offset ({detail})")
    return Check(name, best == 0, detail)


def verify(
    video_path: Path, face_path: Path, audio_path: Path, *, models_dir: Path = weights.DEFAULT_DIR, device: str | None = None
) -> list[Check]:
    """Checks of the render at `video_path` against the portrait and recording it was made from."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    probe = _probe(video_path)
    info = _render_info(probe, video_path)
    streams = {stream["codec_type"]: stream for stream in reversed(probe["streams"])}
    video, sound = streams.get("video"), streams.get("audio")
    if video is None:
        raise LipsyncError(f"{video_path} has no video stream")

    image = load_image(face_path)
    height, width = image.shape[:2]
    recording = audio.decode(audio_path)
    seconds = len(recording) / audio.SAMPLE_RATE
    checks = []

    described = f"{video['codec_name']} {video['pix_fmt']} {video['width']}x{video['height']} at {video['r_frame_rate']} fps"
    expected = (video["codec_name"], video["pix_fmt"], video["width"], video["height"]) == ("h264", "yuv420p", width, height)
    checks.append(Check("video stream", expected, described))

    if sound is None:
        checks.append(Check("audio stream", False, "missing"))
    else:
        duration = float(sound["duration"])
        close = abs(duration - seconds) <= _AUDIO_DURATION_TOLERANCE_S
        checks.append(Check("audio stream", close, f"{sound['codec_name']} {duration:.2f} s (recording is {seconds:.2f} s)"))
    if not expected:
        return checks

    x1, y1, x2, y2 = info.box
    outside = np.ones((height, width, 1), dtype=bool)
    outside[y1:y2, x1:x2] = False
    reference = image.astype(np.float32)

    lower_faces, worst_psnr = [], np.inf
    for index, frame in enumerate(_decoded_frames(video_path, width, height)):
        lower_faces.append(cv2.resize(frame[y1:y2, x1:x2], (FACE_SIZE, FACE_SIZE))[FACE_SIZE // 2 :])
        if index % _BACKGROUND_SAMPLE_EVERY == 0:
            error = np.square(frame - reference, where=outside, out=np.zeros_like(reference)).sum() / (outside.sum() * 3)
            worst_psnr = min(worst_psnr, 10 * np.log10(255**2 / max(error, 1e-12)))

    wanted = audio.frame_count(recording, info.fps)
    count = len(lower_faces)
    checks.append(Check("frame count", count == wanted == info.frames, f"{count} decoded, {wanted} needed to cover the recording"))
    if count == 0:
        return checks
    checks.append(Check("background fidelity", worst_psnr >= _BACKGROUND_PSNR_DB, f"worst PSNR outside the face box {worst_psnr:.1f} dB"))
    if sound is not None:
        # Timing is judged against the muxed track, which is what a player presents.
        expert = load_sync_expert(weights.ensure("sync_expert", models_dir), device)
        checks.append(_sync_check(np.array(lower_faces), audio.decode(video_path), info.fps, expert, device))
    return checks
