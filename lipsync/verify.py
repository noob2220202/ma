"""Check a rendered video against its inputs: container, untouched background, and lip/audio timing."""

from __future__ import annotations

import json
import math
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

_BACKGROUND_BLOCK = 16
_BACKGROUND_MAX_DIFFERENCE = 10.0  # grey levels between block means
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
    command = ["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", "-i", audio.ffmpeg_path(path)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise LipsyncError(f"cannot probe {path}: {result.stderr.strip()}")
    return json.loads(result.stdout)


def _render_info(probe: dict, path: Path) -> RenderInfo:
    try:
        fields = json.loads(probe["format"]["tags"]["comment"])
        box = tuple(int(value) for value in fields["box"])
        info = RenderInfo(box, float(fields["fps"]), int(fields["frames"]), int(fields["width"]), int(fields["height"]))
        x1, y1, x2, y2 = box
        if not (0 <= x1 < x2 <= info.width and 0 <= y1 < y2 <= info.height and 0 < info.fps < math.inf):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise LipsyncError(f"{path} carries no usable render metadata; was it produced by lipsync render?") from None
    return info


def _decoded_frames(path: Path, width: int, height: int) -> Iterator[np.ndarray]:
    command = ["ffmpeg", "-v", "error", "-nostdin", "-i", audio.ffmpeg_path(path), "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    with subprocess.Popen(command, stdout=subprocess.PIPE) as process:
        size = width * height * 3
        while len(buffer := process.stdout.read(size)) == size:
            yield np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)


def _block_means(image: np.ndarray) -> np.ndarray:
    scale = 1 / _BACKGROUND_BLOCK
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA).astype(np.float32)


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
    checks = []

    described = f"{video['codec_name']} {video['pix_fmt']} {video['width']}x{video['height']} at {video['r_frame_rate']} fps"
    expected = (video["codec_name"], video["pix_fmt"], video["width"], video["height"]) == ("h264", "yuv420p", width, height)
    checks.append(Check("video stream", expected, described))

    # The muxed track is judged as decoded, which is what a player presents.
    muxed = audio.decode(video_path) if sound is not None else None
    if muxed is None:
        checks.append(Check("audio stream", False, "missing"))
    else:
        seconds, wanted_seconds = len(muxed) / audio.SAMPLE_RATE, len(recording) / audio.SAMPLE_RATE
        close = abs(seconds - wanted_seconds) <= _AUDIO_DURATION_TOLERANCE_S
        checks.append(Check("audio stream", close, f"{sound['codec_name']} {seconds:.2f} s (recording is {wanted_seconds:.2f} s)"))
    if not expected:
        return checks

    # Block means rather than pixels: encoder noise on a grainy portrait averages out, while a misplaced
    # paste or a colour error moves whole blocks.
    x1, y1, x2, y2 = info.box
    in_box = np.zeros((height, width), dtype=np.float32)
    in_box[y1:y2, x1:x2] = 1
    outside = _block_means(in_box) == 0
    reference = _block_means(image)

    lower_faces, worst_difference = [], 0.0
    for index, frame in enumerate(_decoded_frames(video_path, width, height)):
        lower_faces.append(cv2.resize(frame[y1:y2, x1:x2], (FACE_SIZE, FACE_SIZE))[FACE_SIZE // 2 :])
        if index % _BACKGROUND_SAMPLE_EVERY == 0 and outside.any():
            worst_difference = max(worst_difference, float(np.abs(_block_means(frame) - reference)[outside].max()))

    wanted = audio.frame_count(recording, info.fps)
    count = len(lower_faces)
    checks.append(Check("frame count", count == wanted == info.frames, f"{count} decoded, {wanted} needed to cover the recording"))
    if count == 0:
        return checks
    if outside.any():
        detail = f"largest change outside the face box {worst_difference:.1f} of 255 (means of {_BACKGROUND_BLOCK}x{_BACKGROUND_BLOCK} blocks)"
        checks.append(Check("background fidelity", worst_difference <= _BACKGROUND_MAX_DIFFERENCE, detail))
    else:
        checks.append(Check("background fidelity", None, "the face box fills the image"))
    if muxed is not None:
        expert = load_sync_expert(weights.ensure("sync_expert", models_dir), device)
        checks.append(_sync_check(np.array(lower_faces), audio.at_speech_level(muxed), info.fps, expert, device))
    return checks
