"""Render a talking-head video from one portrait and a speech recording."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from . import LipsyncError, audio, weights
from .detect import detect_face, load_detector
from .wav2lip import FACE_SIZE, face_input, load_generator

log = logging.getLogger(__name__)

_MOTION_SAMPLE = 256  # frames, spread over the clip, used to find what the generator animates
_MOTION_THRESHOLD = 25  # of 255: a pixel moves if its colour swings by more than a tenth of full range
_MOTION_MARGIN = 8  # pixels at generator resolution
_MOTION_FEATHER = 3.0


@dataclass(frozen=True)
class RenderInfo:
    box: tuple[int, int, int, int]
    fps: float
    frames: int
    width: int
    height: int


def load_image(path: Path) -> np.ndarray:
    """The portrait as BGR pixels, cropped by at most one row and column to the even size yuv420p needs."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise LipsyncError(f"cannot read image {path}")
    return image[: image.shape[0] // 2 * 2, : image.shape[1] // 2 * 2]


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def box_mask(height: int, width: int) -> np.ndarray:
    """(height, width) upper bound on the generated face's weight over the original, for a face box of that size.

    The generator only invents the lower half of the box, so the weight is zero above it, and it fades
    to zero at the box edges so that no seam can show there.
    """
    y = np.arange(height, dtype=np.float32)
    x = np.arange(width, dtype=np.float32)
    rows = _smoothstep((y - 0.3 * height) / (0.2 * height)) * _smoothstep((height - 1 - y) / (0.08 * height))
    cols = _smoothstep(x / (0.1 * width)) * _smoothstep((width - 1 - x) / (0.1 * width))
    return np.outer(rows, cols)


def motion_mask(generated: np.ndarray) -> np.ndarray:
    """(96, 96) weight in [0, 1] over what the generator animates across `generated`, (frames, 96, 96, 3) uint8.

    Only the mouth and chin really move. Everywhere else the output is a soft copy of the portrait,
    or jitter along hard edges such as a collar, and the portrait's own pixels look better. The mask
    is the largest moving region, grown by a margin so its feathered edge falls on still skin.
    """
    spread = np.ptp(generated, axis=0).max(axis=-1)
    count, labels = cv2.connectedComponents((spread > _MOTION_THRESHOLD).astype(np.uint8))
    if count == 1:
        return np.zeros(spread.shape, dtype=np.float32)
    main = max(range(1, count), key=lambda label: int(spread[labels == label].sum()))
    margin = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * _MOTION_MARGIN + 1, 2 * _MOTION_MARGIN + 1))
    return cv2.GaussianBlur(cv2.dilate((labels == main).astype(np.float32), margin), (0, 0), _MOTION_FEATHER)


class _VideoWriter:
    """Pipes BGR frames to ffmpeg, muxing them with the audio; the file only appears at `path` on success."""

    def __init__(self, path: Path, info: RenderInfo, audio_path: Path):
        self._path = Path(path)
        self._partial = self._path.with_name(f"{self._path.stem}.partial{self._path.suffix}")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr = tempfile.TemporaryFile()
        command = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-video_size", f"{info.width}x{info.height}",
            "-framerate", str(info.fps), "-i", "pipe:0",
            "-i", audio.ffmpeg_path(audio_path),
            "-map", "0:v:0", "-map", "1:a:0",
            # Without an explicit matrix ffmpeg converts RGB with BT.601 while players assume BT.709 for HD.
            "-vf", "scale=out_color_matrix=bt709:out_range=tv,format=yuv420p",
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709", "-color_range", "tv",
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-metadata", f"comment={json.dumps(asdict(info))}",
            "-movflags", "+faststart",
            audio.ffmpeg_path(self._partial),
        ]
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self._stderr)

    def write(self, frame: np.ndarray) -> None:
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            raise LipsyncError(f"ffmpeg stopped while encoding {self._path}: {self._close()[1]}") from None

    def _close(self) -> tuple[int, str]:
        """Wait for ffmpeg to exit; returns its exit code and error output."""
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        code = self._process.wait()
        self._stderr.seek(0)
        errors = self._stderr.read().decode(errors="replace").strip()
        self._stderr.close()
        return code, errors

    def __enter__(self) -> _VideoWriter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is not None:
            if self._process.poll() is None:
                self._process.kill()
                self._close()
            self._partial.unlink(missing_ok=True)
            return
        code, errors = self._close()
        try:
            if code != 0:
                raise LipsyncError(f"ffmpeg failed while encoding {self._path}: {errors}")
            self._partial.replace(self._path)
        except BaseException:
            self._partial.unlink(missing_ok=True)
            raise


def render(
    face_path: Path,
    audio_path: Path,
    out_path: Path,
    *,
    fps: float = 25.0,
    pads: tuple[int, int, int, int] = (0, 10, 0, 0),
    batch_size: int = 64,
    models_dir: Path = weights.DEFAULT_DIR,
    device: str | None = None,
) -> RenderInfo:
    """Write `out_path`: the portrait at `face_path` speaking the recording at `audio_path`.

    `pads` grows the detected face box by (top, bottom, left, right) pixels; the bottom padding keeps
    the chin inside the region the generator redraws.
    """
    if Path(out_path).suffix.lower() != ".mp4":
        raise LipsyncError(f"output must be an .mp4 file, not {out_path}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    image = load_image(face_path)
    height, width = image.shape[:2]

    wav = audio.decode(audio_path)
    mels = torch.from_numpy(audio.frame_windows(audio.at_speech_level(wav), fps)).unsqueeze(1)

    detector = load_detector(weights.ensure("s3fd", models_dir), device)
    x1, y1, x2, y2 = detect_face(image, detector, device)
    del detector
    top, bottom, left, right = pads
    x1, y1, x2, y2 = max(0, x1 - left), max(0, y1 - top), min(width, x2 + right), min(height, y2 + bottom)
    info = RenderInfo(box=(x1, y1, x2, y2), fps=fps, frames=len(mels), width=width, height=height)
    log.info("face box %s in %dx%d image, %d frames at %g fps on %s", info.box, width, height, info.frames, fps, device)

    generator = load_generator(weights.ensure("wav2lip_gan", models_dir), device)
    face = image[y1:y2, x1:x2]
    box_size = (x2 - x1, y2 - y1)
    with torch.inference_mode():
        features = generator.encode_face(face_input(cv2.resize(face, (FACE_SIZE, FACE_SIZE))).to(device))

        def generate(windows: torch.Tensor) -> np.ndarray:
            """(len(windows), 96, 96, 3) uint8 BGR faces speaking the given mel windows."""
            faces = [generator.decode(windows[start : start + batch_size].to(device), features) for start in range(0, len(windows), batch_size)]
            return (torch.cat(faces).permute(0, 2, 3, 1) * 255).round().byte().cpu().numpy()

        sample = torch.linspace(0, info.frames - 1, min(info.frames, _MOTION_SAMPLE)).long()
        mask = box_mask(*box_size[::-1]) * cv2.resize(motion_mask(generate(mels[sample])), box_size)
        mask = mask[..., None]
        background = face.astype(np.float32) * (1 - mask)
        frame = image.copy()
        with _VideoWriter(out_path, info, audio_path) as writer:
            for start in range(0, info.frames, batch_size):
                for small in generate(mels[start : start + batch_size]):
                    patch = cv2.resize(small, box_size, interpolation=cv2.INTER_CUBIC)
                    frame[y1:y2, x1:x2] = (patch * mask + background + 0.5).astype(np.uint8)
                    writer.write(frame)
                log.info("rendered %d/%d frames", min(start + batch_size, info.frames), info.frames)
    return info
