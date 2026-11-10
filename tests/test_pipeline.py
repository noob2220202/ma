import subprocess

import numpy as np
import pytest

from lipsync import LipsyncError
from lipsync.cli import main
from lipsync.pipeline import RenderInfo, _VideoWriter, box_mask, motion_mask
from lipsync.verify import _decoded_frames, _probe, verify


def test_box_mask_leaves_box_edges_and_upper_face_original():
    mask = box_mask(200, 160)
    assert mask.shape == (200, 160)
    assert mask[:60].max() == 0
    assert mask[-1].max() == 0 and mask[:, 0].max() == 0 and mask[:, -1].max() == 0
    assert mask[100:180, 20:140].min() == 1


def test_motion_mask_covers_the_animated_region_only():
    rng = np.random.default_rng(0)
    still = rng.integers(60, 200, (96, 96, 3), dtype=np.uint8)
    frames = np.repeat(still[None], 8, axis=0)
    frames[::2, 60:76, 36:60] = 20  # a mouth-sized patch that opens and closes
    frames[1::2, 60:76, 36:60] = 220
    frames[::2, 90, 2:12] = 255  # a thin edge that jitters, like a collar
    frames[1::2, 90, 2:12] = 0

    mask = motion_mask(frames)
    assert mask.shape == (96, 96)
    assert mask[60:76, 36:60].min() > 0.99
    assert mask[:40].max() < 0.01
    assert mask[90, 2:12].max() < 0.01


def test_motion_mask_is_empty_when_nothing_moves():
    frames = np.full((4, 96, 96, 3), 128, dtype=np.uint8)
    assert motion_mask(frames).max() == 0


def test_writer_muxes_audio_and_preserves_colour(tmp_path):
    tone = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4", str(tone)], check=True)
    info = RenderInfo(box=(0, 0, 32, 32), fps=25.0, frames=10, width=64, height=48)
    frame = np.zeros((info.height, info.width, 3), dtype=np.uint8)
    frame[:, :32] = (40, 60, 220)  # saturated colours are where a wrong YUV matrix shows
    frame[:, 32:] = (200, 160, 30)
    out = tmp_path / "out.mp4"
    with _VideoWriter(out, info, tone) as writer:
        for _ in range(info.frames):
            writer.write(frame)

    assert not out.with_name("out.partial.mp4").exists()
    assert {stream["codec_type"] for stream in _probe(out)["streams"]} == {"video", "audio"}
    decoded = list(_decoded_frames(out, info.width, info.height))
    assert len(decoded) == info.frames
    assert np.abs(decoded[0].astype(int) - frame).mean() < 3


def test_cli_refuses_a_non_mp4_output_before_doing_any_work(tmp_path, capsys):
    assert main(["render", "face.png", "speech.wav", "-o", str(tmp_path / "out.mkv")]) == 1
    assert "must be an .mp4 file" in capsys.readouterr().err


def test_verify_rejects_a_video_it_did_not_render(tmp_path):
    foreign = tmp_path / "foreign.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=d=0.2:s=64x48", "-metadata", "comment=hello", str(foreign)], check=True)
    with pytest.raises(LipsyncError, match="no render metadata"):
        verify(foreign, tmp_path / "face.png", tmp_path / "speech.wav")


def test_writer_leaves_no_file_when_rendering_fails(tmp_path):
    tone = tmp_path / "tone.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=0.4", str(tone)], check=True)
    info = RenderInfo(box=(0, 0, 32, 32), fps=25.0, frames=10, width=64, height=48)
    out = tmp_path / "out.mp4"
    try:
        with _VideoWriter(out, info, tone) as writer:
            writer.write(np.zeros((info.height, info.width, 3), dtype=np.uint8))
            raise KeyboardInterrupt
    except KeyboardInterrupt:
        pass
    assert list(tmp_path.glob("out*")) == []
