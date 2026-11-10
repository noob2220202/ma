import numpy as np
import pytest
import torch

from lipsync import weights
from lipsync.detect import S3FD, load_detector, nms
from lipsync.wav2lip import FACE_SIZE, SYNC_FRAMES, SyncNet, Wav2Lip, face_input, load_generator, load_sync_expert


def _needs_weight(name):
    path = weights.DEFAULT_DIR / weights.WEIGHTS[name].filename
    return pytest.mark.skipif(not path.exists(), reason=f"{path} not downloaded")


def test_face_input_blanks_the_lower_half_of_the_first_copy():
    stacked = face_input(np.full((FACE_SIZE, FACE_SIZE, 3), 255, dtype=np.uint8))
    assert stacked.shape == (1, 6, FACE_SIZE, FACE_SIZE)
    assert stacked[0, :3, : FACE_SIZE // 2].min() == 1
    assert stacked[0, :3, FACE_SIZE // 2 :].max() == 0
    assert stacked[0, 3:].min() == 1


def test_shared_face_features_match_per_frame_encoding():
    torch.manual_seed(0)
    model = Wav2Lip().eval()
    face = face_input(np.random.default_rng(0).integers(0, 256, (FACE_SIZE, FACE_SIZE, 3), dtype=np.uint8))
    mels = torch.randn(3, 1, 80, 16)
    with torch.inference_mode():
        shared = model.decode(mels, model.encode_face(face))
        per_frame = model(mels, face.expand(3, -1, -1, -1))
    assert shared.shape == (3, 3, FACE_SIZE, FACE_SIZE)
    torch.testing.assert_close(shared, per_frame, atol=1e-5, rtol=0)


def test_s3fd_pyramid_levels():
    with torch.inference_mode():
        outputs = S3FD().eval()(torch.zeros(1, 3, 256, 256))
    # fc6 pads by 3 with a 3x3 kernel, so the map grows from 8 to 12 at the fourth level, as in the trained network.
    sizes = (64, 32, 16, 12, 6, 3)
    assert [tuple(logits.shape) for logits, _ in outputs] == [(1, 2, size, size) for size in sizes]
    assert [tuple(offsets.shape) for _, offsets in outputs] == [(1, 4, size, size) for size in sizes]


def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = np.array([[0, 0, 10, 10, 0.8], [1, 1, 11, 11, 0.9], [50, 50, 60, 60, 0.7]], dtype=np.float32)
    assert nms(boxes, 0.3) == [1, 2]


def test_sync_expert_embeds_faces_and_audio_as_unit_vectors():
    expert = SyncNet().eval()
    with torch.inference_mode():
        faces = expert.embed_faces(torch.rand(2, 3 * SYNC_FRAMES, FACE_SIZE // 2, FACE_SIZE))
        sounds = expert.embed_audio(torch.randn(2, 1, 80, 16))
    assert faces.shape == sounds.shape == (2, 512)
    torch.testing.assert_close(faces.norm(dim=1), torch.ones(2))
    torch.testing.assert_close(sounds.norm(dim=1), torch.ones(2))


@_needs_weight("wav2lip_gan")
def test_published_generator_checkpoint_loads():
    load_generator(weights.ensure("wav2lip_gan"), "cpu")


@_needs_weight("sync_expert")
def test_published_sync_expert_checkpoint_loads():
    load_sync_expert(weights.ensure("sync_expert"), "cpu")


@_needs_weight("s3fd")
def test_published_detector_checkpoint_loads():
    load_detector(weights.ensure("s3fd"), "cpu")
