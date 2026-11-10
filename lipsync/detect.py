"""S3FD face detector (Zhang et al., 2017), laid out so the face-alignment weights load unchanged."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import LipsyncError

_CHANNEL_MEANS = np.array([104, 117, 123], dtype=np.float32)
_MAX_SIDE = 1920
# S3FD's largest anchor is 512 px: as a face approaches that size its box shrinks towards the nose, so a
# close-up is detected again at a scale where the face is comfortably inside the anchor range.
_MAX_FACE = 400
_REDETECT_FACE = 256
_CANDIDATE_SCORE = 0.05
_FACE_SCORE = 0.5
_NMS_OVERLAP = 0.3
_CENTER_VARIANCE, _SIZE_VARIANCE = 0.1, 0.2

_BACKBONE = [
    ("conv1_1", 3, 64), ("conv1_2", 64, 64),
    ("conv2_1", 64, 128), ("conv2_2", 128, 128),
    ("conv3_1", 128, 256), ("conv3_2", 256, 256), ("conv3_3", 256, 256),
    ("conv4_1", 256, 512), ("conv4_2", 512, 512), ("conv4_3", 512, 512),
    ("conv5_1", 512, 512), ("conv5_2", 512, 512), ("conv5_3", 512, 512),
]
_HEADS = [("conv3_3_norm", 256), ("conv4_3_norm", 512), ("conv5_3_norm", 512), ("fc7", 1024), ("conv6_2", 512), ("conv7_2", 256)]


class FaceNotFound(LipsyncError):
    pass


class _L2Norm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-10
        return x / norm * self.weight.view(1, -1, 1, 1)


class S3FD(nn.Module):
    def __init__(self):
        super().__init__()
        for name, cin, cout in _BACKBONE:
            setattr(self, name, nn.Conv2d(cin, cout, 3, 1, 1))
        self.fc6 = nn.Conv2d(512, 1024, 3, 1, 3)
        self.fc7 = nn.Conv2d(1024, 1024, 1, 1, 0)
        self.conv6_1 = nn.Conv2d(1024, 256, 1, 1, 0)
        self.conv6_2 = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv7_1 = nn.Conv2d(512, 128, 1, 1, 0)
        self.conv7_2 = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv3_3_norm = _L2Norm(256)
        self.conv4_3_norm = _L2Norm(512)
        self.conv5_3_norm = _L2Norm(512)
        for index, (name, channels) in enumerate(_HEADS):
            setattr(self, f"{name}_mbox_conf", nn.Conv2d(channels, 4 if index == 0 else 2, 3, 1, 1))
            setattr(self, f"{name}_mbox_loc", nn.Conv2d(channels, 4, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per pyramid level (strides 4 to 128): face/background logits (B, 2, H, W) and box offsets (B, 4, H, W)."""
        sources = {}
        for stage, convs in enumerate((2, 2, 3, 3, 3), start=1):
            for index in range(1, convs + 1):
                x = F.relu(getattr(self, f"conv{stage}_{index}")(x))
            if stage >= 3:
                sources[f"conv{stage}_3_norm"] = getattr(self, f"conv{stage}_3_norm")(x)
            x = F.max_pool2d(x, 2, 2)
        x = sources["fc7"] = F.relu(self.fc7(F.relu(self.fc6(x))))
        x = sources["conv6_2"] = F.relu(self.conv6_2(F.relu(self.conv6_1(x))))
        sources["conv7_2"] = F.relu(self.conv7_2(F.relu(self.conv7_1(x))))

        outputs = []
        for index, (name, _) in enumerate(_HEADS):
            logits = getattr(self, f"{name}_mbox_conf")(sources[name])
            if index == 0:
                # Max-out background label: the finest level scores three background hypotheses.
                logits = torch.cat((logits[:, :3].amax(dim=1, keepdim=True), logits[:, 3:]), dim=1)
            outputs.append((logits, getattr(self, f"{name}_mbox_loc")(sources[name])))
        return outputs


def load_detector(checkpoint: Path, device: str) -> S3FD:
    model = S3FD()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def nms(boxes: np.ndarray, overlap: float) -> list[int]:
    """Indices of (N, 5) [x1, y1, x2, y2, score] rows kept by greedy non-maximum suppression."""
    x1, y1, x2, y2, scores = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        best, rest = order[0], order[1:]
        keep.append(int(best))
        width = np.maximum(0, np.minimum(x2[best], x2[rest]) - np.maximum(x1[best], x1[rest]) + 1)
        height = np.maximum(0, np.minimum(y2[best], y2[rest]) - np.maximum(y1[best], y1[rest]) + 1)
        intersection = width * height
        order = rest[intersection / (areas[best] + areas[rest] - intersection) <= overlap]
    return keep


@torch.inference_mode()
def _faces(image_bgr: np.ndarray, model: S3FD, device: str) -> np.ndarray:
    """(N, 5) confident detections [x1, y1, x2, y2, score] in the pixels of the given image."""
    # The Wav2Lip training crops came from this detector fed RGB-ordered pixels, so the boxes it was
    # trained on are reproduced by doing the same rather than the BGR the Caffe means imply.
    pixels = image_bgr[..., ::-1].astype(np.float32) - _CHANNEL_MEANS
    batch = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device)

    candidates = []
    for level, (logits, offsets) in enumerate(model(batch)):
        stride = 4 * 2**level
        anchor = 4 * stride
        scores = logits.softmax(dim=1)[0, 1]
        ys, xs = torch.nonzero(scores > _CANDIDATE_SCORE, as_tuple=True)
        offsets = offsets[0, :, ys, xs].T
        centers = torch.stack((xs, ys), dim=1) * stride + stride / 2 + offsets[:, :2] * _CENTER_VARIANCE * anchor
        sizes = anchor * torch.exp(offsets[:, 2:] * _SIZE_VARIANCE)
        candidates.append(torch.cat((centers - sizes / 2, centers + sizes / 2, scores[ys, xs, None]), dim=1))
    candidates = torch.cat(candidates).cpu().numpy()

    faces = candidates[nms(candidates, _NMS_OVERLAP)]
    return faces[faces[:, 4] > _FACE_SCORE]


def detect_face(image_bgr: np.ndarray, model: S3FD, device: str) -> tuple[int, int, int, int]:
    """Box (x1, y1, x2, y2) of the largest confidently detected face."""
    scale = min(1.0, _MAX_SIDE / max(image_bgr.shape[:2]))
    while True:
        resized = image_bgr if scale == 1 else cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        faces = _faces(resized, model, device)
        if len(faces) == 0:
            raise FaceNotFound("no face detected in the image")
        largest = faces[np.argmax((faces[:, 2] - faces[:, 0]) * (faces[:, 3] - faces[:, 1]))]
        side = max(largest[2] - largest[0], largest[3] - largest[1])
        if side <= _MAX_FACE:
            x1, y1, x2, y2 = (np.clip(largest[:4], 0, None) / scale).astype(int)
            return int(x1), int(y1), int(x2), int(y2)
        scale *= _REDETECT_FACE / side
