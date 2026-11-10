"""Wav2Lip generator and lip-sync expert (Prajwal et al., 2020), laid out so the published checkpoints load unchanged."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

FACE_SIZE = 96
SYNC_FRAMES = 5


class _Conv(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(cin, cout, kernel_size, stride, padding), nn.BatchNorm2d(cout))
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out = out + x
        return self.act(out)


class _ConvTranspose(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding), nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.conv_block(x))


def _residual(channels):
    return _Conv(channels, channels, 3, 1, 1, residual=True)


def _down(cin, cout, residuals):
    return nn.Sequential(_Conv(cin, cout, 3, 2, 1), *(_residual(cout) for _ in range(residuals)))


def _up(cin, cout):
    return nn.Sequential(_ConvTranspose(cin, cout, 3, 2, 1, output_padding=1), _residual(cout), _residual(cout))


class Wav2Lip(nn.Module):
    def __init__(self):
        super().__init__()
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(_Conv(6, 16, 7, 1, 3)),  # 96
            _down(16, 32, residuals=2),  # 48
            _down(32, 64, residuals=3),  # 24
            _down(64, 128, residuals=2),  # 12
            _down(128, 256, residuals=2),  # 6
            _down(256, 512, residuals=1),  # 3
            nn.Sequential(_Conv(512, 512, 3, 1, 0), _Conv(512, 512, 1, 1, 0)),  # 1
        ])
        self.audio_encoder = nn.Sequential(
            _Conv(1, 32, 3, 1, 1), _residual(32), _residual(32),
            _Conv(32, 64, 3, (3, 1), 1), _residual(64), _residual(64),
            _Conv(64, 128, 3, 3, 1), _residual(128), _residual(128),
            _Conv(128, 256, 3, (3, 2), 1), _residual(256),
            _Conv(256, 512, 3, 1, 0), _Conv(512, 512, 1, 1, 0),
        )
        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(_Conv(512, 512, 1, 1, 0)),  # 1
            nn.Sequential(_ConvTranspose(1024, 512, 3, 1, 0), _residual(512)),  # 3
            _up(1024, 512),  # 6
            _up(768, 384),  # 12
            _up(512, 256),  # 24
            _up(320, 128),  # 48
            _up(160, 64),  # 96
        ])
        self.output_block = nn.Sequential(_Conv(80, 32, 3, 1, 1), nn.Conv2d(32, 3, 1, 1, 0), nn.Sigmoid())

    def encode_face(self, faces: torch.Tensor) -> list[torch.Tensor]:
        """Skip features for (B, 6, 96, 96) inputs built by face_input, coarsest last."""
        features = []
        x = faces
        for block in self.face_encoder_blocks:
            x = block(x)
            features.append(x)
        return features

    def decode(self, mels: torch.Tensor, face_features: list[torch.Tensor]) -> torch.Tensor:
        """(B, 3, 96, 96) faces in [0, 1] for (B, 1, 80, 16) mel windows; single-face features serve the whole batch."""
        x = self.audio_encoder(mels)
        for block, skip in zip(self.face_decoder_blocks, reversed(face_features)):
            x = torch.cat((block(x), skip.expand(len(x), -1, -1, -1)), dim=1)
        return self.output_block(x)

    def forward(self, mels: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        return self.decode(mels, self.encode_face(faces))


class SyncNet(nn.Module):
    """Wav2Lip's lip-sync expert: embeds 0.2 s of lower-face frames and of mel so that in-sync pairs align."""

    def __init__(self):
        super().__init__()
        self.face_encoder = nn.Sequential(
            _Conv(3 * SYNC_FRAMES, 32, 7, 1, 3),
            _Conv(32, 64, 5, (1, 2), 1), _residual(64), _residual(64),
            _Conv(64, 128, 3, 2, 1), _residual(128), _residual(128), _residual(128),
            _Conv(128, 256, 3, 2, 1), _residual(256), _residual(256),
            _Conv(256, 512, 3, 2, 1), _residual(512), _residual(512),
            _Conv(512, 512, 3, 2, 1), _Conv(512, 512, 3, 1, 0), _Conv(512, 512, 1, 1, 0),
        )
        self.audio_encoder = nn.Sequential(
            _Conv(1, 32, 3, 1, 1), _residual(32), _residual(32),
            _Conv(32, 64, 3, (3, 1), 1), _residual(64), _residual(64),
            _Conv(64, 128, 3, 3, 1), _residual(128), _residual(128),
            _Conv(128, 256, 3, (3, 2), 1), _residual(256), _residual(256),
            _Conv(256, 512, 3, 1, 0), _Conv(512, 512, 1, 1, 0),
        )

    def embed_faces(self, faces: torch.Tensor) -> torch.Tensor:
        """Unit vectors for (B, 15, 48, 96) stacks: the lower halves of SYNC_FRAMES consecutive BGR faces in [0, 1]."""
        return F.normalize(self.face_encoder(faces).flatten(1), dim=1)

    def embed_audio(self, mels: torch.Tensor) -> torch.Tensor:
        """Unit vectors for (B, 1, 80, 16) mel windows starting at the first frame of the matching face stack."""
        return F.normalize(self.audio_encoder(mels).flatten(1), dim=1)


def _load(model: nn.Module, checkpoint: Path, device: str) -> nn.Module:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})
    return model.to(device).eval()


def load_generator(checkpoint: Path, device: str) -> Wav2Lip:
    return _load(Wav2Lip(), checkpoint, device)


def load_sync_expert(checkpoint: Path, device: str) -> SyncNet:
    return _load(SyncNet(), checkpoint, device)


def face_input(face_bgr: np.ndarray) -> torch.Tensor:
    """(1, 6, 96, 96) generator input: the face with its lower half blanked, stacked on the intact face."""
    face = torch.from_numpy(face_bgr).permute(2, 0, 1).float() / 255
    masked = face.clone()
    masked[:, FACE_SIZE // 2 :] = 0
    return torch.cat((masked, face)).unsqueeze(0)
