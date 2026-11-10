"""Pinned model weights, fetched on first use and checked against their SHA-256."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_DIR = Path(os.environ.get("LIPSYNC_MODELS", "models"))


@dataclass(frozen=True)
class Weight:
    filename: str
    url: str
    sha256: str


WEIGHTS = {
    "wav2lip_gan": Weight(
        "wav2lip_gan.pth",
        "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth",
        "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8",
    ),
    "sync_expert": Weight(
        "lipsync_expert.pth",
        "https://huggingface.co/Nekochu/Wav2Lip/resolve/main/lipsync_expert.pth",
        "9b9936c721696446eeed353032cab242a8cf0eed8c46cde540366f6ae5493be5",
    ),
    "s3fd": Weight(
        "s3fd.pth",
        "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
        "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, path: Path) -> None:
    log.info("downloading %s", url)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as file:
        shutil.copyfileobj(response, file, 1 << 20)
    partial.replace(path)


def ensure(name: str, models_dir: Path = DEFAULT_DIR) -> Path:
    """Path to the named weight file, downloading it if absent; raises if its checksum is wrong."""
    weight = WEIGHTS[name]
    path = Path(models_dir) / weight.filename
    fresh = not path.exists()
    if fresh:
        _download(weight.url, path)
    digest = _sha256(path)
    if digest != weight.sha256:
        if fresh:
            path.unlink()
        raise RuntimeError(f"{path} has SHA-256 {digest}, expected {weight.sha256}")
    return path
