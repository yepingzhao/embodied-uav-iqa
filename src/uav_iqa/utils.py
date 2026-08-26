import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


_THIRD_PARTY_LOGGERS = (
    "transformers",
    "huggingface_hub",
    "datasets",
    "diffusers",
    "accelerate",
    "peft",
    "safetensors",
    "filelock",
    "torch.distributed",
)


def setup_logging(name: str = None, level: int = logging.INFO):
    """Configure stdlib logging with uniform format for scripts.

    Suppresses verbose INFO logs from third-party libraries (transformers,
    huggingface_hub, etc.) while keeping application logs at the requested
    level.  Returns a logger instance.  Call once at module top-level::

        _log = setup_logging(__name__)
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Silence noisy third-party loggers below WARNING so they don't flood
    # the terminal during model loading / dataset streaming.
    for name_ in _THIRD_PARTY_LOGGERS:
        logging.getLogger(name_).setLevel(logging.WARNING)

    return logging.getLogger(name)


def find_images(
    image_dir: Path,
    exts: Optional[set] = None,
    exclude_dirs: Optional[set] = None,
) -> list:
    """Find all image files in a directory tree, optionally excluding subdirs.

    Args:
        image_dir: Root directory to scan recursively.
        exts: Allowed file extensions (default: .jpg, .jpeg, .png).
        exclude_dirs: Subdirectory names to skip (e.g. {'loss', 'noise'}).
    """
    if exts is None:
        exts = IMAGE_EXTS
    images = []
    for p in sorted(image_dir.rglob("*")):
        if p.suffix.lower() not in exts:
            continue
        if exclude_dirs:
            parts = set(p.relative_to(image_dir).parts)
            if parts & exclude_dirs:
                continue
        images.append(p)
    return images


def load_image_tensor(path: Path, image_size: int = 256) -> torch.Tensor:
    """Load an image and convert to (C, H, W) float tensor in [0, 1].

    Used by the UAV quality dataset and benchmark scripts.
    """
    image = Image.open(path).convert("RGB")
    image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def count_parameters(model) -> tuple:
    """Return (total_params, trainable_params) for the given model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
