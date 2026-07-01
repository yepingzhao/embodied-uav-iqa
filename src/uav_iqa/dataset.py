"""UAV-IQA flat dataset loader.

Loads flat processed JSON files from the data synthesis pipeline. Each JSON
entry is a single (question x distortion) training sample with embedded
paths, distortion info, and cognitive score.
"""

import json
import logging
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from uav_iqa.annotations import SUBTASK_NAME_TO_ID
from uav_iqa.utils import load_image_tensor

_log = logging.getLogger(__name__)


def validate_manifest(manifest_path: Path) -> dict:
    """Validate a manifest JSON file (flat format).

    Returns a dict with keys ``valid`` (bool), ``count`` (int), and
    optional ``warnings`` / ``score_stats``.
    """
    try:
        with open(manifest_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        return {"valid": False, "count": 0, "warnings": [f"Cannot read: {e}"]}

    if not isinstance(data, list):
        return {"valid": False, "count": 0, "warnings": ["Not a JSON array"]}

    return {"valid": True, "count": len(data), "warnings": []}


class UAVIQADataset(Dataset):
    """Multi-image UAV-IQA dataset from flat processed JSONs.

    Each ``__getitem__`` returns a dict with:
      - ``images``: tensor (N_UAV, 3, H, W)
      - ``task_id``: tensor (scalar), subtask id
      - ``score``: tensor (scalar), cognitive_score
      - ``sample_id``: str
      - ``distortion``: str
      - ``num_uavs``: int
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 256,
        augment: bool = False,
        max_uavs: int = 6,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.augment = augment
        self.max_uavs = max_uavs
        self._augment_fn = self._build_augment() if augment else None

        self.samples = self._load()

    def _load(self) -> List[dict]:
        split_dir = self.data_root / self.split
        samples: List[dict] = []

        for fpath in sorted(split_dir.glob("*_VQA_*.json")):
            with open(fpath) as f:
                entries = json.load(f)

            for entry in entries:
                uav_paths = entry.get("uav_paths", {})
                uav_keys = sorted(uav_paths.keys())
                distortion_info = entry.get("distortion_info", {})
                cs = entry.get("cognitive_score")
                cognitive_score = float(cs) if cs is not None else 0.0

                subtask_type = entry.get("subtask_type", "")
                task_id = SUBTASK_NAME_TO_ID.get(subtask_type)
                if task_id is None:
                    _log.warning(
                        "Unknown subtask_type '%s' for sample %s, defaulting to 0",
                        subtask_type,
                        entry.get("sample_id", "?"),
                    )
                    task_id = 0
                samples.append({
                    "uav_paths": uav_paths,
                    "uav_keys": uav_keys,
                    "distorted_uav_paths": entry.get("distorted_uav_paths", {}),
                    "distortion": distortion_info.get("type", "unknown"),
                    "intensity": distortion_info.get("intensity", 0.0),
                    "sample_id": entry.get("sample_id", ""),
                    "subtask_type": subtask_type,
                    "task_id": task_id,
                    "cognitive_score": cognitive_score,
                    "question": entry.get("question", ""),
                })

        _log.info(
            "Loaded %d samples from %s/%s",
            len(samples),
            self.data_root.name,
            self.split,
        )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        uav_keys = sample["uav_keys"]
        distorted_uav = sample["distorted_uav_paths"]

        images = []
        for k in uav_keys:
            dp = distorted_uav.get(k, "")
            img_path = self.data_root / dp.lstrip("/")
            try:
                img = load_image_tensor(img_path, self.image_size)
            except (OSError, IOError):
                _log.warning("Failed to load image: %s", img_path)
                img = torch.zeros(3, self.image_size, self.image_size)
            images.append(img)

        image_tensor = torch.stack(images)

        if self.augment:
            image_tensor = self._augment(image_tensor)

        num_uavs = len(uav_keys)

        return {
            "images": image_tensor,
            "task_id": torch.tensor(sample["task_id"], dtype=torch.long),
            "score": torch.tensor(sample["cognitive_score"], dtype=torch.float32),
            "sample_id": sample["sample_id"],
            "distortion": sample["distortion"],
            "num_uavs": num_uavs,
            "question": sample.get("question", ""),
        }

    @staticmethod
    def _build_augment():
        from torchvision import transforms as T

        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 4:
            return torch.stack([self._augment_fn(img) for img in image])
        return self._augment_fn(image)

    @staticmethod
    def collate_fn(batch: list) -> dict:
        images = torch.nn.utils.rnn.pad_sequence(
            [b["images"] for b in batch], batch_first=True, padding_value=0.0
        )
        task_ids = torch.stack([b["task_id"] for b in batch])
        scores = torch.stack([b["score"] for b in batch])

        return {
            "images": images,
            "task_id": task_ids,
            "score": scores,
            "sample_id": [b["sample_id"] for b in batch],
            "distortion": [b["distortion"] for b in batch],
            "num_uavs": [b["num_uavs"] for b in batch],
            "question": [b.get("question", "") for b in batch],
        }
