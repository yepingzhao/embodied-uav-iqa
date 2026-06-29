"""UAV-IQA multi-image dataset loader.

Loads grouped JSON files from the data synthesis pipeline.  Each training
sample is a ``(scene+frame group, distortion type)`` pair that yields
multiple UAV images with the same distortion, a subtask identifier, and
a cognitive quality score.
"""

import json
import logging
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset

from uav_iqa.annotations import SUBTASK_TO_ID

_log = logging.getLogger(__name__)


def validate_manifest(manifest_path: Path) -> dict:
    """Validate a manifest JSON file (old flat format) or grouped JSON (new format).

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

TASK_NAMES = (
    "scene_description", "scene_comparison", "observing_posture",
    "object_recognition", "object_counting", "object_grounding",
    "object_matching",
    "quality_assessment", "usability_assessment", "causal_assessment",
    "when_to_collaborate", "what_to_collaborate", "who_to_collaborate",
    "why_to_collaborate",
)

TASK_TO_ID: dict[str, int] = {name: i for i, name in enumerate(TASK_NAMES)}

NUM_TASKS = len(TASK_NAMES)


class UAVIQADataset(Dataset):
    """Multi-image UAV-IQA dataset from grouped processed JSONs.

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
                groups = json.load(f)

            for group in groups:
                distortions = group.get("distortions", {})
                uav_paths = group.get("uav_paths", {})
                uav_keys = group.get("uav_keys", [])
                if not uav_paths or not uav_keys or not distortions:
                    continue

                for dist_key, dist_info in distortions.items():
                    distorted_uav = dist_info.get("distorted_uav_paths", {})
                    if not distorted_uav:
                        continue

                    samples.append({
                        "uav_paths": uav_paths,
                        "uav_keys": uav_keys,
                        "num_uavs": group.get("num_uavs", len(uav_keys)),
                        "distortion": dist_info.get("type", "unknown"),
                        "intensity": dist_info.get("intensity", 0.0),
                        "sample_id": dist_info.get("sample_id", ""),
                        "distorted_uav_paths": distorted_uav,
                        "vqa_entries": group.get("vqa_entries", []),
                        "data_root": str(self.data_root.parent),
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
        from uav_iqa.utils import load_image_tensor

        sample = self.samples[idx]
        uav_keys = sample["uav_keys"]
        distorted_uav = sample["distorted_uav_paths"]
        data_root = Path(sample["data_root"])

        images = []
        for k in uav_keys:
            dp = distorted_uav.get(k, "")
            img_path = data_root / dp.lstrip("/")
            try:
                img = load_image_tensor(img_path, self.image_size)
            except (OSError, IOError, Exception):
                img = torch.zeros(3, self.image_size, self.image_size)
            images.append(img)

        image_tensor = torch.stack(images)

        if self.augment:
            image_tensor = self._augment(image_tensor)

        vqa_entries = sample.get("vqa_entries", [])
        if not vqa_entries:
            return {
                "images": image_tensor,
                "task_id": torch.tensor(0, dtype=torch.long),
                "score": torch.tensor(0.0, dtype=torch.float32),
                "sample_id": sample.get("sample_id", ""),
                "distortion": sample.get("distortion", "unknown"),
                "num_uavs": sample.get("num_uavs", len(uav_keys)),
            }

        entry = vqa_entries[0]
        subtask_type = entry.get("subtask_type", "")
        task_id = SUBTASK_TO_ID.get(subtask_type, 0)
        cognitive_score = entry.get("cognitive_score", 0.0)
        if isinstance(cognitive_score, dict):
            cognitive_score = 0.0
        elif not isinstance(cognitive_score, (int, float)):
            cognitive_score = 0.0

        return {
            "images": image_tensor,
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "score": torch.tensor(float(cognitive_score), dtype=torch.float32),
            "sample_id": sample.get("sample_id", ""),
            "distortion": sample.get("distortion", "unknown"),
            "num_uavs": sample.get("num_uavs", len(uav_keys)),
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
        }
