import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

_log = logging.getLogger(__name__)

MANIFEST_REQUIRED_FIELDS = {"path", "task", "distortion", "intensity_level"}
TASK_NAMES = ("tracking", "inspection", "delivery", "sar")
TASK_TO_ID = {name: i for i, name in enumerate(TASK_NAMES)}
VALID_TASKS = set(TASK_NAMES)


def validate_manifest(manifest_path: Path) -> dict:
    """Validate a manifest.json file and return diagnostics.

    Returns:
        dict with keys: valid, n_entries, missing_fields, unknown_tasks,
        missing_paths, score_stats
    """
    import json

    result = {
        "valid": True,
        "n_entries": 0,
        "missing_fields": set(),
        "unknown_tasks": set(),
        "missing_paths": [],
        "score_stats": {},
    }

    if not manifest_path.exists():
        result["valid"] = False
        result["error"] = f"Manifest not found: {manifest_path}"
        return result

    try:
        with open(manifest_path) as f:
            entries = json.load(f)
    except json.JSONDecodeError as e:
        result["valid"] = False
        result["error"] = f"Invalid JSON: {e}"
        return result

    result["n_entries"] = len(entries)
    data_root = manifest_path.parent.parent

    ref_path_missing = 0
    scores = {"vlm_score": [], "vla_score": [], "execution_score": []}

    for i, entry in enumerate(entries):
        missing = MANIFEST_REQUIRED_FIELDS - set(entry.keys())
        if missing:
            result["missing_fields"].update(missing)
            result["valid"] = False

        task = entry.get("task", "")
        if task and task not in VALID_TASKS:
            result["unknown_tasks"].add(task)
            result["valid"] = False

        img_path = entry.get("path", "")
        if img_path and not (data_root / img_path).exists():
            result["missing_paths"].append(img_path)
            result["valid"] = False

        if not entry.get("ref_path"):
            ref_path_missing += 1

        for key in scores:
            val = entry.get(key)
            if val is not None and isinstance(val, (int, float)):
                scores[key].append(float(val))

    for key, vals in scores.items():
        if vals:
            arr = np.array(vals)
            result["score_stats"][key] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }

    if ref_path_missing > 0:
        result["ref_path_missing"] = ref_path_missing
        _log.warning(
            "%d/%d entries lack 'ref_path' — FR benchmarks will fail",
            ref_path_missing,
            len(entries),
        )

    return result


class UAVIQADataset(Dataset):
    """UAV-Embodied-IQA dataset loader.

    Loads distorted image pairs with VLM/VLA/execution annotations.
    """

    TASK_MAP = TASK_TO_ID

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        task: Optional[str] = None,
        annotator_stage: str = "vla",
        image_size: int = 256,
        num_tasks: int = 4,
        augment: bool = False,
        samples: Optional[list] = None,
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.task = task
        self.annotator_stage = annotator_stage
        self.image_size = image_size
        self.num_tasks = num_tasks
        self.augment = augment
        self._augment_fn = self._build_augment() if augment else None

        # Track warnings to avoid flooding per-epoch (capped per category)
        self._warned_missing_score: set = set()
        self._warned_unknown_task: set = set()
        self._warned_corrupt_image: set = set()
        self._MAX_WARNINGS = 50

        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._load_manifest()

    def _load_manifest(self) -> list:
        from uav_iqa.utils import load_manifest

        manifest_path = self.data_root / self.split / "manifest.json"
        if not manifest_path.exists():
            return []
        return load_manifest(manifest_path)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        from uav_iqa.utils import load_image_tensor

        sample = self.samples[idx]

        try:
            image = load_image_tensor(
                self.data_root / sample["path"], self.image_size
            )
        except (OSError, IOError, Exception):
            path = sample.get("path", "?")
            if (
                len(self._warned_corrupt_image) < self._MAX_WARNINGS
                and path not in self._warned_corrupt_image
            ):
                self._warned_corrupt_image.add(path)
                _log.warning(f"Corrupted or missing image replaced with blank: {path}")
            image = torch.zeros(3, self.image_size, self.image_size)

        if self.augment:
            image = self._augment(image)

        task_name = sample.get("task", "tracking")
        task_id = self.TASK_MAP.get(task_name)
        if task_id is None:
            if (
                len(self._warned_unknown_task) < self._MAX_WARNINGS
                and task_name not in self._warned_unknown_task
            ):
                self._warned_unknown_task.add(task_name)
                _log.warning(
                    f"Unknown task '{task_name}' — falling back to tracking (id=0). "
                    f"Valid tasks: {list(self.TASK_MAP.keys())}"
                )
            task_id = 0

        score_key = f"{self.annotator_stage}_score"
        score = sample.get(score_key)
        if score is None:
            score_key_short = score_key.split("_")[0] if "_" in score_key else score_key
            if (
                len(self._warned_missing_score) < self._MAX_WARNINGS
                and score_key_short not in self._warned_missing_score
            ):
                self._warned_missing_score.add(score_key_short)
                _log.warning(
                    f"Score field '{score_key}' missing for sample — defaulting to 0.0. "
                    f"Annotator stage: {self.annotator_stage}, split: {self.split}"
                )
            score = 0.0
        if isinstance(score, list):
            score = np.mean(score)

        result = {
            "image": image,
            "task_id": torch.tensor(task_id, dtype=torch.long),
            "score": torch.tensor(score, dtype=torch.float32),
            "distortion": sample.get("distortion", "unknown"),
            "intensity": sample.get("intensity_level", 0.0),
            "ref_id": sample.get("ref_id", -1),
        }

        for key in ("vlm_score", "vla_score", "execution_score"):
            val = sample.get(key)
            if val is not None:
                if isinstance(val, list):
                    val = float(np.mean(val))
                result[key] = torch.tensor(float(val), dtype=torch.float32)

        return result

    @staticmethod
    def _build_augment():
        from torchvision import transforms as T

        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1),
            ]
        )

    def _augment(self, image: torch.Tensor) -> torch.Tensor:
        return self._augment_fn(image)

    @staticmethod
    def collate_fn(batch: list) -> dict:
        images = torch.stack([b["image"] for b in batch])
        task_ids = torch.stack([b["task_id"] for b in batch])
        scores = torch.stack([b["score"] for b in batch])
        result = {
            "image": images,
            "task_id": task_ids,
            "score": scores,
            "distortion": [b["distortion"] for b in batch],
            "intensity": [b["intensity"] for b in batch],
            "ref_id": [b["ref_id"] for b in batch],
        }
        for key in ("vlm_score", "vla_score", "execution_score"):
            if key in batch[0]:
                result[key] = torch.stack([b[key] for b in batch])
        return result
