from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class UAVIQADataset(Dataset):
    """UAV-Embodied-IQA dataset loader.

    Loads distorted image pairs with VLM/VLA/execution annotations.
    """

    TASK_MAP = {"tracking": 0, "inspection": 1, "delivery": 2, "sar": 3}

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

        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._load_manifest()

    def _load_manifest(self) -> list:
        manifest_path = self.data_root / self.split / "manifest.json"
        if not manifest_path.exists():
            return []
        import json

        with open(manifest_path) as f:
            return json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        image = Image.open(self.data_root / sample["path"]).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)

        if self.augment:
            image = self._augment(image)

        task_id = self.TASK_MAP.get(sample.get("task", "tracking"), 0)

        score = sample.get(f"{self.annotator_stage}_score", 0.0)
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

        return T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.1, contrast=0.1),
        ])

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

    @staticmethod
    def create_dataloader(
        data_root: str,
        split: str = "train",
        batch_size: int = 64,
        num_workers: int = 4,
        **kwargs,
    ):
        dataset = UAVIQADataset(data_root=data_root, split=split, **kwargs)
        from torch.utils.data import DataLoader

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=UAVIQADataset.collate_fn,
        )
