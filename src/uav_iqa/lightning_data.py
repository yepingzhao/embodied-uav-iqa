import json
from pathlib import Path
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader

from .dataset import UAVIQADataset


class UAVIQDataModule(L.LightningDataModule):
    """LightningDataModule wrapping UAVIQADataset with manifest filtering."""

    UAV_DISTORTIONS = {
        "propeller_vibration_blur",
        "atmospheric_scattering_haze",
        "six_dof_viewpoint_blur",
        "communication_packet_loss",
        "low_res_super_resolution",
        "propeller_shadow",
    }

    def __init__(
        self,
        data_root: str = "data/database",
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 256,
        annotator_stage: str = "vla",
        task: Optional[str] = None,
        val_task: Optional[str] = None,
        distortion_filter: Optional[str] = None,
        leave_out_task: Optional[str] = None,
        train_split: float = 0.80,
        val_split: float = 0.10,
        dry_run: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["task", "val_task", "distortion_filter", "leave_out_task"])

        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.annotator_stage = annotator_stage
        self.task = task
        self.val_task = val_task
        self.distortion_filter = distortion_filter
        self.leave_out_task = leave_out_task
        self.dry_run = dry_run

        self.train_dataset: Optional[UAVIQADataset] = None
        self.val_dataset: Optional[UAVIQADataset] = None
        self.test_dataset: Optional[UAVIQADataset] = None

    def _load_and_filter(self, split: str, task_override: Optional[str] = None) -> list:
        manifest_path = self.data_root / split / "manifest.json"
        if not manifest_path.exists():
            return []
        with open(manifest_path) as f:
            samples = json.load(f)
        return self._filter_manifest(samples, task=task_override)

    def _filter_manifest(
        self,
        samples: list,
        task: Optional[str] = None,
    ) -> list:
        filtered = []
        for s in samples:
            sample_task = s.get("task")
            sample_dist = s.get("distortion", "")

            if task and sample_task != task:
                continue
            if self.leave_out_task and sample_task == self.leave_out_task:
                continue

            if self.distortion_filter == "generic":
                if sample_dist in self.UAV_DISTORTIONS:
                    continue
            elif self.distortion_filter == "uav_only":
                if sample_dist not in self.UAV_DISTORTIONS:
                    continue

            filtered.append(s)
        return filtered

    def setup(self, stage: Optional[str] = None) -> None:
        shared_kwargs = dict(
            data_root=str(self.data_root),
            image_size=self.image_size,
            annotator_stage=self.annotator_stage,
        )

        val_task = self.val_task or self.task

        train_samples = self._load_and_filter("train", task_override=self.task)
        val_samples = self._load_and_filter("val", task_override=val_task)
        test_samples = self._load_and_filter("test", task_override=val_task)

        if self.dry_run:
            train_samples = train_samples[:100]
            val_samples = val_samples[:50]
            test_samples = test_samples[:50]

        print(
            f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}"
        )

        self.train_dataset = UAVIQADataset(
            samples=train_samples if train_samples else None,
            **shared_kwargs,
        )
        self.val_dataset = UAVIQADataset(
            samples=val_samples if val_samples else None,
            **shared_kwargs,
        )
        self.test_dataset = UAVIQADataset(
            samples=test_samples if test_samples else None,
            **shared_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=UAVIQADataset.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=UAVIQADataset.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=UAVIQADataset.collate_fn,
        )
