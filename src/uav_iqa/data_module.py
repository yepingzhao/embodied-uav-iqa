"""LightningDataModule for multi-image UAV-IQA grouped dataset."""

import logging
from pathlib import Path
from typing import Optional

import lightning as L
from torch.utils.data import DataLoader

from .dataset import UAVIQADataset
from .distortion import UAVDistortionPipeline

_log = logging.getLogger(__name__)


class UAVIQDataModule(L.LightningDataModule):
    """LightningDataModule for multi-image UAV-IQA grouped JSONs."""

    def __init__(
        self,
        data_root: str = "data/processed",
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 256,
        max_uavs: int = 6,
        subtask_filter: Optional[str] = None,
        distortion_filter: Optional[str] = None,
        dry_run: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.data_root = Path(data_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_uavs = max_uavs
        self.subtask_filter = subtask_filter
        self.distortion_filter = distortion_filter
        self.dry_run = dry_run

        self.train_dataset: Optional[UAVIQADataset] = None
        self.val_dataset: Optional[UAVIQADataset] = None
        self.test_dataset: Optional[UAVIQADataset] = None

    def _make_dataset(self, split: str, augment: bool = False) -> UAVIQADataset:
        ds = UAVIQADataset(
            data_root=str(self.data_root),
            split=split,
            image_size=self.image_size,
            augment=augment,
            max_uavs=self.max_uavs,
        )

        if self.subtask_filter:
            valid_types = {t.strip() for t in self.subtask_filter.split(",")}
            ds.samples = [
                s for s in ds.samples
                if s.get("vqa_entries", [{}])[0].get("subtask_type", "") in valid_types
            ]

        if self.distortion_filter:
            if self.distortion_filter == "generic":
                uav_names = set(UAVDistortionPipeline.get_uav_distortion_names())
                ds.samples = [
                    s for s in ds.samples
                    if s.get("distortion", "") not in uav_names
                ]
            elif self.distortion_filter == "uav_only":
                uav_names = set(UAVDistortionPipeline.get_uav_distortion_names())
                ds.samples = [
                    s for s in ds.samples
                    if s.get("distortion", "") in uav_names
                ]

        if self.dry_run:
            limit = 100 if split == "train" else 50
            ds.samples = ds.samples[:limit]

        return ds

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            self.train_dataset = self._make_dataset("train", augment=True)
            self.val_dataset = self._make_dataset("test", augment=False)
        if stage in (None, "fit", "test"):
            self.test_dataset = self._make_dataset("test", augment=False)

        _log.info(
            "Train: %d, Val: %d, Test: %d",
            len(self.train_dataset) if self.train_dataset else 0,
            len(self.val_dataset) if self.val_dataset else 0,
            len(self.test_dataset) if self.test_dataset else 0,
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
