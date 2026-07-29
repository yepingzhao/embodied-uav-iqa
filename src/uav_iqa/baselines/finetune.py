"""IQA baseline fine-tuning — Lightning modules for FR and NR methods.

Matches the 10 fine-tunable methods from the Embodied-IQA paper (NeurIPS 2025):
  FR (5): AHIQ, CKDN, DISTS, LPIPS, TOPIQ-FR
  NR (5): CLIPIQA, CNNIQA, DBCNN, QualiClip, TOPIQ-NR
"""

import json
import logging
from pathlib import Path

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from uav_iqa.dataset import UAVIQADataset
from uav_iqa.metrics import (
    evaluate_iqa,
    per_distortion_category_metrics,
    per_task_metrics,
)
from uav_iqa.utils import load_flat_samples, load_image_tensor

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FINETUNABLE_METHODS = {
    "ahiq": "FR", "ckdn": "FR", "dists": "FR", "lpips": "FR", "topiq_fr": "FR",
    "clip_iqa": "NR", "cnniqa": "NR", "dbcnn": "NR", "qualiclip": "NR", "topiq_nr": "NR",
}

PYIQA_NAME_MAP = {
    "ahiq": "ahiq", "ckdn": "ckdn", "dists": "dists", "lpips": "lpips",
    "topiq_fr": "topiq_fr",
    "clip_iqa": "clipiqa", "cnniqa": "cnniqa", "dbcnn": "dbcnn",
    "qualiclip": "qualiclip", "topiq_nr": "topiq_nr",
}


# ---------------------------------------------------------------------------
# FR Dataset
# ---------------------------------------------------------------------------


class BaselineFRDataset(Dataset):
    """Single-image dataset for FR IQA baseline fine-tuning.

    Loads (distorted_image, reference_image) -> score pairs from flat JSON samples.
    """

    def __init__(self, data_dir: str, split: str, image_size: int = 256):
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.samples = load_flat_samples(self.data_dir, split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        dist_img = load_image_tensor(self.data_dir / s["path"], self.image_size)
        ref_img = load_image_tensor(self.data_dir / s["ref_path"], self.image_size)
        score = s.get("score", 0.0)
        if isinstance(score, list):
            score = float(np.mean(score))
        return {
            "image": dist_img,
            "ref_image": ref_img,
            "score": torch.tensor(float(score), dtype=torch.float32),
            "sample_id": s.get("sample_id", ""),
            "distortion": s.get("distortion", ""),
            "task": s.get("task", ""),
        }

    @staticmethod
    def collate_fn(batch: list) -> dict:
        return {
            "image": torch.stack([b["image"] for b in batch]),
            "ref_image": torch.stack([b["ref_image"] for b in batch]),
            "score": torch.stack([b["score"] for b in batch]),
            "sample_id": [b["sample_id"] for b in batch],
            "distortion": [b["distortion"] for b in batch],
            "task": [b["task"] for b in batch],
        }


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class BaselineLightningModule(L.LightningModule):
    """Lightweight LightningModule wrapping a pyiqa model for fine-tuning.

    Supports both FR (img+ref -> score) and NR (img -> score) methods.
    """

    def __init__(
        self,
        method_name: str,
        metric_mode: str = "NR",
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters()

        import pyiqa

        pyiqa_name = PYIQA_NAME_MAP[method_name]
        self.metric_mode = metric_mode
        self.model = pyiqa.create_metric(pyiqa_name, device=self.device)
        self.loss_fn = nn.MSELoss()

        self._val_preds = []
        self._val_targets = []
        self._val_tasks = []
        self._val_distortions = []

    def forward(self, x, ref=None):
        if ref is not None and self.metric_mode == "FR":
            return self.model(x, ref).squeeze(-1)
        return self.model(x).squeeze(-1)

    def training_step(self, batch, _):
        img = batch["image"]
        scores = batch["score"]
        if self.metric_mode == "FR":
            ref = batch["ref_image"]
            pred = self(img, ref)
        else:
            pred = self(img)
        loss = self.loss_fn(pred, scores)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, _):
        img = batch["image"]
        scores = batch["score"]
        if self.metric_mode == "FR":
            ref = batch["ref_image"]
            pred = self(img, ref)
        else:
            pred = self(img)
        self._val_preds.append(pred.detach().cpu().numpy())
        self._val_targets.append(scores.detach().cpu().numpy())
        self._val_tasks.extend(batch.get("task", []) or [])
        self._val_distortions.extend(batch.get("distortion", []) or [])

    def on_validation_epoch_end(self):
        if not self._val_preds:
            return
        preds = np.concatenate(self._val_preds)
        targets = np.concatenate(self._val_targets)
        metrics = evaluate_iqa(preds, targets)
        self.log("val/srcc", metrics["srcc"], prog_bar=True)
        self.log("val/plcc", metrics["plcc"])
        self._val_preds.clear()
        self._val_targets.clear()
        self._val_tasks.clear()
        self._val_distortions.clear()

    def configure_optimizers(self):
        hp = self.hparams
        opt = torch.optim.AdamW(self.model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        if hp.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-3, total_iters=hp.warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp.total_epochs - hp.warmup_epochs)
            scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[hp.warmup_epochs])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=hp.total_epochs)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1}}


# ---------------------------------------------------------------------------
# Data Module
# ---------------------------------------------------------------------------


class BaselineDataModule(L.LightningDataModule):
    """Data module for baseline fine-tuning (FR and NR methods)."""

    def __init__(
        self,
        data_dir: str,
        metric_mode: str = "NR",
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 256,
        max_train_samples: int = 0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.metric_mode = metric_mode
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.max_train_samples = max_train_samples

    def _load_split(self, split):
        return load_flat_samples(self.data_dir, split)

    def _make_dataset(self, split: str):
        if self.metric_mode == "FR":
            return BaselineFRDataset(data_dir=str(self.data_dir), split=split, image_size=self.image_size)
        return UAVIQADataset(data_root=str(self.data_dir), split=split, image_size=self.image_size)

    def _get_collate_fn(self):
        if self.metric_mode == "FR":
            return BaselineFRDataset.collate_fn
        return UAVIQADataset.collate_fn

    def setup(self, _stage=None):
        train_samples = self._load_split("train")
        val_split = "val" if (self.data_dir / "val").is_dir() else "test"
        val_samples = self._load_split(val_split)
        test_samples = self._load_split("test")

        self.train_ds = self._make_dataset("train")
        if self.max_train_samples > 0 and hasattr(self.train_ds, "samples"):
            if len(self.train_ds.samples) > self.max_train_samples:
                self.train_ds.samples = self.train_ds.samples[: self.max_train_samples]
                _log.info("Limited training samples to %d", self.max_train_samples)

        self.val_ds = self._make_dataset(val_split)
        self.test_ds = self._make_dataset("test")

        _log.info("Train: %d, Val: %d, Test: %d", len(train_samples), len(val_samples), len(test_samples))

    def _make_loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True, collate_fn=self._get_collate_fn(),
        )

    def train_dataloader(self):
        return self._make_loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._make_loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_ds, shuffle=False)


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------


def evaluate_checkpoint(ckpt_path: str, method_name: str, data_dir: str,
                        output_dir: str, image_size: int, device: str):
    """Evaluate a fine-tuned checkpoint on the test set."""
    _log.info("Evaluating checkpoint on test set...")
    model = BaselineLightningModule.load_from_checkpoint(ckpt_path, method_name=method_name, strict=False)
    model.eval()
    model.to(device)
    is_fr = getattr(model, "metric_mode", "NR") == "FR"

    test_samples = load_flat_samples(Path(data_dir), "test")
    predictions, targets, task_ids, distortion_labels = [], [], [], []

    with torch.no_grad():
        for i, s in enumerate(test_samples):
            img_t = load_image_tensor(Path(data_dir) / s["path"], image_size).unsqueeze(0).to(device)
            if is_fr and s.get("ref_path"):
                ref_t = load_image_tensor(Path(data_dir) / s["ref_path"], image_size).unsqueeze(0).to(device)
                pred = model(img_t, ref_t).item()
            else:
                pred = model(img_t).item()
            score = s.get("score", 0.0)
            if isinstance(score, list):
                score = float(np.mean(score))
            predictions.append(float(pred))
            targets.append(float(score))
            task_ids.append(s.get("task", "scene_description"))
            distortion_labels.append(s.get("distortion", "unknown"))
            if (i + 1) % 1000 == 0:
                _log.info("  %d/%d", i + 1, len(test_samples))

    preds = np.array(predictions)
    targs = np.array(targets)
    metrics = evaluate_iqa(preds, targs)
    per_task = per_task_metrics(preds, targs, task_ids)
    per_cat = per_distortion_category_metrics(preds, targs, distortion_labels)

    results = {
        "method": method_name, "finetuned": True, "checkpoint": str(ckpt_path),
        "srcc": float(metrics["srcc"]), "plcc": float(metrics["plcc"]),
        "rmse": float(metrics["rmse"]), "kendall_tau": float(metrics.get("kendall_tau", 0.0)),
        "per_task": per_task, "per_distortion_category": per_cat,
        "n_test_samples": len(test_samples),
    }
    results_path = Path(output_dir) / f"{method_name}_finetuned_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    _log.info("  Fine-tuned %s: SRCC=%.4f PLCC=%.4f", method_name, metrics["srcc"], metrics["plcc"])
    _log.info("  Results saved to %s", results_path)
    return results
