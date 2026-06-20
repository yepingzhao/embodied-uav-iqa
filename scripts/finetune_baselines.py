#!/usr/bin/env python3
"""M2-R009a: Fine-tune deep-learning IQA baselines on UAV training data.

Supported methods (NR only, FR methods skip fine-tuning — see rationale below):
    brisque, niqe, clipiqa, maniqa, topiq_nr

FR methods (ahiq, topiq_fr) are not fine-tuned because:
  1. They require reference images which the default dataset doesn't batch-load.
  2. Existing IQA literature rarely fine-tunes FR methods; zero-shot is standard.

Usage:
    python scripts/finetune_baselines.py \
        --method clipiqa \
        --data-dir data/processed \
        --output-dir outputs/finetune \
        --max-epochs 30 \
        --lr 1e-4
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import lightning as L
from torch.utils.data import DataLoader

from uav_iqa.metrics import evaluate_iqa, per_task_metrics, per_distortion_category_metrics
from uav_iqa.utils import load_image_tensor, load_manifest, setup_logging

_log = setup_logging(__name__)

FINETUNABLE_METHODS = {
    "brisque": "NR",
    "niqe": "NR",
    "clipiqa": "NR",
    "maniqa": "NR",
    "topiq_nr": "NR",
}

# Our internal name → pyiqa.create_metric name
PYIQA_NAME_MAP = {
    "brisque": "brisque",
    "niqe": "niqe",
    "clipiqa": "clipiqa",
    "maniqa": "maniqa",
    "topiq_nr": "topiq_nr",
}


class BaselineLightningModule(L.LightningModule):
    """Lightweight LightningModule wrapping a pyiqa NR model for fine-tuning."""

    def __init__(
        self,
        method_name: str,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 3,
        total_epochs: int = 30,
    ):
        super().__init__()
        self.save_hyperparameters()

        import pyiqa

        pyiqa_name = PYIQA_NAME_MAP[method_name]
        self.model = pyiqa.create_metric(pyiqa_name, device=self.device)
        self.loss_fn = nn.MSELoss()

        self._val_preds = []
        self._val_targets = []
        self._val_tasks = []
        self._val_distortions = []

    def forward(self, x):
        return self.model(x).squeeze(-1)

    def training_step(self, batch, batch_idx):
        img = batch["image"]
        scores = batch.get("score")

        pred = self(img)
        loss = self.loss_fn(pred, scores)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        img = batch["image"]
        scores = batch.get("score")

        pred = self(img)
        self._val_preds.append(pred.detach().cpu().numpy())
        self._val_targets.append(scores.detach().cpu().numpy())
        self._val_tasks.extend(batch["task_id"].cpu().tolist())
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
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-3, total_iters=self.warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.total_epochs - self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, [warmup, cosine], milestones=[self.warmup_epochs]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.total_epochs)
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }


# Reuse existing collate_fn from UAVIQADataset
from uav_iqa.dataset import UAVIQADataset


class BaselineDataModule(L.LightningDataModule):
    """Data module for baseline fine-tuning (NR methods only)."""

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 8,
        image_size: int = 256,
        annotator_stage: str = "vla",
        max_train_samples: int = 0,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.annotator_stage = annotator_stage
        self.max_train_samples = max_train_samples

    def _load_split(self, split):
        path = self.data_dir / split / "manifest.json"
        if not path.exists():
            return []
        return load_manifest(path)

    def _make_dataset(self, samples):
        return UAVIQADataset(
            data_root=str(self.data_dir),
            image_size=self.image_size,
            annotator_stage=self.annotator_stage,
            samples=samples,
        )

    def setup(self, stage=None):
        train_samples = self._load_split("train")
        val_samples = self._load_split("val")
        test_samples = self._load_split("test")

        if self.max_train_samples > 0:
            train_samples = train_samples[: self.max_train_samples]

        self.train_ds = self._make_dataset(train_samples)
        self.val_ds = self._make_dataset(val_samples)
        self.test_ds = self._make_dataset(test_samples)

        _log.info(
            "Train: %d, Val: %d, Test: %d",
            len(train_samples),
            len(val_samples),
            len(test_samples),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=UAVIQADataset.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=UAVIQADataset.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=UAVIQADataset.collate_fn,
        )


def evaluate_checkpoint(
    ckpt_path: str,
    method_name: str,
    data_dir: str,
    output_dir: str,
    annotator_stage: str,
    image_size: int,
    device: str,
):
    """Evaluate a fine-tuned checkpoint on the test set."""
    _log.info("Evaluating checkpoint on test set...")

    model = BaselineLightningModule.load_from_checkpoint(
        ckpt_path, method_name=method_name, strict=False
    )
    model.eval()
    model.to(device)

    test_path = Path(data_dir) / "test" / "manifest.json"
    if not test_path.exists():
        _log.warning("Test manifest not found: %s", test_path)
        return None

    test_samples = load_manifest(test_path)

    predictions = []
    targets = []
    task_ids = []
    distortion_labels = []

    with torch.no_grad():
        for i, s in enumerate(test_samples):
            img_t = (
                load_image_tensor(Path(data_dir) / s["path"], image_size).unsqueeze(0).to(device)
            )
            pred = model(img_t).item()

            score = s.get(f"{annotator_stage}_score", s.get("score", 0.0))
            if isinstance(score, list):
                score = np.mean(score)
            predictions.append(float(pred))
            targets.append(float(score))
            task_ids.append(s.get("task", "tracking"))
            distortion_labels.append(s.get("distortion", "unknown"))

            if (i + 1) % 1000 == 0:
                _log.info("  %d/%d", i + 1, len(test_samples))

    preds = np.array(predictions)
    targs = np.array(targets)

    metrics = evaluate_iqa(preds, targs)
    per_task = per_task_metrics(preds, targs, task_ids)
    per_cat = per_distortion_category_metrics(preds, targs, distortion_labels)

    results = {
        "method": method_name,
        "finetuned": True,
        "checkpoint": str(ckpt_path),
        "srcc": float(metrics["srcc"]),
        "plcc": float(metrics["plcc"]),
        "rmse": float(metrics["rmse"]),
        "kendall_tau": float(metrics.get("kendall_tau", 0.0)),
        "per_task": per_task,
        "per_distortion_category": per_cat,
        "annotator_stage": annotator_stage,
        "n_test_samples": len(test_samples),
    }

    results_path = Path(output_dir) / f"{method_name}_finetuned_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    _log.info(
        "  Fine-tuned %s: SRCC=%.4f PLCC=%.4f",
        method_name,
        metrics["srcc"],
        metrics["plcc"],
    )
    _log.info("  Results saved to %s", results_path)
    return results


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IQA baselines")
    parser.add_argument("--method", required=True, choices=list(FINETUNABLE_METHODS))
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/finetune")
    parser.add_argument(
        "--annotator-stage",
        default="vla",
        choices=["vlm", "vla", "execution"],
    )
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Limit training samples (0=all)",
    )
    parser.add_argument(
        "--precision",
        default="16-mixed",
        choices=["32-true", "16-mixed", "bf16-mixed"],
    )
    args = parser.parse_args()

    try:
        import pyiqa  # noqa: F401
    except ImportError:
        _log.error("pyiqa is required for fine-tuning. Install: uv sync --group dev")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log.info("Fine-tuning %s", args.method)

    data_module = BaselineDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        annotator_stage=args.annotator_stage,
        max_train_samples=args.max_train_samples,
    )

    model = BaselineLightningModule(
        method_name=args.method,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=max(1, args.max_epochs // 10),
        total_epochs=args.max_epochs,
    )

    ckpt_dir = output_dir / args.method
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir),
        monitor="val/srcc",
        mode="max",
        save_top_k=1,
        filename="best",
    )

    trainer = L.Trainer(
        accelerator=args.device,
        devices=1,
        precision=args.precision,
        max_epochs=args.max_epochs,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        enable_progress_bar=True,
        default_root_dir=str(output_dir),
        callbacks=[ckpt_callback],
        logger=L.pytorch.loggers.CSVLogger(save_dir=str(output_dir), name=args.method),
    )

    trainer.fit(model, datamodule=data_module)

    _log.info(
        "Best checkpoint: %s (val/srcc=%.4f)",
        ckpt_callback.best_model_path,
        ckpt_callback.best_model_score,
    )

    if ckpt_callback.best_model_path:
        evaluate_checkpoint(
            ckpt_path=ckpt_callback.best_model_path,
            method_name=args.method,
            data_dir=args.data_dir,
            output_dir=str(output_dir),
            annotator_stage=args.annotator_stage,
            image_size=args.image_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


if __name__ == "__main__":
    main()
