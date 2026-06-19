#!/usr/bin/env python3
"""M0-R004: Overfit test — train UAV-QANet on 100 images until loss → 0.

Verifies: model implementation is correct, gradients flow, no NaN.
"""

import argparse
import sys
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uav_iqa.lightning_model import UAVQALightningModule


class OverfitDataModule(L.LightningDataModule):
    def __init__(self, n_samples=100, batch_size=8):
        super().__init__()
        self.n_samples = n_samples
        self.batch_size = batch_size

    def setup(self, stage=None):
        x = torch.randn(self.n_samples, 3, 256, 256)
        task_ids = torch.randint(0, 4, (self.n_samples,))
        y = torch.rand(self.n_samples)

        class OverfitDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(x)

            def __getitem__(self, idx):
                return {
                    "image": x[idx],
                    "task_id": task_ids[idx],
                    "score": y[idx],
                    "vlm_score": y[idx],
                    "vla_score": y[idx],
                    "execution_score": y[idx],
                    "distortion": "overfit",
                }

        self.dataset = OverfitDataset()

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)


def main():
    parser = argparse.ArgumentParser(description="M0-R004: Overfit test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    L.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = UAVQALightningModule(
        backbone="mobilenetv4_conv_small",
        lambda_rank=0.0,
        lambda_cross_task=0.0,
        warmup_epochs=2,
        total_epochs=args.epochs,
    )

    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"Model params: {n_params:,}")

    datamodule = OverfitDataModule(n_samples=100, batch_size=args.batch_size)

    trainer = L.Trainer(
        accelerator="auto",
        devices=1,
        max_epochs=args.epochs,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    trainer.fit(model, datamodule=datamodule)

    final_loss = float(trainer.callback_metrics.get("train/loss_epoch", 999))
    print(f"\nFinal loss: {final_loss:.6f}")

    overfit_ok = final_loss < 0.001
    print(f"Overfit check: {'PASSED' if overfit_ok else 'FAILED'} (min_loss={final_loss:.6f} < 0.001)")

    return 0 if overfit_ok else 1


if __name__ == "__main__":
    sys.exit(main())
