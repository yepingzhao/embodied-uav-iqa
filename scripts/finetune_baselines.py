#!/usr/bin/env python3
"""Fine-tune IQA baselines from the Embodied-IQA paper (NeurIPS 2025).

Thin CLI wrapper — see uav_iqa.baselines.finetune for the implementation.
"""

import argparse
import importlib.util
import logging
from pathlib import Path

import lightning as L
import torch

from uav_iqa.baselines.finetune import (
    FINETUNABLE_METHODS,
    BaselineDataModule,
    BaselineLightningModule,
    evaluate_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
_log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune IQA baselines")
    parser.add_argument("--method", required=True, choices=list(FINETUNABLE_METHODS))
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="outputs/finetune")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--precision", default="16-mixed",
                        choices=["32-true", "16-mixed", "bf16-mixed"])
    args = parser.parse_args()

    if importlib.util.find_spec("pyiqa") is None:
        _log.error("pyiqa is required. Install: uv sync --group dev")
        return 1

    metric_mode = FINETUNABLE_METHODS[args.method]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _log.info("Fine-tuning %s (mode=%s)", args.method, metric_mode)

    data_module = BaselineDataModule(
        data_dir=args.data_dir, metric_mode=metric_mode,
        batch_size=args.batch_size, num_workers=args.num_workers,
        image_size=args.image_size, max_train_samples=args.max_train_samples,
    )

    model = BaselineLightningModule(
        method_name=args.method, metric_mode=metric_mode,
        lr=args.lr, weight_decay=args.weight_decay,
        warmup_epochs=max(1, args.max_epochs // 10),
        total_epochs=args.max_epochs,
    )

    ckpt_dir = output_dir / args.method
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=str(ckpt_dir), monitor="val/srcc", mode="max",
        save_top_k=1, filename="best",
    )

    trainer = L.Trainer(
        accelerator=args.device, devices=1, precision=args.precision,
        max_epochs=args.max_epochs, gradient_clip_val=1.0,
        log_every_n_steps=10, enable_progress_bar=True,
        default_root_dir=str(output_dir), callbacks=[ckpt_callback],
        logger=L.pytorch.loggers.CSVLogger(save_dir=str(output_dir), name=args.method),
    )

    trainer.fit(model, datamodule=data_module)
    _log.info("Best: %s (val/srcc=%.4f)", ckpt_callback.best_model_path,
              ckpt_callback.best_model_score)

    if ckpt_callback.best_model_path:
        evaluate_checkpoint(
            ckpt_path=ckpt_callback.best_model_path, method_name=args.method,
            data_dir=args.data_dir, output_dir=str(output_dir),
            image_size=args.image_size,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return None


if __name__ == "__main__":
    main()
