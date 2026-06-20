#!/usr/bin/env python3
"""Training entry point for UAV-IQANet (LightningCLI wrapper).

Usage::

    # Single experiment
    python scripts/train.py fit --config configs/experiments/r013_task_cond.yaml

    # With seed and custom output dir
    python scripts/train.py fit \
        --config configs/experiments/r013_task_cond.yaml \
        --seed_everything 42 \
        --trainer.default_root_dir outputs/r013_task_cond_seed42
"""

from dotenv import load_dotenv

load_dotenv()  # Load .env before Lightning/W&B read env vars

from lightning.pytorch.cli import LightningCLI  # noqa: E402

from uav_iqa.data_module import UAVIQDataModule  # noqa: E402
from uav_iqa.lightning_module import UAVIQALightningModule  # noqa: E402


def main():
    LightningCLI(
        UAVIQALightningModule,
        UAVIQDataModule,
        args=None,  # parse sys.argv
        save_config_callback=None,  # configs are in configs/experiments/
    )


if __name__ == "__main__":
    main()
