#!/usr/bin/env python3
"""Training entry point for UAV-IQANet (LightningCLI wrapper).

Usage::

    # Single experiment
    python scripts/train.py fit --config configs/experiments/full_model.yaml

    # With seed and custom output dir
    python scripts/train.py fit \
        --config configs/experiments/full_model.yaml \
        --seed_everything 42 \
        --trainer.default_root_dir outputs/full_model_seed42
"""

from dotenv import load_dotenv

load_dotenv()  # Load .env before Lightning/W&B read env vars

from lightning.pytorch.cli import LightningCLI  # noqa: E402

from uav_iqa.data import UAVQualityDataModule  # noqa: E402
from uav_iqa.training import UAVQualityTrainingModule  # noqa: E402


def main():
    LightningCLI(
        UAVQualityTrainingModule,
        UAVQualityDataModule,
        args=None,  # parse sys.argv
    )


if __name__ == "__main__":
    main()
