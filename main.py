#!/usr/bin/env python3
"""Unified training entry point for UAV-IQANet.

All hyperparameters come from self-contained YAML config files.
Usage::

    python main.py fit --config configs/experiments/r016_no_fab.yaml
    python main.py fit --config configs/experiments/r024a_tracking.yaml --seed_everything 100
"""

from lightning.pytorch.cli import LightningCLI

from uav_iqa.data_module import UAVIQDataModule
from uav_iqa.lightning_module import UAVIQALightningModule


def main():
    LightningCLI(
        UAVIQALightningModule,
        UAVIQDataModule,
        args=None,  # parse sys.argv
    )


if __name__ == "__main__":
    main()
