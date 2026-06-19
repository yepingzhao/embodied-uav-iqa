#!/usr/bin/env python3
"""M3-R013/R014/R015: Full training pipeline for UAV-IQANet using LightningCLI.

Supports:
- Multi-seed training (--seeds 42 100 200)
- Ablation toggles (--model.init_args.use_fab false)
- Per-task / leave-one-out (--data.init_args.task tracking)
- Distortion filter (--data.init_args.distortion_filter generic)
- Dry run (--dry_run true)

Usage:
    # Standard single-seed training
    python scripts/run_m3_train.py fit --config configs/default.yaml

    # Multi-seed
    python scripts/run_m3_train.py --seeds 42 100 200

    # Ablation — w/o FAB
    python scripts/run_m3_train.py --model.init_args.use_fab false

    # Per-task training
    python scripts/run_m3_train.py --data.init_args.task tracking

    # Leave-one-out
    python scripts/run_m3_train.py --data.init_args.leave_out_task sar

    # Dry run
    python scripts/run_m3_train.py --dry_run true --trainer.max_epochs 3
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Override broken hf-mirror.com with the main HF endpoint for pretrained backbone weights.
# The mirror (hf-mirror.com) returns 308 redirects that timm/huggingface_hub can't follow.
if os.environ.get("HF_ENDPOINT", "").startswith("https://hf-mirror.com"):
    os.environ["HF_ENDPOINT"] = "https://huggingface.co"

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.uav_iqa.cli import UAVIQACLI
from src.uav_iqa.lightning_data import UAVIQDataModule
from src.uav_iqa.lightning_model import UAVIQALightningModule


def main():
    parser = argparse.ArgumentParser(
        description="Train UAV-IQANet with LightningCLI (multi-seed wrapper)",
        add_help=False,
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds")
    parser.add_argument("--output_dir", type=str, default="outputs/training")
    parser.add_argument("--dry_run", action="store_true")

    cli_args, remaining = parser.parse_known_args()

    if cli_args.dry_run:
        remaining.extend(["--data.dry_run", "true"])

    output_dir = Path(cli_args.output_dir)
    all_results = {}

    for seed_idx, seed in enumerate(cli_args.seeds):
        print(f"\n{'='*60}")
        print(f"Seed {seed} ({seed_idx + 1}/{len(cli_args.seeds)})")
        print(f"{'='*60}")

        seed_argv = [
            "fit",
            f"--seed={seed}",
            f"--output_dir={cli_args.output_dir}",
            *remaining,
        ]

        _cli = UAVIQACLI(
            UAVIQALightningModule,
            UAVIQDataModule,
            args=seed_argv,
        )

        results_path = output_dir / f"seed{seed}" / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                all_results[f"seed{seed}"] = json.load(f)

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    import numpy as np

    srccs = [r["test_metrics"]["srcc"] for r in all_results.values()]
    print(f"\n{'='*60}")
    print(f"Summary over {len(cli_args.seeds)} seeds:")
    if srccs:
        print(f"  SRCC: {np.mean(srccs):.4f} +/- {np.std(srccs):.4f}")
    print(f"  Results saved to {summary_path}")


if __name__ == "__main__":
    main()
