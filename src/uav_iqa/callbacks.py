import hashlib
import json
import subprocess
from pathlib import Path
from typing import Dict, List

import lightning as L
import yaml

from .utils import count_parameters


class CurriculumStageCallback(L.Callback):
    """Sets model.curriculum_stage based on current epoch for 3-stage curriculum."""

    def __init__(
        self,
        vlm_epochs: int = 20,
        vla_epochs: int = 20,
        execution_epochs: int = 10,
    ):
        super().__init__()
        self.stage_boundaries = {
            "vlm": (0, vlm_epochs),
            "vla": (vlm_epochs, vlm_epochs + vla_epochs),
            "execution": (
                vlm_epochs + vla_epochs,
                vlm_epochs + vla_epochs + execution_epochs,
            ),
        }

    def on_train_epoch_start(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        epoch = trainer.current_epoch
        for stage, (start, end) in self.stage_boundaries.items():
            if start <= epoch < end:
                prev = getattr(pl_module, "curriculum_stage", None)
                if prev != stage:
                    pl_module.curriculum_stage = stage
                    if not trainer.is_global_zero:
                        continue
                    print(
                        f"\n[Curriculum] Stage: {stage.upper()} (epochs {start+1}-{end})"
                    )
                break


class MetricsHistoryCallback(L.Callback):
    """Records training history and saves to history.json on fit end.

    Captured series:
      - train: loss, mse, rank, cross_task
      - val: srcc, plcc, per-task srcc, per-distortion srcc (uav/generic avg)

    Saves to ``trainer.default_root_dir / "history.json"``.
    Also exposes ``best_val_srcc`` on the pl_module for downstream callbacks.
    """

    def __init__(self):
        super().__init__()
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_mse": [],
            "train_rank": [],
            "train_cross_task": [],
            "val_srcc": [],
            "val_plcc": [],
        }
        self.best_val_srcc: float = -1.0

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        logs = trainer.callback_metrics
        self.history["train_loss"].append(float(logs.get("train/loss_epoch", 0.0)))
        self.history["train_mse"].append(float(logs.get("train/mse_epoch", 0.0)))
        self.history["train_rank"].append(float(logs.get("train/rank_epoch", 0.0)))
        self.history["train_cross_task"].append(
            float(logs.get("train/cross_task_epoch", 0.0))
        )

    def on_validation_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        logs = trainer.callback_metrics
        srcc = float(logs.get("val/srcc", 0.0))
        plcc = float(logs.get("val/plcc", 0.0))
        self.history["val_srcc"].append(srcc)
        self.history["val_plcc"].append(plcc)

        if srcc > self.best_val_srcc:
            self.best_val_srcc = srcc

        # Record per-task SRCC series
        per_task = getattr(pl_module, "val_per_task_srcc", {})
        for task_name, task_srcc in per_task.items():
            key = f"val_srcc_{task_name}"
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(task_srcc)

        # Record UAV/generic SRCC aggregates
        uav_srcc = float(logs.get("val/srcc_uav", 0.0))
        generic_srcc = float(logs.get("val/srcc_generic", 0.0))
        if uav_srcc > 0:
            if "val_srcc_uav" not in self.history:
                self.history["val_srcc_uav"] = []
            self.history["val_srcc_uav"].append(uav_srcc)
        if generic_srcc > 0:
            if "val_srcc_generic" not in self.history:
                self.history["val_srcc_generic"] = []
            self.history["val_srcc_generic"].append(generic_srcc)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        run_dir = Path(trainer.default_root_dir)
        with open(run_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        # Expose for downstream callbacks (e.g. ResultsSavingCallback)
        pl_module.best_val_srcc = self.best_val_srcc


class SetupRunCallback(L.Callback):
    """Logs dataset/environment info at fit start.

    - Computes and prints manifest SHA256 hash for dataset versioning.
    - Prints trainable parameter count.
    - Stores manifest hash on pl_module for downstream callbacks.
    """

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        manifest_hash = self._compute_manifest_hash(trainer)
        if manifest_hash and trainer.is_global_zero:
            print(f"Manifest hash: {manifest_hash}")

        pl_module.manifest_hash = manifest_hash

        n_params = count_parameters(pl_module.model)[0]
        if trainer.is_global_zero:
            print(f"Model params: {n_params:,}")

    @staticmethod
    def _compute_manifest_hash(trainer: L.Trainer) -> str:
        """Compute SHA256 hash of all manifest files for dataset versioning."""
        datamodule = trainer.datamodule
        if datamodule is None:
            return ""
        data_root = getattr(datamodule, "data_root", None)
        if data_root is None:
            return ""
        hasher = hashlib.sha256()
        files_read = 0
        for split in ("train", "val", "test"):
            manifest_path = Path(data_root) / split / "manifest.json"
            if manifest_path.exists():
                with open(manifest_path, "rb") as f:
                    hasher.update(f.read())
                files_read += 1
        return hasher.hexdigest()[:16] if files_read > 0 else ""


class ResultsSavingCallback(L.Callback):
    """On fit end: load best checkpoint, run test, print & save results.

    Reads ``trainer.default_root_dir`` for output path,
    ``pl_module.best_val_srcc`` (set by MetricsHistoryCallback), and
    ``pl_module.manifest_hash`` (set by SetupRunCallback).
    """

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        run_dir = Path(trainer.default_root_dir)

        # Find best checkpoint from ModelCheckpoint
        best_path = None
        for cb in trainer.checkpoint_callbacks:
            if isinstance(cb, L.pytorch.callbacks.ModelCheckpoint):
                best_path = cb.best_model_path
                break

        if best_path and Path(best_path).exists():
            print(f"Best checkpoint: {best_path}")
            trainer.test(pl_module, datamodule=trainer.datamodule, ckpt_path=best_path)
        else:
            trainer.test(pl_module, datamodule=trainer.datamodule)

        test_results = pl_module.get_test_results()
        best_val_srcc = getattr(pl_module, "best_val_srcc", -1.0)
        test_metrics = test_results.get(
            "test_metrics", {"srcc": 0.0, "plcc": 0.0, "rmse": 0.0}
        )

        if trainer.is_global_zero:
            print(
                f"\n  Test: SRCC={test_metrics.get('srcc', 0):.4f} | "
                f"PLCC={test_metrics.get('plcc', 0):.4f} | "
                f"RMSE={test_metrics.get('rmse', 0):.4f}"
            )

        per_task = test_results.get("per_task", {})
        if per_task and trainer.is_global_zero:
            print(
                "  Per-task SRCC:",
                {k: f"{v['srcc']:.4f}" for k, v in per_task.items()},
            )

        n_params = count_parameters(pl_module.model)[0]
        result = {
            "seed": self._read_seed_from_config(run_dir),
            "n_params": n_params,
            "best_val_srcc": best_val_srcc,
            "test_metrics": {k: float(v) for k, v in test_metrics.items()},
            "per_task": per_task,
            "hparams": dict(pl_module.hparams),
            "data_config": {
                k: str(v)
                for k, v in (
                    trainer.datamodule.hparams.items()
                    if trainer.datamodule is not None
                    else []
                )
            },
            "manifest_hash": getattr(pl_module, "manifest_hash", ""),
            "git_commit": self._get_git_commit(),
        }
        with open(run_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)

    @staticmethod
    def _get_git_commit() -> str:
        """Get the current git commit SHA for reproducibility."""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except Exception:
            import logging

            _log = logging.getLogger(__name__)
            _log.warning(
                "Could not determine git commit hash — results.json will lack git_commit"
            )
            return ""

    @staticmethod
    def _read_seed_from_config(run_dir: Path) -> int:
        """Read seed from the config.yaml saved by LightningCLI's SaveConfigCallback."""
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get("seed_everything", 0)
        return 0
