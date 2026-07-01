import hashlib
import json
import logging
import subprocess
import warnings
from pathlib import Path

import lightning as L
import yaml

from .utils import count_parameters

_log = logging.getLogger(__name__)


class CurriculumStageCallback(L.Callback):
    """DEPRECATED: The 3-stage curriculum system (VLM→VLA→Execution) has been
    superseded by direct cognitive_score aggregation. This callback is kept for
    backward compatibility but is now a no-op.
    """

    def __init__(
        self,
        vlm_epochs: int = 20,
        vla_epochs: int = 20,
        execution_epochs: int = 10,
    ):
        super().__init__()

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not getattr(self, "_warned_deprecation", False):
            self._warned_deprecation = True
            warnings.warn(
                "CurriculumStageCallback is deprecated and all parameters "
                "(vlm_epochs, vla_epochs, execution_epochs) are ignored.",
                DeprecationWarning,
                stacklevel=2,
            )


class SetupRunCallback(L.Callback):
    """Logs dataset/environment info at fit start.

    - Computes and prints dataset SHA256 hash for versioning.
    - Prints trainable parameter count.
    """

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        dataset_hash = self._compute_dataset_hash(trainer)
        if dataset_hash and trainer.is_global_zero:
            _log.info("Dataset hash: %s", dataset_hash)

        pl_module.dataset_hash = dataset_hash

        n_params = count_parameters(pl_module.model)[0]
        if trainer.is_global_zero:
            _log.info("Model params: %s", f"{n_params:,}")

    @staticmethod
    def _compute_dataset_hash(trainer: L.Trainer) -> str:
        datamodule = trainer.datamodule
        if datamodule is None:
            return ""
        data_root = getattr(datamodule, "data_root", None)
        if data_root is None:
            return ""
        hasher = hashlib.sha256()
        files_read = 0
        for split in ("train", "test"):
            split_dir = Path(data_root) / split
            if split_dir.is_dir():
                for fpath in sorted(split_dir.glob("*_VQA_*.json")):
                    with open(fpath, "rb") as f:
                        hasher.update(f.read())
                    files_read += 1
        return hasher.hexdigest()[:16] if files_read > 0 else ""


class MetricsHistoryCallback(L.Callback):
    """Tracks epoch-level metrics and writes history.json on fit end."""

    def __init__(self):
        super().__init__()
        self.history = {}
        self.best_val_srcc = -1.0

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logs = trainer.callback_metrics
        for key in ("train/loss_epoch", "val/loss", "val/srcc", "val/plcc", "val/rmse"):
            value = float(logs.get(key, 0.0))
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)

        val_srcc = float(logs.get("val/srcc", -1.0))
        if val_srcc > self.best_val_srcc:
            self.best_val_srcc = val_srcc

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return
        run_dir = Path(trainer.default_root_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
        pl_module.best_val_srcc = self.best_val_srcc


class ResultsSavingCallback(L.Callback):
    """On fit end: load best checkpoint, run test, save results.json."""

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        if not trainer.is_global_zero:
            return

        run_dir = Path(trainer.default_root_dir)

        best_path = None
        for cb in trainer.checkpoint_callbacks:
            if isinstance(cb, L.pytorch.callbacks.ModelCheckpoint):
                bp = getattr(cb, "best_model_path", None) or ""
                if bp and Path(bp).exists():
                    best_path = bp
                    break

        if best_path:
            _log.info("Best checkpoint: %s", best_path)
            trainer.test(pl_module, datamodule=trainer.datamodule, ckpt_path=best_path)
        else:
            trainer.test(pl_module, datamodule=trainer.datamodule)

        cb_metrics = trainer.callback_metrics
        test_metrics = {
            "srcc": float(cb_metrics.get("test/srcc", 0.0)),
            "plcc": float(cb_metrics.get("test/plcc", 0.0)),
        }
        rmse = cb_metrics.get("test/rmse")
        if rmse is not None:
            test_metrics["rmse"] = float(rmse)

        per_task = {}
        for key in cb_metrics:
            if key.startswith("test/srcc_") and key not in (
                "test/srcc_uav",
                "test/srcc_generic",
            ):
                task_name = key.replace("test/srcc_", "")
                per_task[task_name] = {"srcc": float(cb_metrics[key])}

        _log.info(
            "\n  Test: SRCC=%s | PLCC=%s",
            f"{test_metrics.get('srcc', 0):.4f}",
            f"{test_metrics.get('plcc', 0):.4f}",
        )

        best_val_srcc = getattr(pl_module, "best_val_srcc", -1.0)
        n_params = count_parameters(pl_module.model)[0]

        result = {
            "seed": self._read_seed_from_config(run_dir),
            "n_params": n_params,
            "best_val_srcc": best_val_srcc,
            "test_metrics": test_metrics,
            "per_task": per_task,
            "hparams": dict(getattr(pl_module, "hparams", {})),
            "data_config": {
                k: str(v)
                for k, v in (
                    trainer.datamodule.hparams.items() if trainer.datamodule is not None else []
                )
            },
            "dataset_hash": getattr(pl_module, "dataset_hash", ""),
            "git_commit": self._get_git_commit(),
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "results.json", "w") as f:
            json.dump(result, f, indent=2)

    @staticmethod
    def _get_git_commit() -> str:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.stdout.strip()
        except Exception:
            _log.warning("Could not determine git commit hash")
            return ""

    @staticmethod
    def _read_seed_from_config(run_dir: Path) -> int:
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                return config.get("seed_everything", 0)
            except (yaml.YAMLError, OSError):
                _log.warning("Could not parse config.yaml", exc_info=True)
        return 0
