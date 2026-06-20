import hashlib
from pathlib import Path

import lightning as L

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
                    trainer.print(
                        f"\n[Curriculum] Stage: {stage.upper()} (epochs {start+1}-{end})"
                    )
                break


class SetupRunCallback(L.Callback):
    """Logs dataset/environment info at fit start.

    - Computes and prints manifest SHA256 hash for dataset versioning.
    - Prints trainable parameter count.
    """

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        manifest_hash = self._compute_manifest_hash(trainer)
        if manifest_hash:
            trainer.print(f"Manifest hash: {manifest_hash}")

        pl_module.manifest_hash = manifest_hash

        n_params = count_parameters(pl_module.model)[0]
        trainer.print(f"Model params: {n_params:,}")

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


