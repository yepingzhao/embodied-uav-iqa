import json
from pathlib import Path
from typing import Dict, List

import lightning as L


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
            "execution": (vlm_epochs + vla_epochs, vlm_epochs + vla_epochs + execution_epochs),
        }

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = trainer.current_epoch
        for stage, (start, end) in self.stage_boundaries.items():
            if start <= epoch < end:
                prev = getattr(pl_module, "curriculum_stage", None)
                if prev != stage:
                    pl_module.curriculum_stage = stage
                    if not trainer.is_global_zero:
                        continue
                    print(f"\n[Curriculum] Stage: {stage.upper()} (epochs {start+1}-{end})")
                break


class MetricsHistoryCallback(L.Callback):
    """Records training history and saves to history.json on fit end."""

    def __init__(self, output_dir: str = "outputs/training"):
        self.output_dir = Path(output_dir)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_mse": [],
            "train_rank": [],
            "train_cross_task": [],
            "val_srcc": [],
            "val_plcc": [],
        }
        self.best_val_srcc: float = -1.0
        self.best_model_path: str = ""

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logs = trainer.callback_metrics
        self.history["train_loss"].append(float(logs.get("train/loss_epoch", 0.0)))
        self.history["train_mse"].append(float(logs.get("train/mse_epoch", 0.0)))
        self.history["train_rank"].append(float(logs.get("train/rank_epoch", 0.0)))
        self.history["train_cross_task"].append(float(logs.get("train/cross_task_epoch", 0.0)))

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        logs = trainer.callback_metrics
        srcc = float(logs.get("val/srcc", 0.0))
        plcc = float(logs.get("val/plcc", 0.0))
        self.history["val_srcc"].append(srcc)
        self.history["val_plcc"].append(plcc)

        if srcc > self.best_val_srcc:
            self.best_val_srcc = srcc
            ckpt_path = self.output_dir / "best_model.pt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            self.best_model_path = str(ckpt_path)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        with open(self.output_dir / "history.json", "w") as f:
            json.dump(self.history, f, indent=2)
