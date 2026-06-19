from typing import Dict, List, Optional

import lightning as L
import numpy as np
import torch
import torch.nn as nn

from .evaluate import compute_metrics, evaluate_iqa, per_distortion_metrics, per_task_metrics
from .model import UAVIQANet
from .trainer import CrossTaskRegularization, ListMLELoss


class UAVIQALightningModule(L.LightningModule):
    """LightningModule wrapping UAVIQANet with MSE + ListMLE + cross-task loss.

    All __init__ parameters are flat basic types for jsonargparse/LightningCLI compatibility.
    """

    def __init__(
        self,
        backbone: str = "mobilenetv4_conv_small",
        num_tasks: int = 4,
        use_fab: bool = True,
        use_cbam: bool = True,
        use_task_conditioning: bool = True,
        freeze_backbone_stage: int = 2,
        lambda_rank: float = 0.3,
        lambda_cross_task: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
        annotator_stage: str = "vla",
    ):
        super().__init__()
        self.save_hyperparameters()

        model_cfg = dict(
            backbone=backbone,
            num_tasks=num_tasks,
            use_fab=use_fab,
            use_cbam=use_cbam,
            use_task_conditioning=use_task_conditioning,
            freeze_backbone_stage=freeze_backbone_stage,
        )
        self.model = UAVIQANet(**model_cfg)

        self.mse_loss = nn.MSELoss()
        self.rank_loss = ListMLELoss()
        self.cross_task_loss = CrossTaskRegularization()

        self.lambda_rank = lambda_rank
        self.lambda_cross_task = lambda_cross_task
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.annotator_stage = annotator_stage
        self.curriculum_stage = "vlm"

        self._val_preds: List[np.ndarray] = []
        self._val_targets: List[np.ndarray] = []
        self._test_preds: List[np.ndarray] = []
        self._test_targets: List[np.ndarray] = []
        self._test_tasks: List[int] = []
        self._test_distortions: List[str] = []

    def forward(self, x: torch.Tensor, task_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.model(x, task_ids)

    def training_step(self, batch: Dict, batch_idx: int) -> torch.Tensor:
        images = batch["image"]
        task_ids = batch["task_id"]
        score_key = f"{self.curriculum_stage}_score"
        scores = batch.get(score_key, batch.get("score"))

        pred = self.model(images, task_ids)

        loss_mse = self.mse_loss(pred, scores)

        loss_rank = torch.tensor(0.0, device=self.device)
        distortions = batch.get("distortion")
        if distortions:
            # Build a tensor of distortion indices for batched mask construction
            unique_dists = list(set(distortions))
            dist_to_idx = {d: i for i, d in enumerate(unique_dists)}
            dist_ids = torch.tensor(
                [dist_to_idx[d] for d in distortions], device=self.device
            )
            for idx, dist in enumerate(unique_dists):
                mask = dist_ids == idx
                if mask.sum() >= 3:
                    loss_rank = loss_rank + self.rank_loss(pred[mask], scores[mask])

        loss_ct = torch.tensor(0.0, device=self.device)
        if self.model.use_task_conditioning:
            all_task_scores = self.model.forward_all_tasks(images)
            loss_ct = self.cross_task_loss(all_task_scores)

        loss = loss_mse + self.lambda_rank * loss_rank + self.lambda_cross_task * loss_ct

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/mse", loss_mse, on_step=False, on_epoch=True)
        if loss_rank > 0:
            self.log("train/rank", loss_rank, on_step=False, on_epoch=True)
        if loss_ct > 0:
            self.log("train/cross_task", loss_ct, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict, batch_idx: int) -> None:
        images = batch["image"]
        task_ids = batch["task_id"]
        eval_stage = self.annotator_stage
        scores = batch.get(f"{eval_stage}_score", batch.get("score"))
        if scores is None:
            scores = batch.get("score")

        pred = self.model(images, task_ids)

        self._val_preds.append(pred.detach().cpu().numpy())
        self._val_targets.append(scores.detach().cpu().numpy())

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        preds = np.concatenate(self._val_preds)
        targets = np.concatenate(self._val_targets)

        metrics = evaluate_iqa(preds, targets)
        self.log("val/srcc", metrics["SRCC"], prog_bar=True)
        self.log("val/plcc", metrics["PLCC"])

        self._val_preds.clear()
        self._val_targets.clear()

    def test_step(self, batch: Dict, batch_idx: int) -> None:
        images = batch["image"]
        task_ids = batch["task_id"]
        eval_stage = self.annotator_stage
        scores = batch.get(f"{eval_stage}_score", batch.get("score"))
        if scores is None:
            scores = batch.get("score")

        pred = self.model(images, task_ids)

        self._test_preds.append(pred.detach().cpu().numpy())
        self._test_targets.append(scores.detach().cpu().numpy())
        self._test_tasks.extend(batch["task_id"].cpu().tolist())
        self._test_distortions.extend(batch.get("distortion", []) or [])

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        all_preds = np.concatenate(self._test_preds)
        all_targets = np.concatenate(self._test_targets)

        metrics = evaluate_iqa(all_preds, all_targets)
        self.log("test/srcc", metrics["SRCC"])
        self.log("test/plcc", metrics["PLCC"])

        self._test_results = {
            "all_preds": all_preds,
            "all_targets": all_targets,
            "tasks": list(self._test_tasks),
            "distortions": list(self._test_distortions),
        }

        self._test_preds.clear()
        self._test_targets.clear()
        self._test_tasks.clear()
        self._test_distortions.clear()

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        if self.warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-4, total_iters=self.warmup_epochs
            )
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.total_epochs - self.warmup_epochs
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, [warmup, cosine], milestones=[self.warmup_epochs]
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.total_epochs
            )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def get_test_results(self) -> Dict:
        if not hasattr(self, "_test_results") or not self._test_results:
            return {}

        r = self._test_results
        metrics = compute_metrics(r["all_targets"], r["all_preds"])

        results = {
            "test_metrics": metrics,
            "preds": r["all_preds"].tolist(),
            "targets": r["all_targets"].tolist(),
        }

        if r["tasks"]:
            results["per_task"] = per_task_metrics(
                r["all_targets"], r["all_preds"], r["tasks"]
            )

        if r["distortions"]:
            results["per_distortion"] = per_distortion_metrics(
                r["all_preds"], r["all_targets"], r["distortions"]
            )

        return results
