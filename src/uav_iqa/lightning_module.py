from typing import Any, Dict, List, Optional

import lightning as L
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from .distortion import UAV_DISTORTION_NAMES
from .losses import CrossTaskRegularization, ListMLELoss
from .metrics import (
    evaluate_iqa,
    per_distortion_metrics,
    per_task_metrics,
)
from .model import UAVIQANet


class UAVIQALightningModule(L.LightningModule):
    """LightningModule wrapping UAVIQANet with MSE + ListMLE + cross-task loss.

    All __init__ parameters are flat basic types for jsonargparse/LightningCLI compatibility.
    """

    def __init__(
        self,
        backbone: str = "mobilenetv4_conv_small",
        num_tasks: int = 14,
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

        self._val_preds: List[torch.Tensor] = []
        self._val_targets: List[torch.Tensor] = []
        self._val_tasks: List[int] = []
        self._val_distortions: List[str] = []
        self._test_preds: List[torch.Tensor] = []
        self._test_targets: List[torch.Tensor] = []
        self._test_tasks: List[int] = []
        self._test_distortions: List[str] = []

    def _gather_tensor(self, t: torch.Tensor) -> torch.Tensor:
        """Gather a tensor from all DDP processes, trimming padding from unequal splits."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return t
        local_size = torch.tensor([t.shape[0]], device=t.device, dtype=torch.long)
        sizes = self.all_gather(local_size).flatten()  # [world_size]
        gathered = self.all_gather(t)  # [world_size, N_max, ...]
        world_size = dist.get_world_size()
        chunks = [gathered[i][: sizes[i]] for i in range(world_size)]
        return torch.cat(chunks, dim=0)

    def _gather_objects(self, obj_list: list) -> list:
        """Gather arbitrary Python objects from all DDP processes."""
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return list(obj_list)
        world_size = dist.get_world_size()
        output: List[Any] = [None] * world_size
        dist.all_gather_object(output, obj_list)
        result: list = []
        for items in output:
            result.extend(items)
        return result

    def forward(
        self, x: torch.Tensor, task_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        return self.model(x, task_ids)

    def training_step(self, batch: Dict, _: int) -> torch.Tensor:
        images = batch["image"]
        task_ids = batch["task_id"]
        scores = batch.get("score", batch.get("cognitive_score"))

        # Share features: compute backbone+FPN+FAB once, reuse for both pred and cross-task
        f = self.model.forward_features(images)
        pred = self.model(images, task_ids, features=f)

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
            all_task_scores = self.model.forward_all_tasks(images, features=f)
            loss_ct = self.cross_task_loss(all_task_scores)

        loss = (
            loss_mse + self.lambda_rank * loss_rank + self.lambda_cross_task * loss_ct
        )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/mse", loss_mse, on_step=True, on_epoch=True)
        if loss_rank > 0:
            self.log("train/rank", loss_rank, on_step=True, on_epoch=True)
        if loss_ct > 0:
            self.log("train/cross_task", loss_ct, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: Dict, _: int) -> None:
        images = batch["image"]
        task_ids = batch["task_id"]
        scores = batch.get("score", batch.get("cognitive_score"))
        if scores is None:
            scores = batch.get("score")

        pred = self.model(images, task_ids)

        self._val_preds.append(pred.detach().cpu())
        self._val_targets.append(scores.detach().cpu())
        self._val_tasks.extend(batch["task_id"].cpu().tolist())
        self._val_distortions.extend(batch.get("distortion", []) or [])

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        preds_t = torch.cat(self._val_preds).to(self.device)
        targets_t = torch.cat(self._val_targets).to(self.device)
        tasks_t = torch.tensor(self._val_tasks, device=self.device, dtype=torch.long)

        preds = self._gather_tensor(preds_t).cpu().numpy()
        targets = self._gather_tensor(targets_t).cpu().numpy()
        tasks_all = self._gather_tensor(tasks_t).cpu().tolist()
        distortions_all = self._gather_objects(self._val_distortions)

        metrics = evaluate_iqa(preds, targets)
        self.log("val/srcc", metrics["srcc"], prog_bar=True)
        self.log("val/plcc", metrics["plcc"])
        self.log("val/krcc", metrics["kendall_tau"])

        if tasks_all:
            per_task = per_task_metrics(preds, targets, tasks_all)
            for name, info in per_task.items():
                self.log(f"val/srcc_{name}", info["srcc"])

        if distortions_all:
            per_dist = per_distortion_metrics(preds, targets, distortions_all)
            uav_srccs = []
            generic_srccs = []
            for d, info in per_dist.items():
                if d in UAV_DISTORTION_NAMES:
                    uav_srccs.append(info.get("srcc", 0))
                else:
                    generic_srccs.append(info.get("srcc", 0))
            if uav_srccs:
                self.log("val/srcc_uav", float(np.mean(uav_srccs)))
            if generic_srccs:
                self.log("val/srcc_generic", float(np.mean(generic_srccs)))

        self._val_preds.clear()
        self._val_targets.clear()
        self._val_tasks.clear()
        self._val_distortions.clear()

    def test_step(self, batch: Dict, _: int) -> None:
        images = batch["image"]
        task_ids = batch["task_id"]
        scores = batch.get("score", batch.get("cognitive_score"))
        if scores is None:
            scores = batch.get("score")

        pred = self.model(images, task_ids)

        self._test_preds.append(pred.detach().cpu())
        self._test_targets.append(scores.detach().cpu())
        self._test_tasks.extend(batch["task_id"].cpu().tolist())
        self._test_distortions.extend(batch.get("distortion", []) or [])

    def on_test_epoch_end(self) -> None:
        if not self._test_preds:
            return

        preds_t = torch.cat(self._test_preds).to(self.device)
        targets_t = torch.cat(self._test_targets).to(self.device)
        tasks_t = torch.tensor(self._test_tasks, device=self.device, dtype=torch.long)

        all_preds = self._gather_tensor(preds_t).cpu().numpy()
        all_targets = self._gather_tensor(targets_t).cpu().numpy()
        tasks_all = self._gather_tensor(tasks_t).cpu().tolist()
        distortions_all = self._gather_objects(self._test_distortions)

        metrics = evaluate_iqa(all_preds, all_targets)
        self.log("test/srcc", metrics["srcc"])
        self.log("test/plcc", metrics["plcc"])

        if tasks_all:
            per_task = per_task_metrics(all_preds, all_targets, tasks_all)
            for name, info in per_task.items():
                self.log(f"test/srcc_{name}", info["srcc"])

        if distortions_all:
            per_dist = per_distortion_metrics(all_preds, all_targets, distortions_all)
            uav_srccs = []
            generic_srccs = []
            for d, info in per_dist.items():
                if d in UAV_DISTORTION_NAMES:
                    uav_srccs.append(info.get("srcc", 0))
                else:
                    generic_srccs.append(info.get("srcc", 0))
            if uav_srccs:
                self.log("test/srcc_uav", float(np.mean(uav_srccs)))
            if generic_srccs:
                self.log("test/srcc_generic", float(np.mean(generic_srccs)))

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
