import warnings
from typing import Optional, Dict

import torch
import torch.nn as nn

from .evaluate import compute_srcc, compute_plcc


class ListMLELoss(nn.Module):
    """ListMLE ranking loss: model should correctly rank images by quality."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        device = pred.device
        _, indices = torch.sort(target, descending=True)
        pred_sorted = pred[indices]
        log_cumsum = torch.logcumsumexp(
            pred_sorted.flip(0), dim=0
        ).flip(0)
        loss = -torch.sum(pred_sorted - log_cumsum)
        return loss / pred.shape[0]


class CrossTaskRegularization(nn.Module):
    """Encourage task-specific differentiation: penalize identical scores across tasks."""

    def forward(self, task_scores: torch.Tensor) -> torch.Tensor:
        # task_scores: (B, num_tasks)
        B, K = task_scores.shape
        if K < 2:
            return torch.tensor(0.0, device=task_scores.device)
        diffs = 0.0
        count = 0
        for i in range(K):
            for j in range(i + 1, K):
                diffs += torch.mean((task_scores[:, i] - task_scores[:, j]) ** 2)
                count += 1
        return -diffs / count


class UAVQATrainer:
    """Legacy trainer — use UAVQALightningModule + L.Trainer instead."""

    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda",
        lambda_rank: float = 0.3,
        lambda_cross_task: float = 0.1,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        total_epochs: int = 50,
    ):
        warnings.warn(
            "UAVQATrainer is deprecated, use UAVQALightningModule + L.Trainer instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.model = model.to(device)
        self.device = device
        self.lambda_rank = lambda_rank
        self.lambda_cross_task = lambda_cross_task

        self.mse_loss = nn.MSELoss()
        self.rank_loss = ListMLELoss()
        self.cross_task_loss = CrossTaskRegularization()

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs - warmup_epochs
        )
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def train_epoch(self, dataloader, curriculum_stage: str = "vla") -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_rank = 0.0
        total_ct = 0.0
        n_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            scores = batch["score"].to(self.device)
            task_ids = batch.get("task_id")
            if task_ids is not None:
                task_ids = task_ids.to(self.device)

            self.optimizer.zero_grad()

            pred = self.model(images, task_ids)

            loss_mse = self.mse_loss(pred, scores)

            loss_rank = torch.tensor(0.0, device=self.device)
            if batch.get("distortion"):
                unique_dists = set(batch["distortion"])
                for dist in unique_dists:
                    mask = torch.tensor(
                        [d == dist for d in batch["distortion"]], device=self.device
                    )
                    if mask.sum() >= 3:
                        loss_rank += self.rank_loss(pred[mask], scores[mask])

            loss_ct = torch.tensor(0.0, device=self.device)
            if self.model.use_task_conditioning:
                all_task_scores = self.model.forward_all_tasks(images)
                loss_ct = self.cross_task_loss(all_task_scores)

            loss = loss_mse + self.lambda_rank * loss_rank + self.lambda_cross_task * loss_ct
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_mse += loss_mse.item()
            total_rank += loss_rank.item()
            total_ct += loss_ct.item()
            n_batches += 1

        if self.current_epoch >= self.warmup_epochs:
            self.scheduler.step()

        self.current_epoch += 1

        return {
            "loss": total_loss / n_batches,
            "mse": total_mse / n_batches,
            "rank": total_rank / n_batches,
            "cross_task": total_ct / n_batches,
        }

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict:
        self.model.eval()
        all_preds = []
        all_targets = []
        all_tasks = []

        for batch in dataloader:
            images = batch["image"].to(self.device)
            scores = batch["score"].to(self.device)
            task_ids = batch.get("task_id")

            pred = self.model(images, task_ids.to(self.device) if task_ids is not None else None)

            all_preds.append(pred.cpu().numpy())
            all_targets.append(scores.cpu().numpy())
            if task_ids is not None:
                all_tasks.append(task_ids.numpy())

        import numpy as np

        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        tasks = np.concatenate(all_tasks) if all_tasks else None

        metrics = {
            "SRCC": compute_srcc(preds, targets),
            "PLCC": compute_plcc(preds, targets),
        }

        return metrics

    def train_curriculum(self, dataloader_vlm, dataloader_vla, dataloader_exec):
        """3-stage curriculum: VLM → VLA → Execution."""
        print("[Curriculum] Stage 1: VLM annotations (epochs 1-20)")
        for epoch in range(1, 21):
            metrics = self.train_epoch(dataloader_vlm)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: loss={metrics['loss']:.4f}, mse={metrics['mse']:.4f}")

        print("[Curriculum] Stage 2: VLA annotations (epochs 21-40)")
        for epoch in range(21, 41):
            metrics = self.train_epoch(dataloader_vla)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: loss={metrics['loss']:.4f}, mse={metrics['mse']:.4f}")

        print("[Curriculum] Stage 3: Execution annotations (epochs 41-50)")
        for epoch in range(41, 51):
            metrics = self.train_epoch(dataloader_exec)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: loss={metrics['loss']:.4f}, mse={metrics['mse']:.4f}")
