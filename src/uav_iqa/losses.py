import torch
import torch.nn as nn


class ListMLELoss(nn.Module):
    """ListMLE ranking loss: model should correctly rank images by quality."""

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        _, indices = torch.sort(target, descending=True)
        pred_sorted = pred[indices]
        log_cumsum = torch.logcumsumexp(pred_sorted.flip(0), dim=0).flip(0)
        loss = -torch.sum(pred_sorted - log_cumsum)
        return loss / pred.shape[0]


class CrossTaskRegularization(nn.Module):
    """Encourage task-specific differentiation: penalize identical scores across tasks."""

    def forward(self, task_scores: torch.Tensor) -> torch.Tensor:
        # task_scores: (B, num_tasks)
        B, K = task_scores.shape
        if K < 2:
            return torch.tensor(0.0, device=task_scores.device)
        diffs = torch.tensor(0.0, device=task_scores.device)
        count = 0
        for i in range(K):
            for j in range(i + 1, K):
                diffs += torch.mean((task_scores[:, i] - task_scores[:, j]) ** 2)
                count += 1
        return -diffs / count
