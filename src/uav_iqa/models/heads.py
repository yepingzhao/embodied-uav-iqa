import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskConditionedRegressor(nn.Module):
    """FiLM-modulated regression head per task.

    t_k ∈ R^4 → γ_k, β_k ∈ R^128 → h' = γ ⊙ h + β → FC(128→1).
    """

    def __init__(
        self,
        in_features: int = 320,
        hidden_dim: int = 128,
        task_embed_dim: int = 4,
        num_tasks: int = 14,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.task_embed = nn.Embedding(num_tasks, task_embed_dim)
        nn.init.normal_(self.task_embed.weight, std=0.02)

        self.gamma_proj = nn.Linear(task_embed_dim, hidden_dim)
        self.beta_proj = nn.Linear(task_embed_dim, hidden_dim)

        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, f: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        t = self.task_embed(task_ids)
        gamma = self.gamma_proj(t)
        beta = self.beta_proj(t)

        h = F.relu(self.fc1(f))
        h = gamma * h + beta
        q = torch.sigmoid(self.fc2(h))
        return q.squeeze(-1)


def create_shared_regressor(in_features: int, hidden_dim: int = 128) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
        nn.Sigmoid(),
    )
