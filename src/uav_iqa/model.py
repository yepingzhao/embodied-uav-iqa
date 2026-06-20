import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PanetFPN(nn.Module):
    """PANet-style Feature Pyramid Network for multi-scale feature fusion."""

    def __init__(self, in_channels: list, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ic, out_channels, 1) for ic in in_channels]
        )
        self.smooth_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels]
        )

        self.bottom_up_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
                for _ in range(len(in_channels) - 1)
            ]
        )

    def forward(self, features: list) -> list:
        n = len(features)
        lateral = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        td = [lateral[-1]]
        for i in range(n - 1):
            prev = td[-1]
            target_h, target_w = lateral[n - 2 - i].shape[2:]
            upsampled = F.interpolate(
                prev, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            td.append(upsampled + lateral[n - 2 - i])
        td = td[::-1]

        bu = []
        current = td[0]
        bu.append(self.smooth_convs[0](current))
        for i in range(n - 1):
            downsampled = self.bottom_up_convs[i](current)
            current = downsampled + td[i + 1]
            bu.append(self.smooth_convs[i + 1](current))

        return bu


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial attention)."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ca = self.channel_attn(x)
        x = x * ca
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x


class FrequencyAwareBranch(nn.Module):
    """Log-polar FFT branch for rotation/scale-invariant frequency features.

    Input: (B, 3, H, W) raw image
    Output: f_f ∈ (B, 64) frequency feature vector
    """

    def __init__(
        self,
        patch_size: int = 32,
        stride: int = 16,
        angular_bins: int = 32,
        radial_bins: int = 16,
        feat_dim: int = 64,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.angular_bins = angular_bins
        self.radial_bins = radial_bins

        self.tiny_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.proj = nn.Linear(16 * angular_bins * radial_bins, feat_dim)

        # Pre-compute log-polar bin indices for patch_size × patch_size grid (fixed size)
        cy = cx = patch_size / 2
        r_max = min(cy, cx)
        y_coords, x_coords = torch.meshgrid(
            torch.arange(patch_size, dtype=torch.float32),
            torch.arange(patch_size, dtype=torch.float32),
            indexing="ij",
        )
        dy, dx = y_coords - cy, x_coords - cx
        radius = torch.sqrt(dx**2 + dy**2) / r_max * (radial_bins - 1)
        angle = torch.atan2(dy, dx) / (2 * math.pi) * angular_bins
        r_idx = radius.clamp(0, radial_bins - 1).long()
        a_idx = angle.clamp(0, angular_bins - 1).long()
        self.register_buffer(
            "_lp_flat_idx",
            (a_idx * radial_bins + r_idx).view(-1),  # (patch_size * patch_size,)
            persistent=False,
        )

    def _log_polar(self, magnitude: torch.Tensor) -> torch.Tensor:
        B, C, *_ = magnitude.shape
        # Use pre-computed flat_idx, expanding for batch/channel dims
        flat_idx = self._lp_flat_idx.unsqueeze(0).unsqueeze(0).expand(B, C, -1)
        flat_mag = magnitude.reshape(B, C, -1)
        lp_flat = torch.zeros(
            B,
            C,
            self.angular_bins * self.radial_bins,
            device=magnitude.device,
            dtype=magnitude.dtype,
        )
        lp_flat.scatter_add_(2, flat_idx, flat_mag)
        return lp_flat.view(B, C, self.angular_bins, self.radial_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, *_ = x.shape
        x_gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        patches = x_gray.unfold(2, self.patch_size, self.stride).unfold(
            3, self.patch_size, self.stride
        )
        n_h, n_w = patches.shape[2], patches.shape[3]
        n_patches = n_h * n_w
        patches = patches.contiguous().view(
            B, n_patches, self.patch_size, self.patch_size
        )

        fft = torch.fft.fft2(patches.float())
        magnitude = torch.abs(fft)
        magnitude = torch.fft.fftshift(magnitude, dim=(-2, -1))

        # Process all patches at once: (B, n_patches, H, W) → (B*n_patches, 1, H, W)
        mag_all = magnitude.reshape(B * n_patches, 1, self.patch_size, self.patch_size)
        lp_all = self._log_polar(mag_all)  # (B*n_patches, 1, angular_bins, radial_bins)
        lp = lp_all.view(B, n_patches, self.angular_bins, self.radial_bins).mean(dim=1)

        lp = lp.view(B, 1, self.angular_bins, self.radial_bins)

        feats = self.tiny_cnn(lp)
        feats = feats.view(B, -1)
        f_f = self.proj(feats)
        return f_f


class CrossAttentionGate(nn.Module):
    """Gating mechanism: α = σ(W_g · [f_s, f_f]), producing α ∈ R^64."""

    def __init__(
        self, spatial_dim: int = 256, freq_dim: int = 64, hidden_dim: int = 128
    ):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(spatial_dim + freq_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, freq_dim),
            nn.Sigmoid(),
        )

    def forward(self, f_s: torch.Tensor, f_f: torch.Tensor) -> torch.Tensor:
        alpha = self.gate_net(torch.cat([f_s, f_f], dim=-1))
        return alpha * f_f


class TaskConditionedHead(nn.Module):
    """FiLM-modulated regression head per task.

    t_k ∈ R^4 → γ_k, β_k ∈ R^128 → h' = γ ⊙ h + β → FC(128→1).
    """

    def __init__(
        self,
        in_features: int = 320,
        hidden_dim: int = 128,
        task_embed_dim: int = 4,
        num_tasks: int = 4,
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


class UAVIQANet(nn.Module):
    """UAV-IQANet: frequency-aware task-conditioned lightweight NR-IQA model.

    Architecture:
        MobileNetV4-S backbone → PANet FPN → CBAM → f_s ∈ R^256
        FAB (patch FFT + tiny CNN) → f_f ∈ R^64
        Cross-attention gate → fused f ∈ R^320
        Task-conditioned regression heads (FiLM)

    Total: ~5.4M params, INT8 quantized ~1.4MB.
    """

    from .dataset import TASK_TO_ID

    TASK_MAP = TASK_TO_ID

    def __init__(
        self,
        backbone: str = "mobilenetv4_conv_small",
        num_tasks: int = 4,
        freeze_backbone_stage: int = 2,
        use_fab: bool = True,
        use_cbam: bool = True,
        use_task_conditioning: bool = True,
    ):
        super().__init__()
        self.use_fab = use_fab
        self.use_cbam = use_cbam
        self.use_task_conditioning = use_task_conditioning

        import timm

        valid_mobilenetv4 = timm.list_models("*mobilenetv4*")
        if backbone not in valid_mobilenetv4:
            raise ValueError(
                f"Backbone {backbone} not found in timm. "
                f"Available MobileNetV4 variants: {valid_mobilenetv4}"
            )

        try:
            self.backbone = timm.create_model(
                backbone,
                pretrained=True,
                features_only=True,
                out_indices=(2, 3, 4),
            )
        except Exception as e:
            _log = logging.getLogger(__name__)
            _log.warning(
                f"Could not load pretrained weights for {backbone}: {e}. "
                f"Falling back to random initialization — results may differ."
            )
            self.backbone = timm.create_model(
                backbone,
                pretrained=False,
                features_only=True,
                out_indices=(2, 3, 4),
            )

        dummy_in = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.backbone(dummy_in)

        in_channels = [f.shape[1] for f in feats]
        self.fpn = PanetFPN(in_channels)
        self.fpn_proj = nn.Conv2d(256, 256, 3, padding=1)

        if self.use_cbam:
            self.cbam = CBAM(256)

        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_proj = nn.Linear(256, 256)

        if self.use_fab:
            self.fab = FrequencyAwareBranch()
            self.gate = CrossAttentionGate()
            fused_dim = 320
        else:
            fused_dim = 256

        if self.use_task_conditioning:
            self.task_head = TaskConditionedHead(
                in_features=fused_dim,
                hidden_dim=128,
                task_embed_dim=4,
                num_tasks=num_tasks,
            )
        else:
            self.shared_head = nn.Sequential(
                nn.Linear(fused_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        if freeze_backbone_stage > 0:
            self._freeze_backbone_stages(freeze_backbone_stage)

    def _freeze_backbone_stages(self, num_stages: int):
        frozen_count = 0
        for name, param in self.backbone.named_parameters():
            if any(f"stages.{i}" in name for i in range(num_stages)):
                param.requires_grad = False
                frozen_count += 1
        if frozen_count == 0:
            _log = logging.getLogger(__name__)
            _log.warning(
                f"No parameters matched freeze pattern 'stages.N' — "
                f"backbone {self.backbone.__class__.__name__} may use different naming."
            )

    def _extract_fused_features(self, x: torch.Tensor) -> torch.Tensor:
        """Shared backbone → FPN → CBAM → FAB → gate pipeline."""
        feats = self.backbone(x)
        feats = self.fpn(feats)

        x_spatial = self.fpn_proj(feats[0])
        if self.use_cbam:
            x_spatial = self.cbam(x_spatial)

        x_spatial = self.spatial_pool(x_spatial).flatten(1)
        f_s = self.spatial_proj(x_spatial)

        if self.use_fab:
            f_f = self.fab(x)
            f_f = self.gate(f_s, f_f)
            return torch.cat([f_s, f_f], dim=-1)
        return f_s

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract fused features (backbone + FPN + CBAM + FAB) once for reuse."""
        return self._extract_fused_features(x)

    def forward(
        self,
        x: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        f = features if features is not None else self._extract_fused_features(x)

        if self.use_task_conditioning:
            if task_ids is None:
                task_ids = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return self.task_head(f, task_ids)
        else:
            return self.shared_head(f).squeeze(-1)

    def forward_all_tasks(
        self, x: torch.Tensor, features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        f = features if features is not None else self._extract_fused_features(x)

        if self.use_task_conditioning:
            B = x.shape[0]
            scores = []
            for t in range(self.task_head.num_tasks):
                task_ids = torch.full((B,), t, dtype=torch.long, device=x.device)
                scores.append(self.task_head(f, task_ids))
            return torch.stack(scores, dim=1)
        else:
            q = self.shared_head(f).squeeze(-1)
            return q.unsqueeze(-1).expand(-1, 4)
