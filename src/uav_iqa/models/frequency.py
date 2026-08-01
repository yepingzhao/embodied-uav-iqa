import math

import torch
import torch.nn as nn


class LogPolarFrequencyEncoder(nn.Module):
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
        patches = patches.contiguous().view(B, n_patches, self.patch_size, self.patch_size)

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
        return self.proj(feats)


class FrequencyFeatureGate(nn.Module):
    """Gating mechanism: α = σ(W_g · [f_s, f_f]), producing α ∈ R^64."""

    def __init__(self, spatial_dim: int = 256, freq_dim: int = 64, hidden_dim: int = 128):
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
