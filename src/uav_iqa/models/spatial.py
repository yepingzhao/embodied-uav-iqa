import torch
import torch.nn as nn
import torch.nn.functional as F


class PANFeaturePyramid(nn.Module):
    """PANet-style Feature Pyramid Network for multi-scale feature fusion."""

    def __init__(self, in_channels: list, out_channels: int = 256):
        super().__init__()
        self.out_channels = out_channels

        self.lateral_convs = nn.ModuleList([nn.Conv2d(ic, out_channels, 1) for ic in in_channels])
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


class ConvolutionalBlockAttention(nn.Module):
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
        return x * sa
