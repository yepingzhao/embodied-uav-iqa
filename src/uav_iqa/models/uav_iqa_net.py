import logging
import re
from typing import List, Optional

import torch
import torch.nn as nn

from .frequency import FrequencyFeatureGate, LogPolarFrequencyEncoder
from .heads import TaskConditionedRegressor, create_shared_regressor
from .spatial import ConvolutionalBlockAttention, PANFeaturePyramid
from .text import QuestionTextEncoder

_log = logging.getLogger(__name__)


class UAVIQANet(nn.Module):
    """UAV-IQANet: frequency-aware task-conditioned lightweight NR-IQA model.

    Architecture:
        MobileNetV4-S backbone -> PANet FPN -> CBAM -> spatial_features in R^256
        Log-polar frequency encoder (patch FFT + tiny CNN) -> frequency_features in R^64
        Frequency feature gate -> fused in R^320
        (optional) Char-CNN question text encoder -> text_features in R^128
        Task-conditioned regression heads (FiLM)

    Total: ~5.4M params, INT8 quantized ~1.4MB.
    """

    def __init__(
        self,
        backbone: str = "mobilenetv4_conv_small",
        num_tasks: int | None = None,
        freeze_backbone_stage: int = 2,
        use_frequency_encoder: bool = True,
        use_spatial_attention: bool = True,
        use_task_conditioning: bool = True,
        use_text_encoder: bool = True,
        text_dim: int = 128,
    ):
        super().__init__()
        if num_tasks is None:
            from uav_iqa.domain import NUM_SUBTASKS

            num_tasks = NUM_SUBTASKS
        self.use_frequency_encoder = use_frequency_encoder
        self.use_spatial_attention = use_spatial_attention
        self.use_task_conditioning = use_task_conditioning
        self.use_text_encoder = use_text_encoder
        self.num_tasks = num_tasks
        self.text_dim = text_dim

        import timm

        if backbone not in timm.list_models():
            raise ValueError(
                f"Backbone {backbone!r} not found in timm. "
                "Use timm.list_models() to see available backbones."
            )

        probe = timm.create_model(backbone, pretrained=False, features_only=True)
        with torch.no_grad():
            n_stages = len(probe(torch.randn(1, 3, 256, 256)))
        del probe
        out_indices = tuple(range(max(0, n_stages - 3), n_stages))

        try:
            self.backbone = timm.create_model(
                backbone,
                pretrained=True,
                features_only=True,
                out_indices=out_indices,
            )
        except Exception as exc:
            _log.warning(
                "Could not load pretrained weights for %s: %s. "
                "Falling back to random initialization - results may differ.",
                backbone,
                exc,
            )
            self.backbone = timm.create_model(
                backbone,
                pretrained=False,
                features_only=True,
                out_indices=out_indices,
            )

        dummy_in = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            feats = self.backbone(dummy_in)

        in_channels = [f.shape[1] for f in feats]
        self.feature_pyramid = PANFeaturePyramid(in_channels)
        self.pyramid_projection = nn.Conv2d(256, 256, 3, padding=1)

        if self.use_spatial_attention:
            self.spatial_attention = ConvolutionalBlockAttention(256)

        self.spatial_pool = nn.AdaptiveAvgPool2d(1)
        self.spatial_proj = nn.Linear(256, 256)

        if self.use_frequency_encoder:
            self.frequency_encoder = LogPolarFrequencyEncoder()
            self.frequency_gate = FrequencyFeatureGate()
            fused_dim = 320
        else:
            fused_dim = 256

        if self.use_text_encoder:
            self.question_encoder = QuestionTextEncoder(text_dim=text_dim)
            fused_dim = fused_dim + text_dim
            self.null_text_embed = nn.Parameter(torch.randn(1, text_dim) * 0.02)
        else:
            self.question_encoder = None
            self.null_text_embed = None

        if self.use_task_conditioning:
            self.regressor = TaskConditionedRegressor(
                in_features=fused_dim,
                hidden_dim=128,
                task_embed_dim=min(num_tasks, 32),
                num_tasks=num_tasks,
            )
        else:
            self.shared_regressor = create_shared_regressor(fused_dim)

        if freeze_backbone_stage > 0:
            self._freeze_backbone_stages(freeze_backbone_stage)

    def _freeze_backbone_stages(self, num_stages: int) -> None:
        frozen_count = 0
        stage_pat = re.compile(r"(?:^|\.)(?:blocks|stages)[._](\d+)\.")
        for name, param in self.backbone.named_parameters():
            stage_match = stage_pat.search(name)
            if stage_match and int(stage_match.group(1)) < num_stages:
                param.requires_grad = False
                frozen_count += 1
            elif stage_match is None and num_stages > 0:
                prefix = name.split(".")[0]
                if prefix in ("conv_stem", "bn1", "stem"):
                    param.requires_grad = False
                    frozen_count += 1
        if frozen_count == 0:
            sample_names = list(dict(self.backbone.named_parameters()).keys())[:5]
            _log.warning(
                "No parameters matched freeze patterns - backbone %s may use different naming. "
                "Sample param names: %s",
                self.backbone.__class__.__name__,
                sample_names,
            )
        else:
            _log.info("Froze %d backbone parameters (first %d stages)", frozen_count, num_stages)

    def _extract_fused_features(
        self, images: torch.Tensor, text_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if images.dim() == 5:
            B, N, C, H, W = images.shape
            images_flat = images.view(B * N, C, H, W)
        else:
            images_flat = images
            B, N = images.shape[0], 1

        feats = self.backbone(images_flat)
        feats = self.feature_pyramid(feats)

        spatial = self.pyramid_projection(feats[0])
        if self.use_spatial_attention:
            spatial = self.spatial_attention(spatial)

        spatial = self.spatial_pool(spatial).flatten(1)
        spatial_features = self.spatial_proj(spatial)

        if self.use_frequency_encoder:
            frequency_features = self.frequency_encoder(images_flat)
            frequency_features = self.frequency_gate(spatial_features, frequency_features)
            fused = torch.cat([spatial_features, frequency_features], dim=-1)
        else:
            fused = spatial_features

        if N > 1:
            fused = fused.view(B, N, -1).mean(dim=1)

        if self.use_text_encoder and self.null_text_embed is not None:
            if text_features is not None:
                fused = torch.cat([fused, text_features], dim=-1)
            else:
                null_feat = self.null_text_embed.expand(fused.shape[0], -1)
                fused = torch.cat([fused, null_feat], dim=-1)

        return fused

    def forward_features(
        self, images: torch.Tensor, question_text: Optional[List[str]] = None
    ) -> torch.Tensor:
        text_features = self._maybe_encode_text(question_text)
        return self._extract_fused_features(images, text_features=text_features)

    def encode_text(self, texts: List[str]) -> Optional[torch.Tensor]:
        if not self.use_text_encoder or self.question_encoder is None:
            return None
        return self.question_encoder(texts)

    def _maybe_encode_text(self, question_text: Optional[List[str]]) -> Optional[torch.Tensor]:
        if not self.use_text_encoder or self.question_encoder is None:
            return None
        if question_text is not None and len(question_text) > 0:
            return self.question_encoder(question_text)
        return None

    def forward(
        self,
        images: torch.Tensor,
        task_ids: Optional[torch.Tensor] = None,
        features: Optional[torch.Tensor] = None,
        question_text: Optional[List[str]] = None,
    ) -> torch.Tensor:
        fused = (
            features
            if features is not None
            else self._extract_fused_features(
                images, text_features=self._maybe_encode_text(question_text)
            )
        )
        if self.use_task_conditioning:
            if task_ids is None:
                task_ids = torch.zeros(images.shape[0], dtype=torch.long, device=images.device)
            return self.regressor(fused, task_ids)
        return self.shared_regressor(fused).squeeze(-1)

    def forward_all_tasks(
        self,
        images: torch.Tensor,
        features: Optional[torch.Tensor] = None,
        question_text: Optional[List[str]] = None,
    ) -> torch.Tensor:
        fused = (
            features
            if features is not None
            else self._extract_fused_features(
                images, text_features=self._maybe_encode_text(question_text)
            )
        )
        if self.use_task_conditioning:
            B = fused.shape[0]
            scores = []
            for t in range(self.regressor.num_tasks):
                tids = torch.full((B,), t, dtype=torch.long, device=fused.device)
                scores.append(self.regressor(fused, tids))
            return torch.stack(scores, dim=1)
        q = self.shared_regressor(fused).squeeze(-1)
        return q.unsqueeze(-1).expand(-1, self.num_tasks)
