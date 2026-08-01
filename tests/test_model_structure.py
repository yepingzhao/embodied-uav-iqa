"""Characterization tests for the current model component interfaces."""

import timm
import torch

from uav_iqa.model import UAVIQANet
from uav_iqa.models.frequency import FrequencyFeatureGate, LogPolarFrequencyEncoder
from uav_iqa.models.heads import TaskConditionedRegressor
from uav_iqa.models.spatial import ConvolutionalBlockAttention, PANFeaturePyramid
from uav_iqa.models.text import QuestionTextEncoder


def test_model_component_output_shapes():
    """Current standalone model components preserve their documented shapes."""
    with torch.inference_mode():
        fpn = PANFeaturePyramid([16, 32, 64], out_channels=32).eval()
        fpn_outputs = fpn(
            [
                torch.zeros(2, 16, 32, 32),
                torch.zeros(2, 32, 16, 16),
                torch.zeros(2, 64, 8, 8),
            ]
        )
        assert [output.shape for output in fpn_outputs] == [
            (2, 32, 32, 32),
            (2, 32, 16, 16),
            (2, 32, 8, 8),
        ]

        cbam_output = ConvolutionalBlockAttention(32).eval()(torch.zeros(2, 32, 16, 16))
        assert cbam_output.shape == (2, 32, 16, 16)

        frequency_output = LogPolarFrequencyEncoder().eval()(torch.zeros(2, 3, 64, 64))
        assert frequency_output.shape == (2, 64)

        gate_output = FrequencyFeatureGate().eval()(torch.zeros(2, 256), torch.zeros(2, 64))
        assert gate_output.shape == (2, 64)

        head_output = TaskConditionedRegressor().eval()(
            torch.zeros(2, 320), torch.tensor([0, 13])
        )
        assert head_output.shape == (2,)

        text_output = QuestionTextEncoder().eval()(["Inspect the target.", "Is the route clear?"])
        assert text_output.shape == (2, 128)


def test_uav_iqa_net_current_output_shapes_without_pretrained_weights(monkeypatch):
    """UAVIQANet preserves single, multi-UAV, text, and all-task output shapes."""
    original_create_model = timm.create_model

    def create_model_without_pretrained(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs["pretrained"] = False
        return original_create_model(*args, **kwargs)

    monkeypatch.setattr(timm, "create_model", create_model_without_pretrained)

    model = UAVIQANet(num_tasks=14).eval()
    image = torch.zeros(1, 3, 256, 256)
    multi_uav_images = torch.zeros(1, 2, 3, 256, 256)
    question_text = ["Is the target visible?"]

    with torch.inference_mode():
        features = model.forward_features(image, question_text=question_text)
        single_score = model(image, features=features)
        multi_uav_score = model(multi_uav_images, question_text=question_text)
        all_task_scores = model.forward_all_tasks(image, features=features)

    assert features.shape == (1, 448)
    assert single_score.shape == (1,)
    assert multi_uav_score.shape == (1,)
    assert all_task_scores.shape == (1, 14)
    assert torch.all((single_score >= 0) & (single_score <= 1))
    assert torch.all((multi_uav_score >= 0) & (multi_uav_score <= 1))
