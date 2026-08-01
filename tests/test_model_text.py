"""Tests for UAVIQANet with text (question) input alongside images."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from uav_iqa.model import UAVIQANet
from uav_iqa.models.text import QuestionTextEncoder


class TestModelTextInput:
    """Verify UAVIQANet accepts question text as input alongside images."""

    @pytest.fixture
    def model(self):
        """Create a small model instance for testing."""
        return UAVIQANet(
            backbone="mobilenetv4_conv_small",
            num_tasks=14,
            use_fab=True,
            use_cbam=True,
            use_task_conditioning=True,
        )

    @pytest.fixture
    def model_no_task_cond(self):
        """Model without task conditioning."""
        return UAVIQANet(
            backbone="mobilenetv4_conv_small",
            num_tasks=14,
            use_fab=False,
            use_cbam=False,
            use_task_conditioning=False,
        )

    def test_forward_accepts_question_text(self, model):
        """forward() should accept a question_text parameter."""
        B = 4
        x = torch.randn(B, 3, 256, 256)
        task_ids = torch.randint(0, 14, (B,))
        questions = ["What is in the scene?"] * B

        # This should NOT raise TypeError
        scores = model(x, task_ids, question_text=questions)
        assert scores.shape == (B,)

    def test_forward_accepts_multi_image_with_text(self, model):
        """forward() should accept multi-image input (B, N, 3, H, W) with text."""
        B, N = 2, 3
        x = torch.randn(B, N, 3, 256, 256)
        task_ids = torch.randint(0, 14, (B,))
        questions = ["Question A?", "Question B?"]

        scores = model(x, task_ids, question_text=questions)
        assert scores.shape == (B,)

    def test_forward_without_text_still_works(self, model):
        """forward() without question_text should work as before (backward compat)."""
        B = 4
        x = torch.randn(B, 3, 256, 256)
        task_ids = torch.randint(0, 14, (B,))

        scores = model(x, task_ids)  # no question_text
        assert scores.shape == (B,)

    def test_text_encoder_exists(self, model):
        """Model should have a text_encoder attribute after init."""
        assert hasattr(model, "text_encoder"), "model must have text_encoder"

    def test_encode_text_returns_features(self, model):
        """encode_text() should convert question strings to feature vectors."""
        questions = ["What is visible?", "Can you see the target?"]
        text_features = model.encode_text(questions)
        assert isinstance(text_features, torch.Tensor)
        assert text_features.ndim == 2  # (B, dim)
        assert text_features.shape[0] == len(questions)

    def test_text_features_fuse_with_image_features(self, model):
        """Text features should be fused with image features to produce score."""
        B = 3
        x = torch.randn(B, 3, 256, 256)
        task_ids = torch.randint(0, 14, (B,))
        questions = ["Q1?", "Q2?", "Q3?"]

        scores_with_text = model(x, task_ids, question_text=questions)
        scores_without_text = model(x, task_ids)  # no text

        # Scores should differ when text is provided vs not
        assert not torch.allclose(scores_with_text, scores_without_text, atol=1e-4), (
            "Text input should affect the output scores"
        )

    def test_text_encoder_output_dim(self, model):
        """text_encoder should output feature vectors of a known dimensionality."""
        questions = ["A question about UAV imagery."]
        features = model.encode_text(questions)
        # text_dim should be reasonable (e.g., 64 or 128)
        assert 32 <= features.shape[-1] <= 512

    def test_empty_text_handled(self, model):
        """Empty question string should not crash."""
        B = 2
        x = torch.randn(B, 3, 256, 256)
        task_ids = torch.tensor([0, 1], dtype=torch.long)
        questions = ["", "Valid question"]

        scores = model(x, task_ids, question_text=questions)
        assert scores.shape == (B,)

    def test_forward_all_tasks_with_text(self, model):
        """forward_all_tasks should also accept question_text."""
        B = 2
        x = torch.randn(B, 3, 256, 256)
        questions = ["Q1?", "Q2?"]

        all_scores = model.forward_all_tasks(x, question_text=questions)
        assert all_scores.shape == (B, model.num_tasks)

    def test_no_task_conditioning_with_text(self, model_no_task_cond):
        """Model without task conditioning should still accept text."""
        B = 4
        x = torch.randn(B, 3, 256, 256)
        questions = ["Q1?", "Q2?", "Q3?", "Q4?"]

        scores = model_no_task_cond(x, question_text=questions)
        assert scores.shape == (B,)


class TestTextEncoder:
    """Unit tests for the text encoder component (when it exists)."""

    def test_text_encoder_module_exists(self):
        """Verify that a TextEncoder class can be imported after implementation."""
        # This test will pass once the TextEncoder is added to model.py
        encoder = QuestionTextEncoder(text_dim=128)
        assert encoder is not None

    def test_text_encoder_forward(self):
        """encode should handle a list of strings."""
        encoder = QuestionTextEncoder(text_dim=128)
        questions = ["Scene description question?", "Object counting query?"]
        features = encoder(questions)
        assert features.shape == (2, 128)

    def test_text_encoder_tokenization(self):
        encoder = QuestionTextEncoder(max_len=16)
        token_ids = encoder._tokenize(["hello world"])
        assert token_ids.shape == (1, 11)  # 11 printable chars
        assert token_ids.dtype == torch.long

    def test_text_encoder_pad_unk(self):
        encoder = QuestionTextEncoder(max_len=16)
        token_ids = encoder._tokenize(["ab", "abcd"])

        assert token_ids.shape == (2, 4)
        assert token_ids[0, 2] == encoder.PAD_IDX
        assert token_ids[0, 3] == encoder.PAD_IDX

    def test_text_encoder_unknown_chars(self):
        encoder = QuestionTextEncoder(max_len=16)
        token_ids = encoder._tokenize(["\x00\xff\u4e2d", "abc"])
        assert token_ids[0, 0] == encoder.UNK_IDX
        assert token_ids[0, 1] == encoder.UNK_IDX
        assert token_ids[0, 2] == encoder.UNK_IDX

    def test_text_encoder_truncation(self):
        encoder = QuestionTextEncoder(max_len=5)
        token_ids = encoder._tokenize(["hello world"])
        assert token_ids.size(1) <= 5

    def test_text_encoder_empty_string(self):
        encoder = QuestionTextEncoder(text_dim=128)
        token_ids = encoder._tokenize([""])
        assert token_ids.shape == (1, 0)
        assert token_ids.nelement() == 0

    def test_text_encoder_device_move(self):
        encoder = QuestionTextEncoder(text_dim=128, max_len=16)
        encoder.eval()
        out = encoder(["hello"])
        assert out.device == encoder.char_embed.weight.device
