"""Tests for text similarity metrics used in comparison-based scoring."""

import pytest

from uav_iqa.text_metrics import (
    compute_bleu,
    compute_cider,
    compute_cognitive_score,
    compute_rouge_l,
)

# ===========================================================================
# Text metrics — BLEU, ROUGE-L, CIDEr, cognitive score
# ===========================================================================


class TestTextMetrics:
    """Tests for text similarity metrics used in comparison-based scoring."""

    def test_compute_bleu_identical_texts(self):
        text = "Aerial view showing urban landscape with buildings and roads"
        score = compute_bleu(text, text)
        assert score > 0.99

    def test_compute_bleu_different_texts(self):
        ref = "Aerial view showing urban landscape with buildings and roads"
        hyp = "Blurry image with no visible details"
        score = compute_bleu(ref, hyp)
        assert score < 0.5

    def test_compute_bleu_empty_hypothesis(self):
        score = compute_bleu("reference text", "")
        assert score == 0.0

    def test_compute_rouge_l_identical_texts(self):
        text = "Describe the key objects and spatial layout in this aerial view"
        score = compute_rouge_l(text, text)
        assert score > 0.99

    def test_compute_rouge_l_different_texts(self):
        ref = "urban landscape with buildings"
        hyp = "completely unrelated"
        score = compute_rouge_l(ref, hyp)
        assert score < 0.5

    def test_compute_cider_identical_texts(self):
        ref = "Aerial view showing urban landscape with buildings and roads"
        score = compute_cider([ref], ref)
        assert score > 5.0

    def test_compute_cider_empty_inputs(self):
        assert compute_cider([], "text") == 0.0
        assert compute_cider(["text"], "") == 0.0

    def test_compute_cognitive_score_identical_descriptions(self):
        desc = "Aerial view showing urban landscape with buildings and roads"
        refs = [desc, desc, desc]
        dists = [desc, desc, desc]
        result = compute_cognitive_score(refs, dists)
        assert result["cognitive_score"] > 0.9
        assert result["bleu"] > 0.99
        assert result["rouge_l"] > 0.99
        assert result["cider"] > 5.0
        assert len(result["per_prompt"]) == 3
        for ps in result["per_prompt"]:
            assert ps["bleu"] > 0.99
            assert ps["rouge_l"] > 0.99
            assert ps["cider"] > 5.0

    def test_compute_cognitive_score_empty_inputs(self):
        result = compute_cognitive_score([], [])
        assert result["cognitive_score"] == 0.0
        assert result["bleu"] == 0.0
        assert result["rouge_l"] == 0.0
        assert result["cider"] == 0.0

    def test_compute_cognitive_score_length_mismatch(self):
        with pytest.raises(ValueError, match="must have same length"):
            compute_cognitive_score(["a"], ["a", "b"])
