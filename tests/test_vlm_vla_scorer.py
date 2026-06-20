"""Tests for uav_iqa.vlm_vla_scorer — BaseScorer, SyntheticScorer, VLMScorer."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from uav_iqa.vlm_vla_scorer import (
    ALL_TASKS,
    BaseScorer,
    SyntheticScorer,
    VLMScorer,
)

# ===========================================================================
# Helper functions
# ===========================================================================


def _make_dummy_image(path: Path, size: int = 64):
    """Create a small random RGB image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(img).save(str(path))


# ===========================================================================
# BaseScorer abstract class tests
# ===========================================================================


class TestBaseScorer:
    """Verify that BaseScorer can't be instantiated and its interface contract."""

    def test_cannot_instantiate_abstract(self):
        """BaseScorer is abstract — direct instantiation should fail."""
        with pytest.raises(TypeError, match="abstract"):
            BaseScorer()  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_all_methods(self):
        """A subclass missing any abstract method should fail to instantiate."""

        class IncompleteScorer(BaseScorer):
            def score_image(self, image_path, task):
                return {}

            # Missing score_batch and build_ref_lookup

        with pytest.raises(TypeError, match="abstract"):
            IncompleteScorer()  # type: ignore[abstract]


# ===========================================================================
# SyntheticScorer — unit tests
# ===========================================================================


class TestSyntheticScorerUnit:
    """Unit tests for SyntheticScorer — deterministic, reproducible, edge cases."""

    def test_score_image_returns_all_required_keys(self):
        scorer = SyntheticScorer(seed=42)
        result = scorer.score_image("test_image.png", "tracking")

        assert set(result.keys()) == {
            "vlm_score",
            "vla_score",
            "execution_score",
            "annotated",
        }

    def test_score_image_scores_are_in_range(self):
        """All scores must be in [0.0, 1.0]."""
        scorer = SyntheticScorer(seed=42)
        for task in ALL_TASKS:
            result = scorer.score_image("img_001.png", task)
            for key in ("vlm_score", "vla_score", "execution_score"):
                score = result[key]
                assert (
                    0.0 <= score <= 1.0
                ), f"{key}={score} out of range for task={task}"

    def test_score_image_is_deterministic(self):
        """Same input always produces same output."""
        scorer = SyntheticScorer(seed=42)
        r1 = scorer.score_image("img_001.png", "tracking")
        r2 = scorer.score_image("img_001.png", "tracking")
        assert r1 == r2

    def test_score_image_different_images_different_scores(self):
        """Different images should (statistically) get different scores."""
        scorer = SyntheticScorer(seed=42)
        scores = {
            ref: scorer.score_image(f"{ref}.png", "tracking")["vlm_score"]
            for ref in ("ref_001", "ref_002", "ref_003", "ref_004", "ref_005")
        }
        # At least some scores should differ (not all identical)
        assert (
            len(set(scores.values())) > 1
        ), "All scores identical — not diverse enough"

    def test_score_image_different_seeds_different_scores(self):
        """Different seeds produce different scores."""
        s1 = SyntheticScorer(seed=42).score_image("img.png", "tracking")
        s2 = SyntheticScorer(seed=123).score_image("img.png", "tracking")
        assert s1 != s2

    def test_score_image_different_tasks_same_ref(self):
        """Score is NOT task-aware in the synthetic scorer — it's ref-based only."""
        scorer = SyntheticScorer(seed=42)
        # The task parameter is accepted but doesn't affect the synthetic score
        r_track = scorer.score_image("img.png", "tracking")
        r_insp = scorer.score_image("img.png", "inspection")
        # Same ref_id → same scores (synthetic scorer is task-agnostic)
        assert r_track == r_insp

    def test_score_image_strips_clean_suffix(self):
        """Images with _clean suffix use the base stem as ref_id."""
        scorer = SyntheticScorer(seed=42)
        r_normal = scorer.score_image("scene_001.png", "tracking")
        r_clean = scorer.score_image("scene_001_clean.png", "tracking")
        assert r_normal == r_clean

    def test_annotated_is_always_false(self):
        """Synthetic scorer never marks scores as real annotations."""
        scorer = SyntheticScorer(seed=42)
        for task in ALL_TASKS:
            result = scorer.score_image("img.png", task)
            assert result["annotated"] is False

    def test_score_image_with_path_object(self):
        """Accept Path objects as well as strings."""
        scorer = SyntheticScorer(seed=42)
        result = scorer.score_image(Path("test/image.png"), "sar")
        assert "vlm_score" in result

    def test_score_batch_returns_correct_count(self):
        """Batch scoring returns one result per input."""
        scorer = SyntheticScorer(seed=42)
        images = ["img1.png", "img2.png", "img3.png"]
        tasks = ["tracking", "inspection", "delivery"]

        results = scorer.score_batch(images, tasks)
        assert len(results) == 3

        for r in results:
            assert "vlm_score" in r
            assert "annotated" in r

    def test_score_batch_length_mismatch_raises(self):
        """Batch scoring must have matching lengths."""
        scorer = SyntheticScorer(seed=42)
        with pytest.raises(ValueError, match="Length mismatch"):
            scorer.score_batch(["a.png", "b.png"], ["tracking"])

    def test_score_batch_empty_inputs(self):
        """Empty batch returns empty list."""
        scorer = SyntheticScorer(seed=42)
        results = scorer.score_batch([], [])
        assert results == []

    def test_build_ref_lookup_from_real_files(self):
        """Build lookup from actual image files in a directory."""
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            _make_dummy_image(ref_dir / "img001.png")
            _make_dummy_image(ref_dir / "img002.jpg")
            _make_dummy_image(ref_dir / "img003_clean.png")

            scorer = SyntheticScorer(seed=42)
            lookup = scorer.build_ref_lookup(ref_dir)

            assert len(lookup) == 3
            # Check that _clean is stripped
            assert "img003" in lookup
            assert "img003_clean" not in lookup

            for ref_id, scores in lookup.items():
                assert set(scores.keys()) == {
                    "vlm_score",
                    "vla_score",
                    "execution_score",
                    "annotated",
                }
                assert scores["annotated"] is False

    def test_build_ref_lookup_empty_dir(self):
        """Empty directory returns empty lookup."""
        with tempfile.TemporaryDirectory() as tmp:
            scorer = SyntheticScorer(seed=42)
            lookup = scorer.build_ref_lookup(Path(tmp))
            assert lookup == {}

    def test_build_ref_lookup_nonexistent_dir(self):
        """Nonexistent directory logs warning and returns empty lookup."""
        scorer = SyntheticScorer(seed=42)
        lookup = scorer.build_ref_lookup(Path("/nonexistent/path"))
        assert lookup == {}


# ===========================================================================
# SyntheticScorer — integration with existing annotations module
# ===========================================================================


class TestSyntheticScorerIntegration:
    """Verify SyntheticScorer produces scores compatible with annotation pipeline."""

    def test_scores_match_existing_synthetic_ref_scores_structure(self):
        """SyntheticScorer output structure matches annotations.synthetic_ref_scores."""
        from uav_iqa.annotations import synthetic_ref_scores as old_synthetic

        scorer = SyntheticScorer(seed=42)
        result = scorer.score_image("test_ref", "tracking")

        # Same keys
        assert set(result.keys()) == set(old_synthetic("test_ref").keys())

    def test_same_seed_produces_deterministic_across_instances(self):
        """Multiple SyntheticScorer instances with same seed produce identical scores."""
        scorer_a = SyntheticScorer(seed=42)
        scorer_b = SyntheticScorer(seed=42)

        for ref_id in ("ref_a", "ref_b", "ref_c"):
            assert scorer_a.score_image(
                f"{ref_id}.png", "tracking"
            ) == scorer_b.score_image(f"{ref_id}.png", "tracking")


# ===========================================================================
# VLMScorer — unit tests (scaffold, before real VLM implementation)
# ===========================================================================


class TestVLMScorerUnit:
    """Unit tests for VLMScorer — verify interface contract and configuration."""

    def test_vlm_scorer_is_subclass_of_base_scorer(self):
        """VLMScorer must implement BaseScorer interface."""
        assert issubclass(VLMScorer, BaseScorer)

    def test_vlm_scorer_init_defaults(self):
        """Default initialization produces a valid scorer."""
        scorer = VLMScorer()
        assert scorer.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert scorer.backend == "auto"
        assert scorer.device == "cuda"

    def test_vlm_scorer_custom_config(self):
        """Custom model and backend configuration."""
        scorer = VLMScorer(
            model_name="OpenGVLab/InternVL2-8B",
            backend="transformers",
            device="cpu",
            seed=999,
        )
        assert scorer.model_name == "OpenGVLab/InternVL2-8B"
        assert scorer.backend == "transformers"
        assert scorer.device == "cpu"
        assert scorer.seed == 999

    def test_vlm_scorer_has_expected_methods(self):
        """Verify all required interface methods exist."""
        scorer = VLMScorer()
        assert hasattr(scorer, "score_image")
        assert hasattr(scorer, "score_batch")
        assert hasattr(scorer, "build_ref_lookup")
        assert hasattr(scorer, "_build_prompt")
        assert hasattr(scorer, "_call_vlm")

    def test_vlm_scorer_fallback_is_synthetic_scorer(self):
        """When VLM is unavailable, fallback is a SyntheticScorer."""
        scorer = VLMScorer()
        assert isinstance(scorer.fallback, SyntheticScorer)
        assert scorer.fallback.seed == scorer.seed

    def test_vlm_scorer_score_image_is_implemented(self):
        """score_image is now implemented — works with synthetic fallback."""
        scorer = VLMScorer(backend="none")
        result = scorer.score_image("img.png", "tracking")
        assert "vlm_score" in result

    def test_vlm_scorer_score_batch_is_implemented(self):
        """score_batch is now implemented — works with synthetic fallback."""
        scorer = VLMScorer(backend="none")
        results = scorer.score_batch(["img.png"], ["tracking"])
        assert len(results) == 1
        assert "vlm_score" in results[0]

    def test_vlm_scorer_build_ref_lookup_is_implemented(self):
        """build_ref_lookup is now implemented — works with synthetic fallback."""
        scorer = VLMScorer(backend="none")
        lookup = scorer.build_ref_lookup("/tmp")
        assert isinstance(lookup, dict)


# ===========================================================================
# VLMScorer prompt building tests
# ===========================================================================


class TestVLMScorerPrompt:
    """Tests for VLM prompt construction."""

    def test_build_prompt_includes_task(self):
        """Prompt must reference the task type."""
        scorer = VLMScorer()
        prompt = scorer._build_prompt("tracking")
        assert "tracking" in prompt.lower()

    def test_build_prompt_includes_quality_criteria(self):
        """Prompt must ask for quality assessment."""
        scorer = VLMScorer()
        prompt = scorer._build_prompt("inspection")
        assert len(prompt) > 0
        # Should ask for score/rating
        has_quality_keyword = any(
            kw in prompt.lower() for kw in ("quality", "score", "rate", "assess")
        )
        assert (
            has_quality_keyword
        ), f"Prompt missing quality assessment keyword: {prompt}"

    def test_build_prompt_different_per_task(self):
        """Each task type should have a tailored prompt."""
        scorer = VLMScorer()
        prompts = {task: scorer._build_prompt(task) for task in ALL_TASKS}
        assert len(set(prompts.values())) == 4, "All task prompts are identical"


# ===========================================================================
# VLMScorer — backend resolution and response parsing
# ===========================================================================


class TestVLMScorerBackend:
    """Tests for VLM backend resolution and model loading logic."""

    @patch.dict("sys.modules", {"vllm": None, "transformers": None})
    def test_resolve_backend_none_when_no_deps(self):
        """When no VLM deps are available, _resolve_backend returns 'none'."""
        # Force a fresh import check by re-creating the scorer after patching
        scorer = VLMScorer(backend="auto")
        backend = scorer._resolve_backend()
        assert backend == "none"

    def test_resolve_backend_explicit_none(self):
        """Explicitly setting backend='none' always returns 'none'."""
        scorer = VLMScorer(backend="none")
        backend = scorer._resolve_backend()
        assert backend == "none"

    @patch.dict("sys.modules", {"vllm": None, "transformers": None})
    def test_resolve_backend_vllm_unavailable(self):
        """When vllm module doesn't exist, it's not selected."""
        scorer = VLMScorer(backend="auto")
        backend = scorer._resolve_backend()
        assert backend == "none"  # neither vllm nor transformers available

    def test_parse_vlm_response_integer(self):
        """Parse integer score from VLM response text."""
        scorer = VLMScorer()
        score = scorer._parse_vlm_response("4", "tracking")
        assert score == 0.8  # 4/5 normalized

    def test_parse_vlm_response_float(self):
        """Parse float score from VLM response."""
        scorer = VLMScorer()
        score = scorer._parse_vlm_response("3.5", "tracking")
        assert score == 0.7  # 3.5/5

    def test_parse_vlm_response_with_text_prefix(self):
        """Parse score embedded in longer text."""
        scorer = VLMScorer()
        responses = [
            "The quality score is 4 out of 5.",
            "I rate this image as 3/5 for tracking quality.",
            "Score: 5",
            "Based on the criteria, I give this a 4.",
        ]
        expected = [0.8, 0.6, 1.0, 0.8]
        for resp, exp in zip(responses, expected):
            score = scorer._parse_vlm_response(resp, "tracking")
            assert score == exp, f"'{resp}' → expected {exp}, got {score}"

    def test_parse_vlm_response_no_digit_returns_default(self):
        """When no digit found, returns default score (0.5)."""
        scorer = VLMScorer()
        score = scorer._parse_vlm_response("Unable to assess this image.", "tracking")
        assert score == 0.5

    def test_parse_vlm_response_empty_string(self):
        """Empty response returns default score."""
        scorer = VLMScorer()
        score = scorer._parse_vlm_response("", "tracking")
        assert score == 0.5

    def test_parse_vlm_response_out_of_range_clamped(self):
        """Scores outside [0, 1] are clamped."""
        scorer = VLMScorer()
        assert scorer._parse_vlm_response("10", "tracking") == 1.0
        assert scorer._parse_vlm_response("0", "tracking") == 0.0

    def test_parse_vlm_response_fraction_format(self):
        """Parse '4/5' format responses."""
        scorer = VLMScorer()
        assert scorer._parse_vlm_response("4/5", "tracking") == 0.8
        assert scorer._parse_vlm_response("2/5", "inspection") == 0.4


# ===========================================================================
# VLMScorer — fallback scoring (when no VLM backend available)
# ===========================================================================


class TestVLMScorerFallback:
    """Tests for VLMScorer fallback to SyntheticScorer when no VLM is available."""

    def test_score_image_falls_back_to_synthetic(self):
        """When no VLM backend, score_image uses SyntheticScorer fallback."""
        scorer = VLMScorer(backend="none")
        result = scorer.score_image("test_img.png", "tracking")

        assert "vlm_score" in result
        assert "vla_score" in result
        assert "execution_score" in result
        assert result["annotated"] is False
        assert 0.0 <= result["vlm_score"] <= 1.0

    def test_score_image_fallback_is_deterministic(self):
        """Fallback scores are deterministic for same input."""
        scorer = VLMScorer(backend="none")
        r1 = scorer.score_image("img.png", "tracking")
        r2 = scorer.score_image("img.png", "tracking")
        assert r1 == r2

    def test_score_batch_fallback(self):
        """Batch scoring with fallback works correctly."""
        scorer = VLMScorer(backend="none")
        images = ["img1.png", "img2.png"]
        tasks = ["tracking", "inspection"]

        results = scorer.score_batch(images, tasks)
        assert len(results) == 2
        for r in results:
            assert "vlm_score" in r
            assert r["annotated"] is False

    def test_score_batch_fallback_length_mismatch(self):
        """Batch fallback validates input lengths."""
        scorer = VLMScorer(backend="none")
        with pytest.raises(ValueError, match="Length mismatch"):
            scorer.score_batch(["a.png", "b.png"], ["tracking"])

    def test_build_ref_lookup_fallback(self):
        """Reference lookup with fallback scans directory."""
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            _make_dummy_image(ref_dir / "ref001.png")
            _make_dummy_image(ref_dir / "ref002.png")

            scorer = VLMScorer(backend="none")
            lookup = scorer.build_ref_lookup(ref_dir)

            assert len(lookup) == 2
            for ref_id, scores in lookup.items():
                assert scores["annotated"] is False

    def test_build_ref_lookup_fallback_empty_dir(self):
        """Empty directory with fallback returns empty dict."""
        with tempfile.TemporaryDirectory() as tmp:
            scorer = VLMScorer(backend="none")
            lookup = scorer.build_ref_lookup(Path(tmp))
            assert lookup == {}


# ===========================================================================
# ALL_TASKS constant
# ===========================================================================


class TestTaskConstants:
    def test_all_tasks_has_four_tasks(self):
        assert len(ALL_TASKS) == 4

    def test_all_tasks_matches_expected(self):
        assert set(ALL_TASKS) == {"tracking", "inspection", "delivery", "sar"}


# ===========================================================================
# Edge cases and robustness
# ===========================================================================


class TestEdgeCases:
    """Edge case and robustness tests for SyntheticScorer."""

    def test_very_long_filename(self):
        """Handles very long filenames without error."""
        scorer = SyntheticScorer(seed=42)
        long_name = "a" * 500 + ".png"
        result = scorer.score_image(long_name, "tracking")
        assert "vlm_score" in result

    def test_special_characters_in_filename(self):
        """Handles special characters in filenames."""
        scorer = SyntheticScorer(seed=42)
        tricky_names = [
            "path/with/slashes/img.png",
            "img with spaces.png",
            "中文图片.png",
            "image_©®™.png",
            "../../../escape.png",
        ]
        for name in tricky_names:
            result = scorer.score_image(name, "tracking")
            assert "vlm_score" in result
            assert 0.0 <= result["vlm_score"] <= 1.0

    def test_empty_string_task(self):
        """Empty task string — should still work (task doesn't affect synthetic scores)."""
        scorer = SyntheticScorer(seed=42)
        result = scorer.score_image("img.png", "")
        assert "vlm_score" in result

    def test_score_precision_is_four_decimals(self):
        """Scores should be rounded to 4 decimal places."""
        scorer = SyntheticScorer(seed=42)
        result = scorer.score_image("img.png", "tracking")
        for key in ("vlm_score", "vla_score", "execution_score"):
            val = result[key]
            # Check precision: should not have more than 4 decimal places
            assert round(val, 4) == val, f"{key}={val} has more than 4 decimal places"

    def test_large_batch_performance(self):
        """Large batch (1000 images) should complete quickly and correctly."""
        scorer = SyntheticScorer(seed=42)
        images = [f"img_{i:05d}.png" for i in range(1000)]
        tasks = ["tracking"] * 1000

        results = scorer.score_batch(images, tasks)
        assert len(results) == 1000
        # Verify a few random indices
        for i in (0, 500, 999):
            assert "vlm_score" in results[i]


# ===========================================================================
# VLMScorer — coverage of VLM-specific code paths (mocked)
# ===========================================================================


class TestVLMScorerCoverage:
    """Tests that exercise VLM-specific code paths via mocking."""

    def test_resolve_backend_explicit_vllm_installed(self):
        """When vllm is installed, backend='vllm' resolves to 'vllm'."""
        fake_vllm = type(sys)("vllm")
        with patch.dict("sys.modules", {"vllm": fake_vllm}):
            scorer = VLMScorer(backend="vllm")
            backend = scorer._resolve_backend()
            assert backend == "vllm"

    def test_resolve_backend_explicit_vllm_not_installed(self):
        """When vllm is NOT installed, backend='vllm' falls back to 'none'."""
        with patch.dict("sys.modules", {"vllm": None, "transformers": None}):
            scorer = VLMScorer(backend="vllm")
            backend = scorer._resolve_backend()
            assert backend == "none"

    def test_resolve_backend_explicit_transformers_installed(self):
        """When transformers is installed, backend='transformers' resolves correctly."""
        fake_tf = type(sys)("transformers")
        with patch.dict("sys.modules", {"transformers": fake_tf}):
            scorer = VLMScorer(backend="transformers")
            backend = scorer._resolve_backend()
            assert backend == "transformers"

    def test_resolve_backend_explicit_transformers_not_installed(self):
        """When transformers is NOT installed, backend='transformers' falls back."""
        with patch.dict("sys.modules", {"vllm": None, "transformers": None}):
            scorer = VLMScorer(backend="transformers")
            backend = scorer._resolve_backend()
            assert backend == "none"

    def test_resolve_backend_auto_prefers_vllm(self):
        """Auto mode prefers vllm over transformers when both available."""
        fake_vllm = type(sys)("vllm")
        fake_tf = type(sys)("transformers")
        with patch.dict(
            "sys.modules",
            {"vllm": fake_vllm, "transformers": fake_tf},
        ):
            scorer = VLMScorer(backend="auto")
            backend = scorer._resolve_backend()
            assert backend == "vllm"

    def test_resolve_backend_auto_falls_to_transformers(self):
        """Auto mode falls to transformers when vllm not available."""
        fake_tf = type(sys)("transformers")
        with patch.dict(
            "sys.modules",
            {"vllm": None, "transformers": fake_tf},
        ):
            scorer = VLMScorer(backend="auto")
            backend = scorer._resolve_backend()
            assert backend == "transformers"

    def test_load_model_none_backend(self):
        """_load_model with 'none' backend logs and returns without error."""
        scorer = VLMScorer(backend="none")
        scorer._load_model()  # Should not raise, model stays None
        assert scorer._model is None

    @patch.object(VLMScorer, "_resolve_backend", return_value="none")
    def test_score_image_goes_through_fallback_branch(self, mock_resolve):
        """When _resolve_backend returns 'none', fallback path is used."""
        scorer = VLMScorer()
        result = scorer.score_image("img.png", "tracking")
        assert result["annotated"] is False
        assert "vlm_score" in result

    @patch.object(VLMScorer, "_resolve_backend", return_value="none")
    def test_score_batch_goes_through_fallback_branch(self, mock_resolve):
        """Batch scoring with 'none' backend uses fallback."""
        scorer = VLMScorer()
        results = scorer.score_batch(["a.png", "b.png"], ["tracking", "sar"])
        assert len(results) == 2

    @patch.object(VLMScorer, "_resolve_backend", return_value="none")
    def test_build_ref_lookup_fallback_branch(self, mock_resolve):
        """Ref lookup with 'none' backend uses fallback."""
        scorer = VLMScorer()
        lookup = scorer.build_ref_lookup("/tmp")
        assert isinstance(lookup, dict)

    def test_build_prompt_unknown_task_falls_back(self):
        """Unknown task type falls back to tracking prompt."""
        scorer = VLMScorer()
        prompt = scorer._build_prompt("unknown_task")
        assert "tracking" in prompt.lower()

    def test_parse_vlm_response_large_number_div_by_5(self):
        """Numbers > 1 are treated as 1-5 scale and divided by 5."""
        scorer = VLMScorer()
        assert scorer._parse_vlm_response("8", "tracking") == 1.0  # clamped
        assert scorer._parse_vlm_response("2.5", "tracking") == 0.5

    def test_parse_vlm_response_small_decimal(self):
        """Numbers already in [0,1] range kept as-is (with clamping)."""
        scorer = VLMScorer()
        assert scorer._parse_vlm_response("0.75", "tracking") == 0.75
