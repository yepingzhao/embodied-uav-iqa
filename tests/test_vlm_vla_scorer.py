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


# ===========================================================================
# VLMConfig and MODEL_REGISTRY
# ===========================================================================


class TestVLMConfig:
    """Tests for VLMConfig dataclass."""

    def test_create_vlm_config_with_all_fields(self):
        """VLMConfig can be created with all required fields."""
        from uav_iqa.vlm_vla_scorer import VLMConfig

        cfg = VLMConfig(
            short_name="TestModel",
            hf_model_id="org/TestModel-7B",
            family="qwen",
            chat_template="{prompt}",
            model_class_name="AutoModelForVision2Seq",
            processor_class_name="AutoProcessor",
            trust_remote_code=True,
        )
        assert cfg.short_name == "TestModel"
        assert cfg.hf_model_id == "org/TestModel-7B"
        assert cfg.family == "qwen"
        assert cfg.chat_template == "{prompt}"
        assert cfg.model_class_name == "AutoModelForVision2Seq"
        assert cfg.processor_class_name == "AutoProcessor"
        assert cfg.trust_remote_code is True

    def test_vlm_config_default_trust_remote_code(self):
        """trust_remote_code defaults to True."""
        from uav_iqa.vlm_vla_scorer import VLMConfig

        cfg = VLMConfig(
            short_name="M",
            hf_model_id="org/M",
            family="qwen",
            chat_template="{prompt}",
            model_class_name="AutoModelForVision2Seq",
            processor_class_name="AutoProcessor",
        )
        assert cfg.trust_remote_code is True


class TestModelRegistry:
    """Tests for MODEL_REGISTRY — the 15 supported VLMs."""

    def test_registry_has_exactly_15_models(self):
        """MODEL_REGISTRY contains exactly 15 models."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        assert len(MODEL_REGISTRY) == 15

    def test_all_registry_entries_are_vlm_config(self):
        """Every entry in MODEL_REGISTRY is a VLMConfig."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY, VLMConfig

        for name, cfg in MODEL_REGISTRY.items():
            assert isinstance(cfg, VLMConfig), f"{name} is not VLMConfig"
            assert cfg.short_name == name, f"{name} short_name mismatch: {cfg.short_name}"

    def test_registry_short_names_match_expected(self):
        """Registry keys match the 15 expected model short names."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        expected = {
            "Mini-InternVL",
            "InternVL2",
            "InternVL2.5",
            "InternVL3",
            "InternLM-Xcomposer2",
            "InternLM-Xcomposer2.5",
            "Ovis1.5-Gemma",
            "Ovis1.6-Llama",
            "Ovis2",
            "Phi3-Vision",
            "Phi3.5-Vision",
            "Phi4-Multimodal",
            "Qwen2-VL",
            "Qwen2.5-VL",
            "MPlugOwl3",
        }
        assert set(MODEL_REGISTRY.keys()) == expected

    def test_registry_models_have_valid_hf_ids(self):
        """Every model has a valid-looking HuggingFace model ID."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        for name, cfg in MODEL_REGISTRY.items():
            assert "/" in cfg.hf_model_id, f"{name}: hf_model_id missing '/' separator"
            parts = cfg.hf_model_id.split("/")
            assert len(parts) == 2, f"{name}: hf_model_id should be 'org/model'"

    def test_registry_models_have_valid_families(self):
        """All registry entries use a known model family."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        valid_families = {"qwen", "internvl", "internlm_xc", "ovis", "phi", "mplug"}
        for name, cfg in MODEL_REGISTRY.items():
            assert cfg.family in valid_families, (
                f"{name}: unknown family '{cfg.family}'"
            )

    def test_registry_models_have_chat_template_with_prompt_placeholder(self):
        """Every model has a chat template containing {prompt}."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        for name, cfg in MODEL_REGISTRY.items():
            assert "{prompt}" in cfg.chat_template, (
                f"{name}: chat_template missing {{prompt}} placeholder"
            )

    def test_registry_qwen_family_models(self):
        """Qwen family contains Qwen2-VL and Qwen2.5-VL."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        assert MODEL_REGISTRY["Qwen2-VL"].family == "qwen"
        assert MODEL_REGISTRY["Qwen2.5-VL"].family == "qwen"
        assert MODEL_REGISTRY["Qwen2-VL"].hf_model_id == "Qwen/Qwen2-VL-7B-Instruct"
        assert MODEL_REGISTRY["Qwen2.5-VL"].hf_model_id == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_registry_internvl_family_models(self):
        """InternVL family contains Mini, 2, 2.5, 3."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        for name in ("Mini-InternVL", "InternVL2", "InternVL2.5", "InternVL3"):
            assert MODEL_REGISTRY[name].family == "internvl"
            assert MODEL_REGISTRY[name].hf_model_id.startswith("OpenGVLab/")

    def test_registry_internlm_xc_family_models(self):
        """InternLM-Xcomposer family contains 2 and 2.5."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        for name in ("InternLM-Xcomposer2", "InternLM-Xcomposer2.5"):
            assert MODEL_REGISTRY[name].family == "internlm_xc"
            assert MODEL_REGISTRY[name].hf_model_id.startswith("internlm/")

    def test_registry_ovis_family_models(self):
        """Ovis family contains 1.5-Gemma, 1.6-Llama, 2."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        for name in ("Ovis1.5-Gemma", "Ovis1.6-Llama", "Ovis2"):
            assert MODEL_REGISTRY[name].family == "ovis"
            assert MODEL_REGISTRY[name].hf_model_id.startswith("AIDC-AI/")

    def test_registry_phi_family_models(self):
        """Phi family contains Phi3, Phi3.5, Phi4."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        for name in ("Phi3-Vision", "Phi3.5-Vision", "Phi4-Multimodal"):
            assert MODEL_REGISTRY[name].family == "phi"
            assert MODEL_REGISTRY[name].hf_model_id.startswith("microsoft/")

    def test_registry_mplug_owl3(self):
        """MPlugOwl3 in mplug family."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        assert MODEL_REGISTRY["MPlugOwl3"].family == "mplug"
        assert MODEL_REGISTRY["MPlugOwl3"].hf_model_id == "mPLUG/mPLUG-Owl3-7B-240728"


# ===========================================================================
# VLMScorer — registry-based model name resolution
# ===========================================================================


class TestVLMScorerRegistry:
    """Tests for VLMScorer resolving model names from the registry."""

    def test_init_with_registry_short_name(self):
        """VLMScorer accepts a registry short name and resolves config."""
        scorer = VLMScorer(model_name="InternVL2")
        assert scorer.model_name == "OpenGVLab/InternVL2-8B"
        assert scorer.vlm_config is not None
        assert scorer.vlm_config.short_name == "InternVL2"
        assert scorer.vlm_config.family == "internvl"

    def test_init_with_all_registry_names(self):
        """All 15 registry model names can be used to create VLMScorer."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        for short_name in MODEL_REGISTRY:
            scorer = VLMScorer(model_name=short_name)
            assert scorer.vlm_config.short_name == short_name
            assert scorer.vlm_config.hf_model_id == MODEL_REGISTRY[short_name].hf_model_id

    def test_init_with_raw_hf_id_still_works(self):
        """Backward compat: raw HF model ID creates a default VLMConfig."""
        scorer = VLMScorer(model_name="Qwen/Qwen2.5-VL-7B-Instruct")
        assert scorer.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert scorer.vlm_config is not None
        assert scorer.vlm_config.family == "qwen"

    def test_init_with_custom_hf_id_creates_qwen_family_config(self):
        """Custom HF ID not in registry gets a default qwen-family config."""
        scorer = VLMScorer(model_name="org/custom-model-7B")
        assert scorer.model_name == "org/custom-model-7B"
        assert scorer.vlm_config.family == "qwen"
        assert scorer.vlm_config.hf_model_id == "org/custom-model-7B"
        assert scorer.vlm_config.model_class_name == "AutoModelForVision2Seq"

    def test_init_with_nonexistent_registry_name_raises(self):
        """A name not in registry and not a valid HF ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            VLMScorer(model_name="NonExistentModel")

    def test_vlm_scorer_stores_default_short_name_for_registry_models(self):
        """Default model is still in registry and resolves correctly."""
        scorer = VLMScorer()
        assert scorer.vlm_config is not None
        # Default model_name is a full HF ID that happens to also be an entry
        assert scorer.vlm_config.hf_model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert scorer.vlm_config.family == "qwen"

    def test_vlm_scorer_vlm_config_is_vlm_config_instance(self):
        """vlm_config attribute is a VLMConfig instance."""
        from uav_iqa.vlm_vla_scorer import VLMConfig

        scorer = VLMScorer(model_name="Phi3-Vision")
        assert isinstance(scorer.vlm_config, VLMConfig)


# ===========================================================================
# Per-family chat template formatting
# ===========================================================================


class TestChatTemplateFormatting:
    """Tests for per-family chat template formatting in VLMScorer."""

    def test_qwen_family_returns_raw_prompt(self):
        """Qwen family template is just {prompt} — returns raw task prompt."""
        scorer = VLMScorer(model_name="Qwen2-VL")
        formatted = scorer._build_prompt("tracking")
        assert "tracking" in formatted.lower()
        assert "rate" in formatted.lower()
        assert "<|im_start|>" not in formatted
        assert "<|user|>" not in formatted

    def test_internvl_family_formats_with_im_start_tags(self):
        """InternVL family wraps prompt in <|im_start|> chat format."""
        scorer = VLMScorer(model_name="InternVL2")
        formatted = scorer._build_prompt("inspection")
        assert "<|im_start|>system" in formatted
        assert "<|im_start|>user" in formatted
        assert "<|im_start|>assistant" in formatted
        assert "<image>" in formatted
        assert "inspection" in formatted.lower()

    def test_internlm_xc_family_formats_with_user_bot_tags(self):
        """InternLM-Xcomposer family wraps prompt in <|User|>: and <|Bot|>:."""
        scorer = VLMScorer(model_name="InternLM-Xcomposer2")
        formatted = scorer._build_prompt("sar")
        assert formatted.startswith("<|User|>:")
        assert formatted.endswith("<|Bot|>:")
        assert "rescue" in formatted.lower()

    def test_ovis_family_returns_raw_prompt(self):
        scorer = VLMScorer(model_name="Ovis2")
        formatted = scorer._build_prompt("delivery")
        assert "delivery" in formatted.lower()
        assert "<|im_start|>" not in formatted

    def test_phi_family_formats_with_user_assistant_tags(self):
        """Phi family wraps prompt in <|user|>...<|end|>...<|assistant|>."""
        scorer = VLMScorer(model_name="Phi3-Vision")
        formatted = scorer._build_prompt("tracking")
        assert "<|user|>" in formatted
        assert "<|assistant|>" in formatted
        assert formatted.endswith("<|assistant|>\n")
        assert "tracking" in formatted.lower()

    def test_mplug_family_formats_with_user_assistant(self):
        """MPlug family wraps prompt in USER:...ASSISTANT:."""
        scorer = VLMScorer(model_name="MPlugOwl3")
        formatted = scorer._build_prompt("inspection")
        assert formatted.startswith("USER: ")
        assert "ASSISTANT:" in formatted
        assert "<image>" in formatted
        assert "inspection" in formatted.lower()

    def test_all_15_models_produce_non_empty_formatted_prompt(self):
        """Every model in the registry produces a non-empty formatted prompt."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        task_keywords = {
            "tracking": "tracking",
            "inspection": "infrastructure",
            "delivery": "delivery",
            "sar": "rescue",
        }
        for short_name in MODEL_REGISTRY:
            scorer = VLMScorer(model_name=short_name)
            for task, keyword in task_keywords.items():
                formatted = scorer._build_prompt(task)
                assert len(formatted) > 0, f"{short_name}/{task}: empty prompt"
                assert keyword in formatted.lower(), (
                    f"{short_name}/{task}: keyword '{keyword}' not in prompt: {formatted[:80]}..."
                )

    def test_internvl3_format_matches_internvl_pattern(self):
        """InternVL3 uses same format as InternVL2."""
        scorer2 = VLMScorer(model_name="InternVL2")
        scorer3 = VLMScorer(model_name="InternVL3")
        # Both use <|im_start|> format — raw content differs per task
        p2 = scorer2._build_prompt("tracking")
        p3 = scorer3._build_prompt("tracking")
        assert "<|im_start|>system" in p2
        assert "<|im_start|>system" in p3

    def test_phi4_format_matches_phi_pattern(self):
        """Phi4-Multimodal uses same format as Phi3-Vision."""
        scorer3 = VLMScorer(model_name="Phi3-Vision")
        scorer4 = VLMScorer(model_name="Phi4-Multimodal")
        p3 = scorer3._build_prompt("tracking")
        p4 = scorer4._build_prompt("tracking")
        assert "<|user|>" in p3
        assert "<|user|>" in p4
        assert "<|assistant|>" in p3
        assert "<|assistant|>" in p4


# ===========================================================================
# Per-family model loading (transformers backend, mocked)
# ===========================================================================


class TestVLMScorerModelLoading:
    """Tests for per-family model loading in VLMScorer (transformers backend)."""

    @staticmethod
    def _fake_from_pretrained_factory(cls_name, call_tracker):
        """Return a from_pretrained function that records calls and returns
        a mock object supporting .to() and .eval()."""

        class MockModel:
            def to(self, device):
                return self

            def eval(self):
                return self

        def from_pretrained(*args, **kwargs):
            call_tracker.append(cls_name)
            return MockModel()

        return staticmethod(from_pretrained)

    @staticmethod
    def _make_mock_module(class_names, call_tracker):
        """Create a fake transformers module with given class names."""
        mock_tf = type(sys)("transformers")
        for cls_name in class_names:
            mock_cls = type(cls_name, (), {})
            mock_cls.from_pretrained = (
                TestVLMScorerModelLoading._fake_from_pretrained_factory(
                    cls_name, call_tracker
                )
            )
            setattr(mock_tf, cls_name, mock_cls)
        return mock_tf

    def test_load_model_uses_vlm_config_model_class(self):
        """_load_model uses model_class_name from VLMConfig (InternVL -> AutoModel)."""
        import torch

        called = []
        mock_tf = self._make_mock_module(["AutoModel", "AutoTokenizer"], called)

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(
                model_name="InternVL2", backend="transformers", device="cpu"
            )
            scorer._load_model()
            assert "AutoModel" in called
            assert "AutoTokenizer" in called

    def test_load_model_phi_uses_auto_model_for_causal_lm(self):
        """Phi family models use AutoModelForCausalLM."""
        import torch

        called = []
        mock_tf = self._make_mock_module(
            ["AutoModelForCausalLM", "AutoProcessor"], called
        )

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(
                model_name="Phi3-Vision", backend="transformers", device="cpu"
            )
            scorer._load_model()
            assert "AutoModelForCausalLM" in called
            assert "AutoProcessor" in called

    def test_load_model_with_trust_remote_code(self):
        """trust_remote_code from VLMConfig is passed to from_pretrained."""
        import torch

        trust_values = []
        mock_tf = type(sys)("transformers")

        class MockModel:
            def to(self, d):
                return self

            def eval(self):
                return self

        def record_trust(*args, trust_remote_code=True, **kwargs):
            trust_values.append(trust_remote_code)
            return MockModel()

        mock_tf.AutoModel = type("AutoModel", (), {})
        mock_tf.AutoModel.from_pretrained = staticmethod(record_trust)

        mock_tf.AutoTokenizer = type("AutoTokenizer", (), {})
        mock_tf.AutoTokenizer.from_pretrained = staticmethod(
            lambda *a, **kw: MockModel()
        )

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(
                model_name="InternVL2", backend="transformers", device="cpu"
            )
            scorer._load_model()
            assert trust_values == [True]

    def test_load_model_qwen_uses_auto_model_for_vision2seq(self):
        """Qwen family uses AutoModelForVision2Seq (backward compat)."""
        import torch

        called = []
        mock_tf = self._make_mock_module(
            ["AutoModelForVision2Seq", "AutoProcessor"], called
        )

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(
                model_name="Qwen2-VL", backend="transformers", device="cpu"
            )
            scorer._load_model()
            assert "AutoModelForVision2Seq" in called
            assert "AutoProcessor" in called

    def test_load_model_mplug_uses_auto_model_for_vision2seq(self):
        """MPlug family uses AutoModelForVision2Seq."""
        import torch

        called = []
        mock_tf = self._make_mock_module(
            ["AutoModelForVision2Seq", "AutoProcessor"], called
        )

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(
                model_name="MPlugOwl3", backend="transformers", device="cpu"
            )
            scorer._load_model()
            assert "AutoModelForVision2Seq" in called
            assert "AutoProcessor" in called

    def test_load_model_internlm_xc_uses_auto_model_for_causal_lm(self):
        """InternLM-Xcomposer uses AutoModelForCausalLM."""
        import torch

        called = []
        mock_tf = self._make_mock_module(
            ["AutoModelForCausalLM", "AutoTokenizer"], called
        )

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(
                model_name="InternLM-Xcomposer2", backend="transformers", device="cpu"
            )
            scorer._load_model()
            assert "AutoModelForCausalLM" in called
            assert "AutoTokenizer" in called

    def test_load_model_ovis_uses_auto_model_for_vision2seq(self):
        """Ovis family uses AutoModelForVision2Seq."""
        import torch

        called = []
        mock_tf = self._make_mock_module(
            ["AutoModelForVision2Seq", "AutoProcessor"], called
        )

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(
                model_name="Ovis2", backend="transformers", device="cpu"
            )
            scorer._load_model()
            assert "AutoModelForVision2Seq" in called
            assert "AutoProcessor" in called


# ===========================================================================
# Integration — full scoring pipeline with mocked VLM backends
# ===========================================================================


class _MockProcessorOutput(dict):
    """Dict-like that supports .to(device) for transformers pipeline."""

    def to(self, device):
        return self


class _MockProcessor:
    """Mock processor that returns a fixed VLM response text."""

    def __init__(self, response_text="4"):
        self.response_text = response_text
        self._last_texts = []

    def __call__(self, text, images, return_tensors, **kwargs):
        self._last_texts.append(text)
        return _MockProcessorOutput({"input_ids": __import__("torch").tensor([[1]])})

    def decode(self, token_ids, skip_special_tokens=True):
        return self.response_text


class _MockModel:
    """Mock model with .to(), .eval(), .generate()."""

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, **kwargs):
        return [__import__("torch").tensor([1, 2, 3])]


class _MockVLLMText:
    def __init__(self, text="4"):
        self.text = text


class _MockVLLMOutput:
    def __init__(self, text="4"):
        self.outputs = [_MockVLLMText(text)]


class _MockVLLMModel:
    def generate(self, prompts, params):
        return [_MockVLLMOutput()]


class TestVLMScorerPipelineIntegration:
    """Integration tests: public API with mocked real VLM backends."""

    def test_score_image_transformers_mocked_returns_annotated_true(self):
        """Full scoring pipeline with mocked transformers returns annotated=True."""
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.png"
            _make_dummy_image(img_path)

            scorer = VLMScorer(model_name="Qwen2-VL", device="cpu")
            scorer._processor = _MockProcessor("4")
            scorer._model = _MockModel()

            with patch.object(VLMScorer, "_resolve_backend", return_value="transformers"):
                result = scorer.score_image(str(img_path), "tracking")

            assert result["annotated"] is True
            assert result["vlm_score"] == 0.8
            assert result["vla_score"] == 0.72
            assert result["execution_score"] == 0.68

    def test_score_image_vllm_mocked_returns_annotated_true(self):
        """Full scoring pipeline with mocked vllm returns annotated=True."""
        fake_vllm = type(sys)("vllm")
        fake_vllm.LLM = type("LLM", (), {})
        fake_vllm.SamplingParams = type("SamplingParams", (), {})

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.png"
            _make_dummy_image(img_path)

            scorer = VLMScorer(model_name="Qwen2-VL", device="cpu")
            scorer._model = _MockVLLMModel()
            scorer._sampling_params = None

            with patch.dict("sys.modules", {"vllm": fake_vllm}), patch.object(
                VLMScorer, "_resolve_backend", return_value="vllm"
            ):
                result = scorer.score_image(str(img_path), "delivery")

            assert result["annotated"] is True
            assert result["vlm_score"] == 0.8

    def test_all_15_models_produce_valid_scores_mocked(self):
        """Every registered model works end-to-end with mocked transformers."""
        from uav_iqa.vlm_vla_scorer import MODEL_REGISTRY

        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.png"
            _make_dummy_image(img_path)

            for short_name in sorted(MODEL_REGISTRY):
                scorer = VLMScorer(model_name=short_name, device="cpu")
                scorer._processor = _MockProcessor("3")
                scorer._model = _MockModel()

                with patch.object(
                    VLMScorer, "_resolve_backend", return_value="transformers"
                ):
                    result = scorer.score_image(str(img_path), "inspection")

                assert result["annotated"] is True, f"{short_name}: annot mismatch"
                assert result["vlm_score"] == 0.6, f"{short_name}: score mismatch"
                assert 0.0 <= result["vla_score"] <= 1.0
                assert 0.0 <= result["execution_score"] <= 1.0

    def test_score_batch_transformers_mocked_all_annotated(self):
        """Batch scoring with mocked transformers returns annotated=True."""
        with tempfile.TemporaryDirectory() as tmp:
            p1 = Path(tmp) / "a.png"
            p2 = Path(tmp) / "b.png"
            _make_dummy_image(p1)
            _make_dummy_image(p2)

            scorer = VLMScorer(model_name="InternVL2", device="cpu")
            scorer._processor = _MockProcessor("5")
            scorer._model = _MockModel()

            with patch.object(
                VLMScorer, "_resolve_backend", return_value="transformers"
            ):
                results = scorer.score_batch(
                    [str(p1), str(p2)], ["tracking", "sar"]
                )

            assert len(results) == 2
            for r in results:
                assert r["annotated"] is True
                assert r["vlm_score"] == 1.0

    def test_build_ref_lookup_transformers_mocked(self):
        """Reference lookup with mocked transformers builds score cache."""
        with tempfile.TemporaryDirectory() as tmp:
            ref_dir = Path(tmp)
            _make_dummy_image(ref_dir / "ref_a.png")
            _make_dummy_image(ref_dir / "ref_b_clean.png")

            scorer = VLMScorer(model_name="Phi3-Vision", device="cpu")
            scorer._processor = _MockProcessor("4")
            scorer._model = _MockModel()

            with patch.object(
                VLMScorer, "_resolve_backend", return_value="transformers"
            ):
                lookup = scorer.build_ref_lookup(ref_dir)

            assert len(lookup) == 2
            assert "ref_a" in lookup
            assert "ref_b" in lookup
            for ref_id, scores in lookup.items():
                assert scores["annotated"] is True
                assert scores["vlm_score"] == 0.8

    def test_score_image_falls_back_on_vlm_error(self):
        """When _call_vlm raises, score_image falls back to synthetic."""
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.png"
            _make_dummy_image(img_path)

            scorer = VLMScorer(model_name="Qwen2-VL", device="cpu")

            with patch.object(
                VLMScorer, "_resolve_backend", return_value="transformers"
            ):
                # _model is None and _load_model won't find real transformers
                # → _call_vlm will raise, score_image falls back
                result = scorer.score_image(str(img_path), "tracking")

            assert result["annotated"] is False  # fallback
            assert "vlm_score" in result

    def test_different_families_thread_chat_template_to_vlm(self):
        """Each family passes its formatted chat template to the VLM."""
        with tempfile.TemporaryDirectory() as tmp:
            img_path = Path(tmp) / "test.png"
            _make_dummy_image(img_path)

            # Test one model per family
            family_checks = {
                "Qwen2-VL": {
                    "present": ["object tracking"],
                    "absent": ["<|im_start|>", "<|user|>", "USER:", "<|Bot|>"],
                },
                "InternVL2": {
                    "present": ["<|im_start|>system", "<|im_start|>user", "<|im_start|>assistant"],
                },
                "InternLM-Xcomposer2": {
                    "present": ["<|User|>", "<|Bot|>"],
                    "absent": ["<|im_start|>"],
                },
                "Ovis2": {
                    "present": ["delivery"],
                    "absent": ["<|im_start|>", "<|user|>", "USER:"],
                },
                "Phi3-Vision": {
                    "present": ["<|user|>", "<|assistant|>"],
                    "absent": ["<|im_start|>system"],
                },
                "MPlugOwl3": {
                    "present": ["USER:", "ASSISTANT:", "<image>"],
                    "absent": ["<|im_start|>"],
                },
            }

            for short_name, checks in family_checks.items():
                scorer = VLMScorer(model_name=short_name, device="cpu")
                proc = _MockProcessor("4")
                scorer._processor = proc
                scorer._model = _MockModel()

                with patch.object(
                    VLMScorer, "_resolve_backend", return_value="transformers"
                ):
                    task = "delivery" if short_name == "Ovis2" else "tracking"
                    scorer.score_image(str(img_path), task)

                prompt = proc._last_texts[-1]
                for phrase in checks.get("present", []):
                    assert phrase in prompt, (
                        f"{short_name}: expected '{phrase}' in prompt: {prompt[:120]}..."
                    )
                for phrase in checks.get("absent", []):
                    assert phrase not in prompt, (
                        f"{short_name}: '{phrase}' should not be in prompt"
                    )
