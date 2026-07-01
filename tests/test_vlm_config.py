"""Tests for VLMConfig, MODEL_REGISTRY, and VLMScorer model name resolution."""

import pytest

from uav_iqa.vlm import (
    MODEL_REGISTRY,
    VLMConfig,
    VLMScorer,
)

# ===========================================================================
# VLMConfig and MODEL_REGISTRY
# ===========================================================================


class TestVLMConfig:
    """Tests for VLMConfig dataclass."""

    def test_create_vlm_config_with_all_fields(self):
        """VLMConfig can be created with all required fields."""
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
        """trust_remote_code defaults to False (security default)."""
        cfg = VLMConfig(
            short_name="M",
            hf_model_id="org/M",
            family="qwen",
            chat_template="{prompt}",
            model_class_name="AutoModelForVision2Seq",
            processor_class_name="AutoProcessor",
        )
        assert cfg.trust_remote_code is False


class TestModelRegistry:
    """Tests for MODEL_REGISTRY — the 13 supported VLMs."""

    def test_registry_has_exactly_13_models(self):
        """MODEL_REGISTRY contains exactly 13 models."""
        assert len(MODEL_REGISTRY) == 13

    def test_all_registry_entries_are_vlm_config(self):
        """Every entry in MODEL_REGISTRY is a VLMConfig."""
        for name, cfg in MODEL_REGISTRY.items():
            assert isinstance(cfg, VLMConfig), f"{name} is not VLMConfig"
            assert (
                cfg.short_name == name
            ), f"{name} short_name mismatch: {cfg.short_name}"

    def test_registry_short_names_match_expected(self):
        """Registry keys match the 13 expected model short names."""
        expected = {
            "Mini-InternVL",
            "InternVL2",
            "InternVL2.5",
            "InternVL3",
            "InternLM-Xcomposer2.5",
            "Ovis1.5-Gemma",
            "Ovis1.6-Llama",
            "Ovis2",
            "Phi3.5-Vision",
            "Phi4-Multimodal",
            "Qwen2-VL",
            "Qwen2.5-VL",
            "MPlugOwl3",
        }
        assert set(MODEL_REGISTRY.keys()) == expected

    def test_registry_models_have_valid_hf_ids(self):
        """Every model has a valid-looking HuggingFace model ID."""
        for name, cfg in MODEL_REGISTRY.items():
            assert "/" in cfg.hf_model_id, f"{name}: hf_model_id missing '/' separator"
            parts = cfg.hf_model_id.split("/")
            assert len(parts) == 2, f"{name}: hf_model_id should be 'org/model'"

    def test_registry_models_have_valid_families(self):
        """All registry entries use a known model family."""
        valid_families = {"qwen", "internvl", "internlm_xc", "ovis", "phi", "mplug"}
        for name, cfg in MODEL_REGISTRY.items():
            assert (
                cfg.family in valid_families
            ), f"{name}: unknown family '{cfg.family}'"

    def test_registry_models_have_chat_template_with_prompt_placeholder(self):
        """Every model has a chat template containing {prompt}."""
        for name, cfg in MODEL_REGISTRY.items():
            assert (
                "{prompt}" in cfg.chat_template
            ), f"{name}: chat_template missing {{prompt}} placeholder"

    def test_registry_qwen_family_models(self):
        """Qwen family contains Qwen2-VL and Qwen2.5-VL."""
        assert MODEL_REGISTRY["Qwen2-VL"].family == "qwen"
        assert MODEL_REGISTRY["Qwen2.5-VL"].family == "qwen"
        assert MODEL_REGISTRY["Qwen2-VL"].hf_model_id == "Qwen/Qwen2-VL-7B-Instruct"
        assert MODEL_REGISTRY["Qwen2.5-VL"].hf_model_id == "Qwen/Qwen2.5-VL-7B-Instruct"

    def test_registry_internvl_family_models(self):
        """InternVL family contains Mini, 2, 2.5, 3."""
        for name in ("Mini-InternVL", "InternVL2", "InternVL2.5", "InternVL3"):
            assert MODEL_REGISTRY[name].family == "internvl"
            assert MODEL_REGISTRY[name].hf_model_id.startswith("OpenGVLab/")

    def test_registry_internlm_xc_family_models(self):
        """InternLM-Xcomposer family contains 2.5."""
        for name in ("InternLM-Xcomposer2.5",):
            assert MODEL_REGISTRY[name].family == "internlm_xc"
            assert MODEL_REGISTRY[name].hf_model_id.startswith("internlm/")

    def test_registry_ovis_family_models(self):
        """Ovis family contains 1.5-Gemma, 1.6-Llama, 2."""
        for name in ("Ovis1.5-Gemma", "Ovis1.6-Llama", "Ovis2"):
            assert MODEL_REGISTRY[name].family == "ovis"
            assert MODEL_REGISTRY[name].hf_model_id.startswith("AIDC-AI/")

    def test_registry_phi_family_models(self):
        """Phi family contains Phi3.5, Phi4."""
        for name in ("Phi3.5-Vision", "Phi4-Multimodal"):
            assert MODEL_REGISTRY[name].family == "phi"
            assert MODEL_REGISTRY[name].hf_model_id.startswith("microsoft/")

    def test_registry_mplug_owl3(self):
        """MPlugOwl3 in mplug family."""
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
        """All 13 registry model names can be used to create VLMScorer."""
        for short_name in MODEL_REGISTRY:
            scorer = VLMScorer(model_name=short_name)
            assert scorer.vlm_config.short_name == short_name
            assert (
                scorer.vlm_config.hf_model_id == MODEL_REGISTRY[short_name].hf_model_id
            )

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
        assert scorer.vlm_config.model_class_name == "AutoModelForCausalLM"

    def test_init_with_nonexistent_registry_name_raises(self):
        """A name not in registry and not a valid HF ID raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            VLMScorer(model_name="NonExistentModel")

    def test_vlm_scorer_stores_default_short_name_for_registry_models(self):
        """Default model is still in registry and resolves correctly."""
        scorer = VLMScorer()
        assert scorer.vlm_config is not None
        assert scorer.vlm_config.hf_model_id == "Qwen/Qwen2.5-VL-7B-Instruct"
        assert scorer.vlm_config.family == "qwen"

    def test_vlm_scorer_vlm_config_is_vlm_config_instance(self):
        """vlm_config attribute is a VLMConfig instance."""
        scorer = VLMScorer(model_name="Phi3.5-Vision")
        assert isinstance(scorer.vlm_config, VLMConfig)
