"""Tests for uav_iqa.vlm — VLMScorer (prompt, backend, fallback, coverage, pipeline)."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from uav_iqa.domain import SUBTASK_NAMES, SUBTASK_NAME_LIST
from uav_iqa.vlm import (
    MODEL_REGISTRY,
    VLMScorer,
)
from uav_iqa.vlm.vqa_index import VQAIndex

ALL_TASKS = SUBTASK_NAME_LIST


@pytest.fixture(autouse=True)
def _isolate_vqa_index(monkeypatch, tmp_path):
    """Keep scorer tests independent of a checkout's processed dataset."""

    monkeypatch.setattr(
        "uav_iqa.vlm.scorer.VQAIndex",
        lambda _vqa_dir: VQAIndex(str(tmp_path)),
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
# Mock infrastructure for VLM model loading / pipeline tests
# ===========================================================================


class _MockProcessorOutput(dict):
    """Dict-like that supports .to(device) for transformers pipeline."""

    def to(self, device):
        return self

    def pop(self, key, *args):
        return self.get(key, *args)


class _MockProcessor:
    """Mock processor that returns a fixed VLM response text."""

    def __init__(self, response_text="4"):
        self.response_text = response_text
        self._last_texts = []
        self._last_messages = []
        self.tokenizer = self

    def __call__(self, text=None, images=None, messages=None, return_tensors=None, **kwargs):
        if messages is None and isinstance(text, list) and len(text) > 0:
            first = text[0]
            if isinstance(first, dict) and "role" in first:
                messages = text
                text = None
        if text is not None:
            if isinstance(text, list):
                self._last_texts.extend(text)
            else:
                self._last_texts.append(text)
        if messages is not None:
            self._last_messages.append(messages)
        return _MockProcessorOutput({"input_ids": __import__("torch").tensor([[1]])})

    def decode(self, token_ids, skip_special_tokens=True):
        return self.response_text

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        if isinstance(messages, list) and len(messages) > 0:
            content = messages[0].get("content", "")
            if isinstance(content, list):
                text_parts = [c["text"] for c in content if c.get("type") == "text"]
                return "\n".join(text_parts)
        return str(messages)


class _MockVit:
    def __init__(self):
        self.is_loaded = True
        self.forward = self._forward

    def load_model(self):
        pass

    def _forward(self, images):
        return None

    def vision_tower(self, *args, **kwargs):
        return None

    def feature_select(self, x):
        return None

    @property
    def dtype(self):
        return "float16"

    @property
    def device(self):
        return "cpu"


class _MockModelConfig:
    force_image_size = 448
    vocab_size = 151936

    def to_dict(self):
        return {}


class _MockModel:
    """Mock model with .to(), .eval(), .generate(), .chat(), .vis_processor()."""

    def __init__(self, response="4"):
        self._last_query = None
        self._response = response
        self.vit = _MockVit()
        self.prepare_inputs_for_generation = None
        self.tokenizer = None
        self.config = _MockModelConfig()
        self.language_model = None

    def to(self, device):
        return self

    def eval(self):
        return self

    def vis_processor(self, image):
        return __import__("torch").zeros(3, 448, 448)

    def generate(self, **kwargs):
        return [__import__("torch").tensor([1, 2, 3])]

    def chat(
        self,
        tokenizer,
        query=None,
        image=None,
        question=None,
        pixel_values=None,
        generation_config=None,
        **kwargs,
    ):
        self._last_query = query if query is not None else question
        if question is not None:
            return self._response
        return (self._response, [])

    def get_visual_tokenizer(self):
        return self

    def preprocess_image(self, image):
        return (__import__("torch").zeros(1, 3, 384, 384), None)

    def init_processor(self, tokenizer):
        return _MockProcessor(self._response)

    def parameters(self):
        return iter([__import__("torch").tensor([1.0], dtype=__import__("torch").float16)])


class _MockVLLMText:
    def __init__(self, text="4"):
        self.text = text


class _MockVLLMOutput:
    def __init__(self, text="4"):
        self.outputs = [_MockVLLMText(text)]


class _MockVLLMModel:
    def generate(self, prompts, params):
        return [_MockVLLMOutput()]


# ===========================================================================
# ===========================================================================


class TestVLMScorerUnit:
    """Unit tests for VLMScorer — verify interface contract and configuration."""

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
        for method in [
            "_generate_answer",
            "score_image_comparison",
            "score_batch_comparison",
            "_build_description_prompt",
            "_get_description_prompts",
        ]:
            assert hasattr(scorer, method), f"Missing method: {method}"


# ===========================================================================
# VLMScorer — description prompt and distortion name tests
# ===========================================================================


class TestVLMScorerPrompt:
    """Tests for description prompt construction and distortion metadata."""

    def test_get_description_prompts_returns_3_prompts(self):
        """Each task returns exactly 3 description prompts."""
        scorer = VLMScorer()
        for task in ALL_TASKS:
            prompts = scorer._get_description_prompts(task)
            assert len(prompts) == 3, f"{task}: expected 3 prompts, got {len(prompts)}"
            for i, p in enumerate(prompts):
                assert isinstance(p, str), f"{task} prompt {i}: not a string"
                assert len(p) > 0, f"{task} prompt {i}: empty"

    def test_build_description_prompt_is_non_empty(self):
        """_build_description_prompt produces non-empty formatted prompt."""
        scorer = VLMScorer()
        for task in ALL_TASKS:
            formatted = scorer._build_description_prompt(task, 0)
            assert len(formatted) > 0, f"{task}: empty formatted prompt"
            assert (
                "aerial" in formatted.lower()
            ), f"{task}: missing 'aerial' in prompt: {formatted[:80]}..."

    def test_all_tasks_have_3_prompts(self):
        """Every task type returns 3 non-empty description prompts."""
        scorer = VLMScorer()
        for task in ALL_TASKS:
            prompts = scorer._get_description_prompts(task)
            assert len(prompts) == 3
            for i, p in enumerate(prompts):
                assert p and isinstance(p, str), f"{task} prompt {i} is empty or wrong type"


# ===========================================================================
# VLMScorer — backend resolution (no more parsing tests)
# ===========================================================================


class TestVLMScorerBackend:
    """Tests for VLM backend resolution and model loading logic."""

    @patch.dict("sys.modules", {"vllm": None, "transformers": None})
    def test_resolve_backend_none_when_no_deps(self):
        """When no VLM deps are available, _resolve_backend raises RuntimeError."""
        scorer = VLMScorer(backend="auto")
        with pytest.raises(RuntimeError, match="No VLM backend available"):
            scorer._resolve_backend()

    def test_resolve_backend_explicit_none(self):
        """Explicitly setting backend='none' always returns 'none'."""
        scorer = VLMScorer(backend="none")
        backend = scorer._resolve_backend()
        assert backend == "none"

    @patch.dict("sys.modules", {"vllm": None, "transformers": None})
    def test_resolve_backend_vllm_unavailable(self):
        """When vllm module doesn't exist, auto mode fails with RuntimeError."""
        scorer = VLMScorer(backend="auto")
        with pytest.raises(RuntimeError, match="No VLM backend available"):
            scorer._resolve_backend()

    def test_resolve_backend_fails_if_unavailable(self):
        """_resolve_backend raises RuntimeError when no backend is available."""
        scorer = VLMScorer()
        with patch.dict("sys.modules", {"vllm": None, "transformers": None}):
            with pytest.raises(RuntimeError, match="No VLM backend available"):
                scorer._resolve_backend()


# ===========================================================================
# VLMScorer — fallback error handling (comparison-based)
# ===========================================================================


class TestVLMScorerFallback:
    """Tests for VLMScorer error handling when no VLM is available."""

    def test_score_image_comparison_raises_when_no_backend(self):
        """When no VLM backend, score_image_comparison raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.png"
            dist = Path(tmp) / "dist.png"
            _make_dummy_image(ref)
            _make_dummy_image(dist)

            scorer = VLMScorer(backend="none")
            with pytest.raises(RuntimeError, match="No VLM backend"):
                scorer.score_image_comparison(str(ref), str(dist), "tracking")

    def test_score_batch_comparison_raises_when_no_backend(self):
        """When no VLM backend, score_batch_comparison raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.png"
            dist = Path(tmp) / "dist.png"
            _make_dummy_image(ref)
            _make_dummy_image(dist)

            scorer = VLMScorer(backend="none")
            with pytest.raises(RuntimeError, match="No VLM backend"):
                scorer.score_batch_comparison([str(ref)], [str(dist)], ["tracking"])

    def test_score_batch_comparison_length_mismatch(self):
        """Batch comparison validates input lengths."""
        scorer = VLMScorer(backend="none")
        with pytest.raises(ValueError, match="Length mismatch"):
            scorer.score_batch_comparison(["a.png", "b.png"], ["c.png"], ["tracking"])


# ALL_TASKS constant
# ===========================================================================


class TestTaskConstants:
    def test_all_tasks_has_14_subtasks(self):
        assert len(ALL_TASKS) == 14

    def test_all_tasks_matches_subtask_names(self):
        assert set(ALL_TASKS) == set(SUBTASK_NAMES.values())
        assert "scene_description" in ALL_TASKS
        assert "object_recognition" in ALL_TASKS
        assert "quality_assessment" in ALL_TASKS
        assert "when_to_collaborate" in ALL_TASKS


# ===========================================================================
# VLMScorer — backend resolution and new method coverage (mocked)
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
        """When vllm is NOT installed, backend='vllm' raises RuntimeError."""
        with patch.dict("sys.modules", {"vllm": None, "transformers": None}):
            scorer = VLMScorer(backend="vllm")
            with pytest.raises(RuntimeError, match="vLLM backend"):
                scorer._resolve_backend()

    def test_resolve_backend_explicit_transformers_installed(self):
        """When transformers is installed, backend='transformers' resolves correctly."""
        fake_tf = type(sys)("transformers")
        with patch.dict("sys.modules", {"transformers": fake_tf}):
            scorer = VLMScorer(backend="transformers")
            backend = scorer._resolve_backend()
            assert backend == "transformers"

    def test_resolve_backend_explicit_transformers_not_installed(self):
        """When transformers NOT installed, backend='transformers' raises RuntimeError."""
        with patch.dict("sys.modules", {"vllm": None, "transformers": None}):
            scorer = VLMScorer(backend="transformers")
            with pytest.raises(RuntimeError, match="Transformers backend"):
                scorer._resolve_backend()

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
        scorer._load_model()
        assert scorer._model is None

    def test_build_description_prompt_any_task_works(self):
        """_build_description_prompt returns prompt for any task (task-agnostic)."""
        scorer = VLMScorer()
        formatted = scorer._build_description_prompt("unknown_task", 0)
        assert formatted is not None
        assert "aerial" in formatted.lower()


# ===========================================================================
# Per-family description prompt formatting
# ===========================================================================


class TestChatTemplateFormatting:
    """Tests for per-family chat template formatting via _build_description_prompt."""

    def test_qwen_family_build_description_prompt(self):
        """Qwen family wraps prompt in <|im_start|> chat format."""
        scorer = VLMScorer(model_name="Qwen2-VL")
        formatted = scorer._build_description_prompt("tracking", 0)
        assert "<|im_start|>system" in formatted
        assert "<|im_start|>user" in formatted
        assert "<|im_start|>assistant" in formatted
        assert "aerial" in formatted.lower()
        assert "Describe the key objects" in formatted

    def test_internvl_family_build_description_prompt(self):
        """InternVL family wraps prompt in <|im_start|> chat format."""
        scorer = VLMScorer(model_name="InternVL2")
        formatted = scorer._build_description_prompt("inspection", 0)
        assert "<|im_start|>system" in formatted
        assert "<|im_start|>user" in formatted
        assert "<|im_start|>assistant" in formatted
        assert "infrastructure" in formatted.lower()

    def test_internlm_xc_family_build_description_prompt(self):
        """InternLM-Xcomposer family wraps prompt in <|User|>: and <|Bot|>:."""
        scorer = VLMScorer(model_name="InternLM-Xcomposer2.5")
        formatted = scorer._build_description_prompt("sar", 0)
        assert formatted.startswith("<|User|>:")
        assert formatted.endswith("<|Bot|>:")
        assert "aerial" in formatted.lower()

    def test_ovis_family_build_description_prompt(self):
        """Ovis family returns raw description prompt."""
        scorer = VLMScorer(model_name="Ovis2")
        formatted = scorer._build_description_prompt("delivery", 0)
        assert "aerial" in formatted.lower()
        assert "<|im_start|>" not in formatted

    def test_phi_family_build_description_prompt(self):
        """Phi family wraps prompt in <|user|>...<|end|>...<|assistant|>."""
        scorer = VLMScorer(model_name="Phi3.5-Vision")
        formatted = scorer._build_description_prompt("tracking", 0)
        assert "<|user|>" in formatted
        assert "<|assistant|>" in formatted
        assert formatted.endswith("<|assistant|>\n")
        assert "aerial" in formatted.lower()

    def test_mplug_family_build_description_prompt(self):
        """MPlug family template is {prompt} — returns raw description text."""
        scorer = VLMScorer(model_name="MPlugOwl3")
        formatted = scorer._build_description_prompt("inspection", 0)
        assert "aerial" in formatted.lower()
        assert "<|im_start|>" not in formatted

    def test_all_13_models_produce_non_empty_description_prompt(self):
        """Every model in the registry produces a non-empty description prompt."""
        for short_name in MODEL_REGISTRY:
            scorer = VLMScorer(model_name=short_name)
            for task in ALL_TASKS:
                formatted = scorer._build_description_prompt(task, 0)
                assert len(formatted) > 0, f"{short_name}/{task}: empty prompt"
                assert (
                    "aerial" in formatted.lower()
                ), f"{short_name}/{task}: 'aerial' not in prompt: {formatted[:80]}..."

    def test_internvl3_format_matches_internvl_pattern(self):
        """InternVL3 uses same format as InternVL2."""
        scorer2 = VLMScorer(model_name="InternVL2")
        scorer3 = VLMScorer(model_name="InternVL3")
        p2 = scorer2._build_description_prompt("tracking", 0)
        p3 = scorer3._build_description_prompt("tracking", 0)
        assert "<|im_start|>system" in p2
        assert "<|im_start|>system" in p3

    def test_phi4_format_matches_phi_pattern(self):
        """Phi4-Multimodal uses same format as Phi3.5-Vision."""
        scorer3 = VLMScorer(model_name="Phi3.5-Vision")
        scorer4 = VLMScorer(model_name="Phi4-Multimodal")
        p3 = scorer3._build_description_prompt("tracking", 0)
        p4 = scorer4._build_description_prompt("tracking", 0)
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

        class MockVit:
            is_loaded = True

            def load_model(self):
                pass

            def forward(self, images):
                return None

            def vision_tower(self, *args, **kwargs):
                return None

            def feature_select(self, x):
                return x

            @property
            def dtype(self):
                import torch

                return torch.float16

            @property
            def device(self):
                return "cpu"

        class MockModel:
            is_loaded = True

            def to(self, device):
                return self

            def eval(self):
                return self

            def get_model(self):
                return self

            def load_model(self):
                pass

            def prepare_inputs_for_generation(self, *args, **kwargs):
                return ()

        MockModel.vit = MockVit()
        MockModel.tok_embeddings = None

        def from_pretrained(*args, **kwargs):
            call_tracker.append(cls_name)
            model = MockModel()
            model.vit = MockVit()
            model.tok_embeddings = None
            return model

        return staticmethod(from_pretrained)

    @staticmethod
    def _make_mock_module(class_names, call_tracker):
        """Create a fake transformers module with given class names.

        Always includes essential classes that ``_load_model()`` accesses
        unconditionally: ``PreTrainedModel``, ``GenerationMixin``,
        ``AutoConfig``, ``DynamicCache``, ``GenerationConfig``,
        and ``CLIPImageProcessor``.
        """
        mock_tf = type(sys)("transformers")

        _essential = {
            "PreTrainedModel": {},
            "GenerationMixin": {"generate": lambda self, **kw: [__import__("torch").tensor([1])]},
            "AutoConfig": {"register": classmethod(lambda cls, key, *a, **kw: None)},
            "GenerationConfig": {},
            "CLIPImageProcessor": {},
        }
        for cls_name, extra_attrs in _essential.items():
            mock_cls = type(cls_name, (), extra_attrs)
            mock_cls.from_pretrained = classmethod(lambda cls, *a, **kw: type("Mock", (), {})())
            setattr(mock_tf, cls_name, mock_cls)

        _cache_utils = type(sys)("cache_utils")
        _DC = type("DynamicCache", (), {})
        _DC.get_seq_length = lambda self, layer_idx=None: 0
        _cache_utils.DynamicCache = _DC
        mock_tf.cache_utils = _cache_utils

        for cls_name in class_names:
            mock_cls = type(cls_name, (), {})
            mock_cls.from_pretrained = TestVLMScorerModelLoading._fake_from_pretrained_factory(
                cls_name, call_tracker
            )
            mock_cls.get_init_context = classmethod(
                lambda cls, dtype, is_quantized, _is_ds_init_called, allow_all_kernels=None: [
                    __import__("torch").device("meta"),
                ]
            )
            setattr(mock_tf, cls_name, mock_cls)
        return mock_tf

    def test_load_model_uses_vlm_config_model_class(self):
        """_load_model uses model_class_name from VLMConfig (InternVL -> AutoModel)."""
        import torch

        called = []
        mock_tf = self._make_mock_module(["AutoModel", "AutoTokenizer"], called)

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(model_name="InternVL2", backend="transformers", device="cpu")
            scorer._load_model()
            assert "AutoModel" in called
            assert "AutoTokenizer" in called

    def test_load_model_phi_uses_auto_model_for_causal_lm(self):
        """Phi family models use AutoModelForCausalLM."""
        import torch

        called = []
        mock_tf = self._make_mock_module(["AutoModelForCausalLM", "AutoProcessor"], called)

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(model_name="Phi3.5-Vision", backend="transformers", device="cpu")
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

        mock_tf.PreTrainedModel = type("PreTrainedModel", (), {})
        mock_tf.PreTrainedModel._supports_flash_attn_2 = classmethod(lambda cls: False)
        mock_tf.GenerationMixin = type(
            "GenerationMixin", (), {"generate": lambda self, **kw: [torch.tensor([1])]}
        )

        mock_tf.AutoModel = type("AutoModel", (), {})
        mock_tf.AutoModel.from_pretrained = staticmethod(record_trust)

        mock_tf.AutoTokenizer = type("AutoTokenizer", (), {})
        mock_tf.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **kw: MockModel())

        _cache_utils = type(sys)("cache_utils")
        _DC = type("DynamicCache", (), {})
        _DC.get_seq_length = lambda self, layer_idx=None: 0
        _cache_utils.DynamicCache = _DC
        mock_tf.cache_utils = _cache_utils

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(model_name="InternVL2", backend="transformers", device="cpu")
            scorer._load_model()
            assert trust_values == [scorer.vlm_config.trust_remote_code]

    def test_load_model_qwen_uses_auto_model_for_vision2seq(self):
        """Qwen family uses AutoModelForVision2Seq (backward compat)."""
        import torch

        called = []
        mock_tf = self._make_mock_module(["AutoModelForVision2Seq", "AutoProcessor"], called)

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(model_name="Qwen2-VL", backend="transformers", device="cpu")
            scorer._load_model()
            assert "AutoModelForVision2Seq" in called
            assert "AutoProcessor" in called

    def test_load_model_mplug_uses_auto_model_for_causal_lm(self):
        """MPlug family uses AutoModelForCausalLM + AutoTokenizer."""
        import torch

        called = []
        mock_tf = self._make_mock_module(["AutoModelForCausalLM", "AutoTokenizer"], called)

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(model_name="MPlugOwl3", backend="transformers", device="cpu")
            scorer._load_model()
            assert "AutoModelForCausalLM" in called
            assert "AutoTokenizer" in called

    def test_load_model_internlm_xc_uses_auto_model_for_causal_lm(self):
        """InternLM-Xcomposer uses AutoModelForCausalLM."""
        import torch

        called = []
        mock_tf = self._make_mock_module(
            ["AutoModelForCausalLM", "AutoTokenizer", "AutoConfig", "CLIPVisionModel"],
            called,
        )

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(
                model_name="InternLM-Xcomposer2.5", backend="transformers", device="cpu"
            )
            scorer._load_model()
            assert "AutoModelForCausalLM" in called
            assert "AutoTokenizer" in called

    def test_load_model_ovis_uses_auto_model_for_causal_lm(self):
        """Ovis family uses AutoModelForCausalLM + AutoProcessor."""
        import torch

        called = []
        mock_tf = self._make_mock_module(["AutoModelForCausalLM", "AutoProcessor"], called)

        with patch.dict("sys.modules", {"transformers": mock_tf, "torch": torch}):
            scorer = VLMScorer(model_name="Ovis2", backend="transformers", device="cpu")
            scorer._load_model()
            assert "AutoModelForCausalLM" in called
            assert "AutoProcessor" in called


# ===========================================================================
# Integration — comparison-based scoring pipeline with mocked VLM backends
# ===========================================================================


class TestVLMScorerPipelineIntegration:
    """Integration tests: comparison scoring with mocked _generate_answer."""

    def test_score_image_comparison_mocked(self):
        """Mock _generate_answer — verify cognitive score computed from descriptions."""
        DESCRIPTION = "Aerial view showing urban landscape with buildings and roads"

        with tempfile.TemporaryDirectory() as tmp:
            ref = Path(tmp) / "ref.png"
            dist = Path(tmp) / "dist.png"
            _make_dummy_image(ref)
            _make_dummy_image(dist)

            scorer = VLMScorer(model_name="Qwen2-VL", device="cpu")
            with patch.object(scorer, "_generate_answer", return_value=DESCRIPTION):
                result = scorer.score_image_comparison(str(ref), str(dist), "tracking")

            assert result["metadata"]["annotated"] is True
            assert "summary" in result
            assert result["summary"]["cognitive_score"] > 0.9
            assert "bleu" in result["summary"]
            assert "rouge_l" in result["summary"]
            assert "cider" in result["summary"]
            assert result["metadata"]["num_prompts"] == 3
            assert "details" in result
            assert len(result["details"]) == 3
            for detail in result["details"]:
                assert "scores" in detail
                assert "bleu" in detail["scores"]
                assert "rouge_l" in detail["scores"]
                assert "cider" in detail["scores"]
                assert detail["reference"]["answer"] == DESCRIPTION
                assert detail["prediction"]["answer"] == DESCRIPTION
            for prompt in result["details"]:
                assert "prompt" in prompt
            assert len(result["details"]) == 3

    def test_score_batch_comparison_mocked(self):
        """Batch comparison with mocked _generate_answer works correctly."""
        DESCRIPTION = "Aerial view showing urban landscape with buildings and roads"

        with tempfile.TemporaryDirectory() as tmp:
            r1 = Path(tmp) / "ref_a.png"
            d1 = Path(tmp) / "dist_a.png"
            d2 = Path(tmp) / "dist_b.png"
            _make_dummy_image(r1)
            _make_dummy_image(d1)
            _make_dummy_image(d2)

            scorer = VLMScorer(model_name="InternVL2", device="cpu")
            with patch.object(scorer, "_generate_answer", return_value=DESCRIPTION):
                results = scorer.score_batch_comparison(
                    [str(r1), str(r1)],
                    [str(d1), str(d2)],
                    ["tracking", "tracking"],
                    show_progress=False,
                )

            assert len(results) == 2
            for r in results:
                assert r["metadata"]["annotated"] is True
                assert r["summary"]["cognitive_score"] > 0.9
                assert "details" in r
                assert len(r["details"]) == 3
                for detail in r["details"]:
                    assert "prompt" in detail
                    assert "reference" in detail
                    assert "prediction" in detail

    def test_all_13_models_score_comparison_mocked(self):
        """Every registered model works with mocked score_image_comparison."""
        DESCRIPTION = "Aerial view showing urban landscape with buildings and roads"

        with tempfile.TemporaryDirectory() as tmp:
            img1 = Path(tmp) / "ref.png"
            img2 = Path(tmp) / "dist.png"
            _make_dummy_image(img1)
            _make_dummy_image(img2)

            for short_name in sorted(MODEL_REGISTRY):
                scorer = VLMScorer(model_name=short_name, device="cpu")
                with patch.object(scorer, "_generate_answer", return_value=DESCRIPTION):
                    result = scorer.score_image_comparison(str(img1), str(img2), "tracking")

                assert result["metadata"]["annotated"] is True, f"{short_name}: annot mismatch"
                assert (
                    result["summary"]["cognitive_score"] > 0.9
                ), f"{short_name}: low cognitive score {result['summary']['cognitive_score']}"

    def test_different_families_score_comparison_mocked(self):
        """Every model family works with mocked score_image_comparison."""
        DESCRIPTION = "Aerial view showing urban landscape with buildings and roads"

        families = [
            "Qwen2-VL",
            "InternVL2",
            "InternLM-Xcomposer2.5",
            "Ovis2",
            "Phi3.5-Vision",
            "MPlugOwl3",
        ]

        with tempfile.TemporaryDirectory() as tmp:
            img1 = Path(tmp) / "ref.png"
            img2 = Path(tmp) / "dist.png"
            _make_dummy_image(img1)
            _make_dummy_image(img2)

            for short_name in families:
                scorer = VLMScorer(model_name=short_name, device="cpu")
                with patch.object(scorer, "_generate_answer", return_value=DESCRIPTION):
                    result = scorer.score_image_comparison(str(img1), str(img2), "tracking")

                assert result["metadata"]["annotated"] is True, f"{short_name}: annot mismatch"
                assert (
                    result["summary"]["cognitive_score"] > 0.9
                ), f"{short_name}: low cognitive score"


class TestGenerateAnswerMulti:
    """Unit tests for VLMScorer._generate_answer_multi file validation."""

    def test_raises_on_missing_file(self, tmp_path):
        """_generate_answer_multi should raise FileNotFoundError for nonexistent files."""
        scorer = VLMScorer()
        img1 = tmp_path / "real.png"
        _make_dummy_image(img1, 32)
        img_missing = tmp_path / "nonexistent.png"

        with pytest.raises(FileNotFoundError, match="Image not found"):
            scorer._generate_answer_multi(
                [str(img1), str(img_missing)], "vqa", "Describe the scene."
            )

    def test_raises_on_empty_paths(self):
        """_generate_answer_multi should raise RuntimeError for empty image_paths."""
        scorer = VLMScorer()
        with pytest.raises(RuntimeError, match="No image paths"):
            scorer._generate_answer_multi([], "vqa", "Describe the scene.")
