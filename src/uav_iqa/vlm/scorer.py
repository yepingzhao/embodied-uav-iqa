"""Real VLM-based quality scoring for UAV-IQA.

Scoring works via structured description comparison (per proposal design):
VLM generates task-conditioned scene descriptions for both clean reference
and distorted images, compares them via BLEU/ROUGE-L/CIDEr (1:1:0.1) to
compute a cognitive quality score.
"""

import importlib
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

from tqdm import tqdm

from uav_iqa.text_metrics import (
    compute_bleu,
    compute_cider,
    compute_cognitive_score,
    compute_rouge_l,
)
from uav_iqa.vlm.backends import (
    run_transformers_family,
    run_vllm_inference,
    run_vllm_inference_batch,
)
from uav_iqa.vlm.config import DEPRECATED_MODELS, MODEL_REGISTRY, VLMConfig
from uav_iqa.vlm.vqa_index import VQAIndex

try:
    from transformers import AutoConfig as _AutoConfig
    from transformers import PreTrainedModel as _TransformerPTM
    from transformers import CLIPVisionModel as _CLIPVisionModel
except ImportError:
    _AutoConfig = None
    _TransformerPTM = None
    _CLIPVisionModel = None

_log = logging.getLogger(__name__)
_MODEL_PATCHES_APPLIED = False


class VLMScorer:
    """Real VLM-based quality scorer using vLLM or transformers.

    Loads a vision-language model and scores images by prompting the model
    to assess visual quality for a given aerial embodied task.

    Requires vLLM or transformers with VL support (install: uv sync --group dev --extra vlm).
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        backend: str = "auto",  # "auto", "vllm", "transformers", "none"
        device: str = "cuda",
        seed: int = 42,
        cache_dir: Optional[str] = None,
        local_files_only: bool = True,
        vqa_dir: str = "data/processed",
        gpu_memory_utilization: float = 0.5,
        max_model_len: Optional[int] = None,
        max_num_seqs: int = 16,
    ):
        self.backend = backend
        self.device = device
        self.seed = seed
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs
        self._model = None
        self._load_failed = False
        self.vqa_index = VQAIndex(vqa_dir)

        self.vlm_config = self._resolve_model_name(model_name)
        self.model_name = self.vlm_config.hf_model_id

        _log.info(
            "VLMScorer initialized: model=%s backend=%s device=%s",
            self.model_name,
            backend,
            device,
        )

    @staticmethod
    def _resolve_model_name(name: str) -> VLMConfig:
        """Resolve a model name to a VLMConfig.

        Looks up the name in MODEL_REGISTRY first (by short name),
        then by hf_model_id.
        Falls back to checking if it's already a valid HF model ID.
        If the name is not in the registry and doesn't look like a
        HF path (org/model), raises ValueError.
        """
        if name in DEPRECATED_MODELS:
            new_name = DEPRECATED_MODELS[name]
            _log.warning(
                "VLM model '%s' is deprecated, auto-mapping to '%s'",
                name, new_name,
            )
            name = new_name

        if name in MODEL_REGISTRY:
            return MODEL_REGISTRY[name]

        # Also try matching by hf_model_id
        for cfg in MODEL_REGISTRY.values():
            if cfg.hf_model_id == name:
                return cfg

        # Check if it looks like a HuggingFace model ID (contains "/")
        if "/" in name:
            _log.info("Model '%s' not in registry, using default config", name)
            return VLMConfig(
                short_name=name.split("/")[-1],
                hf_model_id=name,
                family="qwen",
                chat_template=(
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    "<|im_start|>user\n{image_tags}{prompt}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                ),
                model_class_name="AutoModelForCausalLM",
                processor_class_name="AutoProcessor",
            )

        raise ValueError(
            f"Unknown model '{name}'. "
            f"Must be a model from MODEL_REGISTRY "
            f"({', '.join(sorted(MODEL_REGISTRY.keys()))}) "
            f"or a HuggingFace model ID (e.g., 'org/model-name')."
        )

    def _resolve_backend(self) -> str:
        """Resolve actual backend capability.

        Checks for available VLM backends in priority order:
        1. vLLM (fastest, recommended for production)
        2. transformers (fallback, more memory)

        Raises RuntimeError if neither backend is available.
        When self.backend is 'auto', auto-detects the best available.
        When set to 'none', always falls back.
        """
        if self.backend == "none":
            return "none"

        # vLLM ships model files for: qwen2_vl, internvl, ovis, phi4mm.
        # NOT supported: internlm_xc (vision model), mplug, phi3-v.
        # We conservatively route phi to transformers (phi3-v not
        # supported) and ovis to vLLM (has dedicated model file).
        vllm_viable = self.vlm_config.family in frozenset({"qwen", "internvl", "ovis"})

        if self.backend == "vllm":
            if not vllm_viable:
                _log.info(
                    "vLLM does not support %s family, using transformers instead",
                    self.vlm_config.family,
                )
                return self._resolve_transformers_backend()
            import importlib.util

            if importlib.util.find_spec("vllm") is not None:
                return "vllm"
            raise RuntimeError(
                "vLLM backend explicitly requested but not installed. "
                "Install with: pip install vllm"
            )

        if self.backend == "transformers":
            return self._resolve_transformers_backend()

        # auto: try vllm first, then transformers, then raise
        if vllm_viable:
            import importlib.util

            if importlib.util.find_spec("vllm") is not None:
                return "vllm"

        return self._resolve_transformers_backend()

    def _resolve_transformers_backend(self) -> str:
        """Resolve the transformers backend."""
        import importlib.util

        if importlib.util.find_spec("transformers") is not None:
            return "transformers"

        if self.backend == "transformers":
            raise RuntimeError(
                "Transformers backend explicitly requested but not installed. "
                "Install with: pip install transformers accelerate"
            )

        # auto mode: no backend available
        raise RuntimeError(
            f"No VLM backend available for {self.model_name}. "
            "Install vLLM (pip install vllm) or "
            "transformers (pip install transformers accelerate)."
        )

    @staticmethod
    def _apply_global_patches(tf_mod):
        """Apply global monkey-patches for transformers 4.57.6 compatibility.

        Patches are guarded by ``_MODEL_PATCHES_APPLIED`` and applied at
        most once per process. Includes DynamicCache shims, Siglip hidden
        states, PreTrainedModel._supports_flash_attn_2, builtins typo fix,
        and LLM backbone generation stubs.
        """
        from transformers.cache_utils import DynamicCache as _DC

        if not hasattr(_DC, "seen_tokens"):
            _DC.seen_tokens = property(lambda self: self.get_seq_length())
        if not hasattr(_DC, "get_usable_length"):
            _DC.get_usable_length = lambda self, seq_length, layer_idx=None: (
                self.get_seq_length() if layer_idx is None else self.get_seq_length(layer_idx)
            )
        _DC.max_cache_len = None
        if not hasattr(_DC, "get_max_length"):
            _DC.get_max_length = lambda self: self.max_cache_len

        try:
            from transformers.models.siglip.modeling_siglip import (
                SiglipEncoder as _SE,
                SiglipVisionTransformer as _SVT,
            )
            from transformers.modeling_outputs import (
                BaseModelOutput,
                BaseModelOutputWithPooling,
            )

            def _patched_se_forward(
                self,
                inputs_embeds,
                attention_mask=None,
                output_hidden_states=False,
                **kwargs,
            ):
                encoder_states = () if output_hidden_states else None
                hidden_states = inputs_embeds
                for layer in self.layers:
                    if output_hidden_states:
                        encoder_states = encoder_states + (hidden_states,)
                    hidden_states = layer(hidden_states, attention_mask, **kwargs)
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                return BaseModelOutput(
                    last_hidden_state=hidden_states,
                    hidden_states=encoder_states,
                )

            _SE.forward = _patched_se_forward

            def _patched_svt_forward(self, pixel_values, interpolate_pos_encoding=False, **kwargs):
                hidden_states = self.embeddings(
                    pixel_values,
                    interpolate_pos_encoding=interpolate_pos_encoding,
                )
                encoder_outputs = self.encoder(inputs_embeds=hidden_states, **kwargs)
                last_hidden_state = encoder_outputs.last_hidden_state
                last_hidden_state = self.post_layernorm(last_hidden_state)
                pooler_output = self.head(last_hidden_state) if self.use_head else None
                return BaseModelOutputWithPooling(
                    last_hidden_state=last_hidden_state,
                    pooler_output=pooler_output,
                    hidden_states=encoder_outputs.hidden_states,
                )

            _SVT.forward = _patched_svt_forward
        except (ImportError, ModuleNotFoundError):
            pass

        _PTM = getattr(tf_mod, "PreTrainedModel")
        _PTM._supports_flash_attn_2 = classmethod(lambda cls: False)

        import builtins as _bi

        if not hasattr(_bi, "NotImplementError"):
            _bi.NotImplementError = NotImplementedError

        for _llm_name in ("Phi3ForCausalLM", "InternLM2ForCausalLM"):
            if hasattr(tf_mod, _llm_name) and not hasattr(getattr(tf_mod, _llm_name), "generate"):
                _llm_cls = getattr(tf_mod, _llm_name)
                _llm_cls.generate = getattr(tf_mod, "GenerationMixin").generate

    @staticmethod
    def _patch_vllm_rope_config():
        """Fix vLLM bug: propagate original_max_position_embeddings into rope_parameters.

        When original_max_position_embeddings is at the top-level config but
        NOT inside rope_scaling (standard Phi-3 layout), vLLM 0.19.1 fails to
        propagate it. This monkey-patches patch_rope_parameters to inject it.
        """
        import vllm.transformers_utils.config as _vllm_cfg

        if getattr(_vllm_cfg.patch_rope_parameters, "_uav_iqa_patched", False):
            return

        _original = _vllm_cfg.patch_rope_parameters

        def _patched_patch_rope_parameters(config):
            _original(config)
            ompe = getattr(config, "original_max_position_embeddings", None)
            rope_params = getattr(config, "rope_parameters", None)
            if ompe is not None and isinstance(rope_params, dict):
                if "original_max_position_embeddings" not in rope_params:
                    rope_params["original_max_position_embeddings"] = ompe

        _patched_patch_rope_parameters._uav_iqa_patched = True  # type: ignore[attr-defined]
        _vllm_cfg.patch_rope_parameters = _patched_patch_rope_parameters

    def _load_model(self):
        """Load the VLM model (lazy initialization)."""
        backend = self._resolve_backend()

        if backend == "vllm":
            self._patch_vllm_rope_config()
            from vllm import LLM

            _log.info("Loading model %s via vLLM on %s...", self.model_name, self.device)
            # Fix vLLM v0.19 CUDA graph memory profiling (can over-estimate
            # by 5000%+, starving the KV cache).  See vLLM issue tracker.
            os.environ.setdefault(
                "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1",
            )
            llm_kwargs: dict = dict(
                model=self.model_name,
                trust_remote_code=self.vlm_config.trust_remote_code,
                max_num_seqs=self.max_num_seqs,
                gpu_memory_utilization=self.gpu_memory_utilization,
                disable_log_stats=True,
            )
            if self.cache_dir:
                llm_kwargs["download_dir"] = self.cache_dir
            if self.max_model_len is not None:
                llm_kwargs["max_model_len"] = self.max_model_len
            self._model = LLM(**llm_kwargs)

        elif backend == "transformers":
            import torch

            tf_mod = importlib.import_module("transformers")
            model_cls = getattr(tf_mod, self.vlm_config.model_class_name)
            proc_cls = getattr(tf_mod, self.vlm_config.processor_class_name)

            _log.info(
                "Loading model %s via transformers on %s (class=%s)...",
                self.model_name,
                self.device,
                self.vlm_config.model_class_name,
            )
            load_kwargs: dict = dict(
                trust_remote_code=self.vlm_config.trust_remote_code,
                local_files_only=self.local_files_only,
            )
            if self.cache_dir:
                load_kwargs["cache_dir"] = self.cache_dir
            # Suppress re-initialization errors for mismatched/missing
            # CLIP vision-model keys (post_layernorm added by newer
            # transformers versions but not present in older checkpoints).
            load_kwargs["ignore_mismatched_sizes"] = True
            # FlashAttention2 is not installed on this system.
            # Force eager attention globally for all models.
            load_kwargs["attn_implementation"] = "eager"

            # Global monkey-patches for transformer 4.57.6 compatibility.
            # Guarded against double-application since they mutate global state.
            global _MODEL_PATCHES_APPLIED
            if not _MODEL_PATCHES_APPLIED:
                self._apply_global_patches(tf_mod)
                _MODEL_PATCHES_APPLIED = True

            # Per-family patches (safe to apply on each load — guarded
            # by model identity checks inside).

            if self.vlm_config.family == "internlm_xc":
                # InternLM-Xcomposer2.5 downloads SimHei.ttf from
                # huggingface.co during __init__, which times out in
                # offline mode.  Patch urllib.request.urlopen to serve
                # the font from the local model cache instead.
                import glob as _glob
                import urllib.request as _urllib_request
                from pathlib import Path as _Path

                _orig_urlopen = _urllib_request.urlopen

                def _resolve_simhei_local() -> _Path:
                    _cache_root = os.environ.get(
                        "HF_HOME",
                        os.path.expanduser("~/.cache/huggingface"),
                    )
                    _pattern = (
                        f"{_cache_root}/hub/models--internlm--internlm-xcomposer2d5-7b/"
                        "snapshots/*/SimHei.ttf"
                    )
                    _matches = _glob.glob(_pattern)
                    if _matches:
                        return _Path(_matches[0])
                    raise FileNotFoundError("SimHei.ttf not found in local cache")

                def _patched_urlopen(url, *args, **kwargs):
                    _url_str = url if isinstance(url, str) else str(url)
                    if "SimHei.ttf" in _url_str and "internlm-xcomposer2d5" in _url_str:
                        return open(str(_resolve_simhei_local()), "rb")
                    return _orig_urlopen(url, *args, **kwargs)

                _urllib_request.urlopen = _patched_urlopen

                # InternLM-Xcomposer2/2.5: PretrainedConfig does not expose
                # max_length as a Python attribute even though config.json
                # defines it.  The model __init__ accesses config.max_length,
                # so we patch it in before loading.
                AutoConfig = getattr(tf_mod, "AutoConfig")
                config = AutoConfig.from_pretrained(self.model_name, **load_kwargs)
                if not hasattr(config, "max_length"):
                    config.max_length = 4096  # default in xcomposer config.json

                # transformers 5.x flattened CLIPVisionModel (removed the
                # .vision_model intermediate wrapper, put components
                # directly on CLIPVisionModel).  internlm-xcomposer2's
                # build_mlp.py accesses .vision_model.embeddings; add a
                # compat property so that code still works.
                # We detect this by checking CLIPVisionModel.__init__
                # source: in 5.x the __init__ does NOT create
                # self.vision_model = CLIPVisionTransformer(config).
                import inspect as _inspect

                if _CLIPVisionModel is not None:
                    _init_src = _inspect.getsource(_CLIPVisionModel.__init__)
                    _needs_vision_model_shim = "self.vision_model" not in _init_src
                    if _needs_vision_model_shim:
                        _CLIPVisionModel.vision_model = property(lambda self: self)

                # transformers >=5.x CLIPVisionEmbeddings.forward enforces a
                # strict image size check (336×336), but internlm-xcomposer
                # resizes inputs to 490×490 with custom position-embedding
                # interpolation.  Force interpolate_pos_encoding=True.
                try:
                    from transformers.models.clip.modeling_clip import (
                        CLIPVisionEmbeddings,
                    )

                    if (
                        "interpolate_pos_encoding"
                        in CLIPVisionEmbeddings.forward.__code__.co_varnames
                    ):
                        _orig_clip_embed_fwd = CLIPVisionEmbeddings.forward

                        def _patched_clip_embed_fwd(
                            self,
                            pixel_values,
                            interpolate_pos_encoding=False,
                        ):
                            return _orig_clip_embed_fwd(
                                self,
                                pixel_values,
                                interpolate_pos_encoding=True,
                            )

                        CLIPVisionEmbeddings.forward = _patched_clip_embed_fwd
                except (ImportError, ModuleNotFoundError, AttributeError):
                    pass  # transformers is mocked in tests

                # internlm-xcomposer2/2.5 loads CLIPVisionModel inside
                # build_vision_tower() *without* a dtype, so CLIP stays
                # float32 while the rest of the model is float16.  Patch
                # CLIPVisionModel.from_pretrained to force float16.
                # Xcomposer2.5 uses internlm/internlm-xcomposer2d5-clip
                # which is not in the local cache; redirect to the cached
                # openai/clip-vit-large-patch14-336 instead (same arch).
                _orig_clip_from_pretrained = _CLIPVisionModel.from_pretrained

                @classmethod
                def _patched_clip_from_pretrained(cls, *args, **kwargs):
                    kwargs.setdefault("torch_dtype", torch.float16)
                    kwargs.setdefault("ignore_mismatched_sizes", True)
                    model = _orig_clip_from_pretrained.__func__(cls, *args, **kwargs)
                    # Fix corrupted position_ids on CLIP vision embeddings.
                    # The position_ids buffer can contain stale/garbage data
                    # because internlm-xcomposer2d5-clip was created with
                    # transformers 4.33.1 (persistent buffer defaults differ
                    # across versions). Re-compute from config to guarantee
                    # correct indices for the position_embedding lookup.
                    try:
                        emb = (
                            model.embeddings
                            if hasattr(model, "embeddings")
                            else model.vision_model.embeddings
                        )
                        n_patches = (emb.image_size // emb.patch_size) ** 2
                        device = getattr(emb.position_ids, "device", torch.device("cpu"))
                        emb.register_buffer(
                            "position_ids",
                            torch.arange(n_patches + 1, device=device).unsqueeze(0),
                            persistent=False,
                        )
                    except Exception:
                        pass
                    return model

                _CLIPVisionModel.from_pretrained = _patched_clip_from_pretrained

                # transformers 5.x wraps model __init__ in torch.device("meta")
                # via PreTrainedModel.get_init_context(), which breaks
                # internlm-xcomposer's nested CLIPVisionModel.from_pretrained()
                # call inside build_vision_tower().  Patch out the meta device
                # so that the nested from_pretrained succeeds.
                _orig_get_init_ctx = _TransformerPTM.get_init_context

                @classmethod
                def _patched_get_init_context(cls, *args, **kwargs):
                    ctxs = _orig_get_init_ctx.__func__(cls, *args, **kwargs)
                    return [
                        c for c in ctxs if not (isinstance(c, torch.device) and str(c) == "meta")
                    ]

                _TransformerPTM.get_init_context = _patched_get_init_context
                try:
                    self._model = model_cls.from_pretrained(
                        self.model_name,
                        config=config,
                        torch_dtype=torch.float16,
                        **load_kwargs,
                    ).to(self.device)
                    self._model.eval()
                    # Fix CLIP vision position_ids corrupted during
                    # loading/.to(device).  The buffer contains garbage
                    # (non-deterministic uninitialized memory) that
                    # causes CUDA device-side asserts in the position
                    # embedding lookup (Bug #4 in internlm_xc series).
                    for _m in self._model.modules():
                        if isinstance(_m, _CLIPVisionModel):
                            _emb = (
                                _m.embeddings
                                if hasattr(_m, "embeddings")
                                else _m.vision_model.embeddings
                            )
                            _n = (_emb.image_size // _emb.patch_size) ** 2
                            _emb.register_buffer(
                                "position_ids",
                                torch.arange(
                                    _n + 1,
                                    device=_emb.position_ids.device,
                                ).unsqueeze(0),
                                persistent=False,
                            )
                    # Move max_length from config to generation_config.
                    # config.max_length is needed by the model __init__
                    # but transformers 5.x raises ValueError when the
                    # config has non-standard attributes at generation
                    # time.  Transfer it to the generation_config where
                    # it belongs in the 5.x API.
                    if hasattr(self._model.config, "max_length"):
                        _ml = self._model.config.max_length
                        if not hasattr(self._model.generation_config, "max_length") or self._model.generation_config.max_length is None:
                            self._model.generation_config.max_length = _ml
                        del self._model.config.max_length
                finally:
                    _urllib_request.urlopen = _orig_urlopen
                    _TransformerPTM.get_init_context = _orig_get_init_ctx
                    _CLIPVisionModel.from_pretrained = _orig_clip_from_pretrained
                # Keep CLIP structure patches alive for scoring lifetime;
                # the model internals need .vision_model shim and
                # interpolate_pos_encoding=True for 490×490 inputs.

                # CLIPVisionTower.forward casts output back to input
                # image dtype (float32), but the rest of the model
                # (vision_proj, LM) is float16 → dtype mismatch.
                # Patch forward to cast output to vision_tower's dtype.
                if "2d5" not in self.model_name:
                    _orig_vit_fwd = self._model.vit.forward

                    def _patched_vit_fwd(self_vit, images, *args, **kwargs):
                        if not self_vit.is_loaded:
                            self_vit.load_model()
                        if type(images) is list:
                            image_features = []
                            for image in images:
                                image_forward_out = self_vit.vision_tower(
                                    image.to(
                                        device=self_vit.device, dtype=self_vit.dtype
                                    ).unsqueeze(0),
                                    output_hidden_states=True,
                                )
                                image_feature = self_vit.feature_select(image_forward_out)
                                image_features.append(image_feature)
                        else:
                            image_forward_outs = self_vit.vision_tower(
                                images.to(device=self_vit.device, dtype=self_vit.dtype),
                                output_hidden_states=True,
                            )
                            image_features = self_vit.feature_select(image_forward_outs)
                        return image_features

                    self._model.vit.forward = _patched_vit_fwd.__get__(self._model.vit)
                # Xcomposer2.5 dtype fix is applied directly in
                # the hub cache source (build_mlp.py: .to(self.dtype)
                # instead of .to(input_imgs.dtype)).

                # transformers 4.57+ passes Cache objects (e.g. DynamicCache)
                # for past_key_values via generate(), but internlm-xcomposer2's
                # prepare_inputs_for_generation expects legacy tuple format.
                # Convert Cache→tuple on entry.
                _orig_prepare = self._model.prepare_inputs_for_generation

                def _patched_prepare(
                    self_m,
                    input_ids,
                    past_key_values=None,
                    attention_mask=None,
                    inputs_embeds=None,
                    im_mask=None,
                    infer_mode="base",
                    **kwargs,
                ):
                    if past_key_values is not None and hasattr(past_key_values, "to_legacy_cache"):
                        past_key_values = past_key_values.to_legacy_cache()
                        # DynamicCache(config=...) pre-allocates None entries
                        # for all layers; treat that as "no cache yet".
                        if (
                            past_key_values
                            and past_key_values[0] is not None
                            and past_key_values[0][0] is None
                        ):
                            past_key_values = None
                    return _orig_prepare(
                        input_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        inputs_embeds=inputs_embeds,
                        im_mask=im_mask,
                        **kwargs,
                    )

                self._model.prepare_inputs_for_generation = _patched_prepare.__get__(self._model)

                # Xcomposer2.5's chat() passes `infer_mode` to generate(),
                # which prepare_inputs_for_generation consumes but forward
                # does not.  transformers 4.57.6 validates all model_kwargs,
                # flagging `infer_mode`.  Make the model tolerate it.
                if "2d5" in self.model_name:
                    self._model._validate_model_kwargs_bak = self._model._validate_model_kwargs
                    self._model._validate_model_kwargs = lambda *a, **kw: None

                # Load tokenizer separately and assign to model.tokenizer.
                # The model.chat() API requires model.tokenizer to be set.
                self._processor = proc_cls.from_pretrained(self.model_name, **load_kwargs)
                self._model.tokenizer = self._processor
            else:
                # Ovis2 uses aimv2 vision encoder whose config type
                # collides with transformers built-in AIMv2Config.
                # Patch AutoConfig.register to accept re-registration
                # BEFORE loading the config (which triggers hub code
                # import that tries to register 'aimv2').
                if self.vlm_config.family == "ovis":
                    AutoConfig = getattr(tf_mod, "AutoConfig")
                    _orig_register = AutoConfig.register

                    def _patched_register(key, *a, **kw):
                        kw.setdefault("exist_ok", True)
                        return _orig_register(key, *a, **kw)

                    AutoConfig.register = _patched_register

                # Phi and Ovis require eager attention; MPlugOwl3 rejects
                # eager and requires sdpa or flash_attention_2.
                if self.vlm_config.family in ("phi", "ovis", "mplug"):
                    _forced_attn = "sdpa" if self.vlm_config.family == "mplug" else "eager"
                    from transformers import AutoConfig as _AC

                    _cfg = _AC.from_pretrained(
                        self.model_name,
                        trust_remote_code=self.vlm_config.trust_remote_code,
                        local_files_only=True,
                    )
                    _cfg._attn_implementation = _forced_attn
                    # Ovis reads self.config.llm_attn_implementation to
                    # pass attn_implementation= to the internal LLM's
                    # from_config() call.
                    for _key in ("llm_attn_implementation",):
                        if hasattr(_cfg, _key):
                            setattr(_cfg, _key, _forced_attn)
                    # Patch nested configs too (belt-and-suspenders).
                    # llm_config can be a config object or a dict (MPlugOwl3).
                    for _attr in ("llm_config", "text_config", "vision_config"):
                        _nc = getattr(_cfg, _attr, None)
                        if _nc is not None:
                            if isinstance(_nc, dict):
                                # PretrainedConfig.__init__ pops
                                # 'attn_implementation' (no underscore)
                                _nc["attn_implementation"] = _forced_attn
                            elif hasattr(_nc, "_attn_implementation"):
                                _nc._attn_implementation = _forced_attn
                    load_kwargs["config"] = _cfg

                if self.vlm_config.family == "ovis":
                    try:
                        self._processor = proc_cls.from_pretrained(self.model_name, **load_kwargs)
                        # Ovis2: AutoProcessor returns bare tokenizer
                        # (no processor_config.json, AIMv2 backbone not
                        # mapped to any registered processor).  Load
                        # CLIPImageProcessor from preprocessor_config
                        # as a separate image processor.
                        if not hasattr(self._processor, "image_processor"):
                            from transformers import CLIPImageProcessor as _CIP

                            self._image_processor = _CIP.from_pretrained(
                                self.model_name,
                                local_files_only=True,
                            )
                        # Ovis2 visual_tokenizer (AIMv2) is loaded via
                        # from_config → stays bf16 checkpoint dtype, but
                        # vte (nn.Linear) gets torch_dtype=float16.
                        # Force bf16 for Ovis2 to keep everything matching.
                        _model_dtype = (
                            torch.bfloat16
                            if self.vlm_config.short_name == "Ovis2"
                            else torch.float16
                        )
                        self._model = model_cls.from_pretrained(
                            self.model_name,
                            torch_dtype=_model_dtype,
                            **load_kwargs,
                        ).to(self.device)
                        self._model.eval()
                    finally:
                        AutoConfig.register = _orig_register
                else:
                    # MPlugOwl3: HyperQwen2Attention.__init__ raises
                    # NotImplementedError when _attn_implementation="eager".
                    # Pre-import modeling_hyper_qwen2.py NOW (shares package
                    # with already-loaded configuration_hyper_qwen2.py) and
                    # redirect eager→sdpa BEFORE model loads.
                    if self.vlm_config.family == "mplug":
                        import sys as _sys
                        import importlib as _il
                        import os as _os

                        for _mn, _mv in list(_sys.modules.items()):
                            _mf = getattr(_mv, "__file__", None) or ""
                            if "configuration_hyper_qwen2" in _mf:
                                _cfg_pkg = _mv.__package__
                                _cfg_dir = _os.path.dirname(_mf)
                                _mm_path = _os.path.join(_cfg_dir, "modeling_hyper_qwen2.py")
                                if _os.path.exists(_mm_path):
                                    _mm_name = f"{_cfg_pkg}.modeling_hyper_qwen2"
                                    _spec = _il.util.spec_from_file_location(
                                        _mm_name,
                                        _mm_path,
                                        submodule_search_locations=[_cfg_dir],
                                    )
                                    _mm = _il.util.module_from_spec(_spec)
                                    _sys.modules[_mm_name] = _mm
                                    _spec.loader.exec_module(_mm)
                                    if hasattr(_mm, "QWEN2_ATTENTION_CLASSES"):
                                        _cls_map = _mm.QWEN2_ATTENTION_CLASSES
                                        if "eager" in _cls_map and "sdpa" in _cls_map:
                                            _cls_map["eager"] = _cls_map["sdpa"]
                                            _log.info("MPlugOwl3: QWEN2 eager→sdpa")
                                break
                    self._processor = proc_cls.from_pretrained(
                        self.model_name,
                        **load_kwargs,
                    )
                    self._model = model_cls.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16,
                        **load_kwargs,
                    ).to(self.device)
                    self._model.eval()

            # InternVLChatModel.generate() delegates to
            # self.language_model.generate(), but the remote-code
            # LLM class may lack GenerationMixin (e.g. stale
            # Phi3ForCausalLM / InternLM2ForCausalLM from hub).
            # Add GenerationMixin to the class bases so generate()
            # and all its helper methods are available.
            if hasattr(self._model, "language_model") and not hasattr(
                self._model.language_model, "generate"
            ):
                from transformers import GenerationConfig, GenerationMixin

                _lm = self._model.language_model
                _lm_cls = type(_lm)
                _lm_cls.__bases__ = (GenerationMixin,) + _lm_cls.__bases__
                if _lm.generation_config is None:
                    _lm.generation_config = GenerationConfig(**self._model.config.to_dict())

                # transformers 4.57+ passes DynamicCache objects for
                # past_key_values, but stale remote-code attention code
                # calls methods (get_usable_length, etc.) that may not
                # exist on DynamicCache, and its compute is often
                # incompatible.  Force legacy tuple cache format.
                _lm_cls._supports_default_dynamic_cache = lambda self: False

                _log.info(
                    "Patched %s with GenerationMixin base + legacy cache",
                    _lm_cls.__name__,
                )

            _log.info("Model loaded successfully: %s", self.model_name)

    @staticmethod
    def _get_description_prompts(task: str = "") -> List[str]:
        """Return 3 task-agnostic aerial image description prompts.

        These prompts ask the VLM to describe the scene rather than rate it
        and are used as a fallback when no VQA prompts are available.
        Each prompts the model to focus on a different aspect:
          1. Scene description (objects, layout, context)
          2. Fine detail assessment (textures, small-scale features)
          3. Visibility/degradation impact

        Note: ``task`` is accepted for API compatibility with future
        task-conditioned description prompts but is currently unused.
        """
        return [
            (
                "Describe the key objects, terrain, structures, and spatial layout "
                "visible in this aerial view. Note any features relevant to aerial "
                "scene understanding, such as moving objects, infrastructure, landing "
                "zones, or vegetation. Output 3-5 sentences."
            ),
            (
                "Identify fine details, textures, edges, and small-scale features "
                "visible in this aerial image. Note image sharpness, color fidelity, "
                "and the distinguishability of individual elements at different scales. "
                "Output 2-3 sentences."
            ),
            (
                "Assess the visibility conditions in this aerial image: describe any "
                "blur, noise, contrast issues, lighting artifacts, or visual degradations "
                "that could affect scene perception and downstream task performance. "
                "Output 2-3 sentences."
            ),
        ]

    def _build_description_prompt(self, task: str, prompt_idx: int) -> Optional[str]:
        """Build a description-generation prompt formatted with chat template.

        Returns None if no prompts are defined for the given task.
        """
        prompts = self._get_description_prompts(task)
        if not prompts:
            return None
        desc_text = prompts[prompt_idx]
        chat_template = self.vlm_config.chat_template
        return chat_template.format(prompt=desc_text, image_tags="")

    def _generate_answer(
        self, image_path: str, task: str, prompt: str, max_tokens: int = 200
    ) -> str:
        """Call the VLM to generate a scene description for an image.

        Delegates to ``_generate_answer_multi`` with a single-element list.
        """
        return self._generate_answer_multi([image_path], task, prompt, max_tokens=max_tokens)

    def ensure_loaded(self) -> None:
        """Eagerly load the model so load failures surface immediately.

        Called by executors at worker startup for fail-fast behavior: a GPU
        OOM or misconfiguration raises here instead of degrading into a
        per-batch "model previously failed to load" cascade across every task.
        """
        if self._model is not None:
            return
        if self._resolve_backend() == "none":
            return
        try:
            self._load_model()
        except Exception:
            self._load_failed = True
            raise
        if self._model is None:
            self._load_failed = True
            raise RuntimeError(f"Failed to load VLM model {self.model_name}")

    def _run_inference(
        self, image_paths: List[str], task: str, prompt: str, max_tokens: int
    ) -> str:
        backend = self._resolve_backend()
        if backend == "none":
            raise RuntimeError("No VLM backend available for description generation")

        if self._model is None:
            if getattr(self, "_load_failed", False):
                raise RuntimeError("VLM model previously failed to load")
            try:
                self._load_model()
            except Exception:
                self._load_failed = True
                raise
            if self._model is None:
                self._load_failed = True
                raise RuntimeError("Failed to load VLM model")

        if backend == "vllm":
            return run_vllm_inference(
                self._model,
                self.vlm_config.chat_template,
                image_paths,
                prompt,
                max_tokens,
                image_placeholder=self.vlm_config.image_placeholder,
            )
        if backend == "transformers":
            return run_transformers_family(
                self.vlm_config,
                self._model,
                self._processor,
                image_paths,
                prompt,
                max_tokens,
                self.device,
            )
        raise RuntimeError(f"Unknown backend: {backend}")

    def _generate_answer_multi(
        self, image_paths: List[str], task: str, prompt: str, max_tokens: int = 200
    ) -> str:
        """Call the VLM with multiple images in a single forward pass.

        All 13 models in the current registry support native multi-image input.
        This method sends all images at once, allowing the VLM to perform
        cross-view reasoning across UAV perspectives.

        Args:
            image_paths: List of paths to image files (2-6 for AirCopBench UAVs).
            task: Task type (e.g. 'vqa', or arbitrary task label).
            prompt: The raw prompt text (not chat template formatted).
            max_tokens: Maximum new tokens for generation.

        Returns:
            Raw text response from the VLM.
        """
        if len(image_paths) == 0:
            raise RuntimeError("No image paths provided for multi-image generation")

        for p in image_paths:
            if not Path(p).is_file():
                raise FileNotFoundError(f"Image not found: {p}")

        result = self._run_inference(image_paths, task, prompt, max_tokens)
        if not result.strip():
            raise RuntimeError(
                f"VLM returned empty output for {len(image_paths)} images, "
                f"first image: {Path(image_paths[0]).name}, "
                f"prompt: {prompt[:100]}..."
            )
        return result

    def _generate_vqa_answer(self, image_path: str, question: str, max_tokens: int = 100) -> str:
        """Call the VLM to answer a VQA question about an image.

        Thin wrapper around _generate_answer that passes the question
        directly as the prompt and uses a shorter default max_tokens.
        """
        return self._generate_answer(image_path, "vqa", question, max_tokens=max_tokens)

    def score_vqa_with_gt(
        self,
        ref_image_path: Union[str, Path],
        dist_image_path: Union[str, Path],
        question: str,
        gt_answer: str,
        task: str = "vqa",
        weights: tuple = (1.0, 1.0, 0.1),
        max_tokens: int = 100,
    ) -> dict:
        """Score a distorted image via VQA with ground-truth anchoring.

        Computes two score sets for a (ref, dist, question, GT) quadruplet:

        **指标1 (GT-normalized)**: BLEU(D,GT)/BLEU(R,GT), ROUGE(D,GT)/ROUGE(R,GT),
        CIDEr(D,GT)/CIDEr(R,GT) — ratios clamped to [0, 1].  Measures how
        much correct-answer fidelity is preserved under distortion, anchored
        by the VLM's baseline performance on the clean reference.

        **指标2 (Embodied-IQA style)**: BLEU(D,R), ROUGE(D,R), CIDEr(D,R) —
        direct comparison between distorted and reference VLM outputs.
        Measures how much the output drifted, independent of ground truth.

        Args:
            ref_image_path: Path to the clean reference image.
            dist_image_path: Path to the distorted image.
            question: The VQA question text.
            gt_answer: Ground-truth answer for the question.
            task: Task label for metadata (default "vqa").
            weights: (bleu, rouge, cider) weight triple for cognitive score.
            max_tokens: Max tokens for VLM answer generation.

        Returns:
            Dict with keys:
              - metadata: image paths, question snippet, VLM answers
              - metric1 (GT-normalized scores)
              - metric2 (Ref-vs-Dist scores)
        """
        eps = 1e-6

        # Stage 1: VLM inference
        answer_ref = self._generate_vqa_answer(str(ref_image_path), question, max_tokens)
        answer_dist = self._generate_vqa_answer(str(dist_image_path), question, max_tokens)

        # Stage 2: Compute raw metric values
        # ---- vs Ground Truth ----
        bleu_r_gt = compute_bleu(gt_answer, answer_ref)
        bleu_d_gt = compute_bleu(gt_answer, answer_dist)
        rouge_r_gt = compute_rouge_l(gt_answer, answer_ref)
        rouge_d_gt = compute_rouge_l(gt_answer, answer_dist)
        cider_r_gt = compute_cider([gt_answer], answer_ref)
        cider_d_gt = compute_cider([gt_answer], answer_dist)

        # ---- Ref vs Dist (Embodied-IQA style) ----
        bleu_rd = compute_bleu(answer_ref, answer_dist)
        rouge_rd = compute_rouge_l(answer_ref, answer_dist)
        cider_rd = compute_cider([answer_ref], answer_dist)

        # Stage 3: Compute GT-normalized ratios (指标1)
        bleu_gt_norm = max(0.0, min(1.0, bleu_d_gt / max(bleu_r_gt, eps)))
        rouge_gt_norm = max(0.0, min(1.0, rouge_d_gt / max(rouge_r_gt, eps)))
        cider_gt_norm = max(0.0, min(1.0, cider_d_gt / max(cider_r_gt, eps)))
        # Flag VLM baseline failure
        baseline_failure = bleu_r_gt < eps

        w_bleu, w_rouge, w_cider = weights
        weight_sum = w_bleu + w_rouge + w_cider
        cognitive_gt = (
            w_bleu * bleu_gt_norm + w_rouge * rouge_gt_norm + w_cider * cider_gt_norm
        ) / weight_sum

        # Stage 4: Compute Ref-vs-Dist cognitive score (指标2)
        cognitive_rd = (w_bleu * bleu_rd + w_rouge * rouge_rd + w_cider * cider_rd) / weight_sum

        return {
            "metadata": {
                "annotated": True,
                "ref_path": str(ref_image_path),
                "dist_path": str(dist_image_path),
                "question": question[:200],
                "ref_answer": answer_ref,
                "dist_answer": answer_dist,
                "gt_answer": gt_answer,
                "baseline_failure": baseline_failure,
                "metric_weights": {
                    "bleu": w_bleu,
                    "rouge_l": w_rouge,
                    "cider": w_cider,
                },
                "task": task,
            },
            "metric1_gt_normalized": {
                "bleu": round(bleu_gt_norm, 6),
                "rouge_l": round(rouge_gt_norm, 6),
                "cider": round(cider_gt_norm, 6),
                "cognitive_score": round(cognitive_gt, 6),
                "raw_bleu_r_gt": round(bleu_r_gt, 6),
                "raw_bleu_d_gt": round(bleu_d_gt, 6),
                "raw_rouge_r_gt": round(rouge_r_gt, 6),
                "raw_rouge_d_gt": round(rouge_d_gt, 6),
                "raw_cider_r_gt": round(cider_r_gt, 6),
                "raw_cider_d_gt": round(cider_d_gt, 6),
            },
            "metric2_ref_vs_dist": {
                "bleu": round(bleu_rd, 6),
                "rouge_l": round(rouge_rd, 6),
                "cider": round(cider_rd, 6),
                "cognitive_score": round(cognitive_rd, 6),
            },
        }

    def score_image_comparison(
        self,
        ref_image_path: Union[str, Path],
        dist_image_path: Union[str, Path],
        task: str,
        weights: tuple = (1.0, 1.0, 0.1),
    ) -> dict:
        """Score a distorted image by comparing VLM descriptions.

        Per the proposal design:
        1. Generate 3 descriptions for the clean reference image
        2. Generate 3 descriptions for the distorted image
        3. Compute BLEU, ROUGE-L, CIDEr per prompt pair
        4. Weighted average (1:1:0.1) -> cognitive quality score

        Args:
            ref_image_path: Path to the clean reference image.
            dist_image_path: Path to the distorted image.
            task: Task type (e.g. 'vqa', any subtask, or arbitrary task label).
            weights: (bleu_weight, rouge_weight, cider_weight).

        Returns:
            Dict with metadata, summary, and details keys.
        """
        ref_path = str(ref_image_path)

        # Use VQA questions for this image; fall back to hardcoded task prompts
        vqa_prompts = self.vqa_index.get_prompts(ref_path)
        if vqa_prompts:
            prompts = vqa_prompts
            _log.info("Using %d VQA questions for %s", len(prompts), ref_path)
        else:
            prompts = self._get_description_prompts(task)
            if not prompts:
                raise ValueError(f"No prompts available for {ref_path}, task={task}")
            _log.info("No VQA questions for %s, using %d task prompts", ref_path, len(prompts))

        ref_descriptions = []
        dist_descriptions = []

        for i, desc_prompt in enumerate(prompts):
            ref_desc = self._generate_answer(str(ref_image_path), task, desc_prompt)
            ref_descriptions.append(ref_desc)

            dist_desc = self._generate_answer(str(dist_image_path), task, desc_prompt)
            dist_descriptions.append(dist_desc)

        metric_results = compute_cognitive_score(
            ref_texts=ref_descriptions,
            dist_texts=dist_descriptions,
            weights=weights,
        )

        return {
            "metadata": {
                "annotated": True,
                "num_prompts": len(prompts),
                "metric_weights": {
                    "bleu": weights[0],
                    "rouge_l": weights[1],
                    "cider": weights[2],
                },
            },
            "summary": {
                "cognitive_score": metric_results["cognitive_score"],
                "bleu": metric_results["bleu"],
                "rouge_l": metric_results["rouge_l"],
                "cider": metric_results["cider"],
            },
            "details": [
                {
                    "id": i,
                    "task": task,
                    "prompt": p,
                    "reference": {"answer": ref_descriptions[i]},
                    "prediction": {"answer": dist_descriptions[i]},
                    "scores": metric_results["per_prompt"][i],
                }
                for i, p in enumerate(prompts)
            ],
        }

    def score_batch_comparison(
        self,
        ref_image_paths: List[str],
        dist_image_paths: List[str],
        task_labels: List[str],
        weights: tuple = (1.0, 1.0, 0.1),
        show_progress: bool = True,
    ) -> List[dict]:
        """Score a batch of distorted images via description comparison.

        Generates descriptions for reference images once (cached per unique
        ref path), then for each distorted image, and computes cognitive
        scores via BLEU/ROUGE-L/CIDEr.

        Args:
            ref_image_paths: Clean reference image paths (aligned with dist).
            dist_image_paths: Distorted image paths.
            task_labels: Task types.
            weights: Metric weights for cognitive score computation.
            show_progress: Whether to show a tqdm progress bar.

        Returns:
            List of dicts, each with metadata, summary, and details keys.
        """
        if not (len(ref_image_paths) == len(dist_image_paths) == len(task_labels)):
            raise ValueError(
                f"Length mismatch: {len(ref_image_paths)} refs, "
                f"{len(dist_image_paths)} dists, {len(task_labels)} tasks"
            )

        # Generate and cache reference descriptions
        ref_desc_cache: Dict[str, List[str]] = {}
        unique_refs = sorted(set(ref_image_paths))

        for ref_path in tqdm(
            unique_refs,
            desc="Generating ref descriptions",
            unit="ref",
            disable=not show_progress,
        ):
            prompts = self.vqa_index.get_prompts(ref_path)
            if prompts:
                descriptions = [self._generate_answer(ref_path, "vqa", p) for p in prompts]
                ref_desc_cache[ref_path] = descriptions

        # Score each distorted image
        results: List[dict] = []
        for ref, dist, task in tqdm(
            zip(ref_image_paths, dist_image_paths, task_labels),
            total=len(dist_image_paths),
            desc="Scoring distorted images",
            unit="img",
            disable=not show_progress,
        ):
            cache_key = ref
            if cache_key not in ref_desc_cache:
                _log.warning("Ref description not found for %s, generating now", ref)
                prompts = self.vqa_index.get_prompts(ref)
                if not prompts:
                    prompts = self._get_description_prompts(task)
                if not prompts:
                    raise ValueError(f"No prompts available for ref={ref}, task={task}")
                ref_desc_cache[cache_key] = [self._generate_answer(ref, "vqa", p) for p in prompts]

            ref_descriptions = ref_desc_cache[cache_key]

            prompts = self.vqa_index.get_prompts(ref)
            if not prompts:
                prompts = self._get_description_prompts(task)
            if not prompts:
                raise ValueError(f"No prompts available for ref={ref}, task={task}")
            dist_descriptions = [self._generate_answer(dist, task, p) for p in prompts]

            metric_results = compute_cognitive_score(
                ref_texts=ref_descriptions,
                dist_texts=dist_descriptions,
                weights=weights,
            )

            results.append(
                {
                    "metadata": {
                        "annotated": True,
                        "num_prompts": len(prompts),
                        "metric_weights": {
                            "bleu": weights[0],
                            "rouge_l": weights[1],
                            "cider": weights[2],
                        },
                    },
                    "summary": {
                        "cognitive_score": round(metric_results["cognitive_score"], 6),
                        "bleu": round(metric_results["bleu"], 6),
                        "rouge_l": round(metric_results["rouge_l"], 6),
                        "cider": round(metric_results["cider"], 6),
                    },
                    "details": [
                        {
                            "id": i,
                            "task": task,
                            "prompt": p,
                            "reference": {"answer": ref_descriptions[i]},
                            "prediction": {"answer": dist_descriptions[i]},
                            "scores": metric_results["per_prompt"][i],
                        }
                        for i, p in enumerate(prompts)
                    ],
                }
            )

        return results

    def score_multi_image(
        self,
        ref_image_paths: List[str],
        dist_image_paths: List[str],
        question: str,
        subtask_type: str = "",
    ) -> Dict[str, Union[float, str]]:
        """Score a multi-UAV scene+frame group via native multi-image VLM inference.

        Sends all UAV images to the VLM in a single forward pass (native
        multi-image), which allows cross-view reasoning.  Computes text
        similarity between the reference and distorted descriptions.

        Args:
            ref_image_paths: Clean reference image paths (one per UAV).
            dist_image_paths: Distorted image paths (one per UAV, same order).
            question: VQA question text used as the VLM prompt.
            subtask_type: Subtask type identifier (e.g. ``"1.1"``), unused for
                scoring but reserved for future task-conditioned weighting.

        Returns:
            Dict with ``cognitive_score``, ``bleu``, ``rouge_l``, ``cider``.  All
            floats in approximately [0, 1].
        """
        if not ref_image_paths or not dist_image_paths:
            return {
                "cognitive_score": 0.0,
                "bleu": 0.0,
                "rouge_l": 0.0,
                "cider": 0.0,
                "ref_description": "",
                "dist_description": "",
                "prompt": "",
            }
        if len(ref_image_paths) != len(dist_image_paths):
            _log.warning(
                "Mismatched UAV counts: %d ref vs %d dist, using min",
                len(ref_image_paths),
                len(dist_image_paths),
            )
            n = min(len(ref_image_paths), len(dist_image_paths))
            ref_image_paths = ref_image_paths[:n]
            dist_image_paths = dist_image_paths[:n]

        prompt = question.strip()

        if not prompt:
            return {
                "cognitive_score": 0.0,
                "bleu": 0.0,
                "rouge_l": 0.0,
                "cider": 0.0,
                "ref_description": "",
                "dist_description": "",
                "prompt": "",
            }

        ref_description = self._generate_answer_multi(ref_image_paths, "vqa", prompt)
        dist_description = self._generate_answer_multi(dist_image_paths, "vqa", prompt)

        result = compute_cognitive_score(
            ref_texts=[ref_description],
            dist_texts=[dist_description],
        )
        return {
            "cognitive_score": round(result["cognitive_score"], 6),
            "bleu": round(result["bleu"], 6),
            "rouge_l": round(result["rouge_l"], 6),
            "cider": round(result["cider"], 6),
            "ref_description": ref_description,
            "dist_description": dist_description,
            "prompt": prompt,
        }

    # ------------------------------------------------------------------
    # Batch inference — sends multiple prompts in one model.generate()
    # call so vLLM continuous batching (max_num_seqs) can interleave them.
    # ------------------------------------------------------------------

    def _generate_answers_batch(
        self,
        prompts: list[tuple[list[str], str]],
        max_tokens: int = 200,
    ) -> list[str]:
        """Generate answers for multiple (image_paths, prompt_text) pairs.

        For the vLLM backend this is a single ``model.generate()`` call;
        for transformers it falls back to sequential :meth:`_generate_answer_multi`.

        Args:
            prompts: List of ``(image_paths, prompt_text)`` tuples.
            max_tokens: Maximum new tokens per generated answer.

        Returns:
            List of generated text strings, same length and order as *prompts*.
        """
        backend = self._resolve_backend()

        if backend == "vllm":
            if self._model is None:
                if getattr(self, "_load_failed", False):
                    raise RuntimeError("VLM model previously failed to load")
                self._load_model()
                if self._model is None:
                    self._load_failed = True
                    raise RuntimeError("Failed to load VLM model")

            return run_vllm_inference_batch(
                self._model,
                self.vlm_config.chat_template,
                prompts,
                max_tokens,
                image_placeholder=self.vlm_config.image_placeholder,
            )

        # transformers / other: fall back to sequential calls
        results: list[str] = []
        for image_paths, prompt_text in prompts:
            results.append(
                self._generate_answer_multi(image_paths, "vqa", prompt_text, max_tokens),
            )
        return results

    def score_batch_multi_image(
        self,
        batch_ref_paths: list[list[str]],
        batch_dist_paths: list[list[str]],
        batch_questions: list[str],
        max_tokens: int = 200,
    ) -> list[dict[str, float | str]]:
        """Batch version of :meth:`score_multi_image` for throughput.

        Groups all reference and distorted generations across *N* entries
        into one (vLLM) or two (transformers) ``model.generate()`` calls,
        then computes text-similarity scores per pair.

        Args:
            batch_ref_paths: Clean reference image paths per entry.
            batch_dist_paths: Distorted image paths per entry (same order).
            batch_questions: VQA question per entry.
            max_tokens: Maximum new tokens per answer.

        Returns:
            List of result dicts with ``cognitive_score``, ``bleu``,
            ``rouge_l``, ``cider``, ``ref_description``, ``dist_description``,
            ``prompt`` — one dict per entry, same order as input.
        """
        if not batch_ref_paths:
            return []

        N = len(batch_ref_paths)
        empty_result: dict[str, float | str] = {
            "cognitive_score": 0.0,
            "bleu": 0.0,
            "rouge_l": 0.0,
            "cider": 0.0,
            "ref_description": "",
            "dist_description": "",
            "prompt": "",
        }

        # Build (image_paths, prompt) pairs for every valid entry.
        # Combine ref + dist into one flat list for a single generate() call
        # when using vLLM; two calls otherwise (ref batch, then dist batch).
        all_prompts: list[tuple[list[str], str]] = []
        valid_indices: list[int] = []
        default_indices: dict[int, dict[str, float | str]] = {}

        for i in range(N):
            ref_paths = batch_ref_paths[i]
            dist_paths = batch_dist_paths[i]
            question = batch_questions[i].strip() if batch_questions[i] else ""

            if not ref_paths or not dist_paths or not question:
                default_indices[i] = dict(empty_result)
                default_indices[i]["prompt"] = question
                continue

            all_prompts.append((list(ref_paths), question))
            all_prompts.append((list(dist_paths), question))
            valid_indices.append(i)

        if not all_prompts:
            return [
                default_indices.get(i, dict(empty_result))
                for i in range(N)
            ]

        # Single batched generate() for all ref + dist prompts
        try:
            all_outputs = self._generate_answers_batch(all_prompts, max_tokens)
        except Exception:
            _log.warning(
                "Batch inference failed (%d prompts), falling back to sequential",
                len(all_prompts),
                exc_info=True,
            )
            # Fall back to per-entry sequential scoring
            results: list[dict[str, float | str]] = []
            for i in range(N):
                if i in default_indices:
                    results.append(default_indices[i])
                else:
                    try:
                        results.append(
                            self.score_multi_image(
                                batch_ref_paths[i],
                                batch_dist_paths[i],
                                batch_questions[i],
                            ),
                        )
                    except Exception:
                        _log.error(
                            "Sequential fallback failed for entry %d (sample_id=%s)",
                            i, batch_ref_paths[i] if i < len(batch_ref_paths) else "?",
                        )
                        raise
            return results

        # all_prompts layout: [ref_0, dist_0, ref_1, dist_1, ...]
        # Pair them up and compute scores.
        results = []
        valid_idx = 0
        for i in range(N):
            if i in default_indices:
                results.append(default_indices[i])
                continue

            ref_idx = valid_idx * 2
            dist_idx = ref_idx + 1
            ref_text = all_outputs[ref_idx] if ref_idx < len(all_outputs) else ""
            dist_text = all_outputs[dist_idx] if dist_idx < len(all_outputs) else ""

            if not ref_text or not dist_text:
                raise RuntimeError(
                    f"Empty VLM output for entry {i}: "
                    f"ref_text={ref_text!r}, dist_text={dist_text!r}, "
                    f"question={batch_questions[i][:100]!r}"
                )

            metric_result = compute_cognitive_score(
                ref_texts=[ref_text],
                dist_texts=[dist_text],
            )
            results.append({
                "cognitive_score": round(metric_result["cognitive_score"], 6),
                "bleu": round(metric_result["bleu"], 6),
                "rouge_l": round(metric_result["rouge_l"], 6),
                "cider": round(metric_result["cider"], 6),
                "ref_description": ref_text,
                "dist_description": dist_text,
                "prompt": batch_questions[i].strip(),
            })

            valid_idx += 1

        return results
