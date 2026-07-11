"""Tests for vLLM load config threading + worker fail-fast behavior.

Covers:
- ``VLMScorer`` threads ``max_model_len`` / ``gpu_memory_utilization`` into
  the vLLM ``LLM(...)`` constructor (and omits ``max_model_len`` when None).
- ``VLMExecutor.prepare()`` loads the model eagerly and fails fast when the
  model cannot load, instead of deferring to a per-batch cascade.
"""

import importlib.machinery
import sys
import types

import pytest

from uav_iqa.vlm.scorer import VLMScorer


class _CapturingLLM:
    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _CapturingLLM.last_kwargs = kwargs


class _RaisingLLM:
    def __init__(self, **kwargs):
        raise ValueError(
            "Free memory on device cuda:0 (39.02/79.14 GiB) on startup is less "
            "than desired GPU memory utilization (0.5, 39.57 GiB)."
        )


def _install_fake_vllm(monkeypatch, llm_cls):
    """Stub the ``vllm`` module so _load_model uses ``llm_cls`` as ``LLM``."""
    fake = types.ModuleType("vllm")
    fake.__spec__ = importlib.machinery.ModuleSpec("vllm", loader=None)
    fake.LLM = llm_cls
    fake.SamplingParams = lambda **k: object()
    monkeypatch.setitem(sys.modules, "vllm", fake)
    # rope patch touches vllm.transformers_utils.config which the stub lacks
    monkeypatch.setattr(
        VLMScorer, "_patch_vllm_rope_config", staticmethod(lambda: None)
    )


def test_scorer_threads_max_model_len_and_gpu_mem(monkeypatch):
    _install_fake_vllm(monkeypatch, _CapturingLLM)
    scorer = VLMScorer(
        model_name="Mini-InternVL",
        backend="vllm",
        gpu_memory_utilization=0.4,
        max_model_len=8192,
    )
    scorer._load_model()

    kw = _CapturingLLM.last_kwargs
    assert kw["gpu_memory_utilization"] == 0.4
    assert kw["max_model_len"] == 8192


def test_scorer_omits_max_model_len_when_none(monkeypatch):
    _install_fake_vllm(monkeypatch, _CapturingLLM)
    scorer = VLMScorer(model_name="Mini-InternVL", backend="vllm")
    scorer._load_model()

    assert "max_model_len" not in _CapturingLLM.last_kwargs


def test_executor_prepare_fails_fast_on_load_error(monkeypatch):
    _install_fake_vllm(monkeypatch, _RaisingLLM)
    from uav_iqa.inference.executor import VLMExecutor

    ex = VLMExecutor(model_name="Mini-InternVL", backend="vllm")
    with pytest.raises(ValueError, match="Free memory on device"):
        ex.prepare()
