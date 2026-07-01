"""Smoke test: load each VLM model and generate a description to verify basic functionality.

Requires GPU and all VLM extra dependencies (transformers, etc.).
Skip with: pytest -m "not smoke"
"""

import gc
import time
from pathlib import Path

import pytest
import torch

from uav_iqa.vlm import MODEL_REGISTRY, VLMScorer

_TEST_IMAGE_CANDIDATES = [
    "data/raw/AirCopBench/Real_2_UAVs/Samples/UAV1/23-00000001-UAV1.jpg",
    "data/raw/AirCopBench/train/Sim_3_UAVs/Samples/UAV1/23-00000001-UAV1.jpg",
    "data/raw/AirCopBench/train/Sim_5_UAVs/Samples/UAV1/23-00000001-UAV1.jpg",
]

TEST_IMAGE = None
for _cand in _TEST_IMAGE_CANDIDATES:
    if Path(_cand).exists():
        TEST_IMAGE = _cand
        break
TEST_TASK = "scene_description"

MODEL_NAMES = list(MODEL_REGISTRY.keys())


def _cleanup_gpu(scorer):
    if scorer._model is not None:
        del scorer._model
        scorer._model = None
    gc.collect()
    torch.cuda.empty_cache()


@pytest.mark.smoke
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for VLM smoke test"
)
@pytest.mark.skipif(
    TEST_IMAGE is None, reason="No test image found in expected locations"
)
@pytest.mark.parametrize("model_name", MODEL_NAMES, ids=MODEL_NAMES)
def test_vlm_model_load_and_generate_answer(model_name):
    scorer = VLMScorer(
        model_name=model_name,
        backend="transformers",
        device="cuda",
    )
    t0 = time.time()
    try:
        prompts = scorer._get_description_prompts(TEST_TASK)
        desc = scorer._generate_answer(TEST_IMAGE, TEST_TASK, prompts[0])
        score_time = time.time() - t0

        assert isinstance(desc, str)
        assert len(desc) > 0

        print(f"\n[{model_name}] OK in {score_time:.1f}s (desc length={len(desc)})")
    finally:
        _cleanup_gpu(scorer)
