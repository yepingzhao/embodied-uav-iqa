"""Tests for uav_iqa.vlm.backends — vLLM multimodal payload construction.

Regression: the vLLM backend must hand ``multi_modal_data['image']`` loaded
PIL.Image objects, not raw file-path strings. Passing strings makes vLLM's
InternVL processor fail with ``'str' object has no attribute 'size'``.
"""

import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image


def _make_dummy_image(path: Path, size: int = 32):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
    Image.fromarray(img).save(str(path))


class _FakeGenOutput:
    def __init__(self, text: str):
        self.text = text
        self.token_ids = [1, 2, 3]
        self.finish_reason = "stop"


class _FakeRequestOutput:
    def __init__(self, text: str):
        self.outputs = [_FakeGenOutput(text)]


class _CapturingModel:
    """Stand-in vLLM engine that records the generate() payload."""

    def __init__(self):
        self.captured_payload = None

    def generate(self, prompts, sampling_params):
        self.captured_payload = prompts
        return [_FakeRequestOutput("hello")]


def _install_fake_vllm(monkeypatch):
    """Provide a lightweight ``vllm`` module so backends can import it."""
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kwargs: object()
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)


def test_vllm_multi_image_payload_contains_pil_images(tmp_path, monkeypatch):
    _install_fake_vllm(monkeypatch)
    from uav_iqa.vlm.backends import run_vllm_inference

    paths = []
    for i in range(1, 4):
        p = tmp_path / f"UAV{i}.jpg"
        _make_dummy_image(p)
        paths.append(str(p))

    model = _CapturingModel()
    run_vllm_inference(
        model,
        chat_template="{image_tags}\n{prompt}",
        image_paths=paths,
        prompt="rate the image",
        max_tokens=8,
        image_placeholder="<image>",
    )

    images = model.captured_payload[0]["multi_modal_data"]["image"]
    assert isinstance(images, list) and len(images) == 3
    for img in images:
        assert isinstance(img, Image.Image), f"expected PIL.Image, got {type(img)}"


def test_vllm_single_image_payload_is_pil_image(tmp_path, monkeypatch):
    _install_fake_vllm(monkeypatch)
    from uav_iqa.vlm.backends import run_vllm_inference

    p = tmp_path / "UAV1.jpg"
    _make_dummy_image(p)

    model = _CapturingModel()
    run_vllm_inference(
        model,
        chat_template="{image_tags}\n{prompt}",
        image_paths=[str(p)],
        prompt="rate the image",
        max_tokens=8,
        image_placeholder="<image>",
    )

    image = model.captured_payload[0]["multi_modal_data"]["image"]
    assert isinstance(image, Image.Image), f"expected PIL.Image, got {type(image)}"
