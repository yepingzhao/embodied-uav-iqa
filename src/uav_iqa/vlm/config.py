"""VLM configuration and model registry for UAV-IQA.

Provides:
- VLMConfig: configuration dataclass for a Vision-Language Model
- MODEL_REGISTRY: curated registry of 15 supported VLMs across 6 families
"""

import logging
from dataclasses import dataclass
from typing import Dict

_log = logging.getLogger(__name__)


@dataclass
class VLMConfig:
    """Configuration for a Vision-Language Model.

    Encapsulates loading strategy, chat template, and metadata for
    a specific VLM model.
    """

    short_name: str
    hf_model_id: str
    family: str  # "qwen", "internvl", "internlm_xc", "ovis", "phi", "mplug"
    chat_template: str  # Python format string with {prompt} and {image_tags}
    model_class_name: str  # "AutoModelForVision2Seq", "AutoModel", etc.
    processor_class_name: str  # "AutoProcessor", "AutoTokenizer", etc.
    trust_remote_code: bool = False
    image_placeholder: str = "<|vision_start|><|image_pad|><|vision_end|>"


MODEL_REGISTRY: Dict[str, VLMConfig] = {
    # --- Qwen family ---
    "Qwen2-VL": VLMConfig(
        short_name="Qwen2-VL",
        hf_model_id="Qwen/Qwen2-VL-7B-Instruct",
        family="qwen",
        chat_template=(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{image_tags}{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        model_class_name="AutoModelForVision2Seq",
        processor_class_name="AutoProcessor",
    ),
    "Qwen2.5-VL": VLMConfig(
        short_name="Qwen2.5-VL",
        hf_model_id="Qwen/Qwen2.5-VL-7B-Instruct",
        family="qwen",
        chat_template=(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{image_tags}{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        model_class_name="AutoModelForVision2Seq",
        processor_class_name="AutoProcessor",
    ),
    # --- InternVL family ---
    "Mini-InternVL": VLMConfig(
        short_name="Mini-InternVL",
        hf_model_id="OpenGVLab/Mini-InternVL-Chat-4B-V1-5",
        family="internvl",
        chat_template=(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{image_tags}\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        model_class_name="AutoModel",
        processor_class_name="AutoTokenizer",
        trust_remote_code=True,
        image_placeholder="<image>",
    ),
    "InternVL2": VLMConfig(
        short_name="InternVL2",
        hf_model_id="OpenGVLab/InternVL2-8B",
        family="internvl",
        chat_template=(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{image_tags}\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        model_class_name="AutoModel",
        processor_class_name="AutoTokenizer",
        trust_remote_code=True,
        image_placeholder="<image>",
    ),
    "InternVL2.5": VLMConfig(
        short_name="InternVL2.5",
        hf_model_id="OpenGVLab/InternVL2_5-8B",
        family="internvl",
        chat_template=(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{image_tags}\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        model_class_name="AutoModel",
        processor_class_name="AutoTokenizer",
        trust_remote_code=True,
        image_placeholder="<image>",
    ),
    "InternVL3": VLMConfig(
        short_name="InternVL3",
        hf_model_id="OpenGVLab/InternVL3-8B",
        family="internvl",
        chat_template=(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n{image_tags}\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        model_class_name="AutoModel",
        processor_class_name="AutoTokenizer",
        trust_remote_code=True,
        image_placeholder="<image>",
    ),
    # --- InternLM-Xcomposer family ---
    "InternLM-Xcomposer2.5": VLMConfig(
        short_name="InternLM-Xcomposer2.5",
        hf_model_id="internlm/internlm-xcomposer2d5-7b",
        family="internlm_xc",
        chat_template="<|User|>:{prompt}<|Bot|>:",
        model_class_name="AutoModelForCausalLM",
        processor_class_name="AutoTokenizer",
    ),
    # --- Ovis family ---
    "Ovis1.5-Gemma": VLMConfig(
        short_name="Ovis1.5-Gemma",
        hf_model_id="AIDC-AI/Ovis1.5-Gemma2-9B",
        family="ovis",
        chat_template="USER: {image_tags}\n{prompt}\nASSISTANT:",
        model_class_name="AutoModelForCausalLM",
        processor_class_name="AutoProcessor",
        image_placeholder="<image>",
    ),
    "Ovis1.6-Llama": VLMConfig(
        short_name="Ovis1.6-Llama",
        hf_model_id="AIDC-AI/Ovis1.6-Llama3.2-3B",
        family="ovis",
        chat_template="USER: {image_tags}\n{prompt}\nASSISTANT:",
        model_class_name="AutoModelForCausalLM",
        processor_class_name="AutoProcessor",
        image_placeholder="<image>",
    ),
    "Ovis2": VLMConfig(
        short_name="Ovis2",
        hf_model_id="AIDC-AI/Ovis2-8B",
        family="ovis",
        chat_template="USER: {image_tags}\n{prompt}\nASSISTANT:",
        model_class_name="AutoModelForCausalLM",
        processor_class_name="AutoProcessor",
        image_placeholder="<image>",
    ),
    # --- Phi family ---
    "Phi3.5-Vision": VLMConfig(
        short_name="Phi3.5-Vision",
        hf_model_id="microsoft/Phi-3.5-vision-instruct",
        family="phi",
        chat_template="<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
        model_class_name="AutoModelForCausalLM",
        processor_class_name="AutoProcessor",
    ),
    "Phi4-Multimodal": VLMConfig(
        short_name="Phi4-Multimodal",
        hf_model_id="microsoft/Phi-4-multimodal-instruct",
        family="phi",
        chat_template="<|user|>\n{prompt}<|end|>\n<|assistant|>\n",
        model_class_name="AutoModelForCausalLM",
        processor_class_name="AutoProcessor",
    ),
    # --- MPlug family ---
    "MPlugOwl3": VLMConfig(
        short_name="MPlugOwl3",
        hf_model_id="mPLUG/mPLUG-Owl3-7B-240728",
        family="mplug",
        chat_template="{prompt}",
        model_class_name="AutoModelForCausalLM",
        processor_class_name="AutoTokenizer",
    ),
}

DEPRECATED_MODELS: dict[str, str] = {
    "InternLM-Xcomposer2": "InternLM-Xcomposer2.5",
    "Phi3-Vision": "Phi3.5-Vision",
}
