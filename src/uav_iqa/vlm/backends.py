"""Per-family VLM inference backends.

Each function handles the image loading, prompt formatting, and generate
call for one VLM family.  They always accept ``image_paths: List[str]``
so single-image and multi-image inference share the same code path.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
from PIL import Image

_log = logging.getLogger(__name__)


# -- top-level helpers -------------------------------------------------------

def _load_images(image_paths: List[str]) -> List[Image.Image]:
    return [Image.open(p).convert("RGB") for p in image_paths]


# -- backend: vLLM -----------------------------------------------------------

def run_vllm_inference(
    model: Any,
    chat_template: str,
    image_paths: List[str],
    prompt: str,
    max_tokens: int,
    *,
    image_placeholder: str = "<|vision_start|><|image_pad|><|vision_end|>",
) -> str:
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    # vLLM 0.19+ requires explicit image placeholder tokens in the prompt
    # for each multi-modal input. Without them, prompt replacement fails
    # with "Failed to apply prompt replacement for mm_items['image'][0]".
    # Placeholders must be embedded INSIDE the user message for models
    # that use conversation templates (Qwen2-VL, Qwen2.5-VL); prepending
    # outside the chat format causes the model to ignore the images and
    # generate only EOS.
    image_tags = "".join([image_placeholder] * len(image_paths))
    try:
        formatted_prompt = chat_template.format(prompt=prompt, image_tags=image_tags)
    except KeyError:
        # Backward compat: templates without {image_tags} placeholder
        formatted_prompt = image_tags + "\n" + chat_template.format(prompt=prompt)
    images = _load_images(image_paths)
    image_data = images[0] if len(images) == 1 else images
    outputs = model.generate(
        [{"prompt": formatted_prompt, "multi_modal_data": {"image": image_data}}],
        sp,
    )
    text = outputs[0].outputs[0].text.strip()
    if not text:
        _log.warning(
            "vLLM returned empty text. "
            "token_ids=%s, finish_reason=%s, text_raw=%r",
            len(getattr(outputs[0].outputs[0], "token_ids", []) or []),
            getattr(outputs[0].outputs[0], "finish_reason", "?"),
            outputs[0].outputs[0].text,
        )
    return text


def run_vllm_inference_batch(
    model: Any,
    chat_template: str,
    prompts: list[tuple[list[str], str]],
    max_tokens: int,
    *,
    image_placeholder: str = "<|vision_start|><|image_pad|><|vision_end|>",
) -> list[str]:
    """Run batched vLLM inference — all prompts in a single ``model.generate()``.

    Unlike :func:`run_vllm_inference` which sends one prompt per call, this
    function packs *prompts* into one ``model.generate()`` call so that
    vLLM's continuous batching (``max_num_seqs``) can interleave them on the
    GPU.

    Args:
        model: vLLM ``LLM`` instance.
        chat_template: Chat template string with ``{image_tags}`` and
            ``{prompt}`` placeholders.
        prompts: List of ``(image_paths, prompt_text)`` tuples — each
            element becomes one sequence in the batch.
        max_tokens: Maximum new tokens for each sequence.
        image_placeholder: Token string used for each image placeholder.

    Returns:
        List of generated text strings, one per element in *prompts*.
    """
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    batch_data: list[dict[str, Any]] = []

    for image_paths, prompt_text in prompts:
        image_tags = "".join([image_placeholder] * len(image_paths))
        try:
            formatted_prompt = chat_template.format(
                prompt=prompt_text, image_tags=image_tags,
            )
        except KeyError:
            formatted_prompt = (
                image_tags + "\n" + chat_template.format(prompt=prompt_text)
            )
        images = _load_images(image_paths)
        image_data = images[0] if len(images) == 1 else images
        batch_data.append({
            "prompt": formatted_prompt,
            "multi_modal_data": {"image": image_data},
        })

    outputs = model.generate(batch_data, sp)

    results: list[str] = []
    for i, output in enumerate(outputs):
        text = output.outputs[0].text.strip()
        if not text:
            raise RuntimeError(
                f"vLLM batch[{i}] returned empty text. "
                f"token_ids={len(getattr(output.outputs[0], 'token_ids', []) or [])}, "
                f"finish_reason={getattr(output.outputs[0], 'finish_reason', '?')}, "
                f"text_raw={output.outputs[0].text!r}"
            )
        results.append(text)

    return results


# -- backend: transformers, per-family ---------------------------------------

def _internlm_xc_gen(
    model: Any,
    processor: Any,
    image_paths: List[str],
    prompt: str,
    max_tokens: int,
) -> str:
    image_tags = "\n".join(["<ImageHere>"] * len(image_paths))
    query = f"{image_tags}\n{prompt}"
    with torch.no_grad():
        response_text, _history = model.chat(
            processor,
            query=query,
            image=image_paths,
            max_new_tokens=max_tokens,
            do_sample=False,
        )
    return response_text


def _internvl_gen(
    model: Any,
    processor: Any,
    image_paths: List[str],
    prompt: str,
    max_tokens: int,
    device: torch.device,
    force_size: Optional[int] = None,
) -> str:
    from torchvision import transforms

    if force_size is None:
        force_size = getattr(model.config, "force_image_size", 448)

    transform = transforms.Compose(
        [
            transforms.Resize((force_size, force_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    pixel_values_list = []
    for p in image_paths:
        image = Image.open(p).convert("RGB")
        pixel_values_list.append(transform(image).unsqueeze(0))
    pixel_values = torch.cat(pixel_values_list, dim=0).to(
        device=device, dtype=torch.float16
    )
    gen_config = {"max_new_tokens": max_tokens, "do_sample": False}
    with torch.no_grad():
        return model.chat(
            processor,
            pixel_values=pixel_values,
            question=prompt,
            generation_config=gen_config,
            num_patches_list=[1] * len(image_paths),
        )


def _phi_gen(
    model: Any,
    processor: Any,
    image_paths: List[str],
    prompt: str,
    max_tokens: int,
    device: torch.device,
    short_name: str = "",
) -> str:
    image_tags = "\n".join(f"<|image_{i + 1}|>" for i in range(len(image_paths)))
    prompt_with_images = (
        f"<|user|>\n{image_tags}\n{prompt}<|end|>\n<|assistant|>\n"
    )
    images = _load_images(image_paths)
    inputs = processor(
        text=prompt_with_images,
        images=images if len(images) > 1 else images[0],
        return_tensors="pt",
    ).to(device)
    if short_name == "Phi4-Multimodal":
        _orig_pifg = model.prepare_inputs_for_generation

        def _patched_pifg(
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            num_logits_to_keep=None,
            **kwargs,
        ):
            return _orig_pifg(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                num_logits_to_keep=1,
                **kwargs,
            )

        model.prepare_inputs_for_generation = _patched_pifg
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    try:
        return processor.decode(outputs[0], skip_special_tokens=True).strip()
    except OverflowError:
        safe_ids = torch.clamp(outputs[0], 0, model.config.vocab_size - 1)
        return processor.decode(safe_ids, skip_special_tokens=True).strip()


def _qwen_gen(
    model: Any,
    processor: Any,
    image_paths: List[str],
    prompt: str,
    max_tokens: int,
    device: torch.device,
) -> str:
    content: List[Dict[str, Any]] = []
    for p in image_paths:
        content.append({"type": "image", "image": p})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    images = _load_images(image_paths)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return processor.decode(outputs[0], skip_special_tokens=True).strip()


def _mplug_gen(
    model: Any,
    processor: Any,
    image_paths: List[str],
    prompt: str,
    max_tokens: int,
    device: torch.device,
) -> str:
    if not hasattr(model, "mplug_processor_cached"):
        model.mplug_processor_cached = model.init_processor(processor)

    image_tags = "\n".join(["<|image|>"] * len(image_paths))
    messages = [{"role": "user", "content": f"{image_tags}\n{prompt}"}]
    images = _load_images(image_paths)
    inputs = model.mplug_processor_cached(
        messages, images=images, return_tensors="pt"
    )
    input_ids = inputs.pop("input_ids").to(device)
    model_kwargs: Dict[str, Any] = {
        "input_ids": input_ids,
        "max_new_tokens": max_tokens,
    }
    for k, v in inputs.items():
        if hasattr(v, "to"):
            model_kwargs[k] = v.to(device)
    model_kwargs["tokenizer"] = processor
    with torch.no_grad():
        outputs = model.generate(**model_kwargs)
    return processor.decode(outputs[0], skip_special_tokens=True).strip()


def _ovis_gen(
    model: Any,
    processor: Any,
    image_paths: List[str],
    prompt: str,
    max_tokens: int,
    device: torch.device,
    chat_template: str,
    image_placeholder: str = "<image>",
) -> str:
    image_tags = "".join([image_placeholder] * len(image_paths))
    try:
        formatted_prompt = chat_template.format(prompt=prompt, image_tags=image_tags)
    except KeyError:
        formatted_prompt = image_tags + "\n" + chat_template.format(prompt=prompt)
    pixel_values_list = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        pv_result = model.get_visual_tokenizer().preprocess_image(image)
        if isinstance(pv_result, tuple):
            pv = pv_result[0]
        else:
            pv = pv_result
        _vt = model.get_visual_tokenizer()
        _vt_param_dtype = next(_vt.parameters()).dtype
        pixel_values_list.append(pv.to(device=device, dtype=_vt_param_dtype))

    text_inputs = processor(text=formatted_prompt, return_tensors="pt")
    input_ids = text_inputs["input_ids"].to(device)
    attention_mask = text_inputs.get(
        "attention_mask",
        torch.ones_like(input_ids),
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values_list,
            max_new_tokens=max_tokens,
        )
    return processor.decode(outputs[0], skip_special_tokens=True).strip()


def _generic_gen(
    model: Any,
    processor: Any,
    image_paths: List[str],
    prompt: str,
    max_tokens: int,
    device: torch.device,
    chat_template: str,
    image_placeholder: str = "<|vision_start|><|image_pad|><|vision_end|>",
) -> str:
    image_tags = "".join([image_placeholder] * len(image_paths))
    try:
        formatted_prompt = chat_template.format(prompt=prompt, image_tags=image_tags)
    except KeyError:
        formatted_prompt = image_tags + "\n" + chat_template.format(prompt=prompt)
    images = _load_images(image_paths)
    inputs = processor(
        text=formatted_prompt,
        images=images if len(images) > 1 else images[0],
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return processor.decode(outputs[0], skip_special_tokens=True).strip()


# -- top-level dispatcher ----------------------------------------------------

def run_transformers_family(
    vlm_config: Any,
    model: Any,
    processor: Any,
    image_paths: List[str],
    prompt: str,
    max_tokens: int,
    device: torch.device,
) -> str:
    """Dispatch to the correct per-family transformers inference function.

    Args:
        vlm_config: VLMConfig instance with ``family``, ``short_name``,
            ``chat_template``.
        model: Loaded HuggingFace model.
        processor: Loaded HuggingFace processor.
        image_paths: List of image file paths (1 for single-image, 2+ for
            multi-image).
        prompt: Raw prompt text (not chat-template formatted).
        max_tokens: Maximum new tokens for generation.
        device: Torch device (cuda / cpu).

    Returns:
        Raw text response from the VLM.
    """
    family = vlm_config.family

    if family == "internlm_xc":
        return _internlm_xc_gen(model, processor, image_paths, prompt, max_tokens)
    if family == "internvl":
        return _internvl_gen(
            model, processor, image_paths, prompt, max_tokens, device
        )
    elif family == "phi":
        return _phi_gen(
            model,
            processor,
            image_paths,
            prompt,
            max_tokens,
            device,
            short_name=vlm_config.short_name,
        )
    elif family == "qwen":
        return _qwen_gen(model, processor, image_paths, prompt, max_tokens, device)
    elif family == "mplug":
        return _mplug_gen(model, processor, image_paths, prompt, max_tokens, device)
    elif family == "ovis":
        return _ovis_gen(
            model,
            processor,
            image_paths,
            prompt,
            max_tokens,
            device,
            chat_template=vlm_config.chat_template,
        )
    else:
        return _generic_gen(
            model,
            processor,
            image_paths,
            prompt,
            max_tokens,
            device,
            chat_template=vlm_config.chat_template,
            image_placeholder=vlm_config.image_placeholder,
        )
