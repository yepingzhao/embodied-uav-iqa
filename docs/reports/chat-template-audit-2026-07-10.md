# Chat Template Audit: 各模型 vs 社区实践

*Generated: 2026-07-10 | Confidence: High*

## Executive Summary

vLLM 后端对每个模型家族使用**不同的图像占位符**。当前代码硬编码了 Qwen 格式的占位符给所有模型，导致 InternVL 和 Ovis 使用了错误的图像标记。另外 Ovis 模型的对话格式也不匹配。需要给 `VLMConfig` 加 `image_placeholder` 字段，`run_vllm_inference` 从配置读取而非硬编码。

---

## 关键发现：vLLM 图像占位符因模型而异

`run_vllm_inference()` 硬编码 `image_placeholder="<|vision_start|><|image_pad|><|vision_end|>"`，但各模型在 vLLM 中期望不同占位符：

| Model Family | vLLM Expected Placeholder | Our Code |
|---|---|---|
| **qwen** | `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` | ✅ Correct |
| **internvl** | `<image>` | ❌ Wrong |
| **ovis** | `<image>` | ❌ Wrong |

来源：[vLLM InternVL](https://docs.vllm.ai/en/v0.17.1/api/vllm/model_executor/models/internvl/), [vLLM Ovis](https://docs.vllm.ai/en/v0.11.0/api/vllm/model_executor/models/ovis.html)

---

## 逐模型审计

### Qwen2-VL / Qwen2.5-VL ✅
- 社区标准：ChatML + `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` 占位符
- 我们的模板：`{image_tags}{prompt}` in ChatML user msg → **正确**

### InternVL family (Mini, 2, 2.5, 3) ⚠️
- 社区标准：ChatML + `<image>\n{prompt}` 占位符
- 我们的模板：格式正确，但 `{image_tags}` 展开为 Qwen 占位符 → **占位符错误**
- 来源：[InternVL Chat Data Format](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html)

### Ovis family (1.5-Gemma, 1.6-Llama, 2) ❌
- 社区标准：`USER: <image>\n{prompt}\nASSISTANT:`
- 我们的模板：ChatML + `{image_tags}` → **格式和占位符都错**
- 来源：[vLLM Ovis model](https://docs.vllm.ai/en/v0.11.0/api/vllm/model_executor/models/ovis.html)

### Phi3.5-Vision / Phi4-Multimodal ⚠️ (transformers-only)
- 社区标准：`<\|user\|>\n<\|image_1\|>\n{prompt}<\|end\|>\n<\|assistant\|>\n`
- 我们的模板：`<\|user\|>\n{prompt}<\|end\|>\n<\|assistant\|>\n` → 缺图像占位符
- 不影响功能（走 transformers 后端）

### InternLM-Xcomposer2.5 ✅ (transformers-only)
- 社区标准：`<\|User\|>:{prompt}<\|Bot\|>:` → 我们的模板一致

### MPlugOwl3 ⚠️ (transformers-only)
- 社区标准：`<\|image\|>\n{prompt}<\|endofthink\|>\n<\|assistant\|>`
- 我们的模板：`{prompt}` → 缺图像占位符
- 不影响功能（走 transformers 后端）

---

## 修复方案

1. **VLMConfig 加 `image_placeholder` 字段**（默认 `"<|vision_start|><|image_pad|><|vision_end|>"`）
2. **InternVL 模型**：`image_placeholder="<image>"`，模板用 `{image_tags}\n{prompt}`
3. **Ovis 模型**：模板改为 `USER: {image_tags}\n{prompt}\nASSISTANT:`，`image_placeholder="<image>"`
4. **`run_vllm_inference`**：从 VLMConfig 读取 `image_placeholder` 而非硬编码

## Sources

1. [vLLM multimodal inputs](https://docs.vllm.ai/en/v0.10.0/features/multimodal_inputs.html)
2. [InternVL Chat Data Format](https://internvl.readthedocs.io/en/latest/get_started/chat_data_format.html)
3. [vLLM InternVL model](https://docs.vllm.ai/en/v0.17.1/api/vllm/model_executor/models/internvl/)
4. [vLLM Ovis model](https://docs.vllm.ai/en/v0.11.0/api/vllm/model_executor/models/ovis.html)
5. [vLLM multimodal inference guideline](https://discuss.vllm.ai/t/multimodal-inference-guideline/698)
6. [mPLUG-Owl3 HuggingFace](https://huggingface.co/mPLUG/mPLUG-Owl3-7B-240728)
