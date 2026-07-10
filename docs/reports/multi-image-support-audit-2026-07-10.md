# Multi-Image Support Audit: 所有注册模型

*Generated: 2026-07-10 | Confidence: High*

## Executive Summary

13 个注册模型中，**11 个支持多图输入，2 个不支持**。不支持的是 Ovis1.5-Gemma 和 Ovis1.6-Llama（仅 Ovis2 支持多图）。AirCopBench 每组 scene+frame 有 2-6 张 UAV 图片，Ovis1.5/1.6 无法在单次推理中处理。

---

## 逐模型确认

### ✅ 支持多图

| Model | 多图方式 | 来源 |
|---|---|---|
| Qwen2-VL | M-RoPE 位置编码，原生多图+视频 | [Qwen2-VL blog](https://qwenlm.github.io/blog/qwen2-vl/) |
| Qwen2.5-VL | 同上 | 同上 |
| Mini-InternVL | InternVL 2.0+ 支持，`<image>` × N | [InternVL docs](https://internvl.readthedocs.io/) |
| InternVL2 | 同上 | 同上 |
| InternVL2.5 | 同上 | 同上 |
| InternVL3 | V2PE 增强多图理解 | [InternVL3 README](https://huggingface.co/OpenGVLab/InternVL3-2B-hf) |
| Ovis2 | Stage 3 训练含多图数据 | [Ovis2 README](https://huggingface.co/AIDC-AI/Ovis2-8B) |
| InternLM-Xcomposer2.5 | `<ImageHere>` 占位符，支持 20+ 张 | [IXC-2.5 paper](https://arxiv.org/abs/2407.03320) |
| Phi3.5-Vision | 从 Phi-3-V 升级，完整多图 | [Phi CookBook](https://deepwiki.com/microsoft/PhiCookBook) |
| Phi4-Multimodal | 多图+音频，128K 上下文 | 同上 |
| MPlugOwl3 | **核心卖点** — 400 张/A100 | [mPLUG-Owl3 paper](https://arxiv.org/abs/2408.04840) |

### ❌ 不支持多图

| Model | 原因 | 替代 |
|---|---|---|
| Ovis1.5-Gemma | 仅单图，无多图训练 | 升级到 Ovis2 |
| Ovis1.6-Llama | 同上 | 升级到 Ovis2 |

来源：[Ovis GitHub](https://github.com/AIDC-AI/Ovis), [Ovis2 commit](https://huggingface.co/AIDC-AI/Ovis2-8B/commit/4aacf1b)

---

## 对当前 Pipeline 的影响

`score_multi_image()` 一次传入所有 UAV 图片（2-6 张）：

- 11 个模型：正常工作 ✅
- Ovis1.5/1.6：会失败 ❌

建议在 MODEL_REGISTRY 中标记多图限制，或在运行时检测 `len(image_paths) > 1` 时对不支持多图的模型降级处理。
