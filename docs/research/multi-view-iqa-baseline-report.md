# 多视角图像质量评估 Baseline：研究报告

*生成日期: 2026-07-29 | 来源: 25+ | 置信度: High*

## 执行摘要

多视角 IQA 是一个新兴方向，目前**没有**成熟的、可直接使用的多视角 IQA 通用 baseline。现有工作分三个子领域：**CrossScore（ECCV 2024）** 是唯一直接对标"多视角→单图质量"的方法，但面向新视角合成（NVS）场景；**360°/全向图像 IQA** 有最成熟的多视口融合技术积累；**光场 IQA** 处理空间-角度双域质量评估。**不存在**针对 UAV 多视角感知质量评估的专用方法或 benchmark——这正是 Embodied-IQA 项目的独特贡献点。

---

## 1. 最直接相关的多视角 IQA 方法

### 1.1 CrossScore — ECCV 2024（Oxford Active Vision Lab）

**核心思路**：用 cross-attention 将 query image 与同一场景的多个 unregistered reference views 进行对比，无需 ground-truth 参考图，预测逐像素的 SSIM-like quality map。

```
架构: DINOv2 → Cross-Attention Transformer Decoder → MLP Head → Score Map
训练: 自监督 — NeRF checkpoint 渲染 → 利用 SSIM(render, clean) 做监督信号
         其他真实视角做 cross-reference set
```

**关键结果**（Active View Selector benchmark, 2025）：
| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| CrossScore-DINOv2 | **21.11** | 0.65 | 0.32 |
| MUSIQ (NR-IQA) | 19.62 | 0.58 | 0.38 |
| MANIQA (NR-IQA) | 19.54 | 0.59 | 0.37 |
| FisherRF (3D-based) | 20.34 | 0.60 | 0.37 |

**局限性**：
- 面向 NVS 场景，需要 scene-specific 的多视角参考图集
- 需要 DINOv2 + cross-attention transformer，推理成本高于单图 NR-IQA
- 不适用于任意单图的零样本评估（需要同场景多视角）

**资源**：[GitHub](https://github.com/ActiveVisionLab/CrossScore) | [arXiv 2404.14409](https://arxiv.org/abs/2404.14409) | [Project Page](https://crossscore.active.vision)

### 1.2 Active View Selector（arXiv 2506.19844, 2025）

CrossScore 团队后续工作，将 CrossScore 用于主动视角选择。在 NVS 和 3D 重建任务中，用 CrossScore 预测每个候选视角的渲染质量。这证明了多视角 IQA 在下游任务中的实用性。

---

## 2. 全向图像（360°）IQA — 多视口融合技术

这是多视角 IQA 最成熟的子领域，提供了丰富的融合策略参考。

### 2.1 融合策略汇总

| 策略 | 方法 | 机制 | 性能 |
|------|------|------|------|
| **Agreement-based Pooling** | PW-360IQA (2024) | 从视口质量分数分布中计算一致性加权 | PLCC/SRCC ~0.95 |
| **Semantic-Guided Adaptive Fusion** | Feng et al. (2025) | 球面均匀采样 + 语义引导自适应融合 | PLCC/SRCC >0.96 |
| **TSM + Self-Channel Attention** | 2025 | 伪时序视口序列 + 时序偏移模块 | 优于固定权重融合 |
| **Hypergraph Convolution** | 2025 | 超图建模视口间高阶关系 + 图影响网络 | 超越简单注意力聚合 |
| **Weight-Shared CNN** | PW-360IQA | 共享主干 + 多通道输入 | 参数从59M降至7.4M |

### 2.2 关键洞察

- **简单平均不是最优的**：learned, context-aware 的聚合策略一致优于 mean pooling
- **空间注意力粒度**：per-location (N×H×W) 的空间加权优于 per-image 或 per-channel 权重
- **视口间关系建模**：从独立处理 → 注意力交互 → 图/超图关系建模是明确的趋势

---

## 3. 光场 IQA

光场图像天然具有多视角结构（空间域 + 角度域），相关方法提供了一些架构参考：

| 方法 | 年份 | 架构 | 特点 |
|------|------|------|------|
| Two-Stream CNN | 2022 | EPI angular + SAI spatial 双流 | 基于极平面图像的角度信息提取 |
| DeLFIQE | 2021 | 多任务 CNN（回归+分类） | 首个 CNN-based LF-IQA |
| ViT-based NR-LFIQA | 2025 | 空间ViT + 角度ViT 双分支 | Transformer 架构，解决空间-角度分辨率差异 |
| Blind LF-QAE | 2023 (TMM) | Group-based + LBP-TOP + Log-Gabor | 角度一致性度量 |

**与本项目的差异**：光场图像的多视角是规则排列的微基线阵列，UAV 多视角是大基线、任意位姿的自由视角——光场方法的角一致性假设不适用。

---

## 4. 点云质量评估中的多视角投影

ViewPCQM（2025年1月）引入 **Multiview Pooling (MVP)** 模块：
- **Cross-View Spatial Fusion (CVSF)**：建模投影视图间的共享交叉信息
- **Set Pooling (SP)**：将多视图特征图池化为单一增强特征图

这提供了"多投影→单质量分数"的一种通用架构思路。

---

## 5. VLM-based 比较式 IQA

大模型驱动的 IQA 可以处理多图像比较，但不做视角融合：

| 方法 | 机制 | 多图能力 |
|------|------|----------|
| **Q-Align** (2024) | 离散评分等级对齐 | 单图（可通过 task_ 切换 quality/aesthetic） |
| **Co-Instruct** (2025) | 指令微调 + 比较推理 | 成对/三图/四图比较 |
| **Compare2Score** | 成对比较 → MAP 估计连续分数 | 通过 pair 扩展到多图 |
| **Q-Insight / VisualQuality-R1** | RL-based (GRPO) | 主要为单图评估 |

**与多视角融合的区别**：这些方法做的是"哪张更好"的比较判断，而非"融合多视角信息推断质量"。

---

## 6. UAV 领域 IQA 现状

### 6.1 ICME 2026 Drone-IQA Grand Challenge

- **6,000 张 UAV 图像**（VisDrone + UAVDT），18 人标注
- 三个维度：Global Quality / Target Quality / Background Quality
- **单图 NR-IQA**，不涉及多视角
- 冠军方案（DroneIQA-VLE）：SigLIP2 + Qwen3.5-9B 集成，PLCC 0.9512, SRCC 0.9450

### 6.2 已有 UAV 多视角数据（但无 IQA 标注）

| 数据集 | 视角数 | 用途 |
|--------|--------|------|
| ClaraVid | 3 同时视角 × 16,917 图 | 3D 重建 |
| AnyVisLoc | 多视角 18,000 图 | 视觉定位 |
| UAVLight (CVPR 2026) | 多时段固定航线 | 光照鲁棒重建 |

**结论**：UAV 多视角 + IQA 标注的数据集目前只有本项目。

---

## 7. 对本项目的启示

### 7.1 Baseline 方法选择建议

当前项目 baseline 全部是单图方法（pyiqa 的 15 个 FR/NR 方法），这是**合理的默认选择**。如果需要加入多视角对比，有以下选项：

| 方案 | 复杂度 | 适应性 | 推荐度 |
|------|--------|--------|--------|
| **A. 保持单图 baseline** | 低 | 直接可用 | ⭐⭐⭐⭐⭐ |
| **B. Per-view 平均 baseline** | 低 | 在现有方法外层包装 | ⭐⭐⭐⭐ |
| **C. CrossScore 迁移** | 高 | 需要场景级多视角参考 | ⭐⭐ |
| **D. 360° IQA 融合方法迁移** | 中-高 | 需要适配和重训 | ⭐⭐ |

### 7.2 推荐方案：A + B

1. **方案 A（当前做法）**：保持 15 个单图 baseline，使用每个样本的第一张 UAV 图像。这回答了 "现有 IQA 能否判断单张 UAV 图像的质量？"

2. **方案 B（补充消融实验）**：在现有方法外层做 per-view 评估 + 聚合（mean/agreement-based pooling），作为额外的消融对比。这回答了 "多视角信息对 IQA 有什么增益？"

```python
# 方案 B 的简单实现思路
def multi_view_score(iqa_model, view_images, pool="mean"):
    per_view_scores = [iqa_model(img) for img in view_images]
    if pool == "mean":
        return np.mean(per_view_scores)
    elif pool == "agreement":
        # agreement-based pooling (from PW-360IQA)
        ...
    return score
```

### 7.3 论文写作角度

多视角融合是 UAV-IQANet 的**核心创新点**（PANet FPN 做多尺度空间金字塔融合），baseline 方法做不到多视角融合正是需要论证的对比。建议：
- 将 "single-view baseline" vs. "multi-view UAV-IQANet" 作为主要对比
- 将 "per-view averaged baseline" 作为消融，证明 multi-view fusion > naive averaging

---

## 8. 关键参考文献

1. [CrossScore: Towards Multi-View Image Evaluation and Scoring](https://arxiv.org/abs/2404.14409) — Wang et al., ECCV 2024
2. [Active View Selector](https://arxiv.org/abs/2506.19844) — Bhalgat et al., arXiv 2025
3. [Omnidirectional IQA via Adaptive Multi-Viewport Fusion](https://doi.org/10.13700/j.bh.1001-5965.2023.0381) — Feng et al., 北航学报 2025
4. [PW-360IQA](https://hal.science/hal-04726828v1) — Sendjasni & Larabi, Sensors 2024
5. [ViewPCQM: Information Exploration of Projected Views for Point Cloud Quality Measurement](https://ieeexplore.ieee.org/document/10841467) — Li & Gao, 2025
6. [Q-Align: Teaching LMMs for Visual Scoring](https://arxiv.org/abs/2312.17090) — Wu et al., 2024
7. [Co-Instruct: Multi-Image Quality Comparison](https://arxiv.org/abs/2501.11311) — 2025
8. [Embodied-IQA](https://arxiv.org/abs/2505.16815) — Li et al., arXiv 2025
9. [Drone-IQA Grand Challenge (ICME 2026)](https://github.com/sunwei925/DroneIQA-VLE)
10. [A Survey on Image Quality Assessment](https://arxiv.org/abs/2502.08540) — Ma et al., 2025

---

## 研究方法

搜索了 12+ 个查询，涵盖多视角 IQA、多图质量评估、360° 视口融合、光场 IQA、UAV IQA benchmark 等方向。分析了 25+ 个独立来源，包括学术论文、arXiv 预印本、开源代码仓库和比赛报告。
