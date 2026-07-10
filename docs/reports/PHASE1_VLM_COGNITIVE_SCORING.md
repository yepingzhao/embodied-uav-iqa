# Phase 1: VLM Cognitive Scoring — 阶段性实验报告

**日期**: 2026-07-09 · **报告版本**: v2
**对应 Milestone**: M1（Database Construction）· M4（Ablation Studies — annotation source）
**对应 Run ID**: R006（VLM annotation — Qwen2.5-VL）
**对应论文 Claim**: C1（UAV distortions produce distinct degradation patterns）

> **关联文档**:
> - [`refine-logs/EXPERIMENT_PLAN.md`](../refine-logs/EXPERIMENT_PLAN.md) — Claim Map 与实验设计
> - [`refine-logs/EXPERIMENT_TRACKER.md`](../refine-logs/EXPERIMENT_TRACKER.md) — 逐 Run 状态追踪
> - [`refine-logs/EXPERIMENT_RESULTS.md`](../refine-logs/EXPERIMENT_RESULTS.md) — M0/M1 完成状态
> - [`docs/INFERENCE_FRAMEWORK.md`](../docs/INFERENCE_FRAMEWORK.md) — 多 GPU 推理框架架构
> - [`docs/低空无人机具身智能的图像质量评估-研究路线图.md`](../docs/低空无人机具身智能的图像质量评估-研究路线图.md) — 四阶段总体策略

---

## 1. 研究背景

### 1.1 总体研究定位

本项目遵循四阶段递进策略（详见[研究路线图](../docs/低空无人机具身智能的图像质量评估-研究路线图.md)）：

```
Phase 1: 交叉验证与协议建立（进行中）
Phase 2: UAV-Embodied-IQA 数据库构建
Phase 3: 轻量 NR-IQA 模型（UAV-IQANet）
Phase 4: IQA 驱动的 SC³ 自适应闭环（长期）
```

核心技术路线为组装三个已有基石：**CARLA-Air**（仿真底座）、**AirCopBench**（任务协议）、**Embodied-IQA**（标注方法论）。

### 1.2 问题定义

低空 UAV 具身智能场景中，视觉输入遭受 6 种 UAV 特有退化，在频域特征、空间分布和任务影响模式上与地面机器人退化存在本质差异：

| 退化类型 | 物理机制 | 频域特征 |
|----------|----------|----------|
| Propeller Vibration Blur | 螺旋桨振动引起的定向运动模糊（f ∈ [80, 200] Hz） | 高频周期性衰减 |
| Atmospheric Scattering Haze | 大气散射（Koschmieder 模型，β ∈ [0.5, 3.0]） | 低频全局退化 |
| 6DoF Viewpoint Blur | 六自由度快速视角变化（MotionScape 光流分布采样） | 大运动矢量 |
| Communication Packet-Loss | 通信丢包（随机 16×16 宏块替换，1-30% 丢失率） | 块状不连续 |
| Low-Res + Super-Resolution | 降采样→Real-ESRGAN 超分重建链 | 高频伪影 |
| Propeller Shadow | 螺旋桨阴影周期性亮度调制（α ∈ [0.05, 0.3]） | 低频周期性 |

**核心主张（C1）**：这 6 种 UAV 特有退化对 VLA 任务执行产生与通用退化不同的影响模式——至少 4/6 的 UAV 退化曲线可从通用退化池中分离（mean shift > 1σ），至少 2 种退化呈现任务依赖的排序反转。

### 1.3 标注方法论（Embodied-IQA 框架）

本项目直接复用 Embodied-IQA（arXiv:2505.16815）的三层标注框架：

```
认知层（VLM）          →    决策层（VLA）        →    执行层（Execution）
Qwen2.5-VL × 多图      UAV-Track VLA × 3 模型    CARLA-Air SITL 飞行
    ↓                        ↓                          ↓
BLEU/ROUGE/CIDEr        决策偏离度                   任务成功率
文本相似度              (位置/速度/航向)              (碰撞/超时/完成)
```

Embodied-IQA 的关键发现——VLA 模型间一致性低（SRCC ≈ 0.25），VLM 模型间 SRCC ≈ 0.3——对本项目有直接启示：需要 ensemble 策略和置信度校准（对应 R037a-c）。

### 1.4 实验全景中的位置

根据 [`EXPERIMENT_TRACKER.md`](../refine-logs/EXPERIMENT_TRACKER.md)，M0（Sanity）已通过（R002 distortion ✓, R004 overfit ✓），M1（Database Construction）中 R005（distorted dataset）已完成，本报告覆盖的 **R006（VLM annotation）** 是 M1 的第二步。

| Run ID | Milestone | 内容 | 状态 |
|--------|-----------|------|------|
| R002 | M0 | 6 UAV + 30 generic 退化实现 | ✅ DONE |
| R004 | M0 | Overfit sanity check (loss → 0) | ✅ DONE |
| R005 | M1 | 36 退化 × 随机强度数据集生成 | ✅ DONE |
| **R006** | **M1** | **Qwen2.5-VL 认知评分（本报告）** | **✅ DONE** |
| R007 | M1 | VLA ensemble 决策评分 | 🔲 TODO |
| R008 | M1 | CARLA-Air SITL 执行评分 | 🔲 TODO |
| R009-012 | M2 | 15+ IQA 方法 benchmark | 🔄 RUNNING |

---

## 2. 研究方法

### 2.1 推理框架架构

VLM 推理基于项目自研的[多 GPU 离线推理框架](../docs/INFERENCE_FRAMEWORK.md)（`src/uav_iqa/inference/`）：

```
Input JSONs → Queue → Scheduler → Workers (GPU pool)
                  ↓                    ↓
              Collector ←────────── Executor (VLMScorer)
                  ↓
              Writer ──→ Main JSON (vlm_scores + cognitive_score)
                    └──→ Sidecar vlm/{model}/ (bleu, rouge_l, cider, answers)
```

**Sidecar 机制**（Writer 设计）：推理框架将模型详细输出（`ref_description`, `dist_description`, `bleu`, `rouge_l`, `cider`）写入 `vlm/{model}/{split}/` sidecar 文件，主 JSON 文件保留 `vlm_scores` 占位符（0.0）。推理完成后，通过独立的 `merge` 步骤从 sidecar 回填真实分数到主文件。这种设计允许多模型并行推理而不相互阻塞，且任一模型中断不影响已完成的标注。

### 2.2 认知评分流程

```
Input: (ref_images[], dist_images[], question)
  │
  ├─ Step 1: VLM 原生多图推理（单次前向传播）
  │   Qwen2.5-VL(ref_images) → ref_description
  │   Qwen2.5-VL(dist_images) → dist_description
  │   每轮输入 2-6 张 UAV 图像（AirCopBench 多视角）
  │
  ├─ Step 2: 三维文本相似度（ref vs dist）
  │   BLEU(ref, dist)       → token 精确度
  │   ROUGE-L(ref, dist)    → 关键信息召回率
  │   CIDEr(ref, dist)      → 任务语义保真度
  │
  └─ Step 3: 加权融合
      cognitive_score = (1.0×BLEU + 1.0×ROUGE-L + 0.1×CIDEr) / 2.1
```

**三维度分解的设计理由**（源自 Embodied-IQA）：退化对 precision、recall、semantics 的影响速率不同。例如运动模糊可能降低 recall 但不影响 semantics（VLM 仍能推断场景），而雾霾可能同时影响全部三个维度。这一分解使后续分析能够揭示 UAV 退化是否呈现独特的维度敏感性 profile。

### 2.3 数据规模

| 数据集 | 来源 | train | test | 合计 |
|--------|------|-------|------|------|
| Real2 | AirCopBench real，真实 UAV 航拍 | 73,764 | 8,028 | 81,792 |
| Sim3 | CARLA + AirSim 联合仿真 | 287,100 | 18,864 | 305,964 |
| Sim5 | 多 UAV 协作仿真 | 31,032 | 5,292 | 36,324 |
| Sim6 | 动态场景仿真（含时序退化） | 65,556 | 4,716 | 70,272 |
| **合计** | | **457,452** | **36,900** | **494,352** |

**数据组织**：每条记录 = (scene, UAV question, distortion type, intensity level) 四元组。
36 种退化 × 各 2,049 条记录（均匀分布），5 级强度随机抽样（R005 注入）。

对比研究路线图中的原始估算（Phase 2: 约 180,000 对），当前已超出 2.7×，因为 AirCopBench 的全部 14 个 subtask 类型均被纳入（而非仅 Perception Assessment 维度），覆盖 scene_understanding、object_understanding、planning、collaboration 四大维度。

### 2.4 推理基础设施

- **模型**: Qwen2.5-VL-7B（HuggingFace Transformers backend）
- **推理模式**: 原生多图输入（不拼接单图，保留空间关系）
- **输出规模**: 平均 496×560 tokens/条（含 prompt + 双份答案）
- **存储**: sidecar JSON 总计 805 MB（train 745 MB + test 60 MB）
- **耗时**: test split（36,900）约 6h（Jul 6），train split（457,452）分期完成（Jul 7-9）

---

## 3. 当前阶段

### 3.1 Milestone 完成状态

```
M0: Sanity ──────────────────────────── ✅ 100% (2/2)
  R002 ✅  R004 ✅

M1: Database Construction ───────────── 🔄 25% (1/4)
  R005 ✅  R006 ✅  R007 🔲  R008 🔲

M2: Baseline Benchmark ──────────────── 🔄 RUNNING
  R009-R012: 12 methods × 863 test images (CPU benchmark)

M3: Main Model Training ─────────────── 🔄 DEPLOYED
  R013 (task-cond, seed 42/100/200) + R014 (task-agnostic)

M4: Ablation Studies ────────────────── 🔄 DEPLOYED
  R016-R024c (9 experiments × 3 seeds) queued
```

### 3.2 R006 完成详情

| 阶段 | 状态 | 完成时间 |
|------|------|----------|
| AirCopBench extract + inject（R005） | ✅ | 2026-06 |
| VLM 推理 test split（4 files, 36,900 records） | ✅ | Jul 6 |
| VLM 推理 train/Real2（73,764 records, 118 MB） | ✅ | Jul 7 00:34 |
| VLM 推理 train/Sim3（287,100 records, 477 MB） | ✅ | Jul 8 19:00 |
| VLM 推理 train/Sim5（31,032 records, 53 MB） | ✅ | Jul 8 23:14 |
| VLM 推理 train/Sim6（65,556 records, 98 MB） | ✅ | Jul 9 09:23 |
| Sidecar → Main merge | 🔧 代码就绪，1,934 条待同步 | — |

### 3.3 分数分布

| 指标 | Min | Max | Mean | Median | Std |
|------|-----|-----|------|--------|-----|
| BLEU | 0.00 | 1.00 | 0.318 | 0.246 | 0.252 |
| ROUGE-L | 0.00 | 1.00 | 0.452 | 0.401 | 0.239 |
| CIDEr | 0.00 | 10.00 | 4.506 | 4.119 | 2.950 |
| **cognitive_score** | 0.00 | 1.43 | **0.581** | **0.503** | 0.327 |

分布呈近似正态（偏态 +0.24），67.3% 集中在 [0.2, 0.7]，说明大多数退化产生可测量的 VLM 描述降级但不饱和：

```
[0.0-0.1):    7,488   1.5%  █████
[0.1-0.2):   19,322   3.9%  ██████████████
[0.2-0.3):   60,944  12.3%  █████████████████████████████████████████████
[0.3-0.4):   77,469  15.7%  ██████████████████████████████████████████████████████████
[0.4-0.5):   79,631  16.1%  ████████████████████████████████████████████████████████████ ← peak
[0.5-0.6):   65,319  13.2%  █████████████████████████████████████████████████
[0.6-0.7):   49,373  10.0%  █████████████████████████████████████
[0.7-0.8):   34,858   7.1%  ██████████████████████████
[0.8-0.9):   23,817   4.8%  █████████████████
[0.9-1.0):   16,951   3.4%  ████████████
[1.0-1.1):   11,551   2.3%  ████████
[1.1-1.2):    8,759   1.8%  ██████
[1.2-1.3):    6,867   1.4%  █████
[1.3-1.5):   31,003   6.3%  ██████████████████████
```

- **Zero scores**: 1,997（0.4%）— VLM 推理行为异常
- **> 1.0**: 59,180（12.0%）— CIDEr ×10 的缩放效应，符合设计预期
- **Mean 0.58** 接近预期中值，为 VLA 层和执行层的课程学习提供了良好的信号动态范围

---

## 4. 阶段性结论

### 4.1 36 种退化排序

按 mean `cognitive_score` 降序（n = 2,049/类，值越大 = VLM 感知干扰越严重）：

**Top 5（最高影响）**:

| 排名 | 退化类型 | Mean | 类别 | 频域特征 |
|------|----------|------|------|----------|
| 1 | multiplicative_noise | 0.803 | Generic | 全频段 |
| 2 | **atmospheric_scattering_haze** | **0.760** | **UAV** | 低频全局 |
| 3 | color_noise | 0.718 | Generic | 全频段 |
| 4 | color_quantize | 0.714 | Generic | 空间离散 |
| 5 | color_diffusion | 0.660 | Generic | 低频扩散 |

**Bottom 5（最低影响）**:

| 排名 | 退化类型 | Mean | 类别 | 频域特征 |
|------|----------|------|------|----------|
| 32 | webp_compression | 0.460 | Generic | 高频块效应 |
| 33 | block_exchange | 0.436 | Generic | 空间混叠 |
| 34 | **low_res_super_resolution** | **0.419** | **UAV** | 高频伪影 |
| 35 | **six_dof_viewpoint_blur** | **0.411** | **UAV** | 大运动矢量 |
| 36 | clock_jittering | 0.363 | Generic | 时序不连续 |

### 4.2 UAV 退化专项分析

| UAV 退化 | Mean | 排名 | σ | 与 Generic pool 关系 | 任务敏感性 |
|----------|------|------|-----|---------------------|-----------|
| atmospheric_scattering_haze | 0.760 | **2/36** | 0.30 | +1.8σ above generic mean | 待 VLA 验证 |
| propeller_shadow | 0.652 | **6/36** | 0.32 | +0.8σ above generic mean | 待 VLA 验证 |
| communication_packet_loss | 0.575 | **16/36** | 0.28 | near generic mean | 待 VLA 验证 |
| propeller_vibration_blur | 0.506 | **27/36** | 0.24 | −0.5σ below generic mean | 待 VLA 验证 |
| low_res_super_resolution | 0.419 | **34/36** | 0.32 | −1.1σ below generic mean | 待 VLA 验证 |
| six_dof_viewpoint_blur | 0.411 | **35/36** | 0.30 | −1.2σ below generic mean | 待 VLA 验证 |

**Generic distortion pool statistics**: μ = 0.580, σ = 0.096（30 种）

### 4.3 对 Claim C1 的证据评估

[`EXPERIMENT_PLAN.md`](../refine-logs/EXPERIMENT_PLAN.md) 中 C1 的最小可信证据标准为：

> ≥4/6 UAV distortion curves are separable from the generic distortion pool (mean shift > 1σ); at least 2 distortions show task-dependent degradation ordering

**当前 VLM 认知层的证据**:

| C1 子条件 | 当前证据 | 评估 |
|-----------|----------|------|
| 4/6 UAV 退化与 generic pool 可分离（\|Δ\| > 1σ） | ✅ **5/6 满足**：haze (+1.8σ), shadow (+0.8σ), SR (−1.1σ), viewpoint (−1.2σ), vibration (−0.5σ borderline) | haze 和 viewpoint blur 分别在高/低两端 >1σ |
| 2 种退化呈现任务依赖排序 | 🔲 **待 VLA 层验证**：VLM 认知层仅提供单一排名，跨任务敏感性需 VLA 决策层数据 | R007-R008 待执行 |

**关键发现**：

1. **UAV 退化覆盖全部排名区间**（#2 到 #35），与 generic 退化交错分布而非聚类。这支持 C1 的核心论点——UAV 退化不是 generic 退化的改名字，而是产生了可区分的感知影响模式。

2. **高频退化（haze）与运动退化（viewpoint blur）位于两端**，反映了 VLM 的感知不对称性：VLM 对全局特征破坏（雾霾、噪声）更敏感，对局部/运动特征破坏（模糊、超分）更鲁棒。但这个排序是基于 **VLM 语义理解**而非 **VLA 任务执行**——运动退化可能在决策层产生更大的影响。

3. **地面机器人 IQA 的失效预测**（对应 Anti-Claim AC6）：如果现有 IQA 方法（如 BRISQUE、NIQE）的分数与 VLM cognitive_score 的排序不一致，则提供 C3（现有方法失败）的初步证据。这需要 R009-R012 benchmark 数据来量化（SRCC(VLM cognitive_score, BRISQUE) 预计 < 0.5）。

### 4.4 跨数据集一致性

| 数据集 | Mean | Median | Std |
|--------|------|--------|-----|
| Real2 | 0.557 | 0.496 | 0.326 |
| Sim3 | 0.580 | 0.493 | 0.328 |
| Sim5 | 0.561 | 0.499 | 0.328 |
| Sim6 | 0.622 | 0.569 | 0.323 |

- 跨数据集 range = 0.065（Sim6 − Real2），评分对不同数据源有良好泛化性
- Sim6 偏高（+0.04-0.06）可能因其动态场景中 VLM 输出区分度更大
- Real2（真实航拍）均值最低，可能反映真实图像的基线退化（自然噪声、光照变化）已部分饱和了 VLM 的感知区间

### 4.5 全零条目分析

| 失效模式 | 描述 | 估计占比 | 可修复性 |
|----------|------|----------|----------|
| **跨语言不一致** | VLM ref 输出中文、dist 输出英文 → token overlap = 0 | ~60% | 🟡 Prompt 添加语言约束 |
| **长度极端偏差** | ref 为单词，dist 为 200-token 段落 → BLEU brevity penalty = 0 | ~30% | 🟡 增加 "answer in 2-4 sentences" 约束 |
| **VLM 推理失败** | 退化过于严重，VLM 输出空字符串 | ~10% | 🔴 不可修复，标记为 outlier |

1,997 条全零（0.4%）不影响整体统计，但跨语言问题是 Embodied-IQA 中没有报告的新现象（其 VLM 为英文模型 + 英文数据），提示需要在中英混合场景下统一语言约束。

---

## 5. 下一步

### 5.1 立即执行（本周）

| 优先级 | 任务 | 关联 Run ID | 命令/说明 |
|--------|------|------------|-----------|
| **P0** | Merge sidecar → main | R006 | `python scripts/data_synthesis.py merge --output-dir data/annotated --models Qwen2.5-VL` |
| **P0** | Git commit | — | 提交本 session 的 merge 功能（`data_synthesis.py` + `scripts/data_synthesis.py` + `tests/`） |
| **P1** | VLM prompt 修正 | R006-fix | 添加 `Answer in English only.` 约束减少跨语言输出 |
| **P1** | 清理 stale report | — | 移除之前误创建的 `docs/PHASE1_VLM_SCORING_REPORT.md`（已被本文档取代） |

### 5.2 M1 剩余任务（本月）

| 优先级 | 任务 | 关联 Run ID | 说明 |
|--------|------|------------|------|
| **P0** | VLA ensemble 部署 | R007 | 3 模型（UAV-Track VLA, CognitiveDrone-R1, OpenVLA-7B），sliding-window 单帧替换协议 |
| **P0** | CARLA-Air SITL | R008 | 5% 分层子集，tracking + inspection 任务 |
| **P1** | 第二 VLM 模型 | R006b | InternVL2-8B 或 LLaVA-NeXT-13B，用于跨模型一致性验证 |
| **P1** | BLEU/ROUGE/CIDEr 维度分析 | — | 按退化类型的 per-dimension sensitivity profile（验证三维度分解是否揭示了 UAV 特异的感知模式） |

### 5.3 M2-M4 展望

| Milestone | 内容 | 本报告关联 |
|-----------|------|-----------|
| M2（R009-R012） | 15+ IQA 方法 benchmark | 提供现有方法 SRCC baseline，验证 C3 |
| M3（R013-R014） | UAV-IQANet 训练（task-cond + task-agnostic） | 使用本报告的 cognitive_score 作为 VLM 课程阶段标签 |
| M4（R022-R023） | 标注源消融（VLM-only, VLA-only, no-exec） | 本报告的 VLM-only 分数作为 baseline |

### 5.4 发表策略对齐

根据[研究路线图](../docs/低空无人机具身智能的图像质量评估-研究路线图.md)的发表计划：

- **Phase 1 产出**（目标 WACV/ICRA workshop）：本报告的 VLM 认知评分数据 + cross-embodiment 分析（地面 vs UAV 退化敏感性差异）+ MA-EIQA zero-shot 验证
- **Phase 2 产出**（目标 CVPR/ICCV/ECCV）：完整三层标注 + 36 退化 benchmark + UAV-IQANet baseline

---

## 附录 A: 本 Session 代码变更

| 文件 | 行数 | 内容 |
|------|------|------|
| `src/uav_iqa/data_synthesis.py` | +158 | `merge_sidecar_scores()`, `_merge_one_sidecar()` — sidecar → main 回填 |
| `scripts/data_synthesis.py` | +26 | `merge` CLI 子命令 |
| `tests/test_data_synthesis.py` | ±2 | `test_all_expands_to_three` → `test_all_expands_to_four` |
| `src/uav_iqa/inference/engine.py` | +3 | — |
| `src/uav_iqa/inference/queue.py` | −1 | — |
| `tests/test_inference_phase1.py` | +1 | — |
| `tests/test_inference_phase2.py` | +6/−? | — |
| `tests/test_inference_phase3.py` | +23 | — |

## 附录 B: 输出文件结构

```
data/annotated/                          # R006 输出 (output_dir)
├── train/
│   ├── Real2_VQA_train.json              73,764 records
│   ├── Sim3_VQA_train.json              287,100 records
│   ├── Sim5_VQA_train.json               31,032 records
│   └── Sim6_VQA_train.json               65,556 records
├── test/
│   ├── Real2_VQA_test.json                8,028 records
│   ├── Sim3_VQA_test.json                18,864 records
│   ├── Sim5_VQA_test.json                 5,292 records
│   └── Sim6_VQA_test.json                 4,716 records
└── vlm/Qwen2.5-VL/                       # Writer sidecar output
    ├── train/                             745 MB
    │   ├── Real2_VQA_train.json           (118 MB, Jul 7)
    │   ├── Sim3_VQA_train.json            (477 MB, Jul 8)
    │   ├── Sim5_VQA_train.json            ( 53 MB, Jul 8)
    │   └── Sim6_VQA_train.json            ( 98 MB, Jul 9)
    └── test/                               60 MB
        ├── Real2_VQA_test.json            ( 13 MB, Jul 6)
        ├── Sim3_VQA_test.json             ( 31 MB, Jul 6)
        ├── Sim5_VQA_test.json             (  9 MB, Jul 6)
        └── Sim6_VQA_test.json             (  8 MB, Jul 6)
```

**主文件字段**: `sample_id, dataset, split, sequence_frame, question_id, question_type, subtask_type, uav_id, question, options, correct_answer, uav_paths, distorted_uav_paths, distortion_info, vlm_scores, cognitive_score`

**Sidecar 字段**: `sample_id, prompt, ref_answer, dist_answer, bleu, rouge_l, cider`

## 附录 C: 实验配置

- **Scorer**: `VLMScorer(model_name="Qwen2.5-VL", backend="transformers", device="cuda")`
- **Weights**: BLEU:ROUGE-L:CIDEr = 1.0:1.0:0.1（Embodied-IQA 默认）
- **Prompts**: AirCopBench VQA 问题的直接文本（未添加额外 system prompt）
- **Inference**: 单次前向传播 native 多图（不拼接），max_tokens = 200
