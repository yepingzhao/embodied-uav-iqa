# UAV-Embodied-IQA 实验文档

> **可执行实验文档** — 每个实验包含描述、目的、验证的 Claim、以及可直接复制执行的多种子训练命令。

**相关文档**:

| 文档 | 内容 |
|------|------|
| [refine-logs/EXPERIMENT_PLAN.md](../refine-logs/EXPERIMENT_PLAN.md) | 实验计划、Claim Map、Anti-Claims、数据流与评估协议 |
| [refine-logs/EXPERIMENT_TRACKER.md](../refine-logs/EXPERIMENT_TRACKER.md) | 逐 run 状态追踪（M0–M4，33 runs） |
| [refine-logs/EXPERIMENT_RESULTS.md](../refine-logs/EXPERIMENT_RESULTS.md) | 初始实验结果与运行状态 |

---

## 前置条件

```bash
# 1. 安装依赖
uv sync --extra wandb

# 2. 设置 W&B API Key（可选，用于云端实验追踪）
export WANDB_API_KEY=your_key_here
# 或创建 .env 文件: echo "WANDB_API_KEY=your_key" > .env

# 3. 合成数据（如果尚未完成）
python scripts/data_synthesis.py all \
    --dataset aircopbench \
    --input-root data/raw/AirCopBench \
    --output-dir data/processed
```

## GPU 环境变量

```bash
# 指定 GPU（多卡用逗号分隔，如 "0,1"）
export CUDA_VISIBLE_DEVICES=0

# 双卡并行训练（覆盖默认单卡配置）
# --trainer.devices 2

# 完整示例：2 号卡 + 双卡
CUDA_VISIBLE_DEVICES=2,3 python scripts/train.py fit \
    --config configs/experiments/r013_task_cond.yaml \
    --trainer.devices 2 \
    --seed_everything 42

# 内存碎片问题时可限制 PyTorch 显存分配
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 训练入口

训练使用 `scripts/train.py`（LightningCLI 封装器），所有超参数由 `configs/experiments/<name>.yaml` 自包含配置驱动。输出目录通过配置中的 `trainer.default_root_dir` 自动设定，无需命令行指定。

```bash
# 基本命令格式（单种子）
CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
    --config configs/experiments/<name>.yaml

# 多种子
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/<name>.yaml \
        --seed_everything $seed
done

# 覆盖超参数示例
CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
    --config configs/experiments/r013_task_cond.yaml \
    --trainer.max_epochs 100 \
    --data.init_args.batch_size 128
```

## 输出结构

```
outputs/<experiment>/
├── wandb/                     # WandbLogger 本地文件
└── version_0/                 # Lightning 版本目录 (多种子递增: version_1, ...)
    ├── metrics.csv            # CSVLogger 训练指标
    ├── hparams.yaml           # 模型超参数
    └── checkpoints/
        ├── best_<val_srcc>.ckpt  # 最佳 checkpoint (按 val/srcc)
        ├── last.ckpt             # 最终 epoch checkpoint
        └── epoch_<NNN>.ckpt      # 每 5 epoch 保存
```

---

## 实验目录

### Claims 对照表

| Claim | 内容 | 类型 | 验证实验 |
|-------|------|------|----------|
| C1 | UAV 失真产生独特的 VLA 性能退化模式 | Primary | B1 (DQUS) |
| C2 | 合成失真与真实 UAV 失真相关 | Primary | B2 (D2RB) |
| C3 | 现有 IQA 方法无法预测 UAV 任务性能 | Primary | B3 (B4H), B4 (ABL) |
| C4 | 多任务训练使能跨任务质量泛化 | Supporting | B4 (XTG) |

### Anti-Claims 对照表

| Anti-Claim | 反驳实验 |
|------------|----------|
| AC1: "增益来自数据量而非 UAV 失真类型" | R021, R021b |
| AC2: "VLA 伪标签不可靠" | R022, R022b, R023 |
| AC3: "频率分支不必要" | R016 |
| AC4: "任务条件化不重要" | R017, R024a-c |
| AC5: "ViT backbone 更好" | R019, R020 |

---

## 步骤1: 主模型训练 (M3)

> **验证 Claim C3**: 数据库能训练出有效的 UAV-IQA 模型。
>
> 完整 UAV-IQANet 训练 + task-agnostic 对照组。

### R013 — 完整 UAV-IQANet (task-conditioned)

完整模型，所有组件启用：FAB + CBAM + FiLM task conditioning + 3-stage curriculum。

**关键配置**: `use_fab=true, use_cbam=true, use_task_conditioning=true`

```bash
# 单种子
CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
    --config configs/experiments/r013_task_cond.yaml \
    --seed_everything 42

# 三种子 (42, 100, 200)
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r013_task_cond.yaml \
        --seed_everything $seed
done
```

**成功标准**: SRCC > 0.65 (vs. VLA ground truth)

---

### R014 — UAV-IQANet task-agnostic 对照组

共享 MLP 头，无任务嵌入连接。作为 R013 的对照组。

**关键配置**: `use_task_conditioning=false`

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r014_task_agnostic.yaml \
        --seed_everything $seed
done
```

**成功标准**: SRCC 低于 R013（验证 task-conditioning 的价值）

---

## 步骤2: 组件消融 (M4)

> **验证 Claim C3 + AC3/AC4/AC5**: 每个设计组件都有独立贡献。
>
> 逐一移除关键组件，测量 SRCC 下降。

### R016 — 消融 FAB（频率感知分支）

移除 patch FFT + log-polar transform + tiny CNN 分支（~30K 参数）。验证 AC3。

**关键配置**: `use_fab=false`

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r016_no_fab.yaml \
        --seed_everything $seed
done
```

**成功标准**: SRCC 下降 > 0.05（证明频率分支的价值）

---

### R017 — 消融 Task Conditioning

移除 FiLM 任务嵌入，所有任务共享同一个 MLP 头。验证 AC4。

**关键配置**: `use_task_conditioning=false`

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r017_no_task_cond.yaml \
        --seed_everything $seed
done
```

**成功标准**: SRCC 下降 > 0.03（证明任务条件化的价值）

---

### R018 — 消融 CBAM

移除通道注意力 + 空间注意力模块。

**关键配置**: `use_cbam=false`

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r018_no_cbam.yaml \
        --seed_everything $seed
done
```

**成功标准**: SRCC 下降 > 0.02

---

### R019 — MobileViT-S Backbone

用 CNN-ViT 混合架构替换 MobileNetV4-S。验证 AC5（ViT 是否优于 CNN）。

**关键配置**: `backbone=mobilevit_s`

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r019_mobilevit_s.yaml \
        --seed_everything $seed
done
```

**成功标准**: SRCC 不显著优于 R013（±0.02，证明 CNN 效率选择合理）

---

### R020 — EfficientViT-B0 Backbone

用纯 ViT 架构替换 MobileNetV4-S。验证 AC5。

**关键配置**: `backbone=efficientvit_b0`

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r020_efficientvit_b0.yaml \
        --seed_everything $seed
done
```

**成功标准**: SRCC 不显著优于 R013（±0.02）

---

## 步骤3: 失真池消融 (M4)

> **验证 AC1**: UAV 失真类型是必要的还是装饰性的？
>
> 分别只用通用失真和只用 UAV 失真训练，对比完整模型。

### R021 — 仅用 30 种通用失真训练

训练时过滤掉所有 6 种 UAV 失真，只用 generic 失真池。

**关键配置**: `distortion_filter=generic`

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r021_generic_only.yaml \
        --seed_everything $seed
done
```

**成功标准**: SRCC 下降 > 0.08（证明 UAV 失真不可替代）

---

### R021b — 仅用 6 种 UAV 失真训练

训练时过滤掉所有 30 种通用失真，只用 UAV 失真池。

**关键配置**: `distortion_filter=uav`

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r021b_uav_only.yaml \
        --seed_everything $seed
done
```

**成功标准**: SRCC > 0.90 × R013 SRCC（UAV 失真池足够支撑有效训练）

---

## 步骤4: 课程消融 (M4, NICE-TO-HAVE)

> **验证 AC2**: VLA 伪标签是否可靠？课程学习的每个阶段贡献有多大？
>
> 分别只用 VLM/VLA 标注，以及去除 execution 层。

### R022 — 仅用 VLM 标注训练

50 epochs 全部使用 VLM 认知评分（无 curriculum transition）。

**关键配置**: `annotator_stage=vlm`

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r022_vlm_only.yaml \
        --seed_everything $seed
done
```

---

### R022b — 仅用 VLA 标注训练

50 epochs 全部使用 VLA 决策评分。

**关键配置**: `annotator_stage=vla`

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r022b_vla_only.yaml \
        --seed_everything $seed
done
```

---

### R023 — 去除 Execution 层

VLM (epochs 1-25) → VLA (epochs 26-50)，无 execution stage。

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r023_no_exec.yaml \
        --seed_everything $seed
done
```

---

## 步骤5: 跨任务泛化 (M4)

> **验证 Claim C4**: 多任务训练使能跨任务质量泛化。
>
> 三种设置：单任务训练、留一任务训练、多任务联合训练。

### R024a — 单任务训练 (4 个实验)

仅在单个任务上训练，测试所有 4 个任务的泛化性能。

| 实验 | 训练任务 | 测试任务 | 验证 |
|------|----------|----------|------|
| R024a_tracking | tracking | tracking, inspection, delivery, SAR | 同任务上限 vs. 跨任务泛化 |
| R024a_inspection | inspection | 同上 | 同上 |
| R024a_delivery | delivery | 同上 | 同上 |
| R024a_sar | SAR | 同上 | 同上 |

**关键配置**: `task=<task_name>`

```bash
# 每个任务分别训练 3 种子
for task in tracking inspection delivery sar; do
    for seed in 42 100 200; do
        CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
            --config "configs/experiments/r024a_${task}.yaml" \
            --seed_everything $seed
    done
done
```

**成功标准**: 跨任务 SRCC 与单任务 SRCC 差距 ≤ 0.05（≥3/4 任务对）

---

### R024b — 留一任务训练 (4 个实验)

在 3 个任务上训练，测试第 4 个留出任务的 zero-shot 泛化。

| 实验 | 训练任务 | 留出测试任务 |
|------|----------|-------------|
| R024b_leave_tracking | inspection + delivery + SAR | tracking |
| R024b_leave_inspection | tracking + delivery + SAR | inspection |
| R024b_leave_delivery | tracking + inspection + SAR | delivery |
| R024b_leave_sar | tracking + inspection + delivery | SAR |

**关键配置**: `leave_out_task=<task_name>`

```bash
for leave_task in tracking inspection delivery sar; do
    for seed in 42 100 200; do
        CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
            --config "configs/experiments/r024b_leave_${leave_task}.yaml" \
            --seed_everything $seed
    done
done
```

---

### R024c — 多任务联合训练

在所有 4 个任务上联合训练，作为跨任务性能的上限参考。

**关键配置**: `task=null, leave_out_task=null`

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r024c_multitask.yaml \
        --seed_everything $seed
done
```

---

## 批量运行

### 运行所有 MUST-RUN 实验（18 configs × 3 seeds = 54 任务）

```bash
# 循环运行所有 MUST-RUN 实验
for config in r013_task_cond r014_task_agnostic r016_no_fab r017_no_task_cond r018_no_cbam \
              r019_mobilevit_s r020_efficientvit_b0 r021_generic_only r021b_uav_only \
              r024a_tracking r024a_inspection r024a_delivery r024a_sar \
              r024b_leave_tracking r024b_leave_inspection r024b_leave_delivery r024b_leave_sar \
              r024c_multitask; do
    for seed in 42 100 200; do
        CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
            --config "configs/experiments/${config}.yaml" \
            --seed_everything $seed
    done
done
```

### 运行所有实验（含 NICE-TO-HAVE，21 configs × 3 seeds = 63 任务）

```bash
# 在上方命令中加入课程消融实验
for config in r013_task_cond r014_task_agnostic r016_no_fab r017_no_task_cond r018_no_cbam \
              r019_mobilevit_s r020_efficientvit_b0 r021_generic_only r021b_uav_only \
              r022_vlm_only r022b_vla_only r023_no_exec \
              r024a_tracking r024a_inspection r024a_delivery r024a_sar \
              r024b_leave_tracking r024b_leave_inspection r024b_leave_delivery r024b_leave_sar \
              r024c_multitask; do
    for seed in 42 100 200; do
        CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
            --config "configs/experiments/${config}.yaml" \
            --seed_everything $seed
    done
done
```

### 运行单个实验

```bash
for seed in 42 100 200; do
    CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
        --config configs/experiments/r013_task_cond.yaml \
        --seed_everything $seed
done
```

### 预览命令（不执行）

```bash
# 设置 dry_run=true 快速验证
CUDA_VISIBLE_DEVICES=0 python scripts/train.py fit \
    --config configs/experiments/r013_task_cond.yaml \
    --data.init_args.dry_run true \
    --trainer.max_epochs 1
```

### 查看运行状态

输出文件位于 `outputs/<experiment>/version_N/` (多种子由 Lightning 自动递增版本号):
- `metrics.csv` — CSVLogger epoch 级训练指标 (SRCC, PLCC, RMSE 等)
- `hparams.yaml` — 模型超参数
- `checkpoints/best_val_srcc.ckpt` — 最佳 checkpoint (按 val/srcc)
- `checkpoints/last.ckpt` — 最终 epoch checkpoint

---

## 实验总结

| 步骤 | 实验 | 验证目标 | 优先级 | 预计 GPU 时间 |
|------|------|----------|--------|---------------|
| 步骤1: 主模型 | R013, R014 | C3 | MUST | ~30h |
| 步骤2: 组件消融 | R016-R020 | C3, AC3/4/5 | MUST | ~75h |
| 步骤3: 失真消融 | R021, R021b | AC1 | MUST | ~30h |
| 步骤4: 课程消融 | R022, R022b, R023 | AC2 | NICE | ~45h |
| 步骤5: 跨任务 | R024a-c | C4 | MUST | ~120h |
| **总计** | **21 configs × 3 seeds** | | | **~300 GPU-h** |

### 实验依赖

```
步骤1 (主模型) ──→ 步骤2 (组件消融) ──→ 步骤4 (课程消融, 可选)
              │
              ├──→ 步骤3 (失真消融)
              │
              └──→ 步骤5 (跨任务泛化)
```

步骤2-5 在步骤1 完成后可以并行运行。

### 成功标准汇总

| 实验 | 标准 | 失败应对 |
|------|------|----------|
| R013 | SRCC > 0.65 | 超参数 sweep (lr, weight decay); 尝试更深 backbone |
| R014 | SRCC < R013 | — |
| R016 | SRCC 下降 > 0.05 | 检查 FAB 实现是否正确融合 |
| R017 | SRCC 下降 > 0.03 | 增加 task embedding 维度 |
| R019/R020 | SRCC ≈ R013 (±0.02) | 如果 ViT 显著更好，重新评估 backbone 选择 |
| R021 | SRCC 下降 > 0.08 | 增加 UAV 失真强度范围 |
| R021b | SRCC > 0.90 × R013 | 增加 UAV 失真样本多样性 |
| R024a | 跨任务 SRCC 差距 ≤ 0.05 | 记录为 limitation |
| R024c | 最接近 R013 (上限) | — |
