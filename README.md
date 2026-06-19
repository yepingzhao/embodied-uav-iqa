# UAV-Embodied-IQA

**Visual Quality Assessment for Aerial Embodied Intelligence**

A research codebase for no-reference image quality assessment (NR-IQA) tailored to UAV-embodied perception. Includes 24 distortion models (6 UAV-specific + 18 generic), a lightweight frequency-aware task-conditioned IQA network (~5.4M params), and a full benchmarking pipeline against 15+ existing IQA methods.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![Lightning 2.0+](https://img.shields.io/badge/lightning-2.0+-purple.svg)](https://lightning.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.11025-b31b1b.svg)](https://arxiv.org/abs/2511.11025)

---

## Key Research Claims

The project tests 4 claims (see [EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md) for details):

| Claim | Statement |
|-------|-----------|
| **C1** | UAV-specific distortions (propeller vibration, atmospheric scattering, 6DoF blur, packet loss, low-res+SR, propeller shadow) are **distinct** from generic distortions in terms of human/VLM quality assessment. |
| **C2** | Synthetic distortions injected into clean AirCopBench frames **correlate with real** UAV-degraded image quality. |
| **C3** | Existing IQA methods (PSNR, SSIM, BRISQUE, CLIP-IQA, MANIQA, etc.) **fail** to accurately assess UAV-specific distortions. |
| **C4** | A task-conditioned IQA model **generalizes across** embodied tasks (tracking, inspection, delivery, SAR). |

---

## Architecture

```
Input (3×256×256)
  ├─ MobileNetV4-S backbone (frozen first 2 stages) → multi-scale features
  ├─ PANet FPN → spatial pyramid fusion
  ├─ CBAM (channel + spatial attention) → f_s ∈ R^256
  ├─ FrequencyAwareBranch (patch FFT + log-polar transform + tiny CNN) → f_f ∈ R^64
  ├─ CrossAttentionGate: α = σ(W·[f_s, f_f]) → α ⊙ f_f
  └─ Concat[f_s, α⊙f_f] → TaskConditionedHead (FiLM: task_embed → γ,β modulate hidden)
       → 1 score per task
```

- **Backbone:** MobileNetV4-S (via `timm`), pretrained, frozen first 2 stages
- **Total params:** ~5.4M (INT8 quantized ~1.4MB)
- **Task types:** tracking (0), inspection (1), delivery (2), sar (3)
- **Training:** 50 epochs, 3-stage curriculum (VLM → VLA → Execution)
- **Loss:** MSE + λ_rank · ListMLE + λ_cross_task · CrossTaskRegularization

---

## Distortion Catalog

### 6 UAV-Specific Distortions

| Distortion | Model | Parameter Range |
|------------|-------|-----------------|
| Propeller Vibration Blur | Directional motion blur + periodic intensity modulation | f ∈ [80,200] Hz, A ∈ [1,8] px |
| Atmospheric Scattering Haze | Koschmieder model with depth estimation | β ∈ [0.5, 3.0] |
| 6DoF Viewpoint Blur | Motion blur from MotionScape empirical flow | μ=36.63 px, σ=25.4 px |
| Communication Packet Loss | 16×16 macroblock loss/replacement | loss rate ∈ [1%, 30%] |
| Low-Res + Super-Resolution | Bicubic downsample + Real-ESRGAN (optional) | scale ∈ [2, 8]× |
| Propeller Shadow | Periodic localized brightness modulation | α ∈ [0.05, 0.3] |

### 18 Generic Distortions

Categories: blur (3), brightness (5), chromatic (3), noise (4), compression (3), spatial (4), and other (4) — applied via **Albumentations**.

---

## Project Layout

```
src/uav_iqa/               # Core library
  distortion.py            # 24 distortion models (UAVDistortionPipeline)
  model.py                 # UAVIQANet (backbone → FPN → CBAM → FAB → task heads)
  dataset.py               # UAVIQADataset — manifest.json loader
  lightning_model.py       # LightningModule with MSE + ListMLE + cross-task loss
  lightning_data.py        # LightningDataModule with manifest filtering
  trainer.py               # ListMLELoss, CrossTaskRegularization
  evaluate.py              # SRCC, PLCC, RMSE, Kendall τ metrics
  callbacks.py             # CurriculumStageCallback, MetricsHistoryCallback
  cli.py                   # UAVIQACLI — LightningCLI with multi-seed support
  utils.py                 # count_parameters()
  __init__.py              # Public API exports

scripts/                   # Executable experiment scripts
  extract_aircopbench_refs.py  # Extract clean ref frames from AirCopBench
  run_distortion.py            # Visual sanity check of all 24 distortions
  run_m1_inject.py             # Batch distortion injection (24×5 per ref)
  run_m1_manifest.py           # Generate train/val/test manifest.json
  run_m1_aircopbench.py        # Full M1 pipeline with AirCopBench annotations
  run_m2_benchmark.py          # Benchmark 15+ existing IQA methods
  run_m3_train.py              # Training (multi-seed, ablations, cross-task, LOO)
  run_overfit.py               # 100-image overfit test (model correctness)

configs/
  default.yaml                 # LightningCLI configuration

tests/
  test_distortion.py           # 8 tests for distortion models
  test_lightning.py            # Tests for LightningModule & DataModule

refine-logs/                   # Research refinement artifacts
  FINAL_PROPOSAL.md            # Method thesis (score 9.0/10)
  EXPERIMENT_PLAN.md           # 33 runs, 6 milestones, 4 claims
  EXPERIMENT_TRACKER.md        # Run-by-run status tracker
  REVIEW_SUMMARY.md            # External review resolution log
```

---

## Setup

### Prerequisites
- Python ≥ 3.10
- CUDA-capable GPU (recommended for training)

### Installation

```bash
# Install with uv (recommended)
uv sync --group dev

# Or with pip
pip install -e .
pip install pytest black ruff  # dev dependencies

# For VLM annotation scoring (optional)
uv sync --group dev --extra vlm

# For benchmarking (pyiqa, optional)
uv sync --group dev --extra benchmark
```

### Data Preparation

Data is **not** included in the repo. Download AirCopBench from [arXiv 2511.11025](https://arxiv.org/abs/2511.11025).

**Data pipeline (execution order):**

```bash
# 1. Extract clean reference frames from AirCopBench
python scripts/extract_aircopbench_refs.py \
  --aircopbench-root data/AirCopBench \
  --output-dir data/database/ref_images

# 2. Inject all 24 distortions × 5 intensity levels
python scripts/run_m1_inject.py \
  --image-dir data/database/ref_images \
  --output-dir data/database/distorted \
  --workers 8

# 3. Generate train/val/test manifest.json
python scripts/run_m1_manifest.py \
  --distorted-dir data/database/distorted \
  --output-dir data/database

# — Or use the AirCopBench annotation pipeline —
python scripts/run_m1_aircopbench.py \
  --data-dir data/AirCopBench \
  --output-dir outputs/m1_aircopbench \
  --workers 4
```

---

## Usage

### Training

```bash
# Standard training with config
python scripts/run_m3_train.py fit --config configs/default.yaml

# Multi-seed (3 seeds)
python scripts/run_m3_train.py --seeds 42 100 200

# Per-task training (tracking only)
python scripts/run_m3_train.py --data.init_args.task tracking

# Leave-one-out (train on 3 tasks, test on SAR)
python scripts/run_m3_train.py --data.init_args.leave_out_task sar

# Distortion filter (generic only or uav_only)
python scripts/run_m3_train.py --data.init_args.distortion_filter generic
python scripts/run_m3_train.py --data.init_args.distortion_filter uav_only

# Dry run (fast verification, 100 train samples)
python scripts/run_m3_train.py --dry_run true --trainer.max_epochs 3
```

### Ablation Study

```bash
# Without Frequency-Aware Branch
python scripts/run_m3_train.py --model.init_args.use_fab false

# Without CBAM
python scripts/run_m3_train.py --model.init_args.use_cbam false

# Without task-conditioned heads
python scripts/run_m3_train.py --model.init_args.use_task_conditioning false
```

### Benchmark

```bash
# Run all 15+ methods on the test set
python scripts/run_m2_benchmark.py \
  --data-dir data/database \
  --output-dir outputs/benchmark

# Selected methods only
python scripts/run_m2_benchmark.py \
  --methods psnr ssim brisque clip_iqa maniqa

# Quick sanity (max 100 samples)
python scripts/run_m2_benchmark.py --max-samples 100
```

### Overfit Test

```bash
python scripts/run_overfit.py
# Verifies loss → 0 on 100 random images. Passes if final loss < 0.001.
```

### Distortion Verification

```bash
python scripts/run_distortion.py --output-dir outputs/m0_distortion_check
# Generates visual grid of all 24 distortions × 5 intensity levels.
```

---

## Evaluation Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **SRCC** | [-1, 1] | Spearman rank correlation (ranking quality) |
| **PLCC** | [-1, 1] | Pearson linear correlation (prediction accuracy) |
| **RMSE** | [0, ∞) | Root mean square error |
| **Kendall τ** | [-1, 1] | Kendall rank correlation |

Support for per-task and per-distortion evaluation via `evaluate.py`.

---

## Configuration

The [default config](configs/default.yaml) controls all aspects:

```yaml
model:
  backbone: mobilenetv4_conv_small
  use_fab: true          # Frequency-Aware Branch
  use_cbam: true         # CBAM attention
  use_task_conditioning: true  # FiLM task heads
  lambda_rank: 0.3       # ListMLE loss weight
  lambda_cross_task: 0.1  # Cross-task regularization

data:
  data_root: data/database
  batch_size: 64
  image_size: 256
  annotator_stage: vla

trainer:
  precision: 16-mixed
  max_epochs: 50
  gradient_clip_val: 1.0
```

---

## Development

### Commands

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/uav_iqa --cov-report=term-missing

# Single test
pytest tests/test_distortion.py::test_pipeline_has_all_distortions -v

# Lint
ruff check src/ tests/ scripts/

# Format
black src/ tests/ scripts/
```

### Key Concepts

- **Manifest format:** JSON list of `{path, task, distortion, intensity_level, ref_id, vlm_score, vla_score, execution_score, annotated}`
- **Distortion naming:** `{name}_L{intensity*10:02d}` (e.g., `propeller_vibration_blur_L04`)
- **3-stage curriculum:** VLM annotations (epochs 1-20) → VLA (21-40) → Execution (41-50)
- **Loss layers:** MSE + λ_rank · ListMLE (per-distortion ranking) + λ_cross_task · CrossTaskRegularization
- **Real-ESRGAN** is optional; falls back to bicubic + sharpen if not installed
- **openVLA/CARLA** are manual installs (not on PyPI); not needed for basic training/inference

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Detailed architecture, data pipeline, commands |
| [AGENTS.md](AGENTS.md) | Quick reference for development agents |
| [refine-logs/FINAL_PROPOSAL.md](refine-logs/FINAL_PROPOSAL.md) | Research method thesis (score 9.0/10) |
| [refine-logs/EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md) | 33 experiments across 6 milestones |
| [refine-logs/EXPERIMENT_TRACKER.md](refine-logs/EXPERIMENT_TRACKER.md) | Run-by-run status |
| [refine-logs/REVIEW_SUMMARY.md](refine-logs/REVIEW_SUMMARY.md) | External review resolutions |
| [docs/](docs/) | Literature reviews & research roadmap |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{uavembodiediqa2025,
  title={UAV-Embodied-IQA: Visual Quality Assessment for Aerial Embodied Intelligence},
  author={UAV-Embodied-IQA Team},
  year={2025},
  archivePrefix={arXiv},
  eprint={2511.11025}
}
```

---

## License

MIT
