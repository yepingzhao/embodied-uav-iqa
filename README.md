# UAV-Embodied-IQA

**Visual Quality Assessment for Aerial Embodied Intelligence**

A research codebase for no-reference image quality assessment (NR-IQA) tailored to UAV-embodied perception. Includes 36 distortion models (6 UAV-specific + 30 generic), a lightweight frequency-aware task-conditioned IQA network (~5.4M params), and a full benchmarking pipeline against 15+ existing IQA methods.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/pytorch-2.1+-red.svg)](https://pytorch.org/)
[![Lightning 2.0+](https://img.shields.io/badge/lightning-2.0+-purple.svg)](https://lightning.ai/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.11025-b31b1b.svg)](https://arxiv.org/abs/2511.11025)

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
- **14 subtask types:** scene_understanding (1.1–1.3), object_understanding (2.1–2.4), planning (3.1–3.3), collaboration (4.1–4.4)
- **Training:** 50 epochs, single `cognitive_score` supervision (replaces 3-stage curriculum)
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

### 30 Generic Distortions

Categories: blur (3), brightness (5), chromatic (3), noise (6), compression (3), spatial (3), transmission (3), and other (4) — applied via **Albumentations**.

---

## Project Layout

```
src/uav_iqa/               # Core library
  __init__.py              # Lightweight package metadata
  domain/                  # AirCopBench taxonomy, parsing, degradation factors, score synthesis
  models/                  # UAVIQANet composition, spatial/frequency/text components, heads
  distortions/             # 36 distortion models (UAVDistortionPipeline)
  data/                    # Dataset adapters, synthesis, samples, dataset, and datamodule
  training/                # Lightning module, MSE/ListMLE/cross-task losses, callbacks
  evaluation/              # IQA and text-similarity metrics
  vlm/                     # VLM scorer, BatchAnnotator, config, backends, VQA index
  baselines/               # Existing IQA evaluation and fine-tuning
  inference/               # Multi-GPU offline inference framework (15 modules)
  utils.py                 # count_parameters, find_images, manifest I/O, logging

scripts/                   # Executable experiment scripts (12 total)
  distortion_synthesis.py              # Unified data synthesis CLI (extract/inject/manifest/annotate/all)
  download_models.py             # Download VLM model weights from HuggingFace Hub
  vlm_annotate.py                # Batch VLM annotation CLI with checkpoint/resume
  benchmark_iqa_methods.py       # Benchmark 15+ existing IQA methods
  finetune_baselines.py          # Fine-tune DL-based IQA baselines on UAV data
  train.py                       # Unified training entry point (LightningCLI)
  inference.py                   # CLI entry point for multi-GPU offline inference

configs/
  README.md                    # Maintained configuration catalog and prerequisites
  experiments/                 # 8 configs; full_model.yaml is the reference

tests/
  test_distortion.py           # 19 tests for distortion models
  test_lightning.py            # Tests for LightningModule & DataModule
  test_distortion_synthesis.py       # 29 tests for data pipeline
  test_text_metrics.py         # Tests for BLEU, ROUGE-L, CIDEr text metrics
  test_vlm_config.py           # Tests for VLMConfig & MODEL_REGISTRY (15 models)
  test_vlm_scorer.py           # 109 tests for VLMScorer, prompts, pipeline
  test_vlm_smoke.py            # GPU smoke tests × 15 parametrized models
  test_batch_annotator.py      # Tests for BatchAnnotator (filter, checkpoint, resume)
  test_annotations.py          # Tests for annotation parsing utilities
  test_dataset.py              # Tests for UAVQualityDataset collation and loading
  test_model_text.py           # Tests for model text/export utilities
  test_inference_phase[1-5].py # 5-phase tests for inference framework

docs/                         # Architecture documentation and literature references
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

**Data pipeline:**

```bash
# Full pipeline (all 4 steps at once)
python scripts/distortion_synthesis.py all \
  --dataset aircopbench \
  --input-root data/raw/AirCopBench \
  --output-dir data/processed

# Or run individual steps:
python scripts/distortion_synthesis.py extract \
  --dataset aircopbench --input-root data/raw/AirCopBench \
  --output-dir data/processed/ref_images

python scripts/distortion_synthesis.py inject \
  --image-dir data/processed/ref_images \
  --output-dir data/processed/distorted --workers 8

python scripts/distortion_synthesis.py manifest \
  --dataset aircopbench --distorted-dir data/processed/distorted \
  --output-dir data/processed

python scripts/distortion_synthesis.py annotate \
  --dataset aircopbench --manifest-dir data/processed \
  --input-root data/raw/AirCopBench
```

---

## Usage

### Training

All hyperparameters live in self-contained YAML configs. Training uses `scripts/train.py` (vanilla LightningCLI). See the [configuration catalog](configs/README.md) for data prerequisites and W&B setup.

```bash
# Standard training (task-conditioned, all components)
python scripts/train.py fit --config configs/experiments/full_model.yaml

# Multi-seed via shell loop
for seed in 42 100 200; do
  python scripts/train.py fit --config configs/experiments/full_model.yaml \
    --seed_everything $seed \
    --trainer.default_root_dir "outputs/full_model_seed${seed}"
done

# Distortion filter (generic only or uav_only)
python scripts/train.py fit --config configs/experiments/generic_only.yaml
python scripts/train.py fit --config configs/experiments/uav_only.yaml

# Override any config key from CLI
python scripts/train.py fit --config configs/experiments/full_model.yaml \
  --data.dry_run true \
  --trainer.max_epochs 3
```

### Ablation Studies

Each ablation has its own self-contained config:

```bash
# Without Frequency-Aware Branch
python scripts/train.py fit --config configs/experiments/no_frequency_encoder.yaml

# Without CBAM
python scripts/train.py fit --config configs/experiments/no_spatial_attention.yaml

# Without task conditioning
python scripts/train.py fit --config configs/experiments/no_task_conditioning.yaml

# Backbone ablations
python scripts/train.py fit --config configs/experiments/backbone_mobilevit_s.yaml
python scripts/train.py fit --config configs/experiments/backbone_efficientvit_b0.yaml
```

### Benchmark

```bash
# Run all 15+ methods on the test set
python scripts/benchmark_iqa_methods.py \
  --data-dir data/processed \
  --output-dir outputs/benchmark

# Selected methods only
python scripts/benchmark_iqa_methods.py \
  --methods psnr ssim brisque clip_iqa maniqa

# Quick sanity (max 100 samples)
python scripts/benchmark_iqa_methods.py --max-samples 100
```

### Overfit Test

```bash
# Verifies loss → 0 on 100 random images. Passes if final loss < 0.001.
```

### Distortion Verification

```bash
# Generates visual grid of all 36 distortions × 1 random intensity level.
```

---

## Evaluation Metrics

| Metric | Range | Description |
|--------|-------|-------------|
| **SRCC** | [-1, 1] | Spearman rank correlation (ranking quality) |
| **PLCC** | [-1, 1] | Pearson linear correlation (prediction accuracy) |
| **RMSE** | [0, ∞) | Root mean square error |
| **Kendall τ** | [-1, 1] | Kendall rank correlation |

Support for per-task and per-distortion evaluation via `metrics.py`.

---

## Configuration

The [main-model config](configs/experiments/full_model.yaml) serves as the reference. Each experiment has its own self-contained YAML in `configs/experiments/`; see the [configuration catalog](configs/README.md) for the eight maintained variants and their scope.

```yaml
seed_everything: 42
model:
  backbone: mobilenetv4_conv_small
  use_frequency_encoder: true            # Frequency-Aware Branch
  use_spatial_attention: true           # CBAM attention
  use_task_conditioning: true  # FiLM task heads
  lambda_rank: 0.3         # ListMLE loss weight
  lambda_cross_task: 0.1   # Cross-task regularization
  lr: 1.2e-3
  total_epochs: 50

data:
  data_root: data/processed
  batch_size: 256
  image_size: 256
  num_workers: 16

trainer:
  accelerator: auto
  precision: 16-mixed
  max_epochs: 50
  gradient_clip_val: 1.0
  logger:
    - class_path: lightning.pytorch.loggers.WandbLogger
    - class_path: lightning.pytorch.loggers.CSVLogger
  callbacks:
    - class_path: uav_iqa.training.SetupRunCallback
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
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

# Run smoke tests only (slow/GPU tests excluded)
pytest tests/ -v -m "not smoke"

# Run GPU tests only
pytest tests/ -v -m gpu

# Lint
ruff check src/ tests/ scripts/

# Format
black src/ tests/ scripts/
```

### Key Concepts

- **Flat JSON format:** `{sample_id, uav_paths, distorted_uav_paths, distortion_info, subtask_type, cognitive_score}` — one entry per question×distortion pair.
- **Distortion naming:** `{name}_L{intensity*10:02d}` (e.g., `propeller_vibration_blur_L04`). Supports `.png` and `.jpg` extensions.
- **Single `cognitive_score` supervision:** Unified quality score from VLM multi-image aggregation (replaces 3-stage curriculum).
- **Loss layers:** MSE + λ_rank · ListMLE (per-distortion ranking) + λ_cross_task · CrossTaskRegularization (negative pairwise score variance)
- **Real-ESRGAN** is optional (`LowResSuperResolution` distortion); falls back to bicubic+sharpen if not installed
- **VLM extras** (`vllm`, `transformers`, `accelerate`) for annotation scoring: `uv sync --group dev --extra vlm`
- **Training entry:** `scripts/train.py` (vanilla LightningCLI). `main.py`, `run_m3_train.py` and `UAVIQACLI` were removed in the 2026-06 refactor.
- **Score annotation:** `scripts/distortion_synthesis.py annotate` scores distorted groups via VLM, `aggregate` computes `cognitive_score` as mean of VLM scores.
- **`scipy` removed as a direct dependency** for distortion models — uses `cv2.filter2D` with manual wrap padding. `scipy` is retained for metric computation.
- **Model download:** `scripts/download_models.py` provides offline VLM model weight download from HuggingFace Hub for VLM annotation scoring. Supports `--all`, `--models <name>`, `--validate`, and `--validate-only`.

---

## Documentation

| Document | Description |
|----------|-------------|
| [CLAUDE.md](CLAUDE.md) | Detailed architecture, data pipeline, commands |
| [AGENTS.md](AGENTS.md) | Quick reference for development agents |
| [docs/CODEMAPS/ARCHITECTURE.md](docs/CODEMAPS/ARCHITECTURE.md) | Detailed architecture diagram and data flow |
| [docs/CODEMAPS/FILES.md](docs/CODEMAPS/FILES.md) | File tree with line counts and dependencies |
| [docs/CODEMAPS/MODULES.md](docs/CODEMAPS/MODULES.md) | Per-module API documentation |
| [docs/CODEMAPS/INFERENCE_FRAMEWORK.md](docs/CODEMAPS/INFERENCE_FRAMEWORK.md) | Multi-GPU offline inference architecture |
| [configs/experiments/](configs/experiments/) | Experiment configurations |
| [docs/README.md](docs/README.md) | Documentation index, literature notes, and source papers |

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
