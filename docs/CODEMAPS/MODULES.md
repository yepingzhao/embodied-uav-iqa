# Module Codemap

**Last Updated:** 2026-06-19

## Package: `uav_iqa` (src/uav_iqa/)

### Module Dependency Graph

```
                     ┌───────────────┐
                     │   model.py    │
                     │  (UAVIQANet)  │
                     └───────┬───────┘
                             │ owns
                             ▼
              ┌──────────────────────────────┐
              │       trainer.py             │
              │  ListMLELoss                 │
              │  CrossTaskRegularization     │
              └──────────────────────────────┘
                      ▲ uses                  ▲ uses
                      │                        │
              ┌───────┴────────────┐   ┌───────┴──────────┐
              │ lightning_model.py │   │  lightning_data.py│
              │ UAVIQALightning    │   │  UAVIQDataModule  │
              │ Module             │   └────────┬──────────┘
              └────────┬───────────┘            │ owns
                       │ uses                   ▼
                       ▼              ┌─────────────────┐
              ┌───────────────┐       │   dataset.py    │
              │ evaluate.py   │       │ UAVIQADataset   │
              │ SRCC, PLCC,   │       └─────────────────┘
              │ RMSE, Kend-τ  │
              └───────────────┘
                       ▲
                       │
              ┌────────┴────────┐
              │    callbacks    │
              │    .py          │
              │ CurriculumStage │
              │ MetricsHistory  │
              └─────────────────┘
                       ▲ added by
                       │
              ┌────────────────┐
              │    cli.py      │
              │  UAVIQACLI     │
              └────────────────┘
                       │ uses
                       ▼
              ┌────────────────┐
              │   utils.py     │
              │ count_params   │
              └────────────────┘

distortion.py (standalone, no internal deps)
  UAVDistortionPipeline ── uses ── PropellerVibrationBlur
                                   AtmosphericScatteringHaze
                                   SixDoFViewpointBlur
                                   CommunicationPacketLoss
                                   LowResSuperResolution
                                   PropellerShadow
                                   GenericDistortions (18 types)
```

---

## Module: `distortion.py`

**Purpose:** 24 distortion models (6 UAV-specific + 18 generic) for injecting quality degradation.

**Location:** `src/uav_iqa/distortion.py`
**Lines:** 665

### Key Classes

| Class | Description |
|-------|-------------|
| `BaseDistortion` | Abstract base: `apply(image, intensity) -> ndarray`, `get_param_range(intensity) -> dict` |
| `PropellerVibrationBlur` | Directional motion blur + periodic intensity modulation (f ∈ [80,200] Hz, A ∈ [1,8] px) |
| `AtmosphericScatteringHaze` | Koschmieder scattering model with depth estimation from luminance |
| `SixDoFViewpointBlur` | Motion blur from MotionScape empirical flow distribution (μ=36.63 px, σ=25.4 px) |
| `CommunicationPacketLoss` | 16×16 macroblock replacement with nearest-valid-block inpainting |
| `LowResSuperResolution` | Bicubic downsample + Real-ESRGAN (optional) or bicubic+sharpen fallback |
| `PropellerShadow` | Periodic brightness modulation with spatially localized mask |
| `GenericDistortions` | 18 types via Albumentations: blur(3), brightness(5), chromatic(3), noise(4), compression(3), spatial(4), other(4) |
| `UAVDistortionPipeline` | Unified orchestrator — applies all 24 × 5 intensity levels; supports parallel batch injection via `ProcessPoolExecutor` |

### Public Exports

```python
from .distortion import (
    UAVDistortionPipeline,
    PropellerVibrationBlur,
    AtmosphericScatteringHaze,
    SixDoFViewpointBlur,
    CommunicationPacketLoss,
    LowResSuperResolution,
    PropellerShadow,
)
```

### Dependencies

- `cv2` (opencv-python) — image I/O, filtering, resizing
- `numpy` — array operations
- `scipy.signal.convolve2d` — blur kernels
- `albumentations` — generic distortion transforms (18 types)
- `basicsr` + `realesrgan` — (optional) super-resolution

---

## Module: `model.py`

**Purpose:** UAVIQANet — frequency-aware task-conditioned lightweight NR-IQA model (~5.4M params).

**Location:** `src/uav_iqa/model.py`
**Lines:** 379

### Key Classes

| Class | Description |
|-------|-------------|
| `PanetFPN` | PANet-style FPN: top-down + bottom-up aggregation with lateral/smooth convs |
| `CBAM` | Convolutional Block Attention Module: channel avg-pool → MLP → sigmoid + spatial avg/max → conv7 → sigmoid |
| `FrequencyAwareBranch` | Patch FFT (32×32, stride 16) → log-polar histogram → 3-layer tiny CNN → proj to 64-dim |
| `CrossAttentionGate` | Gating: α = σ(W·[f_s, f_f]), outputs α ⊙ f_f |
| `TaskConditionedHead` | FiLM modulation: task_embed → γ, β → h' = γ ⊙ h + β → FC(128→1) → sigmoid |
| `UAVIQANet` | Top-level model: MobileNetV4-S → PANet FPN → CBAM → FAB (optional) → CrossAttnGate → TaskConditionedHead (optional) |

### TASK_MAP

```python
{"tracking": 0, "inspection": 1, "delivery": 2, "sar": 3}
```

### Forward Signatures

```python
def forward(self, x: Tensor, task_ids: Optional[Tensor] = None) -> Tensor
    # Returns (B,) quality scores
    # task_ids defaults to all-zeros if None (no task conditioning)

def forward_all_tasks(self, x: Tensor) -> Tensor
    # Returns (B, 4) scores — one per task
    # Requires use_task_conditioning=True
```

### Ablation Toggles

| Flag | Effect |
|------|--------|
| `use_fab=False` | Removes FrequencyAwareBranch; fused dim = 256 (spatial only) |
| `use_cbam=False` | Removes CBAM after FPN projection |
| `use_task_conditioning=False` | Replaces FiLM head with shared FC(320→128→1) MLP |

### Dependencies

- `torch`, `torch.nn`, `torch.nn.functional`
- `timm` — MobileNetV4-S backbone
- `math` — log-polar coordinate computation

---

## Module: `dataset.py`

**Purpose:** UAVIQADataset — loads manifest.json with image paths, task IDs, and annotation scores.

**Location:** `src/uav_iqa/dataset.py`
**Lines:** 134

### Key Class

| Class | Description |
|-------|-------------|
| `UAVIQADataset` | `torch.utils.data.Dataset`: loads images from manifest, resizes to `image_size`, normalizes to [0,1], returns dict of `{image, task_id, score, distortion, intensity, ref_id, optional_vlm/vla/execution_scores}` |

### Manifest Schema

```json
{
  "path": "distorted/ref001__propeller_vibration_blur_L04.png",
  "task": "tracking",
  "distortion": "propeller_vibration_blur",
  "intensity_level": 0.4,
  "ref_id": 1,
  "vlm_score": 0.45,
  "vla_score": 0.52,
  "execution_score": 0.38,
  "annotated": true
}
```

### Key Methods

| Method | Description |
|--------|-------------|
| `collate_fn(batch) -> dict` | Static method: stacks images, task_ids, scores; keeps distortion/intensity/ref_id as lists |
| `create_dataloader(...)` | Static factory for `DataLoader` with pin_memory and collate |

### Dependencies

- `torch`, `torch.utils.data`
- `PIL.Image`
- `numpy`

---

## Module: `trainer.py`

**Purpose:** Custom loss functions for IQA ranking and cross-task regularization.

**Location:** `src/uav_iqa/trainer.py`
**Lines:** 32

### Key Classes

| Class | Description |
|-------|-------------|
| `ListMLELoss` | Listwise ranking loss: sort targets descending, compute log-cumsum-exp on sorted preds, minimize negative log-likelihood |
| `CrossTaskRegularization` | Encourage differentiation: compute MSE between all pairs of task scores, negate (penalize identical scores) |

### Loss Formulas

```
ListMLE: L = -(1/N) * Σᵢ [s_pred[i] - log_cumsum(s_pred[i:])]
  where s_pred is sorted by descending target

CrossTask: L = -(2/(K*(K-1))) * Σ_{i<j} mean((scores[:,i] - scores[:,j])²)
  K = num_tasks; sign is negated to penalize similarity
```

### Dependencies

- `torch`, `torch.nn`

---

## Module: `evaluate.py`

**Purpose:** IQA evaluation metrics.

**Location:** `src/uav_iqa/evaluate.py`
**Lines:** 109

### Public Functions

| Function | Description |
|----------|-------------|
| `compute_srcc(pred, target)` | Spearman Rank Correlation Coefficient |
| `compute_plcc(pred, target)` | Pearson Linear Correlation Coefficient |
| `compute_rmse(pred, target)` | Root Mean Square Error |
| `compute_kendall_tau(pred, target)` | Kendall's τ rank correlation |
| `evaluate_iqa(pred, target) -> dict` | All 4 metrics: {SRCC, PLCC, RMSE, KendallTau} |
| `per_task_metrics(pred, target, task_ids) -> dict` | Per-task breakdown of srcc/plcc/rmse |
| `per_distortion_metrics(pred, target, labels) -> dict` | Per-distortion breakdown of SRCC/PLCC/RMSE |
| `compute_metrics(target, pred) -> dict` | Alias of evaluate_iqa with lowercase keys |

### Dependencies

- `numpy`
- `scipy.stats` (spearmanr, pearsonr, kendalltau)

---

## Module: `lightning_model.py`

**Purpose:** LightningModule wrapping UAVIQANet with training/val/test logic.

**Location:** `src/uav_iqa/lightning_model.py`
**Lines:** 224

### Key Class

| Class | Description |
|-------|-------------|
| `UAVIQALightningModule` | `L.LightningModule`: flat `__init__` params for LightningCLI compat; owns UAVIQANet, MSE, ListMLE, CrossTaskRegularization; handles training_step (composite loss), validation/test_step (metric collection), configure_optimizers (AdamW + warmup/cosine scheduler) |

### Curriculum Integration

```python
self.curriculum_stage  # Set by CurriculumStageCallback
# training_step reads batch["{curriculum_stage}_score"] or falls back to batch["score"]
```

### Test Results

```python
get_test_results() -> dict {
    "test_metrics": {"srcc": ..., "plcc": ..., "rmse": ...},
    "per_task": {"tracking": {"srcc": ..., ...}, ...},
    "per_distortion": {"propeller_vibration_blur_L04": {"SRCC": ..., ...}, ...},
    "preds": [...],
    "targets": [...],
}
```

### Dependencies

- `lightning` (pytorch-lightning)
- `torch`, `torch.nn`

---

## Module: `lightning_data.py`

**Purpose:** LightningDataModule wrapping UAVIQADataset with manifest filtering.

**Location:** `src/uav_iqa/lightning_data.py`
**Lines:** 152

### Key Class

| Class | Description |
|-------|-------------|
| `UAVIQDataModule` | `L.LightningDataModule`: loads train/val/test manifests; applies task/distortion/leave_out filters; supports dry_run subsampling |

### Filter Parameters

| Param | Effect |
|-------|--------|
| `task="tracking"` | Only samples with `task == "tracking"` |
| `val_task="sar"` | Override val/test task (defaults to `task`) |
| `distortion_filter="generic"` | Exclude UAV distortions |
| `distortion_filter="uav_only"` | Exclude generic distortions |
| `leave_out_task="sar"` | Exclude one task from train (leave-one-out) |
| `dry_run=True` | Subsampled: train=100, val=50, test=50 |

### Dependencies

- `lightning`
- `json`, `pathlib`

---

## Module: `callbacks.py`

**Purpose:** PyTorch Lightning callbacks for curriculum and history.

**Location:** `src/uav_iqa/callbacks.py`
**Lines:** 75

### Key Classes

| Class | Description |
|-------|-------------|
| `CurriculumStageCallback` | Sets `pl_module.curriculum_stage` (vlm/vla/execution) based on epoch boundaries |
| `MetricsHistoryCallback` | Records train/val metrics per epoch, saves best model by val SRCC, dumps history.json on fit end |

### Default Curriculum Schedule

| Stage | Epochs |
|-------|--------|
| VLM | 1–20 |
| VLA | 21–40 |
| Execution | 41–50 |

### Dependencies

- `lightning`
- `json`, `pathlib`

---

## Module: `cli.py`

**Purpose:** Custom LightningCLI with multi-seed support, post-fit test, and result saving.

**Location:** `src/uav_iqa/cli.py`
**Lines:** 97

### Key Class

| Class | Description |
|-------|-------------|
| `UAVIQACLI` | Extends `LightningCLI`: adds `--output_dir`, `--seed`, `--dry_run` flags; in `before_fit` creates run dir, seeds everything, adds curriculum/history/checkpoint callbacks; in `after_fit` runs best-checkpoint test, prints results, writes results.json |

### CLI Args (beyond LightningCLI defaults)

| Arg | Default | Description |
|-----|---------|-------------|
| `--output_dir` | `outputs/training` | Base directory for seed subdirs |
| `--seed` | `42` | Random seed |
| `--dry_run` | `false` | Fast verification mode |

### Dependencies

- `lightning.pytorch.cli.LightningCLI`

---

## Module: `utils.py`

**Purpose:** Utility helpers.

**Location:** `src/uav_iqa/utils.py`
**Lines:** 4

### Functions

| Function | Description |
|----------|-------------|
| `count_parameters(model) -> (total, trainable)` | Count total and trainable parameters |

---

## Module: `__init__.py`

**Purpose:** Public API exports.

**Location:** `src/uav_iqa/__init__.py`

### Exports (17 total)

```python
__all__ = [
    # Distortion models
    "UAVDistortionPipeline",
    "PropellerVibrationBlur", "AtmosphericScatteringHaze",
    "SixDoFViewpointBlur", "CommunicationPacketLoss",
    "LowResSuperResolution", "PropellerShadow",
    # Model
    "UAVIQANet",
    # Dataset
    "UAVIQADataset",
    # Metrics
    "compute_srcc", "compute_plcc", "evaluate_iqa",
    # Lightning wrappers
    "UAVIQALightningModule", "UAVIQDataModule",
    # CLI
    "UAVIQACLI",
]
```
