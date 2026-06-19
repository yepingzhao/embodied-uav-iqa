# Module Codemap

**Last Updated:** 2026-06-20

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
               │        losses.py             │
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
               ┌────────┴────────────┐
               │    callbacks.py     │
               │ SetupRunCallback    │
               │ CurriculumStage     │
               │ MetricsHistory      │
               │ ResultsSaving       │
               └────────┬───────────┘
                        │ added by
                        ▼
               ┌──────────────────────┐
               │  main.py (Lightning  │
               │  CLI — no custom CLI)│
               └──────────────────────┘
                        │
               ┌────────┴────────┐
               │   utils.py      │
               │ count_params    │
               └─────────────────┘

distortion.py (standalone, no internal deps)

annotation_utils.py (standalone)
  build_ref_score_lookup
  degradation_factor
  synthetic_ref_scores
  assign_task_label
  parse_quality_score
  parse_usability
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

## Module: `losses.py`

**Purpose:** Custom loss functions for IQA ranking and cross-task regularization (replaces former `trainer.py`).

**Location:** `src/uav_iqa/losses.py`
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

**Purpose:** LightningModule wrapping UAVIQANet with training/val/test logic, per-distortion ListMLE ranking, and feature reuse (forward_features).

**Location:** `src/uav_iqa/lightning_model.py`
**Lines:** 272

### Key Class

| Class | Description |
|-------|-------------|
| `UAVIQALightningModule` | `L.LightningModule`: flat `__init__` params for LightningCLI compat; owns UAVIQANet, MSE, ListMLE, CrossTaskRegularization; handles training_step (composite loss + feature sharing via `forward_features`), validation/test_step (per-task + per-distortion metric aggregation), configure_optimizers (AdamW + warmup/cosine scheduler) |

### Key Parameters (LightningCLI-compatible `__init__`)

| Arg | Default | Description |
|-----|---------|-------------|
| `backbone` | `mobilenetv4_conv_small` | timm backbone name |
| `num_tasks` | `4` | Number of task-specific heads |
| `use_fab` | `True` | Enable FrequencyAwareBranch |
| `use_cbam` | `True` | Enable CBAM attention |
| `use_task_conditioning` | `True` | Enable FiLM task heads |
| `freeze_backbone_stage` | `2` | Freeze stages 0..N-1 |
| `lambda_rank` | `0.3` | ListMLE loss weight |
| `lambda_cross_task` | `0.1` | Cross-task regularization weight |
| `lr` | `3e-4` | AdamW learning rate |
| `total_epochs` | `50` | Total training epochs |
| `annotator_stage` | `vla` | Eval annotation source |

### Feature Sharing Optimization

```python
f = self.model.forward_features(images)  # backbone + FPN + FAB once
pred = self.model(images, task_ids, features=f)  # reuse features for head
all_task_scores = self.model.forward_all_tasks(images, features=f)  # cross-task loss
```

Backbone + FPN + FAB computed once, reused for both quality prediction and cross-task regularization — avoids redundant forward passes.

### Curriculum Integration

```python
self.curriculum_stage  # Set by CurriculumStageCallback
# training_step reads batch["{curriculum_stage}_score"] or falls back to batch["score"]
```

### Per-Distortion ListMLE Ranking

```python
# Grouped by distortion name within batch — applies ListMLE per distortion
for dist in unique_distortions:
    mask = dist_ids == dist
    if mask.sum() >= 3:  # need 3+ samples for ranking
        loss_rank += self.rank_loss(pred[mask], scores[mask])
```

### Test Results

```python
get_test_results() -> dict {
    "test_metrics": {"srcc": ..., "plcc": ..., "rmse": ...},
    "per_task": {"tracking": {"srcc": ..., "plcc": ..., "rmse": ..., "n": ...}, ...},
    "per_distortion": {"propeller_vibration_blur_L04": {"SRCC": ..., "PLCC": ..., "N": ...}, ...},
    "preds": [...],
    "targets": [...],
}
```

### Dependencies

- `lightning` (pytorch-lightning)
- `torch`, `torch.nn`
- `numpy`

---

## Module: `lightning_data.py`

**Purpose:** LightningDataModule wrapping UAVIQADataset with manifest filtering.

**Location:** `src/uav_iqa/lightning_data.py`
**Lines:** 155

### Key Class

| Class | Description |
|-------|-------------|
| `UAVIQDataModule` | `L.LightningDataModule`: loads train/val/test manifests; applies task/distortion/leave_out filters; supports dry_run subsampling |

### Key Parameters

| Param | Default | Effect |
|-------|---------|--------|
| `data_root` | `data/processed` | Base directory with `{train,val,test}/manifest.json` |
| `batch_size` | `256` | Per-device batch size |
| `num_workers` | `16` | Data loading workers |
| `image_size` | `256` | Image resize dimension |
| `annotator_stage` | `vla` | Annotation source for eval (vlm/vla/execution) |
| `task` | `None` | Filter: only samples with this task |
| `val_task` | `None` | Override val/test task (defaults to `task`) |
| `distortion_filter` | `None` | `"generic"` → exclude UAV; `"uav_only"` → exclude generic |
| `leave_out_task` | `None` | Exclude one task from train (leave-one-out) |
| `dry_run` | `False` | Subsampled: train=100, val=50, test=50 |

### Dependencies

- `lightning`
- `json`, `pathlib`

---

## Module: `callbacks.py`

**Purpose:** PyTorch Lightning callbacks for dataset verification, curriculum switching, metric history, and result saving.

**Location:** `src/uav_iqa/callbacks.py`
**Lines:** 234

### Key Classes

| Class | Description |
|-------|-------------|
| `SetupRunCallback` | At fit start: computes manifest SHA256 hash for dataset versioning, prints model param count. Exposes `pl_module.manifest_hash` |
| `CurriculumStageCallback` | Sets `pl_module.curriculum_stage` (vlm/vla/execution) based on epoch boundaries |
| `MetricsHistoryCallback` | Records train/val metrics per epoch, saves `history.json` on fit end, exposes `pl_module.best_val_srcc` |
| `ResultsSavingCallback` | On fit end: loads best checkpoint, runs test, prints & saves results to `results.json` with git commit hash, hparams, data config |

### SetupRunCallback

- Computes SHA256 hash of concatenated train/val/test `manifest.json` files (first 16 chars)
- Useful for verifying dataset version consistency across experiment runs
- Exposed as `pl_module.manifest_hash`

### CurriculumStageCallback

| Stage | Epochs (default) |
|-------|------------------|
| VLM | 1–20 |
| VLA | 21–40 |
| Execution | 41–50 |

Configurable via `vlm_epochs`, `vla_epochs`, `execution_epochs` init args.

### MetricsHistoryCallback

Captured series:
- `train/loss`, `train/mse`, `train/rank`, `train/cross_task`
- `val/srcc`, `val/plcc`
- `val/srcc_{tracking|inspection|delivery|sar}` — per-task SRCC
- `val/srcc_uav`, `val/srcc_generic` — aggregated distortion family SRCC

### ResultsSavingCallback

- Finds best checkpoint from `ModelCheckpoint` (monitors `val/srcc`, mode=max)
- Runs `trainer.test()` with best checkpoint
- Saves `results.json` containing:
  - `seed`, `n_params`, `best_val_srcc`, `test_metrics` (srcc/plcc/rmse)
  - `per_task` breakdown, `hparams`, `data_config`, `manifest_hash`, `git_commit`

### Dependencies

- `lightning`
- `json`, `pathlib`, `hashlib`, `subprocess`, `yaml`

---

## Module: `annotation_utils.py`

**Purpose:** AirCopBench human annotation parsing, degradation factor computation, and synthetic score generation for manifest annotation.

**Location:** `src/uav_iqa/annotation_utils.py`
**Lines:** 155

### Key Functions

| Function | Description |
|----------|-------------|
| `parse_quality_score(quality_str)` | Parse `'Good (4/5)'` → 0.8, `'Excellent (5/5)'` → 1.0, etc. |
| `parse_usability(usability_str)` | Parse `'1 (Available)'` → 1.0, `'3 (Unavailable)'` → 0.25 |
| `degradation_factor(distortion, task)` | Get degradation at max intensity for (distortion, task) pair |
| `build_ref_score_lookup(aircopbench_dir)` | Walk AirCopBench Annotations dirs → dict of `{ref_id: {vlm_score, vla_score, execution_score, annotated}}` |
| `assign_task_label(img_name)` | Deterministic task assignment from image name MD5 hash |
| `synthetic_ref_scores(ref_id)` | Deterministic synthetic scores via MD5 hash for refs without annotations |

### DEGRADATION_FACTORS Table

155-entry dict mapping 31 distortion names × 4 tasks to degradation coefficients. Example:

```python
"propeller_vibration_blur": {
    "tracking": 0.55, "inspection": 0.40, "delivery": 0.70, "sar": 0.50
}
```

Lower values = more severe degradation. The `"none"` entry is 0.95 for all tasks.

### Score Synthesis Model

```
distorted_score = ref_score × (1 - (1 - degradation_factor) × intensity) + noise
```

Combines real AirCopBench annotations (Quality → vlm, Usability → vla, 0.4Q + 0.6U → execution) with degradation physics.

### Dependencies

- `json`, `hashlib`, `re`, `logging`, `pathlib`

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

### Exports (22 total)

```python
__all__ = [
    # Distortion models (7)
    "UAVDistortionPipeline",
    "PropellerVibrationBlur", "AtmosphericScatteringHaze",
    "SixDoFViewpointBlur", "CommunicationPacketLoss",
    "LowResSuperResolution", "PropellerShadow",
    # Model (1)
    "UAVIQANet",
    # Dataset (1)
    "UAVIQADataset",
    # Metrics (3)
    "compute_srcc", "compute_plcc", "evaluate_iqa",
    # Lightning wrappers (2)
    "UAVIQALightningModule", "UAVIQDataModule",
    # Losses (2)
    "ListMLELoss", "CrossTaskRegularization",
    # Annotation utilities (6)
    "parse_quality_score", "parse_usability",
    "degradation_factor", "build_ref_score_lookup",
    "assign_task_label", "synthetic_ref_scores",
]
# Note: UAVIQACLI removed in 2026-06 refactor — use main.py + vanilla LightningCLI
```
