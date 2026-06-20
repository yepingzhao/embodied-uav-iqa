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
                │ lightning_module   │   │  data_module.py  │
                │  .py               │   │  UAVIQDataModule │
                │ UAVIQALightning    │   └────────┬─────────┘
                │ Module             │            │ owns
                └────────┬───────────┘            ▼
                         │ uses         ┌─────────────────┐
                         ▼              │   dataset.py    │
                ┌───────────────┐       │ UAVIQADataset   │
                │  metrics.py   │       │ validate_manifest│
                │ evaluate_iqa  │       └────────┬────────┘
                │ per_task      │                │ uses
                │ per_distortion│                ▼
                │ per_category  │       ┌─────────────────┐
                └───────────────┘       │    utils.py     │
                         ▲             │ find_images      │
                         │             │ load_image_tensor│
                ┌────────┴────────┐    │ count_parameters │
                │  callbacks.py   │    │ split_samples    │
                │ SetupRunCallback│    │ load_manifest    │
                │ CurriculumStage │    │ write_manifest   │
                │ MetricsHistory  │    │ load_task_map    │
                │ ResultsSaving   │    │ setup_logging    │
                └────────┬───────┘    └──────────────────┘
                         │ added by          ▲
                         ▼                   │
                ┌───────────────────────────────────────────┐
                │  scripts/train.py (Lightning CLI — no custom CLI)  │
                └───────────────────────────────────────────┘

distortion.py (standalone)
  → UAV_DISTORTION_NAMES (exported, used by data_module & lightning_module)

annotations.py
  → parse_distortion_key, compute_synthetic_score (NEW)
  → build_ref_score_lookup, degradation_factor, synthetic_ref_scores
  → assign_task_label, parse_quality_score, parse_usability

data_synthesis.py (NEW — 769 lines)
  │
  ├── DatasetFormat (ABC + registry)
  │     ├── AirCopBenchFormat
  │     └── GenericImageDirFormat
  ├── DataSynthesisPipeline (orchestrates 4 steps)
  └── create_pipeline (factory)
       │
       ├── uses → distortion.py (UAVDistortionPipeline)
       ├── uses → annotations.py (build_ref_score_lookup, parse_distortion_key, etc.)
       └── uses → utils.py (find_images, split_samples, write_manifest)
```

---

## Module: `distortion.py`

**Purpose:** 24 distortion models (6 UAV-specific + 18 generic) for injecting quality degradation.

**Location:** `src/uav_iqa/distortion.py`
**Lines:** 685

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

Also exports `UAV_DISTORTION_NAMES` — a `frozenset` of the 6 UAV-specific distortion names, used by `data_module.py` and `lightning_module.py` for distortion-family filtering.

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
**Lines:** 426

### Key Classes

| Class | Description |
|-------|-------------|
| `PanetFPN` | PANet-style FPN: top-down + bottom-up aggregation with lateral/smooth convs |
| `CBAM` | Convolutional Block Attention Module: channel avg-pool → MLP → sigmoid + spatial avg/max → conv7 → sigmoid |
| `FrequencyAwareBranch` | Patch FFT (32×32, stride 16) → log-polar histogram → 3-layer tiny CNN → proj to 64-dim |
| `CrossAttentionGate` | Gating: α = σ(W·[f_s, f_f]), outputs α ⊙ f_f |
| `TaskConditionedHead` | FiLM modulation: task_embed → γ, β → h' = γ ⊙ h + β → FC(128→1) → sigmoid |
| `UAVIQANet` | Top-level model: MobileNetV4-S → PANet FPN → CBAM → FAB (optional) → CrossAttnGate → TaskConditionedHead (optional). **Dynamic stage probing**: probes backbone to determine number of feature stages, selects last 3 as multi-scale output. **Regex-based freeze**: uses `re` matching to freeze stages across naming conventions (`blocks.N`, `stages.N`, `stages_N`) — works with MobileNetV4, EfficientViT, MobileViT, and generic timm backbones. |

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
- `timm` — Backbone (dynamically probed for feature stage count)
- `math` — log-polar coordinate computation
- `re` — regex-based backbone stage freezing (backbone-agnostic naming support)

---

## Module: `dataset.py`

**Purpose:** UAVIQADataset — loads manifest.json with image paths, task IDs, and annotation scores; also provides manifest validation.

**Location:** `src/uav_iqa/dataset.py`
**Lines:** 260

### Key Classes & Functions

| Class / Function | Description |
|------------------|-------------|
| `UAVIQADataset` | `torch.utils.data.Dataset`: loads images from manifest, resizes to `image_size`, normalizes to [0,1], returns dict of `{image, task_id, score, distortion, intensity, ref_id, optional_vlm/vla/execution_scores}`. Supports optional augmentation (RandomHorizontalFlip + ColorJitter). Handles corrupt images gracefully (falls back to blank tensor with warning). |
| `validate_manifest(manifest_path) -> dict` | Validates a manifest.json and returns diagnostics: `{valid, n_entries, missing_fields, unknown_tasks, missing_paths, score_stats}`. Used by `write_manifest()` in utils.py. |

### Module-Level Constants

```python
TASK_NAMES = ("tracking", "inspection", "delivery", "sar")  # ordered tuple
TASK_TO_ID = {"tracking": 0, "inspection": 1, "delivery": 2, "sar": 3}
VALID_TASKS = {"tracking", "inspection", "delivery", "sar"}
MANIFEST_REQUIRED_FIELDS = {"path", "task", "distortion", "intensity_level"}
MANIFEST_OPTIONAL_FIELDS = {"ref_id", "ref_path", "original", "vlm_score", "vla_score", "execution_score", "annotated", "degradation_types"}
```

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
- `numpy`
- `PIL.Image` (via utils.py)

---

## Module: `losses.py`

**Purpose:** Custom loss functions for IQA ranking and cross-task regularization (replaces former `trainer.py`).

**Location:** `src/uav_iqa/losses.py`
**Lines:** 30

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

## Module: `metrics.py`

**Purpose:** IQA evaluation metrics.

**Location:** `src/uav_iqa/metrics.py`
**Lines:** 150

### Public Functions

| Function | Description |
|----------|-------------|
| `compute_srcc(pred, target)` | Spearman Rank Correlation Coefficient |
| `compute_plcc(pred, target)` | Pearson Linear Correlation Coefficient |
| `compute_rmse(pred, target)` | Root Mean Square Error |
| `compute_kendall_tau(pred, target)` | Kendall's τ rank correlation |
| `evaluate_iqa(pred, target) -> dict` | All 4 metrics with lowercase keys: `{srcc, plcc, rmse, kendall_tau}` |
| `per_task_metrics(pred, target, task_ids) -> dict` | Per-task breakdown of srcc/plcc/rmse with `n` count |
| `per_distortion_metrics(pred, target, labels) -> dict` | Per-distortion breakdown of srcc/plcc/rmse/kendall_tau |
| `per_distortion_category_metrics(pred, target, labels) -> dict` | **NEW** — aggregate metrics for UAV-specific vs generic categories: `{UAV: {...}, Generic: {...}}` |

### Key Details

- All functions return lowercase keys (`srcc`, `plcc`, `rmse`, `kendall_tau`)
- `per_task_metrics` accepts int task IDs or string task names
- `per_distortion_category_metrics` auto-detects the 6 UAV distortion names
- All metrics return `0.0` with `n` count when `n < 3` samples

### Dependencies

- `numpy`
- `scipy.stats` (spearmanr, pearsonr, kendalltau)

---

## Module: `lightning_module.py`

**Purpose:** LightningModule wrapping UAVIQANet with training/val/test logic, per-distortion ListMLE ranking, and feature reuse (forward_features).

**Location:** `src/uav_iqa/lightning_module.py`
**Lines:** 280

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

## Module: `data_module.py`

**Purpose:** LightningDataModule wrapping UAVIQADataset with manifest filtering.

**Location:** `src/uav_iqa/data_module.py`
**Lines:** 157

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
**Lines:** 256

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

## Module: `annotations.py`

**Purpose:** AirCopBench human annotation parsing, degradation factor computation, and synthetic score generation for manifest annotation.

**Location:** `src/uav_iqa/annotations.py`
**Lines:** 367

### Key Functions

| Function | Description |
|----------|-------------|
| `parse_distortion_key(key) -> (str, float)` | **NEW** — Parse `'gaussian_blur_L04'` → `('gaussian_blur', 0.4)`. Handles `__` separator, nanme `_L4` / `_L04` / `_L0_5` formats. |
| `parse_quality_score(quality_str)` | Parse `'Good (4/5)'` → 0.8, `'Excellent (5/5)'` → 1.0, etc. |
| `parse_usability(usability_str)` | Parse `'1 (Available)'` → 1.0, `'3 (Unavailable)'` → 0.25 |
| `degradation_factor(distortion, task)` | Get degradation at max intensity for (distortion, task) pair |
| `compute_synthetic_score(distortion, task, intensity)` | **NEW** — Noiseless degradation-model score: `score = 1.0 * (1.0 - alpha * intensity)`. Used by C2 correlation validation. |
| `build_ref_score_lookup(aircopbench_dir)` | Walk AirCopBench Annotations dirs → dict of `{ref_id: {vlm_score, vla_score, execution_score, annotated}}` |
| `assign_task_label(img_name, task_map)` | Extract task label from path (scene_001→tracking, etc.), with configurable `task_map` override |
| `synthetic_ref_scores(ref_id)` | Deterministic synthetic scores via MD5 hash for refs without annotations |

### DEGRADATION_FACTORS Table

248-entry dict mapping 31 distortion names × 4 tasks to degradation coefficients, plus utility keys. Example:

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

**Purpose:** Utility helpers — image I/O, manifest management, logging setup, parameter counting.

**Location:** `src/uav_iqa/utils.py`
**Lines:** 127

### Functions

| Function | Description |
|----------|-------------|
| `setup_logging(name, level)` | **NEW** — Configure stdlib logging with uniform format for scripts |
| `find_images(image_dir, exts, exclude_dirs)` | **NEW** — Recursively find image files, optionally excluding subdirectories |
| `load_image_tensor(path, image_size)` | **NEW** — Load image as `(C, H, W)` float32 tensor in [0,1]; used by both Dataset and benchmark scripts |
| `count_parameters(model) -> (total, trainable)` | Count total and trainable parameters |
| `load_task_map(path)` | **NEW** — Load JSON task-map file (used by data_synthesis.py) |
| `split_samples(samples, ratios, seed)` | **NEW** — Shuffle + split into `{train, val, test}` dicts via numpy.RandomState |
| `load_manifest(manifest_path)` | **NEW** — Load manifest.json entries |
| `write_manifest(entries, manifest_path)` | **NEW** — Write manifest JSON with validation via `validate_manifest()` |

### Dependencies

- `torch`
- `numpy`
- `PIL.Image`

---

## Module: `data_synthesis.py`

**Purpose:** Dataset-agnostic data synthesis pipeline — extract references, inject distortions, generate manifests, annotate scores. Supports AirCopBench and generic image directories.

**Location:** `src/uav_iqa/data_synthesis.py`
**Lines:** 769

### Key Classes

| Class | Description |
|-------|-------------|
| `DatasetFormat` | Abstract base class with explicit `_registry` for dataset-specific logic. Subclasses register via `@DatasetFormat.register`. |
| `AirCopBenchFormat` | Handles AirCopBench nested scene/UAV directory structure + Annotation/*.json parsing. Includes `build_image_index()`, `build_annotation_summary()`, `get_degradation_types()`. |
| `GenericImageDirFormat` | Flat directory of images, no annotations, hash-based task assignment. |
| `DataSynthesisPipeline` | Orchestrates the 4-step pipeline: `extract_references()` → `inject_distortions()` → `generate_manifests()` → `annotate_scores()`. Also provides `run_full()` for end-to-end execution. |
| `create_pipeline(dataset, seed)` | Convenience factory — creates a DataSynthesisPipeline for a named dataset format. |

### Pipeline Steps

| Step | Method | Description |
|------|--------|-------------|
| 1 | `extract_references()` | Symlinks or copies clean reference frames to flat directory |
| 2 | `inject_distortions()` | Applies all 24 distortions × 5 intensities via parallel workers |
| 3 | `generate_manifests()` | Scans distorted directory, parses filenames, builds entries, splits train/val/test |
| 4 | `annotate_scores()` | Assigns VLM/VLA/execution scores using real annotations or synthetic fallback + degradation model |

### Dependencies

- `numpy`
- `uav_iqa.distortion` — `UAVDistortionPipeline`
- `uav_iqa.annotations` — `build_ref_score_lookup`, `parse_distortion_key`, etc.
- `uav_iqa.utils` — `find_images`, `split_samples`, `write_manifest`

---

## Module: `__init__.py`

**Purpose:** Public API exports.

**Location:** `src/uav_iqa/__init__.py`
**Lines:** 83

### Exports (35 total)

```python
__all__ = [
    # Distortion models (7)
    "UAVDistortionPipeline",
    "PropellerVibrationBlur", "AtmosphericScatteringHaze",
    "SixDoFViewpointBlur", "CommunicationPacketLoss",
    "LowResSuperResolution", "PropellerShadow",
    # Model (1)
    "UAVIQANet",
    # Dataset (2)
    "UAVIQADataset", "validate_manifest",          # ← validate_manifest added
    # Metrics (4)
    "compute_srcc", "compute_plcc", "evaluate_iqa",
    "per_distortion_category_metrics",              # ← NEW
    # Lightning wrappers (2)
    "UAVIQALightningModule", "UAVIQDataModule",
    # Losses (2)
    "ListMLELoss", "CrossTaskRegularization",
    # Annotation utilities (8)
    "parse_distortion_key", "parse_quality_score",  # ← parse_distortion_key added
    "parse_usability", "degradation_factor",
    "compute_synthetic_score",                      # ← NEW
    "build_ref_score_lookup", "assign_task_label",
    "synthetic_ref_scores",
    # Utility functions (6)
    "setup_logging", "load_task_map",               # ← NEW category
    "load_manifest", "split_samples",
    "write_manifest", "find_images",
    # Data synthesis (3)
    "DatasetFormat",                                # ← NEW category
    "DataSynthesisPipeline", "create_pipeline",
]
# Note: UAVIQACLI removed in 2026-06 refactor — use scripts/train.py + vanilla LightningCLI
```
