# Module Codemap

**Last Updated:** 2026-07-01

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
                 │  .py               │   │  UAVIQADataModule │
                 │ UAVIQALightning    │   └────────┬─────────┘
                 │ Module             │            │ owns
                 └────────┬───────────┘            ▼
                          │ uses         ┌─────────────────┐
                          ▼              │   dataset.py    │
                 ┌───────────────┐       │ UAVIQADataset   │
                 │  metrics.py   │       │ (flat JSON)     │
                 │ evaluate_iqa  │       └────────┬────────┘
                 │ per_task      │                │ uses
                 │ per_distortion│                ▼
                 │ per_category  │       ┌─────────────────┐
                 └───────────────┘       │  annotations.py │
                          ▲             │ (SUBTASK_NAME_TO_ID, 
                 ┌────────┴────────┐    │  parse_distortion_key,
                 │  callbacks.py   │    │  etc.)
                 │ SetupRunCallback│    └────────┬────────┘
                 │ MetricsHistory  │             │ uses
                 │ ResultsSaving   │             ▼
                 └────────┬───────┘    ┌─────────────────┐
                          │            │    utils.py     │
                          ▼            │ find_images      │
                 ┌────────────────┐    │ load_image_tensor│
                  │ scripts/train  │    │ count_parameters │
                  │ .py (Lightning │    │ split_samples   │
                  │  CLI — no      │    │ load_flat_samples│
                  │  custom CLI)   │    │ load_task_map   │
                  └────────────────┘    └─────────────────┘
                                       └─────────────────┘

distortion.py (standalone)
  → UAVDistortionPipeline, UAV_DISTORTION_NAMES (used by data_module & lightning_module)

annotations.py
  → parse_distortion_key, parse_quality_score, parse_usability
  → build_ref_score_lookup, assign_task_label
  → SUBTASK_NAMES, SUBTASK_TO_ID, SUBTASK_NAME_LIST, SUBTASK_NAME_TO_ID, NUM_SUBTASKS
  → seed_for_distortion, group_by_scene_frame, build_vqa_split_lookup, normalize_subtask_type
  → extract_subtask_type, extract_subtask_id, extract_uav_id_from_question_id, build_sample_id, get_dataset_name

data_synthesis.py (769 lines)
  │
  ├── DatasetFormat (ABC + registry)
  │     ├── AirCopBenchFormat
  │     └── GenericImageDirFormat
  ├── DataSynthesisPipeline (orchestrates 4 steps: extract → inject → annotate → aggregate)
  └── create_pipeline (factory)
       │
       ├── uses → distortion.py (UAVDistortionPipeline)
       ├── uses → annotations.py (build_ref_score_lookup, parse_distortion_key, etc.)
        └── uses → utils.py (find_images, split_samples)

vla_scorer.py (standalone, no internal deps)
  │
  ├── BaseScorer (ABC)
  └── extract_ref_id, task constants

vlm/ (subpackage)
  │
  ├── config.py → VLMConfig (dataclass), MODEL_REGISTRY (dict of 15 models, 6 families)
  ├── vqa_index.py → VQAIndex (AirCopBench VQA question index)
  └── scorer.py → VLMScorer (vLLM / transformers backend)
       │
       └── imports → vla_scorer.py (BaseScorer, extract_ref_id)

batch_annotator.py
  │
  └── uses → vla_scorer.py (BaseScorer)
       │
       ├── BatchAnnotator (orchestrates manifest scoring with checkpoint/resume)
       └── SPLIT_NAMES = ("train", "test")

text_metrics.py (standalone)
  → compute_bleu, compute_rouge_l, compute_cider, compute_cognitive_score
  → used by vlm/scorer.py for text similarity scoring

```

*(Note: `vlm_vla_scorer.py` backward-compat shim was removed — its contents are now split into `vlm/` subpackage, `vla_scorer.py`, `text_metrics.py`, and `batch_annotator.py`.)*

---

## Module: `distortion.py`

**Purpose:** 36 distortion models (6 UAV-specific + 30 generic) for injecting quality degradation.

**Location:** `src/uav_iqa/distortion.py`
**Lines:** 735

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
| `UAVDistortionPipeline` | Unified orchestrator — applies all 36 × 1 random intensity level; supports parallel batch injection via `ProcessPoolExecutor`. Caches distortion instances in `_dist_cache` for reuse. Skips existing output files on re-run. Uses `cv2.imencode` + `write_bytes` for safer file writing. |

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

- `cv2` (opencv-python) — image I/O, filtering (incl. `cv2.filter2D` with manual wrap padding), resizing
- `numpy` — array operations
- `tqdm` — progress bars for batch injection
- `albumentations` — generic distortion transforms (18 types)
- `basicsr` + `realesrgan` — (optional) super-resolution

---

## Module: `model.py`

**Purpose:** UAVIQANet — frequency-aware task-conditioned lightweight NR-IQA model (~5.4M params).

**Location:** `src/uav_iqa/model.py`
**Lines:** 438

### Key Classes

| Class | Description |
|-------|-------------|
| `PanetFPN` | PANet-style FPN: top-down + bottom-up aggregation with lateral/smooth convs |
| `CBAM` | Convolutional Block Attention Module: channel avg-pool → MLP → sigmoid + spatial avg/max → conv7 → sigmoid |
| `FrequencyAwareBranch` | Patch FFT (32×32, stride 16) → log-polar histogram → 3-layer tiny CNN → proj to 64-dim |
| `CrossAttentionGate` | Gating: α = σ(W·[f_s, f_f]), outputs α ⊙ f_f |
| `TaskConditionedHead` | FiLM modulation: task_embed → γ, β → h' = γ ⊙ h + β → FC(128→1) → sigmoid |
| `UAVIQANet` | Top-level model: MobileNetV4-S → PANet FPN → CBAM → FAB (optional) → CrossAttnGate → TaskConditionedHead (optional). **Dynamic stage probing**: probes backbone to determine number of feature stages, selects last 3 as multi-scale output. **Regex-based freeze**: uses `re` matching to freeze stages across naming conventions (`blocks.N`, `stages.N`, `stages_N`) — works with MobileNetV4, EfficientViT, MobileViT, and generic timm backbones. |

### Subtask Constants

Imported at runtime from `annotations.py` (via `lazy_import` to avoid circular deps):

```python
SUBTASK_NAMES = {
    "1.1": "scene_description", "1.2": "scene_comparison", "1.3": "observing_posture",
    "2.1": "object_recognition", "2.2": "object_counting", "2.3": "object_grounding", "2.4": "object_matching",
    "3.1": "quality_assessment", "3.2": "usability_assessment", "3.3": "causal_assessment",
    "4.1": "when_to_collaborate", "4.2": "what_to_collaborate", "4.3": "who_to_collaborate", "4.4": "why_to_collaborate",
}
SUBTASK_NAME_TO_ID = {name: i for i, name in enumerate(SUBTASK_NAMES.values())}  # 14 subtasks
NUM_SUBTASKS = 14
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

**Purpose:** UAVIQADataset — loads flat processed VQA JSONs (`*_VQA_*.json`) with multi-UAV images, subtask types, and cognitive scores; also provides manifest validation.

**Location:** `src/uav_iqa/dataset.py`
**Lines:** 169

### Key Classes & Functions

| Class / Function | Description |
|------------------|-------------|
| `UAVIQADataset` | `torch.utils.data.Dataset`: loads flat processed JSONs (one entry per question×distortion pair), collates multi-UAV images per sample, resizes to `image_size`, normalizes to [0,1], returns dict of `{images, task_id, score, sample_id, distortion, num_uavs}`. Supports optional augmentation (RandomHorizontalFlip + ColorJitter). Handles corrupt images gracefully (falls back to blank tensor with warning). |
| `validate_manifest(manifest_path) -> dict` | Validates a flat manifest JSON and returns diagnostics: `{valid, count, warnings}`. |

### Module-Level Constants

No module-level subtask constants — subtask IDs are resolved via `SUBTASK_NAME_TO_ID` from `annotations.py`.

### Flat JSON Entry Format

```json
[{
  "uav_paths": {"FRONT": "ref/frame_010/uav0.png", "LEFT": "ref/frame_010/uav1.png"},
  "uav_keys": ["FRONT", "LEFT"],
  "distorted_uav_paths": {"FRONT": "distorted/uav0__propeller_vibration_blur_L04.png", "LEFT": "distorted/uav1__propeller_vibration_blur_L04.png"},
  "distortion_info": {"type": "propeller_vibration_blur", "category": "uav", "intensity": 0.4},
  "sample_id": "ref001__propeller_vibration_blur_L04",
  "subtask_type": "scene_description",
  "cognitive_score": 0.85
}]
```

### Key Methods

| Method | Description |
|--------|-------------|
| `collate_fn(batch) -> dict` | Static method: pads multi-UAV image sequences, stacks task_ids/scores, keeps sample_id/distortion/num_uavs as lists |

### Dependencies

- `torch`, `torch.utils.data`
- `uav_iqa.annotations` for `SUBTASK_NAME_TO_ID`
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
**Lines:** 145

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
| `per_distortion_category_metrics(pred, target, labels) -> dict` | Aggregate metrics for UAV-specific vs generic categories: `{UAV: {...}, Generic: {...}}` |

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
**Lines:** 282

### Key Class

| Class | Description |
|-------|-------------|
| `UAVIQALightningModule` | `L.LightningModule`: flat `__init__` params for LightningCLI compat; owns UAVIQANet, MSE, ListMLE, CrossTaskRegularization; handles training_step (composite loss + feature sharing via `forward_features`), validation/test_step (per-task + per-distortion metric aggregation), configure_optimizers (AdamW + warmup/cosine scheduler) |

### Key Parameters (LightningCLI-compatible `__init__`)

| Arg | Default | Description |
|-----|---------|-------------|
| `backbone` | `mobilenetv4_conv_small` | timm backbone name |
| `num_tasks` | `14` | Number of subtask-specific heads |
| `use_fab` | `True` | Enable FrequencyAwareBranch |
| `use_cbam` | `True` | Enable CBAM attention |
| `use_task_conditioning` | `True` | Enable FiLM task heads |
| `freeze_backbone_stage` | `2` | Freeze stages 0..N-1 |
| `lambda_rank` | `0.3` | ListMLE loss weight |
| `lambda_cross_task` | `0.1` | Cross-task regularization weight |
| `lr` | `3e-4` | AdamW learning rate |
| `weight_decay` | `1e-4` | AdamW weight decay |
| `warmup_epochs` | `5` | Linear warmup epochs |
| `total_epochs` | `50` | Total training epochs |

### Feature Sharing Optimization

```python
f = self.model.forward_features(images)  # backbone + FPN + FAB once
pred = self.model(images, task_ids, features=f)  # reuse features for head
all_task_scores = self.model.forward_all_tasks(images, features=f)  # cross-task loss
```

Backbone + FPN + FAB computed once, reused for both quality prediction and cross-task regularization — avoids redundant forward passes.

### Score Supervision

```python
# training_step reads batch["score"] (alias for "cognitive_score")
scores = batch.get("score", batch.get("cognitive_score"))
```

No staged curriculum — the model is supervised on a single unified cognitive_score from VLM multi-image aggregation.

### Per-Distortion ListMLE Ranking

```python
# Grouped by distortion name within batch — applies ListMLE per distortion
unique_dists = list(set(distortions))
dist_to_idx = {d: i for i, d in enumerate(unique_dists)}
dist_ids = torch.tensor([dist_to_idx[d] for d in distortions], device=self.device)
for idx, dist in enumerate(unique_dists):
    mask = dist_ids == idx
    if mask.sum() >= 3:  # need 3+ samples for ranking
        loss_rank += self.rank_loss(pred[mask], scores[mask])
```

### Test Results (DDP-compatible)

DDP gathering uses `_gather_tensor()` and `_gather_objects()` to aggregate predictions across all processes:

- Metrics: `test/srcc`, `test/plcc` (overall)
- Per-subtask: `test/srcc_{subtask_name}` for all 14 subtasks
- Per-distortion category: `test/srcc_uav` (avg of 6 UAV distortions), `test/srcc_generic` (avg of 30 generic distortions)
- All stored via `self.log()` → CSVLogger/metrics.csv + WandbLogger

### Dependencies

- `lightning` (pytorch-lightning)
- `torch`, `torch.nn`
- `numpy`

---

## Module: `data_module.py`

**Purpose:** LightningDataModule wrapping UAVIQADataset with flat VQA JSON loading and subtask/distortion filtering.

**Location:** `src/uav_iqa/data_module.py`
**Lines:** 124

### Key Class

| Class | Description |
|-------|-------------|
| `UAVIQADataModule` | `L.LightningDataModule`: loads train/test VQA JSONs; applies subtask/distortion filters; supports dry_run subsampling and multi-UAV max count |

### Key Parameters

| Param | Default | Effect |
|-------|---------|--------|
| `data_root` | `data/processed` | Base directory with `{train,test}/*_VQA_*.json` |
| `batch_size` | `64` | Per-device batch size |
| `num_workers` | `4` | Data loading workers |
| `image_size` | `256` | Image resize dimension |
| `max_uavs` | `6` | Maximum number of UAV images per sample |
| `subtask_filter` | `None` | Comma-separated subtask types to filter |
| `distortion_filter` | `None` | `"generic"` → exclude UAV; `"uav_only"` → exclude generic |
| `dry_run` | `False` | Subsampled: train=100, val=50, test=50 |

### Dependencies

- `lightning`
- `torch.utils.data`
- `distortion.py` (UAVDistortionPipeline for filter logic)

---

## Module: `callbacks.py`

**Purpose:** PyTorch Lightning callbacks for dataset verification, training metrics tracking, and automated test result saving.

**Location:** `src/uav_iqa/callbacks.py`
**Lines:** 206

### Key Classes

| Class | Description |
|-------|-------------|
| `CurriculumStageCallback` | **DEPRECATED** — no-op. The 3-stage curriculum (VLM→VLA→Execution) has been superseded by single cognitive_score supervision. Kept for backward compatibility. |
| `SetupRunCallback` | At fit start: computes VQA JSON SHA256 hash for dataset versioning, prints model param count. Exposes `pl_module.dataset_hash`. DDP-safe via `trainer.is_global_zero`. |
| `MetricsHistoryCallback` | Tracks epoch-level metrics (`train/loss_epoch`, `val/srcc`, `val/plcc`, `val/rmse`, `val/krcc`) across validation epochs, records best val SRCC, writes `history.json` on fit end. |
| `ResultsSavingCallback` | On fit end: loads best checkpoint, runs test, saves structured `results.json` with test metrics, per-subtask SRCC, model params, git commit hash, seed, and config. |

### SetupRunCallback

- Computes SHA256 hash of train/test VQA JSON files (first 16 chars)
- Useful for verifying dataset version consistency across experiment runs
- Exposed as `pl_module.dataset_hash`
- Prints only on global rank 0 (`trainer.is_global_zero`)

### MetricsHistoryCallback

- Tracks scalar metrics per validation epoch via `on_validation_epoch_end`
- Writes `history.json` to run directory on fit end
- Exposes `pl_module.best_val_srcc` tracked across epochs

### ResultsSavingCallback

- On fit end: finds best checkpoint, runs `trainer.test()` with it
- Saves `results.json` containing:
  - `seed` (read from config.yaml)
  - `n_params`, `best_val_srcc`
  - `test_metrics`: srcc, plcc, rmse
  - `per_task`: per-subtask SRCC
  - `hparams`, `data_config`, `dataset_hash`
  - `git_commit`: short SHA via `git rev-parse`
- Only runs on global rank 0

### Dependencies

- `lightning`
- `pathlib`, `hashlib`, `json`, `logging`, `subprocess`
- `utils.py` (`count_parameters`)
- `yaml` (for reading config.yaml)

---

## Module: `annotations.py`

**Purpose:** AirCopBench human annotation parsing — distortion key extraction, Quality/Usability score parsing, reference score lookup, and task label assignment from path structure.

**Location:** `src/uav_iqa/annotations.py`
**Lines:** 402

### Key Functions

| Function | Description |
|----------|-------------|
| `parse_distortion_key(key) -> (str, float)` | Parse `'gaussian_blur_L04'` → `('gaussian_blur', 0.4)`. Handles `__` separator, `_L4` / `_L04` / `_L0_5` formats. |
| `parse_quality_score(quality_str)` | Parse `'Good (4/5)'` → 0.8, `'Excellent (5/5)'` → 1.0, etc. |
| `parse_usability(usability_str)` | Parse `'1 (Available)'` → 1.0, `'3 (Unavailable)'` → 0.25 |
| `build_ref_score_lookup(aircopbench_dir)` | Walk AirCopBench Annotations dirs → dict of `{ref_id: {cognitive_score, annotated}}` |
| `assign_task_label(img_name, task_map)` | Extract subtask label from path (scene_001→scene_description, etc.), with configurable `task_map` override |

### Score Construction

Reference scores are built from AirCopBench human annotations:
- `Quality` → `cognitive_score` (via `parse_quality_score`)

Unannotated references fall back to deterministic scoring in `vla_scorer.py`.

*(Note: `degradation_factor`, `compute_synthetic_score`, `synthetic_ref_scores`, and `SyntheticScorer` were removed in the 2026-06 refactor. Score synthesis lives in `data_synthesis.py` `DataSynthesisPipeline`.)*

### Dependencies

- `json`, `hashlib`, `re`, `logging`, `pathlib`

---

## Module: `utils.py`

**Purpose:** Utility helpers — image I/O, manifest management, logging setup, parameter counting.

**Location:** `src/uav_iqa/utils.py`
**Lines:** 179

### Functions

| Function | Description |
|----------|-------------|
| `setup_logging(name, level)` | Configure stdlib logging with uniform format for scripts |
| `find_images(image_dir, exts, exclude_dirs)` | Recursively find image files, optionally excluding subdirectories |
| `load_image_tensor(path, image_size)` | Load image as `(C, H, W)` float32 tensor in [0,1]; used by both Dataset and benchmark scripts |
| `count_parameters(model) -> (total, trainable)` | Count total and trainable parameters |
| `load_task_map(path)` | Load JSON task-map file (used by data_synthesis.py) |
| `split_samples(samples, ratios, seed)` | Shuffle + split into `{train, test}` dicts via numpy.RandomState |
| `load_flat_samples(output_dir, split)` | Load processed flat JSON entries into manifest-compatible format for benchmarking |

### Dependencies

- `torch`
- `numpy`
- `PIL.Image`

---

## Module: `data_synthesis.py`

**Purpose:** Dataset-agnostic data synthesis pipeline — extract references, inject distortions, annotate scores with VLMs, aggregate cognitive scores. Supports AirCopBench and generic image directories.

**Location:** `src/uav_iqa/data_synthesis.py`
**Lines:** 769

### Key Classes

| Class | Description |
|-------|-------------|
| `DatasetFormat` | Abstract base class with explicit `_registry` for dataset-specific logic. Subclasses register via `@DatasetFormat.register`. |
| `AirCopBenchFormat` | Handles AirCopBench nested scene/UAV directory structure + Annotation/*.json parsing. Provides `get_degradation_types()` and annotation path mapping. |
| `GenericImageDirFormat` | Flat directory of images, no annotations, hash-based task assignment. |
| `DataSynthesisPipeline` | Orchestrates the 4-step pipeline: `extract_references()` → `inject_distortions()` → `annotate_scores()` → `aggregate_scores()`. Also provides `run_full()` for end-to-end execution. |
| `create_pipeline(dataset, seed)` | Convenience factory — creates a DataSynthesisPipeline for a named dataset format. |

### Pipeline Steps

| Step | Method | Description |
|------|--------|-------------|
| 1 | `extract_references()` | Copies clean reference images to output directory (uses `ThreadPoolExecutor` for parallel I/O). Note: the CLI `extract` command now calls `copy_vqa_files()` instead (copies VQA JSONs only, no images). |
| 2 | `inject_distortions()` | Applies distortions to all UAV images in a group via parallel workers. Supports `fmt="png"\|"jpeg"`. Auto-detects worker count. |
| 3 | `annotate_scores()` | VLM multi-image scoring of distorted groups across all subtasks |
| 4 | `aggregate_scores()` | Computes `cognitive_score` as mean of VLM scores per VQA entry |
| Full | `run_full()` | End-to-end: extract → inject → annotate → aggregate |

All steps include `tqdm` progress bars and support `dry_run` mode.

### Dependencies

- `numpy`
- `tqdm` — progress bars for manifest building and score annotation
- `uav_iqa.distortion` — `UAVDistortionPipeline`
- `uav_iqa.annotations` — `build_ref_score_lookup`, `parse_distortion_key`, etc.
- `uav_iqa.utils` — `find_images`, `split_samples`

---

## Module: `text_metrics.py`

**Purpose:** Text similarity metrics for VLM comparison-based annotation scoring. Implements BLEU, ROUGE-L, and CIDEr in pure Python (no nltk dependency).

**Location:** `src/uav_iqa/text_metrics.py`
**Lines:** 326

### Key Functions

| Function | Description |
|----------|-------------|
| `compute_bleu(reference, hypothesis, max_n=4, smooth=True)` | Sentence-level BLEU score with Chen & Cherry smoothing method 7. Returns `[0, 1]`. |
| `compute_rouge_l(reference, hypothesis, beta=1.0)` | ROUGE-L F-score based on longest common subsequence. Returns `[0, 1]`. |
| `compute_cider(references, hypothesis, corpus=None, max_n=4)` | CIDEr TF-IDF weighted n-gram cosine similarity. Returns `[0, ~10]`. |
| `compute_cognitive_score(ref_texts, dist_texts, weights=(1.0, 1.0, 0.1), cider_corpus=None)` | Combined cognitive quality score via BLEU + ROUGE-L + CIDEr weighting. Averages across prompt pairs. Returns `{bleu, rouge_l, cider, cognitive_score, per_prompt}`. |

### Score Formulas

```
BLEU: BP × exp(Σ log(p_n) / 4), where BP = brevity penalty
ROUGE-L: (1+β²) × recall × precision / (recall + β² × precision), LCS-based
CIDEr: avg(cosine_sim(TF-IDF(ref), TF-IDF(hyp))) × 10
Cognitive: (w_b · BLEU + w_r · ROUGE + w_c · CIDEr) / (w_b + w_r + w_c)
```

### Dependencies

- `math`, `collections`, `numpy` (stdlib + numpy)

---

## Module: `inference/` (subpackage)

**Purpose:** Multi-GPU offline inference framework — lightweight, testable, resumable batch scoring engine for single-machine multi-GPU setups.

**Location:** `src/uav_iqa/inference/`
**Modules:** 15 (14 internal + 1 `__init__.py`)

### Design Principles

1. **Single Responsibility** — each module owns exactly one concern
2. **Stateless Worker** — workers own no file state; receive Task, execute, return Result
3. **Dynamic Task Scheduling** — shared task queue, no static file-to-GPU assignment
4. **Resume First** — SQLite checkpoint at task granularity; resume on restart
5. **Storage Independent** — `BaseStorage` ABC allows JSON/JSONL/Parquet backends
6. **Backend Independent** — `BaseExecutor` ABC allows HuggingFace/vLLM/OpenAI executors

### Module Organization

```
config.py       → InferenceConfig (dataclass: model, paths, chunk/batch sizes, GPU devices)
types.py        → Task, Result, FileRecord, TaskStatus (dataclasses)
queue.py        → BaseQueue ABC + MpQueue (multiprocessing.Queue) + factory functions
storage.py      → BaseStorage ABC + JsonStorage (file scan, chunk load, output write)
checkpoint.py   → CheckpointStore (SQLite-backed task-level checkpoint/resume)
repository.py   → TaskRepository (create, query, update task state — wraps CheckpointStore)
executor.py     → BaseExecutor ABC + DummyExecutor + VLMExecutor
worker.py       → worker_main (multiprocessing worker: load → infer → return)
scheduler.py    → Scheduler (scan input files, generate tasks, enqueue)
collector.py    → Collector (track completed chunks, detect file completion)
writer.py       → Writer (merge chunks, restore order, atomic write)
validator.py    → Validator (verify file count, record count, chunk completeness, order)
metrics.py      → Metrics (queue size, samples/sec, latency, ETA)
engine.py       → InferenceEngine (wire all modules, manage lifecycle, launch workers)
```

### Key Design: Chunk vs Batch

- **Chunk**: scheduling unit (e.g., 256 samples) — how tasks are divided
- **Batch**: GPU forward unit (e.g., 16 samples) — how inference is executed
- A single Chunk(256) → 16 forward passes → one Result

### Key Design: Task Granularity

```python
@dataclass(slots=True)
class Task:
    task_id: str       # unique ID
    file_path: Path    # input JSON file
    chunk_id: int      # chunk index within file
    start: int         # start record index (inclusive)
    end: int           # end record index (exclusive)
```

Tasks use continuous ranges (`start`/`end`), not lists of indices.

### Key Design: SQLite Checkpoint

```sql
tasks (
    task_id TEXT PRIMARY KEY,
    file_path TEXT,
    chunk_id INTEGER,
    start INTEGER,
    end INTEGER,
    status TEXT,      -- PENDING | RUNNING | SUCCESS | FAILED
    retry_count INTEGER,
    updated_at TEXT
)
```

Resume re-queues only `PENDING` and `FAILED` tasks.

### Module Relationships

```
InferenceEngine
  ├── creates → MpQueue (task_queue), MpQueue (result_queue)
  ├── creates → JsonStorage
  ├── creates → CheckpointStore → TaskRepository
  ├── creates → Writer, Collector, Validator, Metrics
  ├── creates → Scheduler (task_queue, repository)
  ├── creates → multiprocessing.Process[worker_main] per GPU
  │                ├── worker receives task_queue, result_queue, storage, executor
  │                └── executor = VLMExecutor(model, device)
  └── runs lifecycle: scheduler → workers → collector → validator → metrics
```

### Public API

```python
from uav_iqa.inference import (
    BaseExecutor, BaseQueue, BaseStorage, CheckpointStore,
    Collector, DummyExecutor, FileRecord, InferenceConfig,
    InferenceEngine, JsonStorage, Metrics, MpQueue, Result,
    Scheduler, Task, TaskRepository, TaskStatus, VLMExecutor,
    Validator, Writer, make_result_queue, make_task_queue, worker_main,
)
```

### Dependencies

- All modules: stdlib only (dataclasses, pathlib, multiprocessing, sqlite3, json, logging)
- `executor.py`: optional `torch`, `transformers`, `vllm` for VLMExecutor
- No dependency on `lightning`, `timm`, or other training libraries

### Related Documentation

- [`docs/INFERENCE_FRAMEWORK.md`](../INFERENCE_FRAMEWORK.md) — full architecture specification with design rationale

---

## Module: `__init__.py`

**Purpose:** Public API exports.

**Location:** `src/uav_iqa/__init__.py`
**Lines:** 100

**Note:** The `inference/` subpackage exports its own 25 symbols via `inference/__init__.py` (see [Inference Subpackage](#module-inference-subpackage) above). These are not re-exported from the top-level `__init__.py`.

### Exports (45 total)

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
    "UAVIQADataset", "validate_manifest",
    # Metrics (4)
    "compute_srcc", "compute_plcc", "evaluate_iqa",
    "per_distortion_category_metrics",
    # Lightning wrappers (2)
    "UAVIQALightningModule", "UAVIQADataModule",
    # Losses (2)
    "ListMLELoss", "CrossTaskRegularization",
    # Annotation utilities (11)
    "parse_distortion_key",
    "build_ref_score_lookup", "build_vqa_split_lookup",
    "group_by_scene_frame",
    "extract_subtask_type", "extract_subtask_id",
    "extract_uav_id_from_question_id",
    "normalize_subtask_type",
    "build_sample_id", "get_dataset_name",
    "seed_for_distortion",
    # Constants (5)
    "SUBTASK_NAMES", "SUBTASK_TO_ID",
    "SUBTASK_NAME_LIST", "SUBTASK_NAME_TO_ID",
    "NUM_SUBTASKS",
    # Utility functions (5)
    "setup_logging", "load_task_map",
    "load_flat_samples", "split_samples",
    "find_images",
    # Data synthesis (3)
    "DatasetFormat",
    "DataSynthesisPipeline", "create_pipeline",
    # VLM/VLA scoring (3)
    "BaseScorer", "VLMScorer", "BatchAnnotator",
]
```

**Note**: `count_parameters` remains importable from `utils.py` but is no longer part of the public `__all__` API. `parse_quality_score`, `parse_usability`, `assign_task_label` were removed from `__all__`; `degradation_factor`, `compute_synthetic_score`, and `synthetic_ref_scores` were fully removed from `annotations.py`. `load_manifest` and `write_manifest` were removed from `utils.py` entirely (replaced by `load_flat_samples`). `seed_for_distortion` was moved from `utils.py` to `annotations.py` and re-exported via `__init__.py`. New constants `SUBTASK_NAME_LIST` and `SUBTASK_NAME_TO_ID` were added. New functions `extract_uav_id_from_question_id` and `normalize_subtask_type` were added.
