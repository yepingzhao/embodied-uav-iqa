# Module Codemap

**Last Updated:** 2026-06-27

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
  → parse_distortion_key, parse_quality_score, parse_usability
  → build_ref_score_lookup, assign_task_label

data_synthesis.py (822 lines)
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

vla_scorer.py (standalone, no internal deps)
  │
  ├── BaseScorer (ABC)
  └── extract_ref_id, task constants

vlm/ (subpackage)
  │
  ├── config.py → VLMConfig (dataclass), MODEL_REGISTRY (dict of 15 models)
  ├── vqa_index.py → VQAIndex (AirCopBench VQA question index)
  └── scorer.py → VLMScorer (vLLM / transformers backend)
       │
       └── imports → vla_scorer.py (BaseScorer, extract_ref_id)

batch_annotator.py
  │
  └── uses → vla_scorer.py (BaseScorer)
       │
       ├── BatchAnnotator (orchestrates manifest scoring with checkpoint/resume)
       └── SPLIT_NAMES = ("train", "val", "test")

text_metrics.py (standalone)
  → compute_bleu, compute_rouge_l, compute_cider, compute_cognitive_score
  → used by vlm/scorer.py for text similarity scoring

```
*(Note: `vlm_vla_scorer.py` backward-compat shim was removed — its contents are now split into `vlm/` subpackage, `vla_scorer.py`, `text_metrics.py`, and `batch_annotator.py`.)*

---

## Module: `distortion.py`

**Purpose:** 36 distortion models (6 UAV-specific + 30 generic) for injecting quality degradation.

**Location:** `src/uav_iqa/distortion.py`
**Lines:** 786

*Note: Pipeline docstring says "36 types" — actual is 36: 6 UAV-specific + 30 generic.*

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
**Lines:** 250

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
**Lines:** 289

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

### Test Results (DDP-compatible)

Test results are now logged via `self.log()` in `on_test_epoch_end()` instead of the removed `get_test_results()` method. DDP gathering uses `_gather_tensor()` and `_gather_objects()` to aggregate predictions across all processes:

- Metrics: `test/srcc`, `test/plcc` (overall)
- Per-task: `test/srcc_{task_name}` for all 4 tasks
- Per-distortion category: `test/srcc_uav` (avg of 6 UAV distortions), `test/srcc_generic` (avg of 30 generic distortions)
- All stored via `self.log()` → CSVLogger/metrics.csv + WandbLogger

### Dependencies

- `lightning` (pytorch-lightning)
- `torch`, `torch.nn`
- `numpy`

---

## Module: `data_module.py`

**Purpose:** LightningDataModule wrapping UAVIQADataset with manifest filtering.

**Location:** `src/uav_iqa/data_module.py`
**Lines:** 176

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

**Purpose:** PyTorch Lightning callbacks for dataset verification and curriculum switching.

**Location:** `src/uav_iqa/callbacks.py`
**Lines:** 215

### Key Classes

| Class | Description |
|-------|-------------|
| `SetupRunCallback` | At fit start: computes manifest SHA256 hash for dataset versioning, prints model param count. Exposes `pl_module.manifest_hash`. DDP-safe via `trainer.is_global_zero`. |
| `CurriculumStageCallback` | Sets `pl_module.curriculum_stage` (vlm/vla/execution) based on epoch boundaries. DDP-safe. |
| `MetricsHistoryCallback` | Tracks epoch-level metrics (`train/loss_epoch`, `val/srcc`, `val/plcc`, etc.) across validation epochs, records best val SRCC, writes `history.json` on fit end. |
| `ResultsSavingCallback` | On fit end: loads best checkpoint, runs test, saves structured `results.json` with test metrics, per-task SRCC, model params, git commit hash, seed, and config. |

### SetupRunCallback

- Computes SHA256 hash of concatenated train/val/test `manifest.json` files (first 16 chars)
- Useful for verifying dataset version consistency across experiment runs
- Exposed as `pl_module.manifest_hash`
- Prints only on global rank 0 (`trainer.is_global_zero`)

### CurriculumStageCallback

| Stage | Epochs (default) |
|-------|------------------|
| VLM | 1–20 |
| VLA | 21–40 |
| Execution | 41–50 |

Configurable via `vlm_epochs`, `vla_epochs`, `execution_epochs` init args. DDP-safe print on global rank 0.

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
  - `per_task`: per-task SRCC
  - `hparams`, `data_config`, `manifest_hash`
  - `git_commit`: short SHA via `git rev-parse`
- Only runs on global rank 0

### Dependencies

- `lightning`
- `pathlib`, `hashlib`, `json`, `logging`, `subprocess`
- `yaml` (for reading config.yaml)

---

## Module: `annotations.py`

**Purpose:** AirCopBench human annotation parsing — distortion key extraction, Quality/Usability score parsing, reference score lookup, and task label assignment from path structure.

**Location:** `src/uav_iqa/annotations.py`
**Lines:** 162

### Key Functions

| Function | Description |
|----------|-------------|
| `parse_distortion_key(key) -> (str, float)` | Parse `'gaussian_blur_L04'` → `('gaussian_blur', 0.4)`. Handles `__` separator, `_L4` / `_L04` / `_L0_5` formats. |
| `parse_quality_score(quality_str)` | Parse `'Good (4/5)'` → 0.8, `'Excellent (5/5)'` → 1.0, etc. |
| `parse_usability(usability_str)` | Parse `'1 (Available)'` → 1.0, `'3 (Unavailable)'` → 0.25 |
| `build_ref_score_lookup(aircopbench_dir)` | Walk AirCopBench Annotations dirs → dict of `{ref_id: {vlm_score, vla_score, execution_score, annotated}}` |
| `assign_task_label(img_name, task_map)` | Extract task label from path (scene_001→tracking, etc.), with configurable `task_map` override |

### Score Construction

Reference scores are built from AirCopBench human annotations:
- `Quality` → `vlm_score` (via `parse_quality_score`)
- `Usibility` → `vla_score` (via `parse_usability`)
- `execution_score = 0.4 × vlm_score + 0.6 × vla_score`

Unannotated references fall back to deterministic scoring in `vla_scorer.py`.

*(Note: `degradation_factor`, `compute_synthetic_score`, `synthetic_ref_scores`, and `SyntheticScorer` were removed in the 2026-06 refactor. Score synthesis lives in `data_synthesis.py` `DataSynthesisPipeline`.)*

### Dependencies

- `json`, `hashlib`, `re`, `logging`, `pathlib`

---

## Module: `utils.py`

**Purpose:** Utility helpers — image I/O, manifest management, logging setup, parameter counting.

**Location:** `src/uav_iqa/utils.py`
**Lines:** 149

### Functions

| Function | Description |
|----------|-------------|
| `setup_logging(name, level)` | Configure stdlib logging with uniform format for scripts |
| `find_images(image_dir, exts, exclude_dirs)` | Recursively find image files, optionally excluding subdirectories |
| `load_image_tensor(path, image_size)` | Load image as `(C, H, W)` float32 tensor in [0,1]; used by both Dataset and benchmark scripts |
| `count_parameters(model) -> (total, trainable)` | Count total and trainable parameters |
| `load_task_map(path)` | Load JSON task-map file (used by data_synthesis.py) |
| `split_samples(samples, ratios, seed)` | Shuffle + split into `{train, val, test}` dicts via numpy.RandomState |
| `load_manifest(manifest_path)` | Load manifest.json entries |
| `write_manifest(entries, manifest_path)` | Write manifest JSON with validation via `validate_manifest()` |

### Dependencies

- `torch`
- `numpy`
- `PIL.Image`

---

## Module: `data_synthesis.py`

**Purpose:** Dataset-agnostic data synthesis pipeline — extract references, inject distortions, generate manifests, annotate scores. Supports AirCopBench and generic image directories.

**Location:** `src/uav_iqa/data_synthesis.py`
**Lines:** 822

### Key Classes

| Class | Description |
|-------|-------------|
| `DatasetFormat` | Abstract base class with explicit `_registry` for dataset-specific logic. Subclasses register via `@DatasetFormat.register`. |
| `AirCopBenchFormat` | Handles AirCopBench nested scene/UAV directory structure + Annotation/*.json parsing. Provides `get_degradation_types()` and annotation path mapping. |
| `GenericImageDirFormat` | Flat directory of images, no annotations, hash-based task assignment. |
| `DataSynthesisPipeline` | Orchestrates the 4-step pipeline: `extract_references()` → `inject_distortions()` → `generate_manifests()` → `annotate_scores()`. Also provides `run_full()` for end-to-end execution. |
| `create_pipeline(dataset, seed)` | Convenience factory — creates a DataSynthesisPipeline for a named dataset format. |

### Pipeline Steps

| Step | Method | Description |
|------|--------|-------------|
| 1 | `extract_references()` | Symlinks or copies clean reference frames to flat directory (uses `ThreadPoolExecutor` for parallel I/O). Supports `max_refs_per_source` (dict of per-source image limits for stratified sampling). |
| 2 | `inject_distortions()` | Applies all 36 distortions × 1 random intensity level via parallel workers. Supports `fmt="png"|"jpeg"` (JPEG quality 92). Auto-detects worker count when `workers=0`. Skips existing outputs. |
| 3 | `generate_manifests()` | Scans distorted directory, parses filenames, builds entries, splits train/val/test |
| 4 | `annotate_scores()` | Assigns VLM/VLA/execution scores using real annotations or synthetic fallback + degradation model |
| Full | `run_full()` | End-to-end pipeline: extract → inject → manifest → annotate. Accepts all per-step parameters (`fmt`, `max_refs_per_source`, etc.). |

All steps include `tqdm` progress bars and support `dry_run` mode.

### Dependencies

- `numpy`
- `tqdm` — progress bars for manifest building and score annotation
- `uav_iqa.distortion` — `UAVDistortionPipeline`
- `uav_iqa.annotations` — `build_ref_score_lookup`, `parse_distortion_key`, etc.
- `uav_iqa.utils` — `find_images`, `split_samples`, `write_manifest`

---

## Module: `text_metrics.py`

**Purpose:** Text similarity metrics for VLM comparison-based annotation scoring. Implements BLEU, ROUGE-L, and CIDEr in pure Python (no nltk dependency).

**Location:** `src/uav_iqa/text_metrics.py`
**Lines:** 318

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

## Module: `__init__.py`

**Purpose:** Public API exports.

**Location:** `src/uav_iqa/__init__.py`
**Lines:** 84

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
    "UAVIQADataset", "validate_manifest",
    # Metrics (4)
    "compute_srcc", "compute_plcc", "evaluate_iqa",
    "per_distortion_category_metrics",
    # Lightning wrappers (2)
    "UAVIQALightningModule", "UAVIQDataModule",
    # Losses (2)
    "ListMLELoss", "CrossTaskRegularization",
    # Annotation utilities (5)
    "parse_distortion_key", "parse_quality_score",
    "parse_usability",
    "build_ref_score_lookup", "assign_task_label",
    # Utility functions (6)
    "setup_logging", "load_task_map",
    "load_manifest", "split_samples",
    "write_manifest", "find_images",
    # Data synthesis (3)
    "DatasetFormat",
    "DataSynthesisPipeline", "create_pipeline",
    # VLM/VLA scoring (3)
    "BaseScorer", "VLMScorer", "BatchAnnotator",
]
```

**Note**: Exports removed since last update: `count_parameters` remains importable from `utils.py` but is no longer part of the public `__all__` API. `degradation_factor`, `compute_synthetic_score`, and `synthetic_ref_scores` were fully removed from `annotations.py` — score synthesis now lives in `vla_scorer.py` (SyntheticScorer) and `data_synthesis.py` (DataSynthesisPipeline).
