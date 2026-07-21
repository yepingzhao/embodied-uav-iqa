# Module Codemap

<!-- Generated: 2026-07-21 | Files scanned: ~90 source files | Token estimate: ~1000 -->

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
                 │ UAVIQALightning    │   └────────┬─────────┐
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

annotations.py (standalone, no internal deps)
  → parse_distortion_key, parse_quality_score, parse_usability
  → build_ref_score_lookup, assign_task_label
  → SUBTASK_NAMES, SUBTASK_TO_ID, SUBTASK_NAME_LIST, SUBTASK_NAME_TO_ID, NUM_SUBTASKS
  → seed_for_distortion, group_by_scene_frame, build_vqa_split_lookup, normalize_subtask_type
  → extract_subtask_type, extract_subtask_id, extract_uav_id_from_question_id, build_sample_id, get_dataset_name

distortion_synthesis.py (466 lines) — renamed from data_synthesis.py
  │
  ├── DatasetFormat (ABC + registry)
  │     ├── AirCopBenchFormat
  │     └── GenericImageDirFormat
  ├── DistortionSynthesisPipeline (orchestrates: extract → inject)
  └── create_pipeline (factory)
       │
       ├── uses → distortion.py (UAVDistortionPipeline)
       ├── uses → annotations.py (build_ref_score_lookup, parse_distortion_key, etc.)
       └── uses → utils.py (find_images, split_samples)

vla_scorer.py (standalone, no internal deps)
  │
  ├── BaseScorer (ABC)
  └── extract_ref_id, task constants

vlm/ subpackage (5 files, ~1678 lines)
  │
  ├── config.py (175 lines) — VLMConfig + MODEL_REGISTRY (15 models, 6 families)
  ├── backends.py — ChatTemplate processor + vLLM/transformers inference helpers
  ├── scorer.py (1419+ lines) — VLMScorer with score_vqa_with_gt, score_multi_image,
  │                             score_batch_multi_image, vLLM/transformers backends
  └── vqa_index.py (77 lines) — VQAIndex for question-answer retrieval

batch_annotator.py (228 lines)
  │
  ├── BatchAnnotator — manifest I/O, filtering, checkpoint/resume
  └── Uses → VLMScorer (for type hints), BaseScorer

text_metrics.py (326 lines)
  │
  ├── BLEU, ROUGE-L, CIDEr (pure Python, no nltk)
  └── compute_cognitive_score(ref_texts, dist_texts, weights=(1,1,0.1))
       → cognitive = (1.0*bleu + 1.0*rouge_l + 0.1*cider) / 2.1

inference/ subpackage (15 modules)
  │
  ├── types.py — Task, Result, FileRecord, TaskStatus (dataclasses)
  ├── config.py — InferenceConfig: model, paths, batch sizes, GPU devices
  ├── queue.py — BaseQueue ABC + MpQueue (multiprocessing.Queue)
  ├── storage.py — BaseStorage ABC + JsonStorage (file scan, chunk load, output write)
  ├── checkpoint.py — CheckpointStore (SQLite task-level checkpoint/resume)
  ├── repository.py — TaskRepository (CRUD + state transitions)
  ├── executor.py — BaseExecutor ABC + DummyExecutor + VLMExecutor (real inference)
  ├── worker.py — worker_main: load data → execute → return result
  ├── scheduler.py — Scheduler: scan inputs → generate tasks → enqueue
  ├── collector.py — Collector: track completions → notify Writer
  ├── writer.py — Writer: merge chunks → atomic file write
  ├── validator.py — Validator: file count, record count, chunk completeness
  ├── metrics.py — Metrics: runtime stats, queue size, samples/sec, ETA
  ├── engine.py — InferenceEngine: wire all modules, manage lifecycle
  └── __init__.py — Public API: 25 exported symbols
```

### Module Line Counts (Top-Level)

| File | Lines | Purpose |
|------|-------|---------|
| `__init__.py` | 99 | Public API: exports **44 symbols** |
| `distortion.py` | **735** | 36 distortion models (6 UAV + 30 generic) + UAVDistortionPipeline |
| `model.py` | **438** | UAVIQANet: backbone → FPN → CBAM → FAB → FiLM task heads |
| `distortion_synthesis.py` | 466 | Dataset-agnostic distortion injection pipeline (renamed from data_synthesis.py, VLM steps removed) |
| `annotations.py` | **402** | AirCopBench annotation parsing, subtask mapping, score synthesis |
| `text_metrics.py` | **326** | BLEU, ROUGE-L, CIDEr, cognitive_score computation |
| `lightning_module.py` | 282 | UAVIQALightningModule: MSE/ListMLE/cross-task loss, DDP |
| `batch_annotator.py` | 228 | Batch annotation engine with checkpoint/resume |
| `callbacks.py` | 206 | SetupRunCallback, MetricsHistoryCallback, ResultsSavingCallback |
| `utils.py` | 179 | count_parameters, find_images, load_image_tensor, split_samples |
| `dataset.py` | 169 | UAVIQADataset: flat JSON loader + validate_manifest |
| `metrics.py` | 145 | SRCC, PLCC, RMSE, Kendall τ, per-task/per-distortion metrics |
| `data_module.py` | 124 | UAVIQADataModule: manifest filtering, DDP-compatible |
| `losses.py` | 30 | ListMLELoss + CrossTaskRegularization |
| `vla_scorer.py` | 60 | BaseScorer ABC + task constants |
| `vlm/` | ~1678 | VLM scoring subpackage (5 files) |
| `inference/` | ~2500 | Multi-GPU offline inference (15 modules) |

### Per-File Dependency Chain

```
__init__.py
  → distortion.py, model.py, dataset.py, metrics.py,
    losses.py, annotations.py, utils.py, distortion_synthesis.py,
    lightning_module.py, data_module.py,
    vla_scorer.py, batch_annotator.py
  (12 internal deps, VLMScorer no longer in __init__)

text_metrics.py
  → math, collections, numpy (stdlib + numpy)

distortion.py
  → cv2, numpy, tqdm, albumentations
  → basicsr + realesrgan (optional, LowResSuperResolution)

model.py
  → torch, timm, math, re

distortion_synthesis.py (renamed from data_synthesis.py)
  → numpy, json, shutil, logging, pathlib, abc, typing
  → uav_iqa.distortion (UAVDistortionPipeline)
  → uav_iqa.annotations, uav_iqa.utils

vlm/config.py
  → logging, dataclasses (stdlib)

vlm/scorer.py
  → vlm/config.py (VLMConfig, MODEL_REGISTRY)
  → vlm/vqa_index.py (VQAIndex)
  → vla_scorer.py (BaseScorer)
  → vllm (optional), transformers (optional), PIL, torch, tqdm

vlm/backends.py
  → vllm (optional), transformers (optional), torch, PIL

batch_annotator.py
  → vla_scorer.py (BaseScorer)
  → vlm/scorer.py (VLMScorer)
  → json, tempfile, os, time, pathlib (stdlib)

lightning_module.py
  → lightning, torch, numpy
  → metrics.py, model.py, losses.py, distortion.py

data_module.py
  → lightning, dataset.py, distortion.py

callbacks.py
  → lightning, hashlib, pathlib, json, yaml
  → utils.py (count_parameters)

inference/ — see INFERENCE_FRAMEWORK.md for full dependency tree
```
