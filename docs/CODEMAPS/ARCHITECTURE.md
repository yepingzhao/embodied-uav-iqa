# Architecture Codemap

**Last Updated:** 2026-07-01

## High-Level System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                   DATA PREPARATION (M1)                              │
│                                                                      │
│  Dataset-agnostic pipeline (data_synthesis.py)                       │
│                                                                      │
│  Input Source ───► DatasetFormat ──► DataSynthesisPipeline           │
│  (AirCopBench /    (registry key:     │                              │
│   GenericDir)       "aircopbench",    ├── extract (copy VQA JSONs)   │
│                     "generic")        ├── inject (apply distortion   │
│                                        │    to all UAVs in group)    │
│                                        ├── annotate (VLM multi-image │
│                                        │    scoring)                 │
│                                        └── aggregate (compute        │
│                                             cognitive_score)         │
│                                             │                        │
│                                             ▼                        │
│                                     *_VQA_*.json                     │
│                                     (grouped: uav_paths, distortions, │
│                                      vqa_entries, cognitive_score)   │
└─────────────────────────────────────┼────────────────────────────────┘
                                       │
                                       ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      TRAINING (M3)                                   │
│                                                                      │
│  UAVIQADataset ──► UAVIQALightningModule ──► UAVIQANet              │
│  (manifest +        │                              │                 │
│   augmentation)     │  Loss: MSE + ListMLE +       │                 │
│                     │  CrossTaskRegularization     │                 │
│                     └──────────────────────────────┘                 │
│                                 │                                    │
│                             Output                                   │
│                       (best_*.ckpt, last.ckpt, metrics.csv)         │
└──────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    VLM ANNOTATION (M1.5)                             │
│                                                                      │
│  scripts/vlm_annotate.py ──► BatchAnnotator ──► VLMScorer           │
│  (multi-model CLI)          │                       │                │
│                              ├── filter_entries     ├── MODEL_REGISTRY│
│                              ├── annotate_manifest   │   (15 models,  │
│                              │   (checkpoint/resume) │    6 families) │
│                              ├── save/load_checkpoint├── score_image  │
│                              └── write_groups       └── score_batch  │
│                                        │                  via vLLM   │
│                                        ▼                  or         │
│                                vlm_annotated/              transformers│
│                                {model}/*_VQA_*.json          │
│                                                                      │
│  scripts/download_models.py ── snapshot_download all 15 models       │
│  (HF Hub download + optional validate)                               │
└──────────────────────────────────┬───────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    BENCHMARKING (M2)                                 │
│                                                                      │
│  Test manifest ──► benchmark_iqa_methods.py ──► metrics table        │
│                      (pyiqa, 15+ zero-shot methods)                  │
│  Test manifest ──► finetune_baselines.py  ──► fine-tuned metrics     │
│                      (brisque/niqe/clipiqa/maniqa/topiq_nr)         │
│  UAVIQANet       ──► metrics.py            (SRCC, PLCC, RMSE,       │
│                        evaluate_iqa()       Kendall τ,              │
│                        per_task_metrics()    per-task &             │
│                        per-distortion_       per-distortion         │
│                          category_metrics()   category metrics)     │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Reference Image (3×H×W, uint8)
    │
    ▼
┌──────────────────────────────────────────────────┐
│           UAVDistortionPipeline                   │
│                                                    │
│  6 UAV-specific ────┐                              │
│  ┌─ PropellerVibrationBlur    ─── directional     │
│  │                          motion blur +          │
│  │                          intensity modulation   │
│  ├─ AtmosphericScatteringHaze ─── Koschmieder      │
│  │                          model + depth est.     │
│  ├─ SixDoFViewpointBlur     ─── MotionScape flow   │
│  ├─ CommunicationPacketLoss ─── 16×16 block loss   │
│  ├─ LowResSuperResolution   ─── bicubic + ESRGAN   │
│  └─ PropellerShadow         ─── periodic           │
│                              brightness modulation  │
│                                                     │
│  30 Generic ───────────┐                            │
│  (via Albumentations)   │                            │
│  ├─ blur (3)            │                            │
│  ├─ brightness (6)      │                            │
│  ├─ chromatic (3)       │                            │
│  ├─ noise (6)           │                            │
│  ├─ compression (3)     │                            │
│  ├─ spatial (3)         │                            │
│  ├─ transmission (3)    │                            │
│  └─ other (3)           │                            │
│                                                     │
│  1 randomly selected intensity per distortion              │
└──────────────────────────────────────────────────┘
    │
    ▼
Grouped entry: [{dataset, sequence_frame, uav_paths, num_uavs,
                  distortions: {dist_key: {type, category, intensity, ...}},
                  vqa_entries: [{subtask_type, cognitive_score, ...}]}]
```

## Model Inference Flow

```
Input (B × 3 × 256 × 256)
    │
    ▼
┌──────────────────────────────────────────────────┐
│              timm Backbone (dynamically probed)    │
│  (MobileNetV4-S / EfficientViT / MobileViT)       │
│  n_stages probed → last 3 as multi-scale output   │
│  regex-based freeze (blocks.N / stages.N /        │
│   stages_N naming conventions)                    │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│              PANet FPN (PanetFPN)                 │
│  Top-down + bottom-up aggregation                │
│  Smooth convs on output                          │
└──────────────────────┬───────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────┐
│               CBAM (optional)                     │
│  Channel attention + spatial attention           │
│  Output: f_s ∈ R^256                              │
└──────┬───────────────────────────────────────────┘
       │                                           
       │  ┌──────────────────────────────────────┐
       │  │  FrequencyAwareBranch (FAB, optional) │
       │  │  Gray → patch FFT (32×32, stride 16) │
       │  │  → log-polar transform                │
       │  │  → tiny CNN → proj → f_f ∈ R^64      │
       │  └──────────────┬───────────────────────┘
       │                 │
       ▼                 ▼
    ┌──────────────────────────┐
    │  CrossAttentionGate      │
    │  α = σ(W·[f_s, f_f])    │
    │  → α ⊙ f_f               │
    └────────────┬─────────────┘
                 │
                 ▼
           ┌──────────┐
           │  Concat   │
           │ f ∈ R^320  │
           └─────┬─────┘
                 │
                 ▼
┌──────────────────────────────────────────────────┐
│         TaskConditionedHead (optional)            │
│  FiLM: task_embed → γ, β                         │
│  h' = γ ⊙ (W·f + b) + β → σ(W·h')               │
│  → 1 score per task                               │
└──────────────────────────────────────────────────┘
    │
    ▼
Output: (B,) quality scores ∈ [0, 1]
```

## Training Architecture

```
┌─────────────────────────────────────────────┐
│          UAVIQALightningModule               │
│                                              │
│  ┌──────────────────────────────────────┐   │
│  │         Loss Computation              │   │
│  │                                      │   │
│  │  pred = UAVIQANet(images, task_ids)  │   │
│  │                                      │   │
│  │  L_mse = MSE(pred, scores)           │   │
│  │  L_rank = ListMLE(pred, scores)      │   │
│  │  L_ct   = CrossTaskRegularization    │   │
│  │          (all_task_scores)           │   │
│  │                                      │   │
│  │  L = L_mse + λ_r·L_rank + λ_ct·L_ct │   │
│  └──────────────────────────────────────┘   │
│                                              │
│  Optimizer: AdamW (lr=3e-4, wd=1e-4)       │
│  Scheduler: warmup (5) → cosine (45)       │
│                                              │
│  Supervision: single cognitive_score             │
│  (unified quality score from VLM aggregation)     │
│                                              │
│  Logging: self.log() → WandbLogger         │
│           (cloud: wandb.ai)                  │
│           + CSVLogger (local: metrics.csv)   │
└─────────────────────────────────────────────┘
```

## Multi-GPU Offline Inference Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                  InferenceEngine (lifecycle manager)                │
│                                                                    │
│  Main Process:                    Worker Processes (1 per GPU):    │
│  ┌──────────────────────────┐     ┌───────────────────────────┐   │
│  │  Scheduler               │     │  worker_main               │   │
│  │  ├── scan input files    │     │  ┌─────────────────────┐  │   │
│  │  ├── generate tasks      │     │  │ task_queue.get()    │  │   │
│  │  └── TaskRepository      │     │  │        ↓            │  │   │
│  │       └── CheckpointStore│     │  │ JsonStorage         │  │   │
│  │            (SQLite)      │     │  │  .load_chunk()      │  │   │
│  └───────────┬──────────────┘     │  │        ↓            │  │   │
│              │ tasks              │  │ Executor.infer()    │  │   │
│              ▼                    │  │        ↓            │  │   │
│  ┌──────────────────────┐         │  │ result_queue.put()  │  │   │
│  │  TaskQueue (MpQueue)  │◄───────┤  └─────────────────────┘  │   │
│  │  [multiprocessing]    │         └───────────────────────────┘   │
│  └──────────────────────┘                                         │
│              │                                                     │
│              ▼                                                     │
│  ┌──────────────────────┐         ┌───────────────────────────┐   │
│  │  Collector           │◄────────┤  ResultQueue (MpQueue)    │   │
│  │  ├── track chunks    │         └───────────────────────────┘   │
│  │  ├── detect complete │                                         │
│  │  └── notify Writer   │                                         │
│  └───────────┬──────────┘                                         │
│              ▼                                                     │
│  ┌──────────────────────┐                                         │
│  │  Writer              │                                         │
│  │  ├── merge chunks    │                                         │
│  │  ├── restore order   │                                         │
│  │  └── atomic write    │                                         │
│  └──────────────────────┘                                         │
│              │                                                     │
│              ▼                                                     │
│  ┌──────────────────────┐                                         │
│  │  Validator           │  file/record count, order, identity     │
│  └──────────────────────┘                                         │
│              │                                                     │
│              ▼                                                     │
│  ┌──────────────────────┐                                         │
│  │  Metrics             │  samples/sec, latency, ETA              │
│  └──────────────────────┘                                         │
└────────────────────────────────────────────────────────────────────┘

Key properties:
- Input: N JSON files → Task chunking → dynamic GPU scheduling
- Output: N JSON files (same names, same record count & order)
- Resume: SQLite checkpoint at task granularity
- Backend-agnostic Executor: DummyExecutor / VLMExecutor / future
- Storage-agnostic: JsonStorage / future JsonlStorage / Parquet
```

## Component Relationships

```
scripts/data_synthesis.py
  └── calls → DataSynthesisPipeline (data_synthesis.py)
                  │
                  ├── uses → DatasetFormat (registry: AirCopBenchFormat, GenericImageDirFormat)
                  ├── uses → UAVDistortionPipeline (distortion.py)
                  ├── uses → annotations.py (build_ref_score_lookup, assign_task_label, etc.)
                   └── uses → utils.py (find_images, split_samples)

scripts/finetune_baselines.py
  ├── uses → UAVIQADataset (dataset.py) — reuses existing data loading
  ├── uses → metrics.py (evaluate_iqa, per_task_metrics, per_distortion_category_metrics)
  ├── uses → utils.py (load_image_tensor, load_flat_samples, setup_logging)
  ├── owns → BaselineLightningModule (standalone wrapper around pyiqa model)
  │            └── owns → pyiqa model (brisque/niqe/clipiqa/maniqa/topiq_nr)
  └── owns → BaselineDataModule (standalone, no LightningCLI)
                 └── owns → UAVIQADataset (reuses manifest data loading)

scripts/vlm_annotate.py
  └── calls → BatchAnnotator (batch_annotator.py)
                  │
                  ├── uses → VLMScorer (vlm/scorer.py)
                  │              ├── owns → MODEL_REGISTRY (vlm/config.py)
                  │              ├── uses → BaseScorer ABC (vla_scorer.py)
                  │              ├── uses → text_metrics.py (compute_cognitive_score)
                  │              ├── resolves → backend (vllm / transformers / none)
                  │              └── dynamic patches per model family
                  └── uses → utils.py (checkpoint I/O)

scripts/download_models.py
  └── uses → MODEL_REGISTRY (via importlib lazy-import from vlm/config.py)
  └── uses → huggingface_hub.snapshot_download

scripts/vlm_cognitive_score_vqa.py
  ├── uses → VLMScorer (vlm/scorer.py)
  ├── uses → text_metrics.py (compute_bleu, compute_rouge_l, compute_cider)
  ├── reads → manifest JSON (ref_img, dist_img, question, gt_answer)
  └── writes → two JSON output files (指标1 GT-normalized + 指标2 direct-comparison)

scripts/fix_configs.py
  └── utility — batch-converts experiment YAML configs from nested to flat format

scripts/inference.py
  └── calls → InferenceEngine (inference/engine.py)
                  │
                  ├── creates → Scheduler (inference/scheduler.py)
                  │                └── uses → TaskRepository (inference/repository.py)
                  │                       └── uses → CheckpointStore (inference/checkpoint.py)
                  ├── creates → TaskQueue + ResultQueue (inference/queue.py)
                  ├── creates → JsonStorage (inference/storage.py)
                  ├── creates → Writer (inference/writer.py)
                  ├── creates → Collector (inference/collector.py)
                  ├── creates → Validator (inference/validator.py)
                  ├── creates → Metrics (inference/metrics.py)
                  └── spawns → Worker processes (inference/worker.py)
                                  └── owns → Executor (inference/executor.py:
                                                DummyExecutor / VLMExecutor)

LightningCLI (scripts/train.py)
  ├── --config → configs/experiments/<name>.yaml
  ├── configures → WandbLogger + CSVLogger (dual logger)
  ├── calls → UAVIQALightningModule
  │              ├── owns → UAVIQANet
  │              │            ├── owns → MobileNetV4-S (timm)
  │              │            ├── owns → PanetFPN
  │              │            ├── owns → CBAM (optional)
  │              │            ├── owns → FrequencyAwareBranch (optional)
  │              │            ├── owns → CrossAttentionGate
  │              │            └── owns → TaskConditionedHead (optional)
  │              ├── uses → ListMLELoss
  │              ├── uses → CrossTaskRegularization
  │              └── uses → metrics.py (evaluate_iqa, per_task_metrics, per_distortion_metrics)
  ├── calls → UAVIQADataModule
  │              └── owns → UAVIQADataset
  │                     └── uses → utils.py (load_image_tensor)
  ├── adds → SetupRunCallback (dataset SHA256, param count, DDP-safe)
  ├── adds → MetricsHistoryCallback (epoch metrics → history.json)
  ├── adds → ResultsSavingCallback (test results → results.json with git hash)
  ├── adds → ModelCheckpoint (val/srcc, top-1)
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| MobileNetV4-S backbone | Lightweight (~5.4M total params), suitable for edge deployment |
| PANet FPN over simple FPN | Bottom-up path improves small object feature propagation |
| Frequency-Aware Branch | Captures periodic UAV artifacts (vibration, shadow) missed by spatial CNN |
| Cross-Attention Gate | Adaptively weights frequency vs. spatial features per input |
| FiLM task conditioning | Enables multi-task with minimal parameter overhead (4-dim embed) |
| Single cognitive_score | Unified quality score from VLM multi-image aggregation replaces staged curriculum |
| ListMLE loss | Per-distortion ranking signal improves relative quality ordering |
| Feature sharing (forward_features) | Backbone+FPN+FAB computed once, reused for pred + cross-task loss |
| Dynamic stage probing | Probes backbone forward pass to determine n_stages, selects last 3 — compatible with MobileNetV4, EfficientViT, MobileViT |
| Regex-based backbone freeze | `_freeze_backbone_stages` matches `blocks.N`/`stages.N`/`stages_N` via `re` — supports diverse timm backbones |
| scripts/train.py + LightningCLI | Replaces custom UAVIQACLI; self-contained YAML per experiment |
| WandbLogger + CSVLogger | Cloud + local dual logging; no cloud dependency for local runs |
| Inference framework with multiprocessing.Queue | Lightweight multi-GPU scheduling without Ray/Redis dependency |
| SQLite checkpoint storage | Zero-config task-level resume; single file per run |
| Chunk/batch separation | Chunk = scheduling unit, batch = GPU forward unit; independent sizing |
| Backend-agnostic Executor ABC | Test with DummyExecutor, deploy with VLMExecutor, extend to vLLM/OpenAI |
| Storage-agnostic ABCs | JsonStorage for MVP; swap to JsonlStorage/Parquet without changing workers |
