# Architecture Codemap

<!-- Generated: 2026-07-21 | Files scanned: ~90 source files | Token estimate: ~900 -->

## High-Level System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                   DISTORTION SYNTHESIS (M1)                          │
│                                                                      │
│  Dataset-agnostic pipeline (distortion_synthesis.py)                 │
│                                                                      │
│  Input Source ───► DatasetFormat ──► DistortionSynthesisPipeline     │
│  (AirCopBench /    (registry key:     │                              │
│   GenericDir)       "aircopbench",    ├── extract (copy VQA JSONs)   │
│                     "generic")        └── inject (apply distortion   │
│                                            to all UAVs in group)     │
│                                             │                        │
│                                             ▼                        │
│                                     *_VQA_*.json                     │
│                                     (grouped: uav_paths,             │
│                                      distorted_uav_paths,            │
│                                      distortion_info, vqa_entries)   │
└─────────────────────────────────────┬────────────────────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    VLM ANNOTATION (M1.5)                             │
│                                                                      │
│  BatchAnnotator ──► VLMScorer (score_multi_image)                    │
│  │                       │                                           │
│  ├── annotate_manifest   ├── vLLM backend (batch inference)         │
│  ├── filter_entries      ├── transformers backend                   │
│  ├── checkpoint/resume   └── MODEL_REGISTRY (15 models, 6 families) │
│  └── write_groups                                                    │
│           │                                                          │
│           ▼                                                          │
│  data/annotated/vlm/{model}/*_VQA_*.json                             │
│           │                                                          │
│           ▼ (update_vlm_scores.py: aggregate per-model scores)       │
│  data/annotated/{split}/*_VQA_*.json                                 │
│  (vlm_scores: {model: score}, cognitive_score: mean)                │
│                                                                      │
│  scripts/download_models.py ── snapshot_download all 15 models       │
│  scripts/vlm_cognitive_score_vqa.py ── VQA-paradigm GT-normalized    │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      TRAINING (M3)                                   │
│                                                                      │
│  UAVIQADataset ──► UAVIQALightningModule ──► UAVIQANet              │
│  (flat JSON +       │                              │                 │
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
│                    BENCHMARKING (M2)                                 │
│                                                                      │
│  Test manifest ──► benchmark_iqa_methods.py ──► metrics table        │
│                      (pyiqa, 15+ zero-shot methods)                  │
│  Test manifest ──► finetune_baselines.py  ──► fine-tuned metrics     │
│  UAVIQANet       ──► metrics.py            (SRCC, PLCC, RMSE,       │
│                        evaluate_iqa()       Kendall τ,              │
│                        per_task_metrics()    per-task &             │
│                        per_distortion_       per-distortion         │
│                          category_metrics()   category metrics)     │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│              MULTI-GPU OFFLINE INFERENCE (M3+)                       │
│                                                                      │
│  scripts/inference.py ──► InferenceEngine                            │
│  (CLI entry point)       │                                           │
│                          ├── Scheduler → scans inputs, generates     │
│                          │   tasks, enqueues via TaskRepository      │
│                          ├── MpQueue → multiprocessing task queue    │
│                          ├── VLMExecutor / DummyExecutor → workers   │
│                          │   process tasks, return Results           │
│                          ├── Collector → tracks completion,          │
│                          │   notifies Writer                         │
│                          ├── Writer → merge chunks, atomic write     │
│                          ├── Validator → file/record/chunk verify   │
│                          ├── Metrics → runtime stats, ETA            │
│                          └── CheckpointStore → SQLite resume         │
│                                                                      │
│  scripts/reset_empty_tasks.py ── reset hung/zero-output tasks        │
└──────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
AirCopBench (raw)            VLM Models (HF Hub)
      │                              │
      ▼                              ▼
distortion_synthesis.py     download_models.py
  extract → inject                │
      │                            │
      ▼                            ▼
data/processed/              cached model weights
  *_VQA_*.json                      │
  (grouped entries)                 │
      │                            │
      └──────────┬─────────────────┘
                 │
                 ▼
          BatchAnnotator
          (score_multi_image:
           ref → dist text sim)
                 │
                 ▼
          data/annotated/vlm/
          {model}/{split}/
          *_VQA_*.json
          (bleu, rouge_l, cider)
                 │
                 ▼
          update_vlm_scores.py
          (compute cognitive_score,
           average across models)
                 │
                 ▼
          data/annotated/
          {split}/*_VQA_*.json
          (vlm_scores, cognitive_score)
                 │
                 ▼
          scripts/train.py
          (LightningCLI)
                 │
                 ▼
          outputs/{run}/
          (checkpoints, metrics.csv)

Benchmarking (separate):
  test set → benchmark_iqa_methods.py → pyiqa metrics
  test set → finetune_baselines.py → fine-tuned baselines
```

## Component Relationships

```
scripts/train.py (LightningCLI)
  └── configs/experiments/r*.yaml
        └── model: UAVIQANet (model.py)
        │     ├── backbone (MobileNetV4-S, MobileViT-S, EfficientViT-B0)
        │     ├── PANet FPN
        │     ├── CBAM (channel + spatial attention)
        │     ├── FrequencyAwareBranch (patch FFT + log-polar + CNN)
        │     └── TaskConditionedHead (FiLM modulation per task)
        ├── data: UAVIQADataModule (data_module.py)
        │     └── UAVIQADataset (dataset.py) → flat JSON
        └── trainer: Lightning Trainer + CSVLogger/WandbLogger

BatchAnnotator (batch_annotator.py)
  └── VLMScorer (vlm/scorer.py)
        └── score_multi_image()
              ├── _generate_answer_multi() → VLM inference
              └── compute_cognitive_score() → BLEU/ROUGE-L/CIDEr
                    (text_metrics.py)

InferenceEngine (inference/engine.py)
  ├── Scheduler → TaskRepository → MpQueue → Worker
  ├── Worker → VLMExecutor → Result → MpQueue → Collector
  └── Collector → Writer → Validator → Metrics
```

## Key Changes Since Last Update (2026-07-01)

| Change | Detail |
|--------|--------|
| `data_synthesis.py` → `distortion_synthesis.py` | Renamed; VLM annotate/aggregate removed (now in BatchAnnotator) |
| `vlm_annotate.py` deleted | CLI replaced by BatchAnnotator API + `update_vlm_scores.py` |
| `figures/` added | 6 paper figures + gen_all_figures.py + paper_plot_style.py + latex_includes.tex |
| `reset_empty_tasks.py` added | Reset hung/zero-output tasks in inference checkpoint |
| `update_vlm_scores.py` added | Aggregate per-model VLM scores into multi-model average |
| `PAPER_PLAN.md`, `MANIFEST.md` added | Paper planning and output manifest |
| Inference vLLM improvements | Batch inference, continuous batching, CUDA graph fix |
