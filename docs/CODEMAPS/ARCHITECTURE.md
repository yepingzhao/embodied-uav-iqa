# Architecture Codemap

**Last Updated:** 2026-06-21

## High-Level System Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                   DATA PREPARATION (M1)                              │
│                                                                      │
│  Dataset-agnostic pipeline (data_synthesis.py)                       │
│                                                                      │
│  Input Source ───► DatasetFormat ──► DataSynthesisPipeline           │
│  (AirCopBench /    (registry key:     │                              │
│   GenericDir)       "aircopbench",    ├── extract_references         │
│                     "generic")        ├── inject_distortions         │
│                                        │    (33 × 5 per ref)         │
│                                        ├── generate_manifests        │
│                                        │    (train/val/test splits)  │
│                                        └── annotate_scores           │
│                                             │                        │
│                                             ▼                        │
│                                     manifest.json                    │
│                                     (path, task, distortion,         │
│                                      intensity, scores)              │
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
│                   CurriculumStageCallback                            │
│                   (VLM → VLA → Execution)                           │
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
│  27 Generic ───────────┐                            │
│  (via Albumentations)   │                            │
│  ├─ blur (3)            │                            │
  │  ├─ brightness (6)      │                            │
│  ├─ chromatic (3)       │                            │
│  ├─ noise (4)           │                            │
│  ├─ compression (3)     │                            │
│  ├─ spatial (4)         │                            │
│  └─ other (4)           │                            │
│                                                     │
│  5 intensity levels per distortion                   │
└──────────────────────────────────────────────────┘
    │
    ▼
Manifest entry: {path, task, distortion, intensity_level,
                 ref_id, vlm_score, vla_score, execution_score, annotated}
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
│  3-Stage Curriculum:                          │
│    Epochs 1-20:   vlm_score                  │
│    Epochs 21-40:  vla_score                  │
│    Epochs 41-50:  execution_score            │
│                                              │
│  Logging: self.log() → WandbLogger         │
│           (cloud: wandb.ai)                  │
│           + CSVLogger (local: metrics.csv)   │
└─────────────────────────────────────────────┘
```

## Component Relationships

```
scripts/data_synthesis.py
  └── calls → DataSynthesisPipeline (data_synthesis.py)
                  │
                  ├── uses → DatasetFormat (registry: AirCopBenchFormat, GenericImageDirFormat)
                  ├── uses → UAVDistortionPipeline (distortion.py)
                  ├── uses → annotations.py (build_ref_score_lookup, assign_task_label, etc.)
                  └── uses → utils.py (find_images, split_samples, write_manifest)

scripts/finetune_baselines.py
  ├── uses → UAVIQADataset (dataset.py) — reuses existing data loading
  ├── uses → metrics.py (evaluate_iqa, per_task_metrics, per_distortion_category_metrics)
  ├── uses → utils.py (load_image_tensor, load_manifest, setup_logging)
  ├── owns → BaselineLightningModule (standalone wrapper around pyiqa model)
  │            └── owns → pyiqa model (brisque/niqe/clipiqa/maniqa/topiq_nr)
  └── owns → BaselineDataModule (standalone, no LightningCLI)
                 └── owns → UAVIQADataset (reuses manifest data loading)

scripts/fix_configs.py
  └── utility — batch-converts experiment YAML configs from nested to flat format

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
  ├── calls → UAVIQDataModule
  │              └── owns → UAVIQADataset
  │                     └── uses → utils.py (load_image_tensor, load_manifest)
  ├── adds → SetupRunCallback (manifest SHA256, param count)
  ├── adds → CurriculumStageCallback (VLM→VLA→Execution)
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
| 3-stage curriculum | Progressive supervision: cheap VLM → medium VLA → expensive execution |
| ListMLE loss | Per-distortion ranking signal improves relative quality ordering |
| Feature sharing (forward_features) | Backbone+FPN+FAB computed once, reused for pred + cross-task loss |
| Dynamic stage probing | Probes backbone forward pass to determine n_stages, selects last 3 — compatible with MobileNetV4, EfficientViT, MobileViT |
| Regex-based backbone freeze | `_freeze_backbone_stages` matches `blocks.N`/`stages.N`/`stages_N` via `re` — supports diverse timm backbones |
| scripts/train.py + LightningCLI | Replaces custom UAVIQACLI; self-contained YAML per experiment |
| WandbLogger + CSVLogger | Cloud + local dual logging; no cloud dependency for local runs |
