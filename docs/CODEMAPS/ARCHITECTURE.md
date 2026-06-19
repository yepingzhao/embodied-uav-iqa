# Architecture Codemap

**Last Updated:** 2026-06-19

## High-Level System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA PREPARATION (M1)                      │
│                                                                  │
│  AirCopBench ──► extract_refs ──► inject_distortions ──►        │
│  Dataset         (clean frames)    (24 × 5 per ref)              │
│                                       │                          │
│                                       ▼                          │
│                               manifest.json                      │
│                               (train/val/test splits)            │
│                                       │                          │
└───────────────────────────────────────┼──────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────┐
│                        TRAINING (M3)                            │
│                                                                  │
│  UAVIQADataset ──► UAVIQALightningModule ──► UAVIQANet          │
│  (manifest)          │                              │            │
│                      │  Loss: MSE + ListMLE +       │            │
│                      │  CrossTaskRegularization     │            │
│                      └──────────────────────────────┘            │
│                                  │                               │
│                    CurriculumStageCallback                       │
│                    (VLM → VLA → Execution)                      │
│                                  │                               │
│                              Output                              │
│                        (best_model.pt, results.json)             │
└───────────────────────────────────┬──────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                     BENCHMARKING (M2)                            │
│                                                                  │
│  Test manifest ──► pyiqa (15+ methods) ──► metrics table        │
│  UAVIQANet       ──► evaluate.py           (SRCC, PLCC, RMSE)   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
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
│  18 Generic ───────────┐                            │
│  (via Albumentations)   │                            │
│  ├─ blur (3)            │                            │
│  ├─ brightness (5)      │                            │
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
│              MobileNetV4-S Backbone               │
│  (timm, pretrained, stage 0-1 frozen)            │
│  out_indices=(2, 3, 4) → 3 feature scales        │
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
└─────────────────────────────────────────────┘
```

## Component Relationships

```
UAVIQACLI (LightningCLI)
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
  │              └── uses → evaluate_iqa
  ├── calls → UAVIQDataModule
  │              └── owns → UAVIQADataset
  ├── adds → CurriculumStageCallback
  ├── adds → MetricsHistoryCallback
  └── adds → ModelCheckpoint
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
