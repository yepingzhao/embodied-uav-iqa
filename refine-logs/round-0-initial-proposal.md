# Research Proposal: Task-Aware Embodied IQA for Low-Altitude UAV Perception

## Problem Anchor

- **Bottom-line problem**: In low-altitude UAV embodied intelligence, visual inputs suffer from UAV-specific degradations (propeller vibration blur, atmospheric scattering, 6DoF viewpoint change blur, communication packet-loss artifacts, low-resolution+super-resolution artifacts, propeller shadow modulation) that differ fundamentally from both human-centric and ground-robot distortions. Yet no quality assessment framework exists to predict whether a given visual input is adequate for a specific UAV downstream task.

- **Must-solve bottleneck**: Current embodied IQA databases (EPD, Embodied-IQA) cover only fixed-base manipulator robots in indoor settings. UAV scenarios introduce (a) 6 novel distortion types with distinct frequency signatures, (b) extreme SWaP constraints requiring <10M parameter models, (c) multi-task conditioning (inspection/tracking/delivery/SAR have different quality requirements), and (d) the need for real-time closed-loop integration (<10ms inference). No existing method addresses all four simultaneously.

- **Non-goals**: This is NOT about (a) image restoration or enhancement — we assess quality, not fix images; (b) general UAV perception benchmark — we focus on IQA, not detection/tracking accuracy per se; (c) multi-UAV collaboration algorithms — we use AirCopBench's multi-view data but don't design new collaboration strategies; (d) full SC³ system integration — the closed-loop deployment is Phase 4 future work.

- **Constraints**: 
  - Compute: University GPU cluster (A100-level) for training; Jetson Orin for deployment benchmarking
  - Data: Must assemble from existing open-source sources (AirCopBench, CARLA-Air, MotionScape) — no budget for new real-world UAV data collection in Phase 1-2
  - Time: 14-16 months for database + model (Phase 2+3 combined)
  - Model size: <10M parameters, INT8 quantized <2MB for onboard deployment
  - Annotation: VLM/VLA-based annotation via existing models (no human annotator budget for large-scale labeling)

- **Success condition**: A trained NR-IQA model that (a) predicts task-specific quality scores correlating with VLA decision degradation at SRCC > 0.7 on held-out UAV tasks, (b) runs at <10ms on Jetson Orin, and (c) the UAV-specific distortion types are shown to be necessary (ablation: removing them degrades prediction).

## Technical Gap

### Why current methods fail

1. **Embodied-IQA (2505.16815) and EPD/MA-EIQA (2412.18774)** established the paradigm: define image quality by robot task success, not human preference. But their scope is limited to fixed-base manipulators (UR5 arm, SAPIEN simulator) with 25-30 generic distortion types (Gaussian blur, JPEG, color shifts, etc.). These distortions capture indoor manipulation degradation but miss the UAV-specific frequency-domain degradations that dominate aerial perception.

2. **AirCopBench (2511.11025)** provides multi-UAV collaborative perception data with built-in perception assessment VQA protocols (quality/availability assessment across 14.6k questions). It includes real-world UAV degradations (motion blur, noise, data loss) in its sim+real data. But it uses generic MLLM scoring for quality assessment — it has NO dedicated IQA model, NO task-specific quality metrics, and NO distortion-level annotations. It's a perception benchmark, not an IQA database.

3. **The gap**: Embodied-IQA provides the annotation methodology (VLM→VLA→robot pipeline) and the theoretical framework (Mertonian perception-cognition-decision-execution pipeline). AirCopBench provides UAV-specific task protocols and multi-view degraded data. But these two lines of work have never been combined. No one has applied embodied IQA methodology to UAV perception data.

### Why naive fixes are insufficient

- **Just adding UAV images to existing IQA databases**: Doesn't work because the annotation must be task-conditioned. A blur that's acceptable for scene description may be catastrophic for precision tracking.

- **Just using AirCopBench's VQA scores as IQA labels**: AirCopBench's perception assessment VQA produces discrete accuracy scores, not continuous quality scores. The 2 assessment dimensions (quality/availability) are MLLM-centric, not task-performance-centric.

- **Just fine-tuning MA-EIQA on UAV data**: MA-EIQA's 48.83M parameters are too large for UAV onboard deployment. Its ResNet50 backbone has no frequency-aware processing for UAV-specific distortions. Its single regression head cannot handle multi-task conditioning.

- **Just stacking more modules**: Adding a frequency branch + task heads + temporal consistency + uncertainty estimation would create a bloated model that violates the <10M parameter constraint.

### Smallest adequate intervention

A **task-conditioned, frequency-aware lightweight NR-IQA model** trained on a purpose-built UAV-Embodied-IQA database. The key insight: UAV-specific distortions have distinct **frequency-domain signatures** (propeller vibration = periodic high-frequency, atmospheric scattering = low-frequency attenuation, block artifacts from packet loss = grid-pattern high-frequency), and different UAV tasks are sensitive to **different frequency bands**. A compact frequency branch alongside a spatial CNN backbone provides the minimal additional capacity needed to capture these signatures, while task embeddings modulate the quality regression to produce task-specific scores from a shared feature representation.

### Core technical claim

**Frequency-aware task conditioning with <2 new trainable components can bridge the embodied IQA → UAV domain gap, achieving SRCC > 0.7 on UAV task performance prediction with <10M parameters.**

This is a mechanism-level claim: the frequency branch + task-conditioned regression head are the minimal sufficient additions to adapt embodied IQA to UAV scenarios. The claim is falsifiable — if removing either component doesn't degrade performance on UAV-specific distortions, the claim is wrong.

### Required evidence

1. UAV-specific distortion types are necessary (ablation: model trained without them underperforms on real UAV data)
2. Frequency branch improves prediction specifically for frequency-domain distortions (vibration blur, packet-loss blocks)
3. Task conditioning improves per-task correlation vs. a single task-agnostic head
4. The full model <10M parameters achieves SRCC > 0.7 on held-out tasks, outperforming both generic NR-IQA and unadapted MA-EIQA

## Method Thesis

- **One-sentence thesis**: A frequency-aware task-conditioned lightweight NR-IQA model, trained on the first UAV-Embodied-IQA database that marries Embodied-IQA's VLM→VLA annotation pipeline with AirCopBench's multi-UAV task protocols, can predict task-specific visual input adequacy for aerial embodied intelligence under extreme SWaP constraints.

- **Why this is the smallest adequate intervention**: We reuse (1) Embodied-IQA's three-stage annotation pipeline, (2) AirCopBench's task protocols and multi-view data, (3) MobileNetV4 as a strong lightweight backbone, and (4) the PANet+CBAM architecture pattern from MA-EIQA. We add only two novel trainable components: a frequency-aware branch (for UAV-specific distortion signatures) and task-conditioned regression heads (for multi-task quality prediction). Everything else is assembly, not invention.

- **Why this route is timely in the foundation-model era**: VLM/VLA models (Qwen2.5-VL, UAV-Track VLA, CognitiveDrone) are now capable enough to serve as annotation oracles for embodied IQA — this was not true 2 years ago. CARLA-Air provides stable UAV simulation (357 resets, zero crashes) that makes distortion injection experiments reproducible. The convergence of (a) mature VLM/VLA annotators, (b) stable UAV simulation, and (c) established embodied IQA methodology means this database+model can be built now by assembling existing components, not by inventing new infrastructure.

## Contribution Focus

- **Dominant contribution**: The first UAV-Embodied-IQA database that (a) defines 6 UAV-specific distortion types with mathematical models, (b) provides VLM+VLA task-performance annotations across 4 UAV task categories, and (c) establishes the first benchmark for UAV-specific NR-IQA.

- **Optional supporting contribution**: UAV-IQANet — a <10M parameter frequency-aware task-conditioned NR-IQA model that achieves SRCC > 0.7 on UAV task performance prediction and runs at <10ms on Jetson Orin.

- **Explicit non-contributions**: 
  - We do NOT claim novelty in the annotation pipeline (reuses Embodied-IQA's three-stage protocol)
  - We do NOT claim novelty in the backbone architecture (uses MobileNetV4 + PANet + CBAM)
  - We do NOT claim novelty in multi-UAV collaboration (uses AirCopBench's existing protocols)
  - We do NOT claim to improve VLM/VLA models themselves

## Proposed Method

### Complexity Budget

- **Frozen / reused backbone**: MobileNetV4-S (pre-trained on ImageNet-21k), PANet-style FPN for multi-scale feature fusion, CBAM-style channel+spatial attention
- **New trainable components** (≤2):
  1. **Frequency-Aware Branch (FAB)**: A lightweight FFT-based branch that extracts frequency-domain features from input patches, processes them through a tiny 2-layer CNN, and fuses with spatial features via cross-attention gating
  2. **Task-Conditioned Regression Head (TCRH)**: A task embedding layer (4-dimensional learned embedding per task type) that modulates the final regression MLP via FiLM (feature-wise linear modulation)
- **Tempting additions intentionally not used**:
  - Temporal consistency regularizer (adds complexity, only useful for video)
  - Uncertainty estimation head (adds parameters, not needed for threshold-based control)
  - Multi-UAV fusion module (Phase 4 work, not this paper)
  - Distortion classification auxiliary head (frequency branch already captures this implicitly)

### System Overview

```
Input Image (H×W×3)
    │
    ├──→ MobileNetV4-S Backbone (frozen stem, trainable stage 3-5)
    │         │
    │         ├──→ C3 (stage 3 features, 1/8 resolution)
    │         ├──→ C4 (stage 4 features, 1/16 resolution)  
    │         └──→ C5 (stage 5 features, 1/32 resolution)
    │                   │
    │                   ▼
    │         PANet-style FPN (top-down + bottom-up)
    │                   │
    │                   ▼
    │         Multi-scale fused features {P3, P4, P5}
    │                   │
    │                   ▼
    │         CBAM Attention (channel + spatial)
    │                   │
    │                   ▼
    │         Spatial Feature Vector f_s ∈ R^256
    │
    └──→ Patch-wise FFT (32×32 patches, stride 16)
              │
              ▼
         Frequency magnitude spectrum |F(u,v)|
              │
              ▼
         Log-polar transform (rotation/scale invariant)
              │
              ▼
         Tiny CNN (Conv3×3→BN→ReLU→Conv3×3→BN→ReLU)
              │
              ▼
         Global Average Pool → f_f ∈ R^64
              │
              ▼
         Cross-Attention Gate: α = σ(W_g · [f_s, f_f])
              │
              ▼
         Fused Feature: f = Concat[f_s, α ⊙ f_f] ∈ R^320
              │
              ▼
    ┌────────┴────────┬────────┬────────┐
    │                 │        │        │
    ▼                 ▼        ▼        ▼
  TCRH              TCRH     TCRH     TCRH
  (track)           (inspect) (deliver) (SAR)
    │                 │        │        │
    ▼                 ▼        ▼        ▼
  q_track          q_inspect q_deliver q_SAR

TCRH details (per task):
  t_k ∈ R^4 (learned task embedding for task k)
  FiLM: h' = γ(t_k) ⊙ h + β(t_k)  where h = f after first FC
  2-layer MLP: f → FC(320→128) → FiLM → FC(128→1) → q_k
```

### Core Mechanism

**1. Frequency-Aware Branch (FAB) — the main novelty**

- **Input/Output**: Takes the raw RGB image; outputs a 64-dim frequency feature vector.
- **Architecture**: 
  - Divide image into 32×32 patches with stride 16 (overlapping patches to avoid boundary artifacts)
  - Per-patch 2D FFT → magnitude spectrum |F(u,v)|
  - Log-polar transform for rotation/scale invariance (UAV viewpoint changes cause rotation/scaling of frequency patterns)
  - Feed through a tiny 2-layer CNN (3×3 conv, 8 channels → 16 channels) with BatchNorm and ReLU
  - Global average pool across patches → 64-dim vector
- **Why FFT and not learned**: Vibration blur, atmospheric scattering, and block artifacts have well-defined mathematical frequency signatures. An FFT front-end provides strong inductive bias with zero learned parameters in the transform itself. The tiny CNN only needs to learn to discriminate frequency patterns, not to rediscover the Fourier transform.
- **Fusion with spatial branch**: A lightweight cross-attention gate computes a scalar gating factor α per frequency feature dimension, conditioned on the spatial feature. This allows the model to ignore frequency features when they're uninformative (e.g., for color distortions that have weak frequency signatures).

**2. Task-Conditioned Regression Head (TCRH) — the supporting novelty**

- **Input/Output**: Takes the fused spatial-frequency feature (320-dim); outputs a task-specific quality score q_k ∈ [0, 1].
- **Task embedding**: Each of K tasks has a learned 4-dim embedding t_k. 4 dimensions suffice because tasks differ primarily along 2 axes: (a) precision requirement (tracking needs sub-pixel accuracy vs. scene description is coarse), (b) temporal sensitivity (tracking is frame-by-frame critical vs. inspection can tolerate occasional bad frames).
- **Modulation mechanism**: FiLM (feature-wise linear modulation) after the first FC layer:
  - γ(t_k) = W_γ · t_k + b_γ  (scale, shape: 128)
  - β(t_k) = W_β · t_k + b_β   (shift, shape: 128)
  - h' = γ ⊙ h + β
- **Why FiLM and not separate heads**: Completely separate heads per task would multiply parameters by K. FiLM shares the core regression MLP across tasks while allowing task-specific modulation — this is parameter-efficient and forces the shared feature to be genuinely task-relevant.

### Training Plan

**Data Construction**:
- **Reference images**: ~1,500 UAV frames from AirCopBench (sim + real) + CARLA-Air new captures + MotionScape high-dynamic samples
- **Distortion injection**: 18 generic (from Embodied-IQA catalog) + 6 UAV-specific (see below), each at 5 intensity levels → ~1,500 × 24 × 5 = 180,000 distorted images
- **Pseudo-label generation**: 
  - Stage 1 (Cognitive): 3 VLMs (Qwen2.5-VL-7B, InternVL2-8B, LLaVA-NeXT-13B) run AirCopBench perception assessment VQA on each distorted image → BLEU/ROUGE/CIDEr text similarity vs. clean reference → cognitive quality score
  - Stage 2 (Decision): 3 VLAs (UAV-Track VLA, CognitiveDrone-R1, Qwen-VLA) run tracking/cognitive tasks on distorted images → trajectory deviation (position/velocity/heading error vs. clean reference) → decision quality score
  - Stage 3 (Execution, optional subset): CARLA-Air SITL flight with distorted input → task success rate → execution quality score
  - Final MOS: Weighted average of cognitive (0.3) + decision (0.5) + execution (0.2) scores, per task
- **Dataset split**: 70% train / 15% val / 15% test, stratified by distortion type and task

**UAV-Specific Distortion Models** (the 6 novel additions):

1. **Propeller Vibration Blur** — modeled as directional motion blur with periodic intensity modulation:
   ```
   I_vib(x,y) = ∫_0^T I(x + A·sin(2πft)·cos(θ), y + A·sin(2πft)·sin(θ)) · w(t) dt
   ```
   where f ∈ [80, 200] Hz (propeller RPM range), A ∈ [1, 8] pixels, θ ∼ Uniform(0, 2π)

2. **Atmospheric Scattering / Haze** — Koschmieder model with altitude-dependent parameters:
   ```
   I_haze(x) = I(x) · e^(-β·d(x)) + A_∞ · (1 - e^(-β·d(x)))
   ```
   where β ∈ [0.5, 3.0] (scattering coefficient, higher for low-altitude haze), A_∞ is atmospheric light

3. **6DoF Fast Viewpoint Change Blur** — motion vectors sampled from MotionScape's empirical optical flow distribution (mean 36.63 pixels):
   ```
   I_6dof = I ⊗ PSF(v),  v ∼ MotionScapeFlowDistribution(μ=36.63, σ=25.4)
   ```
   where PSF is a linear motion kernel with direction and magnitude sampled from the distribution

4. **Communication Packet-Loss Block Artifacts** — random macroblock replacement simulating SC³ data packet loss:
   ```
   I_comm(x,y) = M(x,y) ⊙ I(x,y) + (1-M(x,y)) ⊙ I_nearest(x,y)
   ```
   where M is a random macroblock mask (16×16 blocks), loss rate ∈ [1%, 30%], replacement with nearest valid block

5. **Low-Resolution + Super-Resolution Artifacts** — downsample → SR upscale chain:
   ```
   I_lr-sr = Real-ESRGAN( BicubicDownsample(I, s) ),  s ∈ [2, 8]
   ```
   capturing the oversharpening, hallucinated textures, and ringing artifacts of real SR models

6. **Propeller Shadow** — periodic localized brightness modulation:
   ```
   I_shadow(x,y) = I(x,y) · (1 - α · Π(f_prop · t mod 1, w(x,y)))
   ```
   where Π is a periodic rectangular pulse with spatial frequency matching propeller rotation, α ∈ [0.05, 0.3] (shadow depth), affected region w(x,y) depends on sun angle

**Training Recipe**:
- **Loss**: L = L_MSE + λ_rank · L_ListMLE + λ_task · L_cross-task
  - L_MSE: MSE between predicted q̂_k and pseudo-label q_k
  - L_ListMLE: ListMLE ranking loss within each distortion type (model should correctly rank 5 intensity levels)
  - L_cross-task: Regularization that penalizes predicting identical scores across tasks for the same image
    ```
    L_cross-task = -Σ_{i,j, i≠j} |q̂_i - q̂_j|  (encourages task-specific differentiation)
    ```
- **Hyperparameters**: λ_rank = 0.3, λ_task = 0.1, AdamW with lr=1e-4, batch_size=64, cosine schedule
- **Training stages**:
  1. Warm-up: Train only TCRH (freeze backbone + FAB) for 5 epochs — learn task embeddings
  2. Joint training: Unfreeze all, train for 50 epochs with cosine decay
  3. Fine-tuning: 10 additional epochs on hardest 20% samples (highest L_ListMLE)
- **Data augmentation**: Random horizontal flip, rotation ±15°, brightness jitter ±0.1 (mild — we're training on distorted images already)
- **Validation**: Per-task SRCC on validation split; early stopping with patience=10 based on mean SRCC
- **Quantization**: Post-training INT8 quantization via ONNX Runtime; calibration on 1000 validation samples

### Integration into Downstream Pipeline

At inference time:
```
Camera Frame → UAV-IQANet (frequency + spatial branches)
                    ↓
            {q_track, q_inspect, q_deliver, q_SAR}
                    ↓
            Task Selector (based on current mission)
                    ↓
            q_current = q_{active_task}
                    ↓
            Threshold-based action:
              q > 0.7 → normal operation (lightweight perception model)
              q ∈ [0.4, 0.7] → degraded mode (robust model, reduced speed)
              q < 0.4 → emergency (hover, request collaborative view)
```

The model is deployed as an ONNX Runtime graph on Jetson Orin, running asynchronously alongside the main perception pipeline.

### Failure Modes and Diagnostics

- **Failure mode 1**: Frequency branch overfits to simulation-specific vibration patterns and fails on real UAV data.
  - **Detection**: Monitor FAB activation patterns; if real UAV images produce near-zero α gating values, the branch is ignoring real frequency signatures.
  - **Mitigation**: Include MotionScape real-world high-dynamic video frames in training (already planned); add frequency-domain data augmentation (random spectral perturbation).

- **Failure mode 2**: Task embeddings collapse to near-identical vectors, losing task specificity.
  - **Detection**: Compute pairwise cosine similarity of learned t_k embeddings; if >0.9, collapse has occurred.
  - **Mitigation**: Increase λ_task; add explicit task-discrimination loss (contrastive loss between task embeddings).

- **Failure mode 3**: VLA-based pseudo-labels are too noisy (model disagreement SRCC ≈ 0.25 per Embodied-IQA).
  - **Detection**: Compute inter-VLA annotation consistency on validation set; if per-task SRCC < 0.3, labels are unreliable.
  - **Mitigation**: Use only the best-performing VLA per task (instead of averaging); add execution-layer real robot validation on a subset.

- **Failure mode 4**: Model is still too large after pruning/quantization.
  - **Detection**: Measure ONNX model size and Jetson Orin inference latency.
  - **Mitigation**: Reduce MobileNetV4 width multiplier; remove PANet top-down path (use only bottom-up); reduce task embedding dimension to 2.

### Novelty and Elegance Argument

**Closest work — MA-EIQA (2412.18774)**: 
- MA-EIQA has ResNet50 backbone (48.83M), PANet FPN, CBAM attention, single regression head.
- Our key differences: (1) frequency-aware branch for UAV-specific distortions (MA-EIQA is purely spatial), (2) task-conditioned regression via FiLM (MA-EIQA outputs one score for all tasks), (3) MobileNetV4 backbone (<5M vs. 23M+), (4) trained on UAV-specific distortions that MA-EIQA has never seen.
- These are mechanism-level differences, not just "we changed the backbone and added data."

**Closest work — Embodied-IQA database (2505.16815)**:
- Their database covers ground manipulation (UR5 arm, SAPIEN simulator, 30 generic distortions).
- Our database covers aerial embodied intelligence (UAV, CARLA-Air, 18 generic + 6 UAV-specific distortions), with multi-task annotation (4 task types vs. their 2 manipulation tasks).
- We reuse their annotation methodology but apply it to a fundamentally different embodiment with novel distortion types.

**Why this is focused, not a module pile-up**:
- The spatial branch (MobileNetV4 + PANet + CBAM) is a standard strong baseline — we don't claim novelty here.
- The frequency branch is the ONE new perception mechanism, justified by the frequency-domain nature of UAV distortions.
- The task-conditioned head is the ONE new architectural pattern, justified by multi-task UAV deployment.
- Two new components, one clear story: "UAV distortions are frequency-domain phenomena, and UAV tasks have different quality requirements; our model handles both."

## Claim-Driven Validation Sketch

### Claim 1: UAV-specific distortions are necessary for predicting UAV task performance
- **Minimal experiment**: Train two versions of UAV-IQANet — one with all 24 distortion types, one with only 18 generic types. Compare SRCC on real UAV test data (AirCopBench real subset + MotionScape).
- **Baselines / ablations**: Full model vs. generic-only model; also test zero-shot transfer of MA-EIQA (trained on EPD) to UAV data.
- **Metric**: SRCC / PLCC between predicted quality scores and VLA decision deviation (trajectory error).
- **Expected evidence**: Full model SRCC at least 0.10 higher than generic-only model; MA-EIQA zero-shot SRCC < 0.3 on UAV data.

### Claim 2: Frequency-aware branch improves prediction for frequency-domain distortions
- **Minimal experiment**: Ablation study — remove FAB (spatial-only baseline), compare per-distortion SRCC across all 24 distortion types.
- **Baselines / ablations**: Full model vs. no-FAB vs. FAB with random frequency features (control for just "more parameters").
- **Metric**: Per-distortion SRCC, with focus on the 6 UAV-specific frequency-domain distortions vs. 18 generic distortions.
- **Expected evidence**: FAB provides >5% SRCC improvement on propeller vibration blur, atmospheric scattering, and packet-loss blocks specifically; minimal improvement (<2%) on color/compression distortions.

### Claim 3: Task conditioning improves per-task correlation
- **Minimal experiment**: Train with and without TCRH (single shared regression head vs. FiLM-modulated heads), compare per-task SRCC.
- **Baselines / ablations**: Single-head baseline, separate-heads baseline (independent heads, more parameters), FiLM-modulated shared head (ours).
- **Metric**: Per-task SRCC (tracking/inspection/delivery/SAR); macro-averaged across tasks.
- **Expected evidence**: TCRH achieves higher per-task SRCC than single-head baseline with fewer parameters than separate-heads baseline.

## Experiment Handoff Inputs

- **Must-prove claims**: 3 claims above (UAV distortion necessity, frequency branch benefit, task conditioning benefit)
- **Must-run ablations**: FAB removal, TCRH removal, UAV distortion removal, backbone scale ablation (MobileNetV4-S vs. -XS vs. -XXS)
- **Critical datasets / metrics**: AirCopBench real subset for external validation; SRCC/PLCC for correlation; Jetson Orin latency for deployment feasibility
- **Highest-risk assumptions**: VLA pseudo-label reliability (need execution-layer validation subset); sim-to-real transfer of frequency signatures (need real UAV vibration data from MotionScape)

## Compute & Timeline Estimate

- **Estimated GPU-hours**: 
  - Distortion injection pipeline: ~50 GPU-hours (primarily Real-ESRGAN for SR artifacts)
  - VLM annotation (3 models × 90,000 images): ~500 GPU-hours on A100 (vLLM batch inference)
  - VLA annotation (3 models × 90,000 images): ~800 GPU-hours on A100 (more expensive, trajectory rollout)
  - Model training (including ablations): ~200 GPU-hours on A100
  - Total: ~1,550 GPU-hours (feasible on a 4×A100 cluster in ~3 weeks)
- **Data / annotation cost**: $0 (all open-source data and models)
- **Timeline**: 14-16 months total
  - Months 1-3: Environment setup + distortion pipeline + data collection
  - Months 4-7: VLM + VLA annotation (can parallelize across models)
  - Months 8-10: Model implementation + training + ablation studies
  - Months 11-12: Quantization + Jetson deployment benchmarking
  - Months 13-16: Paper writing + revision + supplementary experiments
