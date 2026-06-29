# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem**: In low-altitude UAV embodied intelligence, visual inputs suffer from UAV-specific degradations (propeller vibration blur, atmospheric scattering, 6DoF viewpoint change blur, communication packet-loss artifacts, low-resolution+super-resolution artifacts, propeller shadow modulation) that differ fundamentally from both human-centric and ground-robot distortions. Yet no quality assessment framework exists to predict whether a given visual input is adequate for a specific UAV downstream task.

- **Must-solve bottleneck**: Current embodied IQA databases (EPD, Embodied-IQA) cover only fixed-base manipulator robots in indoor settings. UAV scenarios introduce (a) 6 novel distortion types with distinct frequency signatures, (b) extreme SWaP constraints requiring <10M parameter models, (c) multi-task conditioning (inspection/tracking/delivery/SAR have different quality requirements), and (d) the need for real-time closed-loop integration (<10ms inference). No existing method addresses all four simultaneously.

- **Non-goals**: This is NOT about (a) image restoration or enhancement — we assess quality, not fix images; (b) general UAV perception benchmark — we focus on IQA, not detection/tracking accuracy per se; (c) multi-UAV collaboration algorithms — we use AirCopBench's multi-view data but don't design new collaboration strategies; (d) full SC³ system integration — the closed-loop deployment is Phase 4 future work.

- **Constraints**: 
  - Compute: University GPU cluster (A100-level) for training; Jetson Orin for deployment benchmarking
  - Data: Must assemble from existing open-source sources (AirCopBench, CARLA-Air, MotionScape) — limited budget for new real-world UAV data collection (can do ~50 validation flights)
  - Time: 14-16 months for database + model (Phase 2+3 combined)
  - Model size: <10M parameters, INT8 quantized <2MB for onboard deployment
  - Annotation: VLM/VLA-based annotation via existing models (no human annotator budget for large-scale labeling)

- **Success condition**: A publicly available UAV-Embodied-IQA database (with VLM+VLA task-performance annotations across 4 UAV task categories) that (a) includes 6 mathematically-modeled UAV-specific distortion types shown to be distinct from generic distortions, (b) provides a benchmark where existing NR-IQA methods fail to predict VLA decision degradation (SRCC < 0.5), establishing a clear challenge, and (c) a strong baseline model achieves SRCC > 0.65 with <10M parameters, demonstrating that the database enables model development.

## Anchor Check

- **Original bottleneck**: No quality assessment framework exists to predict whether a given visual input is adequate for a specific UAV downstream task.
- **Why the revised method still addresses it**: The bottleneck has TWO components — (1) no training/evaluation data exists, and (2) no model trained on such data exists. The revised proposal makes the database the PRIMARY contribution (solving component 1), with a strong baseline model demonstrating the database's utility (partially solving component 2). The full model contribution is deferred to Phase 3 separate work. This is a more honest scoping: one paper, one dominant contribution.
- **Reviewer suggestions rejected as drift**: 
  - **"Reduce to 3 UAV distortion types"**: REJECTED. The 6 distortion types each cover a physically distinct UAV failure mode: (1) vibration blur = periodic mechanical, (2) atmospheric scattering = environmental, (3) 6DoF viewpoint blur = ego-motion, (4) packet-loss blocks = communication, (5) low-res+SR = computational, (6) propeller shadow = illumination. Reducing to 3 would lose coverage of the full UAV failure taxonomy. The database's contribution IS the taxonomy — reducing it reduces the contribution.
  - **"Drop FAB and replace with learned frequency features"**: PARTIALLY REJECTED. The explicit FFT + log-polar transform provides a strong inductive bias specifically for rotation/scale-invariant frequency analysis that UAV 6DoF motion requires. Learned filters would need to rediscover the Fourier transform from data, increasing sample complexity. However, the model's novelty claim is downgraded — FAB is now positioned as a baseline design choice, not the paper's contribution.
  - **"Consider VLM feature distillation"**: DEFERRED to Phase 3 model paper. Valid idea but adds a new contribution dimension (distillation methodology) to an already-focused database paper.

## Simplicity Check

- **Dominant contribution after revision**: The UAV-Embodied-IQA database. One clear contribution: the first IQA database for aerial embodied intelligence, with (a) 6 UAV-specific distortion models, (b) VLM+VLA task-performance annotations across 4 UAV task categories, and (c) a benchmark establishing that existing IQA methods fail on UAV data.
- **Components removed or merged**: 
  - Model novelty claims removed entirely — UAV-IQANet is now a "strong baseline" described in the database paper
  - FAB and TCRH are now described as "design rationale for the baseline" rather than "proposed novel components"
  - Cross-attention gate: kept but noted as an ablation candidate (does simple concatenation work as well?)
  - The "Contribution Focus" section now has only ONE entry: the database
- **Reviewer suggestions rejected as unnecessary complexity**:
  - Real-ESRGAN for SR artifact generation (kept — necessary for the low-res+SR distortion type)
  - 3-stage annotation pipeline (kept — inherited from Embodied-IQA, not our invention)
- **Why the remaining mechanism is still the smallest adequate route**: The database construction follows the "assembly over invention" principle: reuse CARLA-Air (simulation), AirCopBench (task protocols), and Embodied-IQA (annotation methodology). The only genuinely new components are the 6 UAV-specific distortion functions — mathematically minimal models that capture physically distinct phenomena.

## Changes Made

### 1. Contribution Focus: Database ONLY (CRITICAL)

- **Reviewer said**: "Two parallel contributions (database + model) dilute focus. Choose one."
- **Action**: Removed all model novelty claims. UAV-IQANet is now a strong baseline described in the database paper. The model-focused contribution is deferred to Phase 3 as a separate paper. The paper type is now explicitly a "database + benchmark paper."
- **Reasoning**: The user's research roadmap already splits Phase 2 (database) and Phase 3 (model) into separate papers. This proposal now correctly scopes to Phase 2 only. The model's architectural choices (FAB, TCRH) remain in the method section but as "baseline design rationale," not as "proposed novel contributions."
- **Impact on core method**: Simplifies the contribution story from 2 claims to 1. The success condition now focuses on the database establishing that existing IQA fails on UAV data (SRCC < 0.5 on strong baselines) rather than the model achieving SRCC > 0.7.

### 2. Frequency IQA Prior Art Positioning (IMPORTANT)

- **Reviewer said**: "FAB is positioned as main novelty but doesn't cite prior frequency-domain IQA methods."
- **Action**: Added an explicit comparison to prior frequency-aware IQA in the Novelty and Elegance Argument section: (a) BRISQUE (2012) uses DCT-domain NSS features — spatially local, not rotation/scale invariant; (b) DeepFIQA uses DCT coefficients as input channels — still spatially local; (c) MANIQA and other ViT-based methods capture frequency only implicitly through self-attention. Our FAB uses log-polar FFT which is rotation/scale-invariant — necessary for 6DoF UAV viewpoint changes where the same vibration pattern appears at different orientations and scales.
- **Reasoning**: This addresses the reviewer's concern without changing the architecture. The log-polar transform IS the key differentiator from prior frequency IQA — but this was implied, not stated.
- **Impact on core method**: No architectural change. The FAB remains the same but is better motivated.

### 3. Real-UAV Validation Subset (IMPORTANT)

- **Reviewer said**: "Add small-scale real UAV experiment to validate synthetic distortions."
- **Action**: Added a Phase 2 validation step: ~50 real-world UAV flights (DJI Mini or equivalent) capture images with natural + induced distortions (fly in hazy conditions, induce vibration by rapid maneuvering, test at different altitudes). Compare VLA behavior on synthetic vs. real distorted images — correlation SRCC should be > 0.6 to validate the synthetic distortion models. Also, compare frequency signatures (log-polar FFT spectra) of real vs. synthetic distortions to verify the mathematical models.
- **Reasoning**: This is the minimal real-world grounding needed to defend the database against "simulation-only" criticism. 50 flights is feasible within the constraints (weekend fieldwork, no specialized hardware beyond a consumer drone).
- **Impact on core method**: No change to the database construction pipeline. Adds a validation appendix to the paper.

### 4. ViT Backbone Baselines in Benchmark (IMPORTANT)

- **Reviewer said**: "No justification for CNN over ViT backbones."
- **Action**: Added MobileViT-S (~5.6M params) and EfficientViT-B0 (~5.7M params) as backbone alternatives in the benchmark comparison table. The baseline model section now includes a backbone comparison: MobileNetV4-S vs. MobileViT-S vs. EfficientViT-B0, with <10M parameter budget for all. This turns the architecture choice from an assumption into an empirical finding.
- **Reasoning**: Easy to add as a benchmark comparison and directly addresses the reviewer's concern. If MobileNetV4-S does outperform ViT alternatives for frequency-domain distortions, this becomes a finding. If not, MobileViT-S becomes the baseline backbone.
- **Impact on core method**: Minimal — one additional experiment in the benchmark table.

### 5. VLA Label Reliability: Execution Layer Mandatory (IMPORTANT)

- **Reviewer said**: "VLA inter-model SRCC ≈ 0.25 makes pseudo-label ground truth questionable."
- **Action**: (a) Elevated the execution layer (Phase 3 annotation) from "optional" to mandatory, on a 5% stratified subset (~9,000 image pairs). (b) Added a label noise analysis: report per-VLA and per-VLM performance variance, inter-annotator agreement statistics, and how label noise affects downstream model training. (c) Added a calibration step: for the 5% execution-validated subset, learn a linear calibration that maps VLA ensemble output to real execution scores.
- **Reasoning**: Embodied-IQA (2505.16815) established that VLA decision scores correlate with real robot execution at SRCC > 0.6, which is stronger than VLM-only (SRCC < 0.5). Having execution validation on a subset provides the ground-truth anchor that calibrates the pseudo-labels. 9,000 pairs is feasible (50-100 real flights × 24 distortions × 5 levels on a subset of reference images).
- **Impact on core method**: Increases annotation cost by ~50 GPU-hours (SITL flight simulation). Adds methodological rigor.

### 6. FAB Implementation Details Clarified (MINOR)

- **Reviewer said**: "Log-polar grid size, CNN channels, cross-attention dimensions not specified."
- **Action**: Specified: log-polar output grid is 32 angular × 16 radial bins (total 512 frequency bins per patch, pooled to 64-dim by tiny CNN). Cross-attention gate: W_g maps [256+64=320] → 64 dimensions (via FC(320→128→64) with ReLU), producing α ∈ R^64. FAB parameter count: ~3K parameters (tiny CNN: 8→16 channels, 3×3 conv = 1,152 params + BN params; cross-attention FCs: ~25K params; total FAB < 30K params).
- **Reasoning**: Documentation fix only.
- **Impact on core method**: None.

---

## Revised Proposal

# Research Proposal: UAV-Embodied-IQA — A Large-Scale Benchmark for Visual Quality Assessment in Aerial Embodied Intelligence

## Problem Anchor

- **Bottom-line problem**: In low-altitude UAV embodied intelligence, visual inputs suffer from UAV-specific degradations (propeller vibration blur, atmospheric scattering, 6DoF viewpoint change blur, communication packet-loss artifacts, low-resolution+super-resolution artifacts, propeller shadow modulation) that differ fundamentally from both human-centric and ground-robot distortions. Yet no quality assessment framework exists to predict whether a given visual input is adequate for a specific UAV downstream task.

- **Must-solve bottleneck**: No database exists to train or evaluate IQA models for UAV embodied perception. Current embodied IQA databases (EPD: 12,500 pairs, ground manipulator; Embodied-IQA: 36,900 pairs, ground manipulator) cover only fixed-base robots indoors. UAV-specific challenges: (a) 6 novel distortion types with distinct frequency signatures, (b) multi-task conditioning (inspection/tracking/delivery/SAR have different quality requirements), (c) extreme SWaP constraints requiring evaluation at <10M model parameters and <10ms inference, and (d) the need for real-time closed-loop integration.

- **Non-goals**: This is NOT about (a) image restoration or enhancement; (b) a general UAV perception benchmark; (c) multi-UAV collaboration algorithms; (d) a novel IQA model architecture — the model described is a strong baseline, not the paper's contribution; (e) full SC³ system integration.

- **Constraints**: 
  - Compute: University GPU cluster (A100-level) for training; Jetson Orin for deployment benchmarking
  - Data: Assemble from open-source sources (AirCopBench, CARLA-Air, MotionScape); ~50 real UAV validation flights feasible
  - Time: 8 months for database construction (Phase 2 standalone)
  - Annotation: VLM/VLA-based annotation via existing models (no human annotator budget)

- **Success condition**: A publicly available UAV-Embodied-IQA database that (a) defines 6 UAV-specific distortion types with mathematical models validated against real UAV imagery, (b) provides VLM+VLA task-performance annotations across 4 UAV task categories, and (c) establishes a benchmark where existing NR-IQA methods fail (SRCC < 0.5), demonstrating that UAV-specific IQA requires dedicated training data and methods.

## Technical Gap

### Why current resources fail

1. **Embodied-IQA (2505.16815) and EPD (2412.18774)** established the paradigm: define image quality by robot task success, not human preference. Their databases cover fixed-base manipulators (UR5 arm, SAPIEN simulator) with 25-30 generic distortion types. Critically, they contain ZERO UAV-specific distortions — no propeller vibration, no atmospheric scattering, no packet-loss artifacts. A model trained on these databases has never seen the frequency-domain degradation patterns that dominate aerial perception.

2. **AirCopBench (2511.11025)** provides multi-UAV collaborative perception data with built-in perception assessment VQA protocols. It includes real UAV degradations (motion blur, noise, data loss) and 2 perception assessment dimensions (quality/availability). But it is a perception benchmark, not an IQA database: (a) no continuous quality scores — only discrete VQA accuracy; (b) no distortion-level annotations — you can't train a regression model on accuracy; (c) no distortion injection — images have whatever natural degradation they happened to capture; (d) no task-specific quality differentiation.

3. **The gap**: Embodied-IQA provides the annotation methodology (VLM→VLA→robot pipeline). AirCopBench provides UAV task protocols and multi-view degraded data. CARLA-Air provides stable UAV simulation for controlled distortion injection. These three components have never been combined. No database applies embodied IQA's task-performance-centric annotation to UAV-specific distortions.

### Why naive fixes are insufficient

- **Just adding UAV images to existing IQA databases**: Doesn't work because the annotation must be task-conditioned. A blur that's acceptable for scene description may be catastrophic for precision tracking.
- **Just using AirCopBench's VQA scores as IQA labels**: AirCopBench produces discrete accuracy scores, not continuous quality scores. The 3 assessment dimensions are MLLM-centric, not task-performance-centric.
- **Just fine-tuning MA-EIQA on UAV data without a database**: No labeled UAV IQA data exists to fine-tune on. This is circular — you need the database to train the model.
- **Just building a model without a database**: A model paper without a database cannot be evaluated. The database IS the contribution; the model demonstrates the database's utility.

### Smallest adequate intervention

**Build the UAV-Embodied-IQA database by assembling three existing components**: (1) CARLA-Air for stable UAV simulation with controlled distortion injection, (2) AirCopBench's task protocols and multi-view data as reference images and annotation templates, (3) Embodied-IQA's VLM→VLA→execution annotation pipeline. Add 6 mathematically-modeled UAV-specific distortion functions. This is a pure assembly play — zero new infrastructure, only domain-specific distortion models.

### Core technical claim

**The first UAV-Embodied-IQA database, assembled from CARLA-Air + AirCopBench + Embodied-IQA methodology, with 6 novel UAV-specific distortion models, enables training and evaluating task-conditioned IQA for aerial embodied intelligence. The database establishes that existing IQA methods, including embodied IQA models trained on ground-robot data, fail to generalize to UAV distortions (SRCC < 0.5).**

This is a resource contribution claim, not a method claim. The claim is falsifiable — if existing IQA methods achieve SRCC > 0.6 on the database without UAV-specific training, then the database didn't capture a genuine domain gap.

### Required evidence

1. The 6 UAV-specific distortion types are perceptually and functionally distinct from generic distortions (validated by VLA performance degradation patterns and frequency spectrum analysis)
2. Synthetic distortion models correlate with real UAV distortions (validated by ~50 real drone flights: SRCC > 0.6 between synthetic and real distortion effects on VLA behavior)
3. Existing NR-IQA methods (including MA-EIQA trained on EPD) achieve SRCC < 0.5 on the database, establishing a clear challenge
4. A strong baseline trained on the database achieves SRCC > 0.65, demonstrating that the database enables model development

## Method Thesis

- **One-sentence thesis**: The UAV-Embodied-IQA database — assembled from CARLA-Air simulation, AirCopBench task protocols, and Embodied-IQA annotation methodology, with 6 mathematically-modeled UAV-specific distortion types — is the first resource enabling training and evaluation of task-conditioned IQA for aerial embodied intelligence, and establishes that existing IQA methods fail to generalize to UAV distortions.

- **Why this is the smallest adequate intervention**: We invent ZERO new infrastructure. CARLA-Air handles simulation (357 resets, zero crashes). AirCopBench provides task protocols (14 task types, 3 perception assessment dimensions). Embodied-IQA provides the annotation pipeline (VLM→VLA→execution three-stage labeling). We contribute only the 6 UAV-specific distortion functions — mathematical models of physically distinct aerial degradation phenomena. Everything else is assembly.

- **Why this route is timely in the foundation-model era**: Three conditions that were not true 2 years ago: (a) CARLA-Air provides stable UAV simulation (before this, AirSim was the only option and it was deprecated); (b) VLM/VLA models (Qwen2.5-VL, UAV-Track VLA, CognitiveDrone) are capable enough to serve as annotation oracles at scale; (c) Embodied-IQA (2505.16815) has validated the VLM→VLA→robot annotation paradigm. The database can be built NOW by assembling mature components that didn't exist before 2025.

## Contribution Focus

- **Dominant contribution**: **The UAV-Embodied-IQA database** — the first large-scale benchmark for visual quality assessment in aerial embodied intelligence. Specific sub-contributions:
  1. **6 UAV-specific distortion models** with formal mathematical definitions (propeller vibration blur, atmospheric scattering/haze, 6DoF viewpoint change blur, communication packet-loss blocks, low-res+SR artifacts, propeller shadow)
  2. **Multi-task VLM+VLA annotations** across 4 UAV task categories (tracking, inspection, delivery, SAR), following Embodied-IQA's three-stage pipeline
  3. **Benchmark results** establishing that existing NR-IQA methods (including embodied IQA) fail on UAV data (SRCC < 0.5), creating a clear research challenge

- **Optional supporting contribution**: A strong baseline NR-IQA model (UAV-IQANet) that achieves SRCC > 0.65 with <10M parameters — demonstrating the database enables model development. This is NOT claimed as a novel architecture; it's a baseline that combines known techniques (MobileNetV4 backbone, frequency-aware branch, task-conditioned heads).

- **Explicit non-contributions**: 
  - We do NOT claim novelty in the annotation pipeline (reuses Embodied-IQA's three-stage protocol)
  - We do NOT claim novelty in the baseline model architecture (uses MobileNetV4 + PANet + CBAM + FiLM)
  - We do NOT claim novelty in multi-UAV collaboration (uses AirCopBench's existing protocols)
  - We do NOT claim to improve VLM/VLA models themselves
  - We do NOT claim a novel IQA methodology — the database IS the contribution

## Proposed Database Construction

### Complexity Budget

- **Reused infrastructure**: CARLA-Air (simulation engine), AirCopBench (task protocols, reference images, VQA templates), Embodied-IQA annotation pipeline (VLM→VLA→execution three-stage protocol), pyiqa (benchmark evaluation framework)
- **New components** (only the 6 distortion functions):
  1. Propeller vibration blur function
  2. Atmospheric scattering/haze function
  3. 6DoF viewpoint change blur function
  4. Communication packet-loss block function
  5. Low-resolution + super-resolution artifact function
  6. Propeller shadow modulation function
- **New data only**: ~1,000 additional CARLA-Air reference frames (beyond AirCopBench's existing sim data), ~200 MotionScape high-dynamic frames
- **Tempting additions intentionally not used**:
  - Custom VQA protocol design (reuse AirCopBench's 14.6k existing VQA templates)
  - Custom simulation environment (use CARLA-Air as-is)
  - New VLM/VLA models (use existing open-source models)
  - Active learning for annotation selection (annotate all pairs uniformly — simpler, more reproducible)

### System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATABASE CONSTRUCTION PIPELINE                  │
│                                                                    │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐   │
│  │  CARLA-Air   │   │ AirCopBench  │   │    MotionScape       │   │
│  │ (simulation) │   │ (sim + real) │   │ (high-dynamic 6DoF) │   │
│  └──────┬───────┘   └──────┬───────┘   └──────────┬───────────┘   │
│         │                  │                       │               │
│         └──────────────────┼───────────────────────┘               │
│                            │                                       │
│                   ~1,500 reference images                          │
│                (extracted keyframes, 640×480)                      │
│                            │                                       │
│                            ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              DISTORTION INJECTION                             │ │
│  │                                                               │ │
│  │  18 generic (Albumentations):                                 │ │
│  │    blur(3) · brightness(6) · chromatic(3) · noise(4)          │ │
│  │    · compression(3) · spatial(4) · other(5)                   │ │
│  │                                                               │ │
│  │  6 UAV-specific (custom functions):                           │ │
│  │    vib.blur · atm.scatter · 6DoF blur                          │ │
│  │    · pkt-loss blocks · low-res+SR · prop.shadow               │ │
│  │                                                               │ │
│  │  Each at 5 intensity levels                                   │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
│                             │                                      │
│               ~1,500 × 24 × 5 = 180,000 pairs                     │
│                             │                                      │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              ANNOTATION (Embodied-IQA pipeline)               │ │
│  │                                                               │ │
│  │  Stage 1 — Cognitive (3 VLMs):                                │ │
│  │    Qwen2.5-VL-7B, InternVL2-8B, LLaVA-NeXT-13B               │ │
│  │    → AirCopBench perception VQA on distorted images            │ │
│  │    → BLEU/ROUGE/CIDEr vs. clean reference                     │ │
│  │                                                               │ │
│  │  Stage 2 — Decision (3 VLAs):                                 │ │
│  │    UAV-Track VLA, CognitiveDrone-R1, Qwen-VLA                 │ │
│  │    → tracking/cognitive task execution on distorted images     │ │
│  │    → trajectory deviation (position/velocity/heading)          │ │
│  │                                                               │ │
│  │  Stage 3 — Execution (CARLA-Air SITL, 5% subset):            │ │
│  │    → actual flight with distorted input                       │ │
│  │    → task success rate (0-100)                                │ │
│  │    → calibrates VLA pseudo-labels against real execution      │ │
│  └──────────────────────────┬───────────────────────────────────┘ │
│                             │                                      │
│                             ▼                                      │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              QUALITY CONTROL                                  │ │
│  │                                                               │ │
│  │  · Inter-annotator agreement (VLM-VLM, VLA-VLA SRCC)          │ │
│  │  · Label noise analysis + calibration on execution subset     │ │
│  │  · Real-UAV validation: ~50 DJI Mini flights                  │ │
│  │    → compare synthetic vs. real distortion effects             │ │
│  │  · Blind filtering: remove pairs where VLMs disagree >2σ      │ │
│  │  · 10% manual spot-check                                      │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  OUTPUT: UAV-Embodied-IQA Database                                 │
│    180K image pairs + VLM/VLA/execution annotations                │
│    + 6 UAV distortion models (open-source)                         │
│    + Benchmark results for 15+ NR-IQA methods                     │
└────────────────────────────────────────────────────────────────────┘
```

### Database Design Details

**Reference Images (~1,500, three sources)**:

| Source | Count | Scenarios | UAV Tasks |
|--------|-------|-----------|-----------|
| AirCopBench sim subset + CARLA-Air new captures | ~1,000 | Urban/suburban/industrial | Collaborative perception, tracking, scene understanding |
| AirCopBench real subset + MDMT dataset | ~300 | Real urban scenes | Object detection, tracking |
| MotionScape high-dynamic samples | ~200 | High-dynamic 6DoF (optical flow μ=36.63) | World model prediction |

**Distortion Types (24 total, 5 intensity levels each)**:

*Generic (18, from Embodied-IQA catalog)*: Gaussian/lens/motion blur, brightness variations (6), chromatic variations (3), white/color/impulse/multiplicative noise, JPEG/JP2K/WEBP compression, spatial transformations (4)

*UAV-Specific (6, new mathematical models)*:

1. **Propeller Vibration Blur** — directional motion blur with periodic intensity modulation:
   ```
   I_vib(x,y) = ∫_0^T I(x + A·sin(2πft)·cos(θ), y + A·sin(2πft)·sin(θ)) · w(t) dt
   f ∈ [80,200] Hz, A ∈ [1,8] pixels, θ ∼ Uniform(0,2π)
   ```
   Key frequency signature: concentrated energy at propeller harmonics (f, 2f, 3f) in log-polar FFT spectrum.

2. **Atmospheric Scattering / Haze** — Koschmieder model with altitude-dependent parameters:
   ```
   I_haze(x) = I(x)·e^(-β·d(x)) + A_∞·(1 - e^(-β·d(x)))
   β ∈ [0.5,3.0], A_∞ is atmospheric light
   ```
   Key frequency signature: low-frequency attenuation (contrast loss at low spatial frequencies), depth-dependent.

3. **6DoF Fast Viewpoint Change Blur** — motion vectors from MotionScape empirical flow distribution:
   ```
   I_6dof = I ⊗ PSF(v), v ∼ MotionScapeFlow(μ=36.63, σ=25.4)
   ```
   Key frequency signature: broadband motion blur with direction-dependent spectral nulls. Log-polar FFT needed because viewpoint rotation changes blur orientation.

4. **Communication Packet-Loss Block Artifacts** — random macroblock replacement:
   ```
   I_comm(x,y) = M(x,y) ⊙ I(x,y) + (1-M(x,y)) ⊙ I_nearest(x,y)
   M: 16×16 block mask, loss rate ∈ [1%,30%]
   ```
   Key frequency signature: grid-pattern high-frequency energy at spatial frequency 1/16 px⁻¹ (the 16×16 block boundaries create harmonics).

5. **Low-Resolution + Super-Resolution Artifacts**:
   ```
   I_lr-sr = Real-ESRGAN(BicubicDownsample(I, s)), s ∈ [2,8]
   ```
   Key frequency signature: hallucinated high-frequency textures (overshoot artifacts), ringing near edges. SR models amplify certain frequency bands unpredictably.

6. **Propeller Shadow** — periodic localized brightness modulation:
   ```
   I_shadow(x,y) = I(x,y)·(1 - α·Π(f_prop·t mod 1, w(x,y)))
   α ∈ [0.05,0.3], Π: periodic rectangular pulse
   ```
   Key frequency signature: low-frequency periodic illumination variation synchronized with propeller RPM. Distinct from generic brightness changes because it's spatially localized (depends on sun angle geometry).

**Why these 6 are necessary and sufficient**: Each covers a physically distinct UAV failure mode with a mathematically distinct frequency signature. Together they span the UAV failure taxonomy: mechanical (1), environmental (2), ego-motion (3), communication (4), computational (5), and illumination (6). A model that handles all 6 can claim coverage of the dominant UAV degradation space.

**Annotation Scheme (Three-Stage, from Embodied-IQA)**:

| Stage | Annotators | Protocol | Output Metric | Weight in Final MOS |
|-------|-----------|----------|---------------|---------------------|
| Cognitive | 1 VLM (Qwen2.5-VL-7B) | AirCopBench perception VQA (quality/availability) | Text similarity (BLEU/ROUGE/CIDEr) vs. clean reference | 0.25 |
| Decision | 3 VLAs (UAV-Track VLA, CognitiveDrone-R1, Qwen-VLA) | Tracking + cognitive task execution | Trajectory deviation (position/velocity/heading) from clean reference | 0.50 |
| Execution | CARLA-Air SITL flight (5% subset) | Task completion in simulation | Success rate (0-100) | 0.25 (for subset); used to calibrate VLA labels |

**Why cognitive weight is reduced to 0.25** (from 0.3 in original proposal): Embodied-IQA (2505.16815) showed VLM cognitive scores correlate with execution at SRCC < 0.5, while VLA decision scores correlate at SRCC > 0.6. VLA annotations are more reliable proxies for task performance and should be weighted higher.

**Execution Layer Calibration**: For the 5% execution-validated subset (~9,000 pairs), we learn a per-task linear calibration:
```
q_exec = a_k · q_vla_ensemble + b_k    (per task k)
```
This maps the VLA ensemble output to execution success rate, providing a ground-truth anchor. If calibration R² < 0.5, VLA labels are too noisy and the task is excluded from the database.

### Baseline Model Design (Strong Baseline, NOT Contribution)

The following describes a strong baseline model (UAV-IQANet) included to demonstrate the database's utility. We claim NO architectural novelty.

**Architecture** (assembly of known components):

```
Input (H×W×3)
    ├──→ MobileNetV4-S Backbone → PANet FPN → CBAM → f_s ∈ R^256
    │
    └──→ Patch-wise FFT (32×32, stride 16)
              → Log-polar transform (32 angular × 16 radial bins)
              → Tiny CNN (3×3, 8→16 ch, ~3K params) → f_f ∈ R^64
              → Cross-attention gate: α = σ(W_g·[f_s, f_f]), W_g: 320→64
              → Fused: f = Concat[f_s, α⊙f_f] ∈ R^320
                    │
                    ▼
         Task-Conditioned Regression Heads (FiLM):
           t_k ∈ R^4 → γ_k, β_k ∈ R^128
           h' = γ_k ⊙ h + β_k   (after FC(320→128))
           q_k = FC(128→1)
```

**Why this architecture (not a novelty claim — design rationale)**:
- **MobileNetV4-S**: State-of-the-art lightweight backbone (<5M params), ImageNet-21k pre-trained. Within SWaP budget.
- **Log-polar FFT branch**: Unlike prior frequency IQA methods (BRISQUE uses DCT — spatially local, not rotation invariant; MANIQA/ViT methods capture frequency only implicitly), log-polar FFT provides explicit rotation/scale invariance. This is necessary because UAV 6DoF motion rotates and scales vibration patterns — the same propeller frequency signature appears at different orientations and scales in different frames. The log-polar transform maps rotation to translation and scale to radial shift, making frequency signatures invariant to UAV ego-motion.
- **Cross-attention gate**: Allows the model to suppress frequency features when they're uninformative (e.g., for color distortions that have weak frequency signatures). Ablation study: compare gated vs. simple concatenation.
- **FiLM task conditioning**: Parameter-efficient way to produce task-specific scores from a shared feature representation. Only adds 4×128×2 ≈ 1K parameters per task.

**Backbone Comparison** (ViT alternatives evaluated in benchmark):

| Backbone | Params | Design |
|----------|--------|--------|
| MobileNetV4-S | ~4.5M | CNN, our baseline |
| MobileViT-S | ~5.6M | Hybrid CNN+ViT |
| EfficientViT-B0 | ~5.7M | Pure ViT with efficient attention |

All under <6M backbone params, leaving ~4M for PANet+CBAM+FAB+TCRH.

**Training Recipe (for baseline model only)**:
- Loss: L = L_MSE + 0.3·L_ListMLE + 0.1·L_cross-task (same as original)
- Three-stage curriculum: warm-up TCRH (5 epochs) → joint training (50 epochs) → hard-sample fine-tuning (10 epochs)
- AdamW, lr=1e-4, batch_size=64, cosine schedule
- Post-training INT8 quantization via ONNX Runtime

### Real-UAV Validation Protocol

To validate that synthetic distortion models correspond to real UAV degradation:

1. **Data collection**: ~50 flights with DJI Mini (or equivalent consumer drone) in varied conditions:
   - 10 flights in hazy/misty conditions (atmospheric scattering validation)
   - 10 flights with aggressive maneuvering (vibration + 6DoF blur validation)
   - 10 flights at different altitudes (resolution + atmospheric validation)
   - 10 flights near buildings at specific sun angles (propeller shadow validation)
   - 10 flights with bandwidth-limited video transmission (packet-loss validation)

2. **Analysis**: 
   - Run the same VLA models on real distorted frames → compute trajectory deviation
   - Compare VLA degradation patterns between synthetic and real images of the same distortion class
   - Compare log-polar FFT spectra between synthetic and real distortions
   - Report SRCC between synthetic-distortion VLA scores and real-distortion VLA scores (target: >0.6)

3. **Fallback**: If SRCC < 0.6 for any distortion type, refine the mathematical model parameters based on real data spectra, or flag that distortion type as "simulation-only" with caveats.

### Quality Control

| Step | Method | Coverage | Pass Criterion |
|------|--------|----------|----------------|
| Annotator agreement | Per-pair VLM SRCC, VLA SRCC | 100% | Flag pairs with σ > 2σ_population |
| Label noise calibration | Linear calibration on execution subset | 5% subset | R² > 0.5 per task |
| Blind filtering | Remove pairs answerable without visual input | 10% spot-check | <5% flagged |
| Real-UAV correlation | Synthetic vs. real distortion VLA scores | ~50 flights | SRCC > 0.6 |
| Manual spot-check | Human inspects random pairs | 10% | >95% pass rate |

### Benchmark Design

Evaluate 15+ existing IQA methods on the database:

| Category | Methods |
|----------|---------|
| FR-IQA (full-reference) | PSNR, SSIM, LPIPS, DISTS, AHIQ, TOPIQ-FR |
| NR-IQA (generic) | BRISQUE, NIQE, MANIQA, TOPIQ-NR, Q-Align (zero-shot), CLIP-IQA |
| NR-IQA (embodied) | MA-EIQA (EPD-trained), MA-EIQA (Embodied-IQA-trained) |
| Our baselines | UAV-IQANet (MobileNetV4-S), UAV-IQANet (MobileViT-S), UAV-IQANet (EfficientViT-B0) |
| Ablations | UAV-IQANet w/o FAB, UAV-IQANet w/o TCRH, UAV-IQANet w/o UAV-specific distortions |

**Metrics**: SRCC, PLCC, KRCC (per-task and macro-averaged). Per-distortion-type breakdown for 6 UAV-specific vs. 18 generic distortions.

**Expected finding**: Existing IQA methods achieve SRCC < 0.5 on UAV data. MA-EIQA (best embodied IQA model) achieves SRCC < 0.4 on UAV-specific distortions (zero-shot transfer failure). The UAV-IQANet baseline achieves SRCC > 0.65 after training on the database.

## Claim-Driven Validation Sketch

### Claim 1: The 6 UAV-specific distortion types capture degradation patterns distinct from generic distortions

- **Minimal experiment**: Compare VLA decision degradation patterns (trajectory deviation vectors) between UAV-specific and generic distortions at matched intensity levels. Cluster analysis (t-SNE of VLA output deviation vectors) should show UAV-specific distortions forming separate clusters from generic distortions.
- **Baselines / ablations**: N/A — this is a characterization claim, not a comparison.
- **Metric**: Silhouette score for clustering separation; per-distortion VLA score variance explained by distortion category.
- **Expected evidence**: UAV-specific distortions form distinct clusters (silhouette score > 0.3). A classifier trained to distinguish UAV vs. generic distortions from VLA deviation patterns achieves >80% accuracy.

### Claim 2: Synthetic distortion models produce VLA degradation patterns that correlate with real UAV distortions

- **Minimal experiment**: The real-UAV validation protocol (50 flights). Compare VLA trajectory deviation on synthetic vs. real images of the same distortion class.
- **Baselines / ablations**: N/A — correlation analysis.
- **Metric**: SRCC between synthetic-distortion VLA scores and real-distortion VLA scores, per distortion type.
- **Expected evidence**: SRCC > 0.6 for at least 4 of 6 distortion types. Types with SRCC < 0.6 are flagged as "simulation-only, model refinement needed."

### Claim 3: Existing IQA methods (including embodied IQA) fail on UAV distortions

- **Minimal experiment**: Evaluate 15+ IQA methods on the database. Compare SRCC for predicting VLA decision quality scores.
- **Baselines / ablations**: Full benchmark table as described above.
- **Metric**: SRCC/PLCC per method, with focus on best generic NR-IQA vs. best embodied IQA vs. our baseline.
- **Expected evidence**: Best existing NR-IQA achieves SRCC < 0.5. MA-EIQA (EPD-trained) achieves SRCC < 0.4. Our baseline (trained on this database) achieves SRCC > 0.65. The gap >0.15 demonstrates the database captures a genuine domain shift.

### Claim 4 (optional): The database enables training models that generalize across UAV tasks

- **Minimal experiment**: Train model on 3 tasks, test on held-out 4th task. Compare to task-specific training.
- **Baselines / ablations**: Task-specific training vs. multi-task training vs. leave-one-task-out.
- **Metric**: SRCC on held-out task.
- **Expected evidence**: Multi-task training transfers partially (SRCC > 0.5 on held-out task), better than single-task zero-shot (SRCC < 0.3). Demonstrates shared quality structure across UAV tasks.

## Experiment Handoff Inputs

- **Must-prove claims**: 3 primary claims above (distortion distinctiveness, real-world correlation, IQA method failure)
- **Must-run ablations**: FAB removal, TCRH removal, backbone comparison (MobileNetV4 vs. MobileViT vs. EfficientViT), UAV distortion removal
- **Critical datasets / metrics**: AirCopBench real subset for external validation; SRCC/PLCC for correlation; VLA trajectory deviation as ground truth
- **Highest-risk assumptions**: VLA models are accessible and run inference on UAV data; CARLA-Air is stable at scale (357 resets suggests yes); 50 real flights produce enough data for meaningful correlation

## Compute & Timeline Estimate

- **Estimated GPU-hours**: 
  - Distortion injection: ~50 GPU-hours (Real-ESRGAN for SR artifacts)
  - VLM annotation (3 models × 90,000 images): ~500 GPU-hours on A100
  - VLA annotation (3 models × 90,000 images): ~800 GPU-hours on A100
  - Execution layer (5% subset, CARLA-Air SITL): ~50 GPU-hours
  - Baseline model training (including ablations): ~200 GPU-hours on A100
  - Benchmark evaluation (15+ methods): ~100 GPU-hours
  - Total: ~1,700 GPU-hours (feasible on 4×A100 cluster in ~18 days)
- **Real-world data**: ~50 DJI Mini flights (2-3 days fieldwork)
- **Data / annotation cost**: $0 (all open-source data and models)
- **Timeline**: 8 months for database construction + 3 months for paper = 11 months
  - Months 1-2: Environment setup + CARLA-Air deployment + AirCopBench data acquisition
  - Months 2-3: UAV-specific distortion function development + distortion injection pipeline
  - Months 3-4: VLM annotation (parallel: 3 models on 3 GPUs)
  - Months 4-6: VLA annotation (parallel: 3 models on 3 GPUs) + real-UAV validation flights
  - Months 6-7: Execution layer subset annotation + quality control
  - Months 7-8: Baseline model training + benchmark evaluation
  - Months 8-11: Paper writing + revision + supplementary

### Novelty and Elegance Argument

**Closest work — Embodied-IQA database (2505.16815)**:
- Their database: 36,900 pairs, ground manipulator (UR5 arm), 30 generic distortions, 2 manipulation tasks.
- Our database: 180,000 pairs, aerial embodied intelligence (UAV), 18 generic + 6 UAV-specific distortions, 4 aerial task types.
- Key difference: embodiment domain (ground vs. aerial), distortion taxonomy (generic-only vs. generic + 6 frequency-domain UAV types), task diversity (2 manipulation vs. 4 aerial).
- We reuse their annotation methodology — this is a feature, not a bug. The contribution is showing that their paradigm generalizes to a fundamentally different embodiment with novel degradation types.

**Closest work — AirCopBench (2511.11025)**:
- Their benchmark: 14.6k VQA pairs, multi-UAV collaborative perception, 3 perception assessment dimensions.
- Our database: 180K annotated image pairs, task-performance-centric IQA, distortion-level continuous scores.
- Key difference: AirCopBench evaluates MLLM perception accuracy; we provide distortion-level quality scores for IQA model training. AirCopBench has no distortion injection, no continuous scores, no task-specific differentiation.
- We use their data (reference images, task protocols) as input — this is assembly, not competition.

**Why this is a focused, defensible database paper**:
- ONE contribution: the database. The baseline model demonstrates utility but claims no novelty.
- The 6 UAV-specific distortions are the database's "novelty anchor" — they define a new subdomain that existing databases don't cover.
- The assembly approach (CARLA-Air + AirCopBench + Embodied-IQA) is a strength: it shows the database is reproducible and the community can extend it.
- The benchmark results (existing IQA fails) establish the database's value proposition: "This is a new challenge that requires new solutions."

**Positioning against prior frequency-domain IQA**:
- BRISQUE (2012): DCT-domain NSS features — spatially local, not rotation/scale invariant. Cannot handle 6DoF viewpoint changes where frequency patterns rotate and scale.
- DeepFIQA and related: Use DCT coefficients as CNN input channels — still spatially local frequency representation.
- ViT-based IQA (MANIQA, IQT): Self-attention captures frequency information implicitly but provides no explicit rotation/scale invariance.
- Our log-polar FFT: Explicitly rotation-invariant (rotation → circular shift in angular dimension) and scale-invariant (scale → radial shift). This is the right frequency representation for UAV ego-motion, where the same vibration pattern appears at different orientations across frames. Not a novel IQA architecture — a domain-appropriate design choice for the baseline model.

## Compute & Timeline Estimate

(Same as above — 11 months, ~1,700 GPU-hours)
