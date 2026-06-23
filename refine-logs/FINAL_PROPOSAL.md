# UAV-Embodied-IQA: A Benchmark for Visual Quality Assessment in Aerial Embodied Intelligence

## Problem Anchor

- **Bottom-line problem**: In low-altitude UAV embodied intelligence, visual inputs suffer from UAV-specific degradations (propeller vibration blur, atmospheric scattering, 6DoF viewpoint change blur, communication packet-loss artifacts, low-resolution+super-resolution artifacts, propeller shadow modulation) that differ fundamentally from both human-centric and ground-robot distortions. Yet no quality assessment framework exists to predict whether a given visual input is adequate for a specific UAV downstream task.

- **Must-solve bottleneck**: Current embodied IQA databases (EPD, Embodied-IQA) cover only fixed-base manipulator robots in indoor settings. UAV scenarios introduce (a) 6 novel distortion types with distinct frequency signatures, (b) extreme SWaP constraints requiring <10M parameter models, (c) multi-task conditioning (inspection/tracking/delivery/SAR have different quality requirements), and (d) the need for real-time closed-loop integration (<10ms inference). Furthermore, Embodied-IQA (2505.16815) revealed that VLA inter-model agreement is critically low (SRCC ≈ 0.25), and EPD (2412.18774) demonstrated the Moravec paradox in IQA — human quality ratings correlate with robot task performance at PLCC < 0.22 — yet no UAV-specific validation of this gap exists. No existing method addresses all these challenges simultaneously.

- **Non-goals**: NOT about image restoration/enhancement, general UAV perception benchmarking, multi-UAV collaboration algorithms, full SC³ system integration (future work), or claiming architectural novelty in the baseline model.

- **Constraints**: University GPU cluster (A100-level); data assembled from existing open-source sources (AirCopBench, CARLA-Air, MotionScape, MDMT); 11 months; VLM/VLA-based annotation; ~50 real-UAV validation flights.

- **Success condition**: A publicly available UAV-Embodied-IQA database with: (a) ~3,000 reference images across 4 UAV task categories, (b) 24 distortion types × 5 intensity levels with VLM+VLA+Execution task-performance annotations, (c) benchmark results for 15+ existing IQA methods showing they fail on UAV data (SRCC < 0.5), (d) small-scale real-UAV validation confirming synthetic distortion fidelity, and (e) a strong baseline model (SRCC > 0.65) demonstrating the database enables training effective UAV-specific IQA models.

## Technical Gap

Four existing resource lines have not yet intersected:

1. **Embodied-IQA (2505.16815)**: Provides the VLM→VLA→Execution annotation methodology and the Mertonian perception-cognition-decision-execution theory, but only covers ground manipulators with generic distortions. Critically, it reveals VLA inter-model SRCC ≈ 0.25 (low annotation consistency) and VLM inter-model SRCC ≈ 0.3 — problems that must be addressed for UAV-scale annotation reliability.

2. **AirCopBench (2511.11025)**: Provides multi-UAV task protocols across 14 task types in 4 dimensions, including Perception Assessment (quality/availability/causal VQA, ~5,000 pairs). Key finding: causal assessment (identifying degradation causes) is the cognitive ability most correlated with overall performance. Yet AirCopBench lacks distortion-level annotations and continuous quality scores.

3. **EPD + MA-EIQA (2412.18774)**: First demonstration of the Moravec paradox in IQA — human MOS vs. robot task score PLCC < 0.22, proving human-centric IQA is fundamentally misaligned with robot needs. Color distortion impacts tasks most, noise least (opposite of human perception). This gap has never been quantified for UAV scenarios.

4. **Frequency-aware IQA methods** (BRISQUE/DCT, DeepFIQA): Capture frequency information but are designed for human perceptual quality with spatially local frequency representations — they lack the rotation/scale invariance needed for 6DoF UAV ego-motion.

The UAV-Embodied-IQA database bridges these four lines: AirCopBench task protocols + Embodied-IQA annotation methodology (with VLA calibration) + EPD's Moravec paradox quantification for UAV + 6 novel UAV-specific frequency-domain distortion models + compound distortion modeling.

## Contribution

**Single contribution**: The UAV-Embodied-IQA database — the first benchmark for visual quality assessment in aerial embodied intelligence. Components:
1. 6 UAV-specific distortion types with physically-motivated mathematical models (+ compound distortion pairs)
2. ~3,000 reference images from AirCopBench, CARLA-Air, MotionScape, MDMT, spanning 4 UAV task categories
3. ~180,000 distorted+annotated image pairs with VLM (cognitive + causal assessment), VLA (decision, with ensemble calibration), and CARLA-Air Execution quality scores; plus human MOS on a 5% stratified subset for Moravec paradox quantification
4. Comprehensive benchmark of 15+ existing NR-IQA and FR-IQA methods on UAV data, plus cross-database validation against Embodied-IQA and EPD
5. Strong baseline model (UAV-IQANet, <5.5M params) demonstrating database utility; zero-shot cross-database testing against Embodied-IQA/EPD to establish domain specificity

## Database Design

### Reference Images (~3,000)
- AirCopBench sim + CARLA-Air: ~1,800 (urban/suburban/industrial, varied altitudes)
- AirCopBench real + MDMT: ~600 (real urban scenes, natural degradation)
- MotionScape: ~600 (high-dynamic 6DoF, optical flow μ=36.63 px)

### UAV-Specific Distortions (6 types, with mathematical models)
1. **Propeller Vibration Blur**: Directional motion blur with periodic intensity modulation (f ∈ [80,200] Hz)
2. **Atmospheric Scattering / Haze**: Koschmieder model, altitude-dependent β ∈ [0.5, 3.0]
3. **6DoF Fast Viewpoint Change Blur**: Motion vectors from MotionScape empirical flow distribution
4. **Communication Packet-Loss Block Artifacts**: Random 16×16 macroblock replacement (loss rate 1-30%)
5. **Low-Resolution + Super-Resolution Artifacts**: Bicubic downsample → Real-ESRGAN upscale chain
6. **Propeller Shadow**: Periodic localized brightness modulation, α ∈ [0.05, 0.3]

Plus 18 generic distortion types inherited from Embodied-IQA catalog. All at 5 intensity levels.
Plus 10 compound distortion pairs (UAV-specific × UAV-specific pairwise combinations) at 3 intensity level pairs, applied to a 20% stratified subset (~600 reference images → ~18,000 compound-distorted pairs).

### Annotation Pipeline
- **Cognitive (VLM) — Two Dimensions**:
  - *Quality Assessment*: Qwen2.5-VL-7B, InternVL2-8B, LLaVA-NeXT-13B → AirCopBench perception assessment VQA → cognitive quality score (as in Embodied-IQA)
  - *Causal Assessment*: Same VLMs → AirCopBench causal assessment VQA → distortion cause identification accuracy. **Rationale**: AirCopBench found causal assessment is the cognitive ability most correlated with overall MLLM performance; adding this dimension as a VLM annotation task provides diagnostic value beyond quality scores alone
- **Decision (VLA)**: UAV-Track VLA, CognitiveDrone-R1, Qwen-VLA (fallback: OpenVLA-7B × 3 seeds) → sliding-window single-frame replacement protocol → decision score
  - **VLA Ensemble Calibration**: Embodied-IQA revealed VLA inter-model SRCC ≈ 0.25. We address this via: (a) per-model confidence weighting derived from execution-layer validation on the 5% SITL subset, (b) reporting ICC(3,k) per task/distortion for transparency, (c) weighted mean with outlier rejection (models outside 1.5σ of ensemble mean are down-weighted)
- **Execution (CARLA-Air SITL)**: Tracking + Inspection tasks (5% stratified subset) → execution score
- **Human MOS (5% subset)**: ~9,000 pairs rated by 15+ human subjects → quantify the UAV-specific Moravec paradox (SRCC between human MOS and VLA/Execution scores). **Rationale**: EPD (2412.18774) demonstrated PLCC < 0.22 for ground robots; establishing this gap for UAV tasks is essential to justify why UAV-specific IQA is needed
- **Training labels**: 3-stage curriculum (VLM epochs 1-20 → VLA epochs 21-40 → Execution epochs 41-50)
- **Compound distortions**: 10 pairwise combinations of UAV-specific distortions (e.g., haze + vibration, packet-loss + low-res SR) at 3 intensity level pairs each, applied to a 20% stratified subset. **Rationale**: AirCopBench images exhibit multiple simultaneous degradations; Embodied-IQA JND analysis showed compound effects can be fatal at low individual intensities

### Pre-Start Gate
Verify ≥2 target VLA models have public weights. If not, activate OpenVLA-7B × 3-seed backup plan.

### Real-UAV Validation
- Sequential protocol: 20 flights (scattering + vibration) → gate check → 30 flights (remaining 4 types)
- SRCC target: >0.6 between real and synthetic distortion effects on VLA behavior

## Baseline Model: UAV-IQANet

Architecture: MobileNetV4-S backbone → PANet FPN → CBAM attention → frequency-aware branch (patch FFT + tiny CNN) → task embedding concatenation → shared regression MLP. Total: ~5.4M params (3.5M trainable), INT8 quantized ~1.4MB.

Positioned as a strong baseline — no architectural novelty claimed. Benchmarked against MobileViT-S and EfficientViT-B0 as backbone alternatives.

## Benchmark Design

15+ methods evaluated: FR-IQA (PSNR, SSIM, LPIPS, DISTS, AHIQ, TOPIQ), NR-IQA (BRISQUE, NIQE, MANIQA, Q-Align, CLIP-IQA), Frequency-aware (BRISQUE/DCT, DeepFIQA), Embodied (MA-EIQA zero-shot and fine-tuned), UAV baseline (UAV-IQANet with ablations).

## Key Claims

1. **UAV-specific distortions produce distinct VLA performance degradation patterns vs. generic distortions** — supported by causal assessment showing distortion-type-conditional degradation signatures that differ from both ground-robot (EPD/Embodied-IQA) and human perception patterns
2. **Synthetic UAV distortions correlate with real-world UAV distortion effects** (SRCC > 0.6) — validated via AirCopBench-inspired sim-to-real protocol with telemetry-based parameter estimation
3. **Existing IQA methods fail to predict UAV task performance** (SRCC < 0.5); the UAV-Embodied-IQA database enables training effective UAV-IQA models (SRCC > 0.65) — with cross-database validation confirming that Embodied-IQA/EPD-trained models also fail on UAV data and vice versa
4. **(Optional) Multi-task training enables cross-task quality generalization**
5. **(Supporting) The Moravec paradox holds for UAV embodied IQA** — human MOS correlates with UAV task performance at SRCC < 0.3, quantitatively establishing the gap that justifies task-performance-based annotation
6. **(Supporting) Causal assessment (identifying distortion causes) provides annotation signal beyond simple quality scoring** — VLM causal assessment accuracy is predictive of VLA decision degradation patterns

## Compute & Timeline
- ~2,000 GPU-hours on A100 (~22 days on 4×A100)
- 50 real-UAV flights (2-3 days fieldwork)
- Human MOS collection: ~9,000 pairs × 15 subjects, estimated 3-4 weeks via crowdsourcing
- 11 months total: 8 months construction + 3 months paper
