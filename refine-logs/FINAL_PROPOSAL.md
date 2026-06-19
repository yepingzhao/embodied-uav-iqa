# UAV-Embodied-IQA: A Benchmark for Visual Quality Assessment in Aerial Embodied Intelligence

## Problem Anchor

- **Bottom-line problem**: In low-altitude UAV embodied intelligence, visual inputs suffer from UAV-specific degradations (propeller vibration blur, atmospheric scattering, 6DoF viewpoint change blur, communication packet-loss artifacts, low-resolution+super-resolution artifacts, propeller shadow modulation) that differ fundamentally from both human-centric and ground-robot distortions. Yet no quality assessment framework exists to predict whether a given visual input is adequate for a specific UAV downstream task.

- **Must-solve bottleneck**: Current embodied IQA databases (EPD, Embodied-IQA) cover only fixed-base manipulator robots in indoor settings. UAV scenarios introduce (a) 6 novel distortion types with distinct frequency signatures, (b) extreme SWaP constraints requiring <10M parameter models, (c) multi-task conditioning (inspection/tracking/delivery/SAR have different quality requirements), and (d) the need for real-time closed-loop integration (<10ms inference). No existing method addresses all four simultaneously.

- **Non-goals**: NOT about image restoration/enhancement, general UAV perception benchmarking, multi-UAV collaboration algorithms, full SC³ system integration (future work), or claiming architectural novelty in the baseline model.

- **Constraints**: University GPU cluster (A100-level); data assembled from existing open-source sources (AirCopBench, CARLA-Air, MotionScape, MDMT); 11 months; VLM/VLA-based annotation; ~50 real-UAV validation flights.

- **Success condition**: A publicly available UAV-Embodied-IQA database with: (a) ~3,000 reference images across 4 UAV task categories, (b) 24 distortion types × 5 intensity levels with VLM+VLA+Execution task-performance annotations, (c) benchmark results for 15+ existing IQA methods showing they fail on UAV data (SRCC < 0.5), (d) small-scale real-UAV validation confirming synthetic distortion fidelity, and (e) a strong baseline model (SRCC > 0.65) demonstrating the database enables training effective UAV-specific IQA models.

## Technical Gap

Three existing resource lines have not yet intersected: (1) Embodied-IQA (2505.16815) provides the VLM→VLA→Execution annotation methodology but only covers ground manipulators with generic distortions, (2) AirCopBench (2511.11025) provides multi-UAV task protocols and perception assessment VQA but lacks distortion-level annotations and continuous quality scores, (3) frequency-aware IQA methods (BRISQUE/DCT, DeepFIQA) capture some frequency information but are designed for human perceptual quality with spatially local frequency representations — they lack the rotation/scale invariance needed for 6DoF UAV ego-motion.

The UAV-Embodied-IQA database bridges these three lines: AirCopBench task protocols + Embodied-IQA annotation methodology + 6 novel UAV-specific frequency-domain distortion models.

## Contribution

**Single contribution**: The UAV-Embodied-IQA database — the first benchmark for visual quality assessment in aerial embodied intelligence. Components:
1. 6 UAV-specific distortion types with physically-motivated mathematical models
2. ~3,000 reference images from AirCopBench, CARLA-Air, MotionScape, MDMT, spanning 4 UAV task categories
3. ~180,000 distorted+annotated image pairs with VLM (cognitive), VLA (decision), and CARLA-Air Execution quality scores
4. Comprehensive benchmark of 15+ existing NR-IQA and FR-IQA methods on UAV data
5. Strong baseline model (UAV-QANet, <5.5M params) demonstrating database utility

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

### Annotation Pipeline
- **Cognitive (VLM)**: Qwen2.5-VL-7B, InternVL2-8B, LLaVA-NeXT-13B → AirCopBench perception assessment VQA → cognitive score
- **Decision (VLA)**: UAV-Track VLA, CognitiveDrone-R1, Qwen-VLA (fallback: OpenVLA-7B × 3 seeds) → sliding-window single-frame replacement protocol → decision score
- **Execution (CARLA-Air SITL)**: Tracking + Inspection tasks (5% stratified subset) → execution score
- **Training labels**: 3-stage curriculum (VLM epochs 1-20 → VLA epochs 21-40 → Execution epochs 41-50)

### Pre-Start Gate
Verify ≥2 target VLA models have public weights. If not, activate OpenVLA-7B × 3-seed backup plan.

### Real-UAV Validation
- Sequential protocol: 20 flights (scattering + vibration) → gate check → 30 flights (remaining 4 types)
- SRCC target: >0.6 between real and synthetic distortion effects on VLA behavior

## Baseline Model: UAV-QANet

Architecture: MobileNetV4-S backbone → PANet FPN → CBAM attention → frequency-aware branch (patch FFT + tiny CNN) → task embedding concatenation → shared regression MLP. Total: ~5.4M params (3.5M trainable), INT8 quantized ~1.4MB.

Positioned as a strong baseline — no architectural novelty claimed. Benchmarked against MobileViT-S and EfficientViT-B0 as backbone alternatives.

## Benchmark Design

15+ methods evaluated: FR-IQA (PSNR, SSIM, LPIPS, DISTS, AHIQ, TOPIQ), NR-IQA (BRISQUE, NIQE, MANIQA, Q-Align, CLIP-IQA), Frequency-aware (BRISQUE/DCT, DeepFIQA), Embodied (MA-EIQA zero-shot and fine-tuned), UAV baseline (UAV-QANet with ablations).

## Key Claims
1. UAV-specific distortions produce distinct VLA performance degradation patterns vs. generic distortions
2. Synthetic UAV distortions correlate with real-world UAV distortion effects (SRCC > 0.6)
3. Existing IQA methods fail to predict UAV task performance (SRCC < 0.5)
4. (Optional) Multi-task training enables cross-task quality generalization

## Compute & Timeline
- ~1,700 GPU-hours on A100 (~18 days on 4×A100)
- 50 real-UAV flights (2-3 days fieldwork)
- 11 months total: 8 months construction + 3 months paper
