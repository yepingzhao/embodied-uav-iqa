# Paper Plan

**Title**: UAV-Embodied-IQA: A Benchmark for Visual Quality Assessment in Aerial Embodied Intelligence
**Working short title**: UAV-Embodied-IQA
**One-sentence contribution**: We introduce UAV-Embodied-IQA, the first benchmark and database for visual quality assessment in aerial embodied intelligence, featuring 6 physically-motivated UAV-specific distortion models, 36-distortion database with VLM/VLA/Execution annotations, and evidence that 15+ existing IQA methods fundamentally fail to predict UAV task performance (SRCC < 0.5), while our database enables training effective UAV-IQA models (SRCC > 0.65).
**Venue**: NeurIPS Datasets & Benchmarks Track (primary); CVPR (backup)
**Type**: Benchmark / Empirical
**Date**: 2026-07-13
**Page budget**: 9 pages (main body to Conclusion end, excluding references & appendix)
**Section count**: 7
**Status**: IN PROGRESS — experiments deployed, results pending

> **Competitive landscape update (2026-07-27)**: Embodied-IQA (arXiv 2505.16815, SJTU/Shanghai AI Lab) accepted to **ICLR 2026** — this is the direct predecessor. Venue target adjusted to NeurIPS D&B (primary) / CVPR (backup). Differentiation strategy: UAV platform (6DoF outdoor) vs. ground manipulators (indoor); 6 physically-derived UAV distortion models vs. generic catalog; 4 aerial task categories; ~108K vs. 36.9K annotated pairs. See §2 Related Work for full positioning.

---

## Claims-Evidence Matrix

| # | Claim | Evidence | Status | Target Section |
|---|-------|----------|--------|----------------|
| C1 | UAV-specific distortions produce distinct VLA performance degradation patterns vs. generic distortions | Per-distortion ΔVLA curves (B1/DQUS); frequency signature visualization (B5/QUFD); task interaction ANOVA | **Pending** — VLA annotation (R006-R008) not yet run | §4.1 |
| C2 | Synthetic UAV distortions correlate with real-world UAV distortion effects (SRCC > 0.6) | 50 real-UAV flights with telemetry-based parameter estimation; synthetic replication evaluation (B2/D2RB) | **Pending** — real flights (R025-R028) not yet run | §4.2 |
| C3 | Existing IQA methods fail to predict UAV task performance (SRCC < 0.5); database enables training effective UAV-IQA models (SRCC > 0.65) | 15+ method benchmark (B3/B4H); UAV-IQANet full training with 3 seeds (R013); component ablations (R016-R020); distortion pool ablations (R021/R021b); cross-database validation (R036a-R036c) | **Partially supported** — benchmark running (R009-R012); model training deployed (R013-R024c); no final metrics yet | §4.3, §4.4 |
| C4 | Multi-task training enables cross-task quality generalization | Per-task training × 4 (R024a); leave-one-task-out × 4 (R024b); multi-task joint training (R024c) | **Pending** — experiments deployed, results pending | §4.4 |
| C5 | The Moravec paradox holds for UAV embodied IQA — human MOS misaligned with UAV task performance (SRCC < 0.3) | Human MOS collection on 5% subset (~9K pairs); pairwise SRCC between human/VLM/VLA/Execution (R038-R039) | **Pending** — human study not yet conducted | §5 |
| AC1 | Gain comes from database size, not UAV distortion types | Train on generic-only (R021) vs. UAV-only (R021b) vs. full (R013) | **Pending** — ablations deployed | §4.4 |
| AC2 | VLA pseudo-labels are unreliable | VLM-only (R022) vs. VLA-only (R022b) vs. no-execution (R023) curriculum comparison; VLA ensemble ICC (R037a-c) | **Pending** | §4.4 |
| AC3 | Frequency branch is unnecessary | W/o FAB ablation (R016) vs. full model (R013) | **Pending** — ablation deployed | §4.4 |
| AC4 | Task-conditioning is unnecessary | Task-agnostic variant (R017) vs. task-conditioned (R013); cross-task matrix (R024a-c) | **Pending** — experiments deployed | §4.4 |
| AC5 | ViT backbones are the real gain | MobileViT-S (R019) and EfficientViT-B0 (R020) vs. MobileNetV4-S (R013) | **Pending** — experiments deployed | §4.4 |
| AC6 | UAV distortions are just renamed ground-robot distortions | Cross-database: MA-EIQA on UAV data (R036a); UAV-IQANet on Embodied-IQA/EPD (R036b) | **Pending** — not yet deployed | §4.3 |
| AC7 | A single strong VLA is sufficient for labels | VLA ensemble (weighted mean) vs. single-best-VLA (R037a-c) | **Pending** — not yet deployed | §3.3 |

### Evidence Completeness Summary

| Category | Complete | Partial | Pending |
|----------|----------|---------|---------|
| Sanity checks | R002, R004 | — | — |
| Database construction | R005 (distortion pipeline) | — | R006-R008 (VLM/VLA/Execution scoring) |
| Baseline benchmark | — | R009-R012 (running) | — |
| Main model training | — | R013-R014 (deployed) | — |
| Component ablations | — | R016-R020 (deployed) | — |
| Distortion ablations | — | R021-R021b (deployed) | — |
| Curriculum ablations | — | R022-R023 (deployed) | — |
| Cross-task generalization | — | R024a-R024c (deployed) | — |
| Real-UAV validation | — | — | R025-R028 |
| Cross-database validation | — | — | R036a-R036c |
| VLA calibration | — | — | R037a-R037c |
| Human MOS / Moravec | — | — | R038-R039 |
| Compound distortions | — | — | R040a-R040c |

---

## Structure

### §0 Abstract

- **What we achieve**: We introduce UAV-Embodied-IQA, the first benchmark for visual quality assessment in aerial embodied intelligence. The database comprises ~3,000 reference images from 4 UAV task categories, 36 distortion types (6 UAV-specific + 30 generic), and multi-layer annotations (VLM cognitive, VLA decision, Execution task-performance).
- **Why it matters / is hard**: Low-altitude UAV visual inputs suffer from degradations — propeller vibration blur, atmospheric scattering, 6DoF viewpoint change blur, packet-loss block artifacts, low-res+SR artifacts, and propeller shadow — that differ fundamentally from human-centric and ground-robot distortions. No quality assessment framework exists to predict whether a visual input is adequate for a specific UAV downstream task.
- **How we do it**: We design 6 physically-motivated UAV distortion models with explicit frequency-domain signatures, assemble reference images spanning 4 UAV tasks (inspection, tracking, delivery, SAR), annotate quality via a multi-model VLM/VLA/Execution pipeline, and benchmark 15+ existing IQA methods against a lightweight task-conditioned baseline (UAV-IQANet, ~5.4M parameters).
- **Evidence**: 15+ existing IQA methods achieve SRCC < 0.5 on UAV data; UAV-IQANet trained on our database achieves SRCC > 0.65; per-distortion degradation analysis reveals task-dependent patterns unique to UAV distortions; ablation studies validate each model component; cross-database evaluation confirms UAV-specific domain gap.
- **Most remarkable result**: Existing NR-IQA and FR-IQA methods, including those designed for frequency-aware and embodied quality assessment, fail to predict UAV task performance (SRCC < 0.5). Our database closes this gap, enabling compact models (<5.5M params) to exceed SRCC 0.65 across all 4 UAV tasks.
- **Estimated length**: 180-220 words
- **Self-contained check**: Yes — defines problem, approach, evidence, and key result without external context.

### §1 Introduction (1.5 pages)

- **Opening hook**: Low-altitude UAVs increasingly perform safety-critical embodied tasks — infrastructure inspection, search-and-rescue, package delivery, and target tracking — where visual input quality directly determines task success. Yet the visual degradations that affect UAV cameras (propeller-induced vibration, atmospheric scattering at varying altitudes, aggressive 6DoF ego-motion, wireless video packet loss) differ fundamentally from the distortions studied in conventional image quality assessment.
- **Gap / challenge**: Current embodied IQA databases (EPD, Embodied-IQA) cover only fixed-base ground manipulators in indoor settings. Human-centric IQA methods correlate poorly with robot task performance (EPD: PLCC < 0.22 — the "Moravec paradox" of IQA). No resource exists that (a) models UAV-specific degradation physics, (b) annotates quality in terms of downstream task success, and (c) spans the diversity of aerial embodied tasks.
- **One-sentence contribution**: We introduce UAV-Embodied-IQA, the first benchmark and database for visual quality assessment in aerial embodied intelligence, with 6 physically-motivated UAV-specific distortion models, 36-distortion database, and evidence that existing IQA methods fundamentally fail on UAV data.
- **Approach overview**: Rather than collecting new flight data (prohibitively expensive at scale), we assemble reference images from existing UAV task datasets and apply 6 physically-modeled distortion types whose mathematical forms are derived from UAV flight physics. We annotate quality through a 3-layer pipeline: VLM-based cognitive scoring (text similarity against clean-reference outputs), VLA-based decision scoring (task success under single-frame replacement), and Execution scoring (CARLA-Air SITL for a calibrated subset). A lightweight task-conditioned NR-IQA model (UAV-IQANet) serves as a strong baseline.
- **Key questions**:
  1. Do UAV-specific distortions produce degradation patterns that differ from generic distortions in ways that matter for downstream tasks? (C1)
  2. Do synthetically modeled distortions faithfully reproduce real UAV flight phenomenology? (C2)
  3. Do existing IQA methods fail to predict UAV task performance, and can our database train effective models? (C3)
  4. Can task-conditioned IQA generalize across diverse UAV task types? (C4)
- **Contributions**:
  1. 6 UAV-specific distortion types with physically-motivated mathematical models and explicit frequency-domain signatures, plus compound distortion pairs
  2. UAV-Embodied-IQA database: ~3,000 reference images, 36 distortion types, ~108,000 annotated pairs with VLM/VLA/Execution quality scores across 4 task categories
  3. Comprehensive benchmark of 15+ IQA methods establishing their failure on UAV data (SRCC < 0.5), with cross-database validation
  4. UAV-IQANet: a lightweight task-conditioned baseline (<5.5M params, SRCC > 0.65) demonstrating database utility, with ablation studies validating frequency-aware architecture, task conditioning, and UAV distortion taxonomy
- **Results preview**: All 15+ benchmarked IQA methods — including state-of-the-art NR-IQA (MANIQA, Q-Align, CLIP-IQA), FR-IQA (LPIPS, DISTS, TOPIQ), and frequency-aware methods — achieve SRCC < 0.5 on UAV data. UAV-IQANet trained on our database exceeds SRCC 0.65, with the frequency-aware branch contributing >0.05 SRCC improvement and task conditioning contributing >0.03 SRCC.
- **Hero figure**: Figure 1 should show a 3-panel overview: (Left) 4 UAV task exemplars (inspection, tracking, delivery, SAR) with clean vs. UAV-distorted image pairs, (Center) degradation impact — bar chart of ΔVLA success rate for 6 UAV distortions vs. 6 representative generic distortions, color-coded by task, (Right) benchmark summary — scatter plot of model size vs. SRCC for all methods, with UAV-IQANet highlighted at the Pareto frontier (highest SRCC, smallest model). Caption: "UAV-Embodied-IQA overview. Left: UAV tasks suffer from 6 UAV-specific visual degradations not captured by conventional IQA. Center: UAV distortions cause task-dependent VLA performance degradation that differs from generic distortions. Right: Existing IQA methods fail on UAV data (SRCC < 0.5); UAV-IQANet trained on our database achieves SRCC > 0.65 at <5.5M parameters."
- **Estimated length**: 1.5 pages
- **Key citations**: AirCopBench (2511.11025), Embodied-IQA (2505.16815, **ICLR 2026** — direct predecessor), EPD/MA-EIQA (2412.18774), BRISQUE (2012), NIQE (2013), MANIQA (2022), Q-Align (2024), CLIP-IQA (2023)
- **Front-loading check**: A skim reader who reads only the title, abstract, Figure 1, and Introduction will know: (1) UAV-specific visual quality assessment is an unsolved problem, (2) we built the first benchmark for it, (3) existing methods fail badly, (4) our database enables effective models.

### §2 Related Work (1 page)

- **Subtopics**:
  1. **Image Quality Assessment (IQA)** — NR-IQA methods (BRISQUE, NIQE, MANIQA, Q-Align, CLIP-IQA), FR-IQA methods (PSNR, SSIM, LPIPS, DISTS, AHIQ, TOPIQ). Position: these are designed for human perceptual quality; we show they fail for UAV task-oriented quality.
  2. **Embodied IQA and Task-Oriented Quality** — Embodied-IQA (2505.16815, **ICLR 2026**) establishes the VLM→VLA→Execution annotation pipeline and demonstrates that existing IQA methods fail to predict robot task performance on ground manipulators (SRCC < 0.65). EPD/MA-EIQA (2412.18774) quantifies the Moravec paradox for indoor fixed-base robots. Position: Embodied-IQA is our **direct predecessor** — we adopt its annotation methodology but differ fundamentally on three axes: (1) platform (fixed-base indoor manipulators vs. 6DoF outdoor UAV), (2) distortion types (generic catalog vs. 6 physically-derived UAV-specific models), (3) task diversity (single manipulation task type vs. 4 aerial task categories). The Moravec paradox we establish for aerial embodied intelligence is more severe and mechanistically distinct from the ground-robot case.
  3. **Frequency-Aware IQA and Distortion Modeling** — BRISQUE/DCT, DeepFIQA, log-polar frequency representations. Position: existing frequency IQA methods use spatially local transforms unsuitable for 6DoF UAV motion; our frequency-aware branch uses patch-FFT with log-polar transform for rotation/scale invariance.
  4. **UAV Perception and Benchmarking** — AirCopBench (2511.11025) multi-UAV task protocols, CARLA-Air simulation, UAV-specific degradation studies. Position: AirCopBench provides task protocols but lacks distortion-level annotations; we add continuous quality scores and degradation modeling.
  5. **Lightweight and Task-Conditioned Architectures** — MobileNetV4, MobileViT, EfficientViT, FiLM conditioning, CBAM attention. Position: we compose these into a strong lightweight baseline without claiming architectural novelty.
- **Positioning**: UAV-Embodied-IQA bridges 4 previously separate lines: AirCopBench task protocols + Embodied-IQA annotation methodology (ICLR 2026) + EPD's Moravec quantification extended to UAV + 6 novel UAV-specific frequency-domain distortion models. The key distinction from Embodied-IQA (the closest prior work) is the shift from fixed-base indoor manipulation to 6DoF outdoor aerial embodied intelligence, which introduces physically-distinct distortions (propeller vibration, atmospheric scattering, 6DoF motion blur) that require new distortion models, a larger-scale database (~108K vs. 36.9K annotated pairs), and task-conditioned evaluation across 4 diverse aerial task categories. No existing resource combines (a) UAV-specific distortions with physical models, (b) task-conditioned multi-layer annotations, and (c) comprehensive IQA benchmarking.
- **Organization rule**: Organized by research community / method family, not paper-by-paper. Each paragraph synthesizes 3-5 papers, states what they achieved, and identifies the gap our work fills.
- **Minimum length**: 1 full page (4-5 paragraphs with substantive synthesis)

### §3 The UAV-Embodied-IQA Database (2 pages)

- **Notation**: Key symbols: $\mathcal{D} = \{I_r, I_d, \mathbf{s}, t\}$ where $I_r$ = reference image, $I_d$ = distorted image, $\mathbf{s} = (s_{cog}, s_{dec}, s_{exec})$ = annotation score vector, $t \in \{1..4\}$ = task type. Distortion parameters: $\theta_k \in \Theta_k$ for each of $K=36$ distortion types. UAV-specific distortion set: $\mathcal{U} \subset \{1..36\}, |\mathcal{U}|=6$.
- **Problem formulation**: Given a clean reference image $I_r$ from UAV task $t$, apply distortion $d_k$ with intensity $\alpha \in [0.2,1.0]$ to produce $I_d$. Annotate $I_d$ with cognitive score $s_{cog}$ (VLM text similarity), decision score $s_{dec}$ (VLA task success retention), and execution score $s_{exec}$ (SITL validation on 5% subset). Training target: $y = \text{cognitive\_score}$ (aggregated from raw scores via curriculum schedule).
- **Database sections**:
  - **§3.1 Reference Image Sources** (~3,000 images): AirCopBench sim + CARLA-Air (~1,800, urban/suburban/industrial, varied altitudes); AirCopBench real + MDMT (~600, real urban scenes); MotionScape (~600, high-dynamic 6DoF). Span 4 task categories: infrastructure inspection, target tracking, package delivery, search-and-rescue.
  - **§3.2 UAV-Specific Distortion Models** (major subsection, ~1 page):
    1. *Propeller Vibration Blur*: Directional motion blur with periodic intensity modulation, $f \in [80,200]$ Hz, derived from rotor frequency harmonics
    2. *Atmospheric Scattering / Haze*: Koschmieder model, altitude-dependent $\beta \in [0.5, 3.0]$, transmission $t(x) = e^{-\beta d(x)}$
    3. *6DoF Fast Viewpoint Change Blur*: Motion vectors sampled from MotionScape empirical optical flow distribution ($\mu = 36.63$ px), 6DoF trajectory integration
    4. *Communication Packet-Loss Block Artifacts*: Random $16\times16$ macroblock replacement, loss rate $p \in [0.01, 0.30]$, simulating wireless video transmission errors
    5. *Low-Resolution + Super-Resolution Artifacts*: Bicubic downsampling → Real-ESRGAN upscaling chain, producing characteristic overshoot and hallucination artifacts
    6. *Propeller Shadow*: Periodic localized brightness modulation, spatial frequency matched to rotor geometry, $\alpha \in [0.05, 0.3]$
    - Plus 30 generic distortion types (Gaussian blur, noise, JPEG compression, etc.) inherited from Embodied-IQA catalog
    - Each distortion applied at 1 randomly selected intensity level from {0.2, 0.4, 0.6, 0.8, 1.0}
  - **§3.3 Annotation Pipeline** (~0.5 pages):
    - *Cognitive (VLM)*: Qwen2.5-VL-7B → task-conditioned structured description → BLEU/ROUGE-L/CIDEr text similarity (1:1:0.1 weighting) → cognitive quality score
    - *Decision (VLA)*: VLA ensemble (UAV-Track VLA, CognitiveDrone-R1, Qwen-VLA; fallback: OpenVLA-7B × 3 seeds) → sliding-window single-frame replacement → decision score. Ensemble calibration via per-model confidence weights from SITL execution regression. ICC(3,k) reported per task/distortion.
    - *Execution (CARLA-Air SITL)*: Tracking + Inspection tasks on 5% stratified subset → execution score
    - *Training label curriculum*: VLM (epochs 1-20) → VLA (epochs 21-40) → Execution (epochs 41-50)
  - **§3.4 Database Statistics** (0.25 pages): 494,352 distorted-annotated pairs (full AirCopBench), train/val/test = 6,889/861/863. Per-distortion coverage, per-task distribution, intensity level coverage.

### §4 Benchmark and Baseline (1.5 pages)

- **§4.1 UAV-IQANet Architecture** (~0.75 pages):
  - Architecture diagram (Figure 3): MobileNetV4-S backbone (frozen first 2 stages) → PANet Feature Pyramid Network → CBAM attention (channel + spatial) → Frequency-Aware Branch (patch-FFT + log-polar transform + tiny CNN, $f_f \in \mathbb{R}^{64}$) → Cross-Attention Gate ($\alpha = \sigma(W \cdot [f_s, f_f])$, $\alpha \odot f_f$) → Concat$[f_s, \alpha \odot f_f]$ → Task-Conditioned FiLM Head ($\gamma_t, \beta_t$ modulate hidden features per task $t$) → single score per task. Total: ~5.4M params (3.5M trainable).
  - Design rationale: CNN backbone for frequency-domain locality (versus ViT's global attention which smooths high-frequency degradation signatures); FAB for rotation/scale-invariant frequency representation (log-polar maps 6DoF rotation to translation); CBAM for distortion-aware attention; FiLM for task-conditioned quality regression.
  - Baselines: MobileViT-S and EfficientViT-B0 as backbone alternatives (parameter-matched comparison).
- **§4.2 Benchmark Protocol** (~0.5 pages):
  - 15+ methods: FR-IQA (PSNR, SSIM, LPIPS, DISTS, AHIQ, TOPIQ), NR-IQA (BRISQUE, NIQE, MANIQA, Q-Align, CLIP-IQA), Frequency-aware (BRISQUE/DCT, DeepFIQA), Embodied (MA-EIQA zero-shot and fine-tuned on UAV data)
  - Evaluation: SRCC, PLCC, RMSE against VLA decision scores (primary) and cognitive scores (secondary)
  - Cross-database: MA-EIQA (EPD-trained) → UAV data; UAV-IQANet → Embodied-IQA/EPD test splits
  - Statistical significance: Wilcoxon signed-rank test with Holm-Bonferroni correction
- **§4.3 Training Setup** (~0.25 pages):
  - Loss: $\mathcal{L} = \mathcal{L}_{MSE} + \lambda_{rank} \cdot \mathcal{L}_{ListMLE} + \lambda_{cross} \cdot \mathcal{L}_{CrossTaskReg}$
  - Optimizer: AdamW, lr=3e-4, weight decay=1e-4, batch size=64, 50 epochs
  - Hardware: 1×A100, ~30 GPU-hours per full training run
  - 3 seeds (42, 100, 200) per experiment; mean ± std reported

### §5 Experiments and Analysis (2 pages)

- **§5.1 Do existing IQA methods work on UAV data? (C3)** (~0.75 pages):
  - Table 2: Main results — SRCC/PLCC/RMSE for all 15+ methods vs. UAV-IQANet
  - Figure 4: Per-distortion SRCC breakdown (grouped bar, 36 distortions, top-5 methods + UAV-IQANet)
  - Key result: All existing methods SRCC < 0.5; UAV-IQANet SRCC > 0.65
  - Cross-database: Table in appendix showing MA-EIQA fails on UAV data, UAV-IQANet fails on Embodied-IQA/EPD (domain specificity confirmed)

- **§5.2 Are UAV-specific distortions genuinely distinct? (C1)** (~0.75 pages):
  - Figure 5: Per-distortion ΔVLA degradation curves (6 panels for 6 UAV distortions, 4 task lines per panel, mean ± std across intensity levels)
  - Table 3: Pool separation statistics — Cohen's d between UAV and generic distortion pools per task
  - Figure 6: Frequency signature comparison — log-polar FFT magnitude maps for 6 UAV vs. 2 nearest generic distortions
  - Key result: ≥4/6 UAV distortions show task-dependent ranking (Kendall's τ < 0.8 across ≥2 task pairs); mean ΔVLA separation with d > 0.8

- **§5.3 Ablation studies: What makes the model work? (C3-support)** (~0.5 pages):
  - Table 4: Component ablation — ΔSRCC from removing FAB, CBAM, task conditioning
  - Table 5: Distortion pool ablation — generic-only vs. UAV-only vs. full (validates AC1)
  - Table 6: Backbone comparison — MobileNetV4-S vs. MobileViT-S vs. EfficientViT-B0
  - Figure 7: Curriculum comparison — VLM-only vs. VLA-only vs. full 3-stage (validates AC2)
  - Key result: FAB contributes >0.05 SRCC; UAV distortion pool accounts for >0.08 SRCC gain over generic-only; task conditioning contributes >0.03 SRCC; ViT backbones do not outperform CNN

- **§5.4 Cross-task generalization (C4)** (supplementary, ~0.25 pages in main, full matrices in appendix):
  - Per-task SRCC matrix (train task → test task, 4×4) — in appendix
  - Leave-one-task-out summary — in appendix
  - Multi-task joint training results

### §6 Discussion (0.5 pages)

- **Limitations** (honest assessment):
  1. Synthetic distortions approximate but do not fully replicate real UAV phenomenology — validated on only 50 flights, insufficient for statistical power beyond initial correlation
  2. Database scale (3,000 reference images) is competitive but not dominant — the 6 UAV-specific distortion types are the primary differentiator
  3. VLA models are evolving targets — model availability may change, requiring re-annotation
  4. Execution ground truth limited to tracking + inspection tasks (CARLA-Air SITL scope); delivery and SAR rely on VLA ensemble without SITL calibration
  5. Compound distortions explored only on 20% subset with 10 pairs — not comprehensive
  6. Temporal quality dynamics (how quality varies within a flight trajectory) not modeled
- **Broader impact**: Enables quality-aware UAV autonomy — models trained on this benchmark could gate visual inputs before they reach downstream planners, preventing task failures from degraded perception
- **Future work**: (1) SC³ closed-loop integration — IQA-driven control adaptation; (2) cross-embodiment transfer to ground robots; (3) VLM distillation for further model compression; (4) temporal quality modeling for video streams

### §7 Conclusion (0.5 pages)

- **Restatement**: We introduced UAV-Embodied-IQA, the first benchmark for visual quality assessment in aerial embodied intelligence. The database comprises 6 physically-modeled UAV-specific distortion types, ~3,000 reference images across 4 task categories, 36-distortion coverage, and VLM/VLA/Execution multi-layer annotations. Our benchmark of 15+ existing IQA methods reveals they fundamentally fail on UAV data (SRCC < 0.5), while our lightweight task-conditioned baseline UAV-IQANet (<5.5M params) achieves SRCC > 0.65.
- **Limitations**: [condensed from §6]
- **Future work**: [condensed from §6]

---

## Figure Plan

| ID | Type | Description | Data Source | Priority | Status |
|----|------|-------------|-------------|----------|--------|
| Fig 1 | Hero/Overview (3-panel) | See hero figure description above | Manual + benchmark + distortion analysis | **CRITICAL** | TODO — compile after benchmark results |
| Fig 2 | Architecture diagram | UAV-IQANet full architecture (MobileNetV4-S → PANet → CBAM → FAB → FiLM heads) | Manual / TikZ | HIGH | TODO |
| Fig 3 | Distortion gallery | 6 UAV distortions × 4 intensity levels, example images | `scripts/visualize_distortions.py` | HIGH | TODO — run visualization script |
| Fig 4 | Per-distortion SRCC | Grouped bar chart: 36 distortions × top-5 methods + UAV-IQANet | R029 per-distortion breakdown | HIGH | Pending R029 |
| Fig 5 | Degradation curves | 6-panel per-UAV-distortion ΔVLA vs. intensity, 4 task lines each | B1/DQUS VLA evaluation | HIGH | Pending R006-R007 |
| Fig 6 | Frequency signatures | Log-polar FFT magnitude maps: 6 UAV + 2 nearest generic distortions | R030 frequency visualization | MEDIUM | Pending R030 |
| Fig 7 | Curriculum comparison | Line plot: SRCC vs. epoch for VLM-only, VLA-only, full 3-stage | R022/R022b/R023 training logs | MEDIUM | Pending R022-R023 |
| Table 1 | Database statistics | Reference images, distortions, annotated pairs, task distribution | R005 output | HIGH | Partial (R005 done) |
| Table 2 | Main benchmark results | All 15+ methods: SRCC, PLCC, RMSE (mean ± std over 3 seeds) | R009-R012 + R013 | **CRITICAL** | Pending benchmark completion |
| Table 3 | Pool separation statistics | Cohen's d, task interaction ANOVA for UAV vs. generic distortion pools | B1/DQUS analysis | HIGH | Pending |
| Table 4 | Component ablation | ΔSRCC from removing FAB, CBAM, task conditioning | R016/R017/R018 results | HIGH | Pending |
| Table 5 | Distortion pool ablation | Generic-only vs. UAV-only vs. full, SRCC comparison | R021/R021b results | HIGH | Pending |
| Table 6 | Backbone comparison | MobileNetV4-S vs. MobileViT-S vs. EfficientViT-B0 | R019/R020 results | MEDIUM | Pending |
| Table 7 | Cross-database validation | MA-EIQA → UAV, UAV-IQANet → Embodied-IQA/EPD | R036a/R036b | HIGH | Pending |

### Hero Figure (Fig 1) Detailed Specification

**Layout**: 3-column horizontal layout

**Left panel — "UAV Tasks and Distortions"**:
- 4 rows (one per task: Inspection, Tracking, Delivery, SAR)
- Each row: clean reference image (left) → UAV-distorted version (right)
- Distortion examples cycle through the 6 types across the 4 rows
- Caption sub-label: "Task-conditioned visual quality"

**Center panel — "Degradation Impact"**:
- Grouped bar chart, x-axis = 12 distortion types (6 UAV + 6 representative generic)
- y-axis = ΔVLA (success rate drop from clean reference)
- 4 colored bars per distortion = 4 tasks
- Key visual: UAV distortions show larger spread across tasks (task-dependent) than generic distortions
- Insets: propeller vibration and atmospheric scattering show opposite task rankings

**Right panel — "Benchmark Summary"**:
- Scatter plot: x-axis = model parameters (log scale), y-axis = SRCC
- All benchmarked methods as gray circles, labeled
- UAV-IQANet as a large colored star in upper-left (high SRCC, low params) — Pareto-optimal
- Horizontal dashed line at SRCC = 0.5 labeled "Existing methods upper bound"
- Horizontal dashed line at SRCC = 0.65 labeled "UAV-IQANet"

**Caption**: "Figure 1: UAV-Embodied-IQA overview. (Left) Four UAV embodied tasks suffer from 6 UAV-specific visual degradations — examples show propeller vibration on inspection, atmospheric scattering on tracking, 6DoF blur on delivery, and packet-loss blocks on SAR. (Center) UAV distortions produce task-dependent VLA performance degradation (colored bars show 4 tasks per distortion); the spread across tasks is significantly larger for UAV distortions than for generic ones. (Right) Benchmark of 15+ IQA methods: all existing methods achieve SRCC < 0.5 on UAV data (gray region), while UAV-IQANet trained on our database achieves SRCC > 0.65 at <5.5M parameters (Pareto-optimal, starred)."

**Why this figure matters**: A skim reader sees in one figure: (1) what problem we're solving (UAV-specific degradations), (2) why it's different from existing IQA (task-dependent degradation patterns), and (3) the headline result (existing methods fail, our approach succeeds). This figure alone should make the paper's contribution legible.

---

## Citation Plan

### §1 Introduction (8-12 citations)
- AirCopBench (2511.11025) — UAV task protocols
- Embodied-IQA (2505.16815) — VLM→VLA→Execution methodology
- EPD / MA-EIQA (2412.18774) — Moravec paradox in IQA
- BRISQUE (Mittal et al., 2012) — classical NR-IQA baseline
- MANIQA (Yang et al., 2022) — ViT-based NR-IQA
- Q-Align (Wu et al., 2024) — VLM-guided IQA
- CLIP-IQA (Wang et al., 2023) — zero-shot IQA
- LPIPS (Zhang et al., 2018) — perceptual FR-IQA
- DISTS (Ding et al., 2020) — texture-aware FR-IQA

### §2 Related Work (15-20 citations)
- **IQA methods**: BRISQUE, NIQE (Mittal et al., 2013), ILNIQE, MANIQA, Q-Align, CLIP-IQA, PSNR, SSIM, LPIPS, DISTS, AHIQ, TOPIQ, DeepFIQA, BRISQUE/DCT
- **Embodied IQA**: Embodied-IQA (2505.16815), EPD/MA-EIQA (2412.18774)
- **Frequency analysis**: BRISQUE/DCT (Saad et al., 2012), DeepFIQA (Chen et al., 2023)
- **UAV perception**: AirCopBench (2511.11025), CARLA-Air, MotionScape, MDMT
- **Lightweight backbones**: MobileNetV4 (Qin et al., 2024), MobileViT (Mehta & Rastegari, 2022), EfficientViT (Cai et al., 2023)
- **Architecture components**: CBAM (Woo et al., 2018), FiLM (Perez et al., 2018), PANet (Liu et al., 2018)

### §3 Database (5-8 citations)
- AirCopBench — reference image sources, task definitions
- CARLA-Air — SITL execution environment
- MotionScape — 6DoF motion statistics
- Embodied-IQA — annotation methodology
- Koschmieder (1924) — atmospheric scattering model
- Real-ESRGAN (Wang et al., 2021) — SR upscaling

### §4 Benchmark and Baseline (8-12 citations)
- All baseline IQA method citations
- MobileNetV4-S, PANet, CBAM, FiLM — architecture components
- ListMLE (Xia et al., 2008) — ranking loss
- Embodied-IQA — cross-database protocol

### §5 Experiments (4-6 citations for comparison context)
- Mostly self-referential (our own results and tables)
- Cross-database comparisons cite Embodied-IQA and EPD

### Citation Rules Applied
1. All citations above are from known published works or verified arXiv preprints
2. Key papers to [VERIFY] before final submission:
   - DeepFIQA exact venue and year
   - MotionScape paper reference (check arXiv)
   - MDMT paper reference
   - Qwen2.5-VL-7B technical report
   - UAV-Track VLA and CognitiveDrone-R1 availability
3. Prefer published versions: BRISQUE (TIP 2012), NIQE (SPL 2013), LPIPS (CVPR 2018), DISTS (ECCV 2020), MANIQA (CVPR 2022), CLIP-IQA (ICLR 2023)
4. Use `\citep` for parenthetical, `\citet` for textual (ICLR uses natbib)

---

## Reviewer Feedback

**Source**: GPT-5.6-Sol (xhigh reasoning effort) via Codex MCP — 2026-07-22T (supersedes self-assessment from 2026-07-13)

### Scores

| Criterion | Score | Key Finding |
|-----------|------:|-------------|
| 1. Logical flow | 7/10 | Skeleton natural; C2 has no §5.x home; C5 has no section; §4 model oversized vs. benchmark evidence |
| 2. Claim-evidence alignment | 4/10 | C1/C2/C3 all pending; C1 statistical design insufficient; C2 not structured as held-out validation |
| 3. Experimental completeness | 4/10 | Missing: label validity check, leakage-safe split, fair supervised baseline, clustered uncertainty, synthetic-to-real model transfer |
| 4. Positioning | 5/10 | Direction correct; overlap with Embodied-IQA/EPD/AirCopBench deeper than acknowledged; "first benchmark" must be narrowed |
| 5. Page budget feasibility | 3/10 | §4 too large (1.5p → 0.75p needed); §5 too small (2.5p → 3.3p needed); no room for captions/statistics |
| 6. Front matter strength | 6/10 | Title/problem clear; hero figure has potential; abstract writes pending thresholds as facts |

**Overall submission readiness: 2-3/10 with all claims pending; 7.5-8/10 if P0 fixes are applied and results support the story.**

### P0 Blockers (must fix before submission)

1. **Freeze dataset before evaluating.** Abstract and contribution sentence write `SRCC < 0.5` / `SRCC > 0.65` / `SRCC > 0.6` as established facts. These are success criteria, not results. Freeze distortion parameters, split manifests, and primary metrics before running any baselines. If a baseline exceeds 0.5, update the claim — not the test set. Replace "fundamentally fail" with "show limited zero-shot alignment with task utility."

2. **Define the single primary ground truth.** Clarify: (a) one primary label or three separate leaderboards? (b) for the 95% without SITL execution labels, what is the training target? (c) if training target is `cognitive_score` (VLM-derived), the claim "predict UAV task performance" overstates — say "predict a VLM/VLA-derived task-utility proxy."

3. **Fix information asymmetry.** UAV-IQANet receives task ID + question text; most baselines do not. Either (a) provide same inputs to all supervised baselines, or (b) report the IQA-only variant (no task ID) as the primary comparable row.

4. **Define source/scene-disjoint split explicitly.** Confirm no reference image appears in multiple splits; describe how scenes are assigned to train/val/test. Benchmark papers require scene-disjoint test splits.

5. **Restructure page allocation** (recommended reallocation):

   | Section | Recommended |
   |---------|------------:|
   | Title + Abstract | 0.45p |
   | §1 Introduction + hero figure | 1.15p |
   | §2 Related Work | 0.65p |
   | §3 Database (sources + distortions + labels + splits) | 2.10p |
   | §4 Baseline + evaluation protocol | **0.75p** (was 1.5p) |
   | §5 Experiments | **3.35p** (was 2.5p) |
   | §6+§7 Discussion + Conclusion (merged) | 0.55p |

### Structural Revisions Required

- **Add §5.2 "Synthetic-to-Real Validation" (C2).** C2 currently has no corresponding §5.x. It belongs before the ablation section.
- **Remove C5 (Moravec paradox) from main paper.** Human MOS collection blocks timeline and has zero evidence. Move to future work or appendix.
- **Narrow C4 to supplementary.** Per-task SRCC matrix and leave-one-out results in appendix; one paragraph summary in §5.
- **Split C3 into two questions**: C3a = zero-shot transfer gap (off-the-shelf IQA); C3b = supervised learnability (trained on UAV labels).
- **Revise hero figure Panel C.** Replace params-vs-SRCC Pareto (model paper framing) with zero-shot vs. supervised leaderboard, split clearly, with 95% CI bars.

### Revised One-Sentence Contribution (template — fill actual numbers after results)

> We introduce UAV-Embodied-IQA, a source-disjoint benchmark for predicting aerial task utility under six UAV-relevant degradation operators and 30 generic distortions, with tiered VLM/VLA annotations and a held-out execution audit; representative off-the-shelf IQA methods achieve [actual range], while equally supervised task-aware baselines achieve [actual range].

### Potential Reviewer Concerns (updated)

1. **"Why not collect real distorted data instead of synthesizing?"** → Address in §3.2: real-UAV flights at scale infeasible; 50-flight validation (§5.2) establishes synthetic fidelity. The choice is pragmatic, not lazy.

2. **"Are VLA labels reliable enough as ground truth?"** → Address in §3.3: report ICC(3,k) per task/distortion; ensemble calibration via SITL on 5% subset; compare VLM-only vs. VLA-only curriculum; acknowledge noise.

3. **"How is this different from Embodied-IQA?"** → Address in §1/§2/§3: (a) UAV-specific distortions with physical models, (b) aerial tasks vs. fixed-base manipulation, (c) 6 novel distortion operators, (d) frequency-domain signatures for UAV ego-motion, (e) cross-database domain specificity validation.

4. **"The model architecture is incremental."** → Address in §4: UAV-IQANet is a strong baseline, not a contribution. Purpose: demonstrate benchmark learnability. No architectural novelty claimed.

5. **"Database scale is small (3K reference images)."** → Address in §6: 36 distortions × ~3K = 108K annotated pairs; 6 UAV distortion types are the differentiator, not scale; assembly approach is reproducible.

---

## Next Steps

### P0 — Before any further writing (blockers)
- [ ] **Freeze split manifests** — confirm scene-disjoint train/val/test; no reference image in multiple splits
- [ ] **Freeze distortion parameters** — do not tune after seeing baseline numbers
- [ ] **Define single primary label** — decide: one leaderboard (cognitive_score) or three? document the target
- [ ] **Add task-ID-aware supervised baseline** for fair apples-to-apples comparison with UAV-IQANet

### P1 — Critical path experiments (must complete for submission)
- [ ] **R006 VLM annotation** — pre-requisite for all C1/C3 evidence
- [ ] **R007 VLA annotation** — pre-requisite for C1 ΔVLA curves and C3
- [ ] **R008 Execution annotation** (CARLA-Air SITL, 5% subset) — ICC calibration
- [ ] **R009-R012 baseline benchmark** — confirms C3a zero-shot transfer gap
- [ ] **R013 UAV-IQANet full training** — confirms C3b supervised learnability
- [ ] **R025-R028 real-UAV flights** (50 flights) — C2 synthetic-to-real; do not commit to `SRCC > 0.6` before running

### P2 — Structure fixes to apply before /paper-write
- [ ] **Add §5.2 Synthetic-to-Real** subsection for C2 evidence
- [ ] **Remove C5 (Moravec paradox)** from main claims; move to future work
- [ ] **Narrow C4** to one supplementary paragraph + appendix tables
- [ ] **Revise hero figure Panel C** — zero-shot vs. supervised leaderboard with 95% CI, not params-vs-SRCC
- [ ] **Rebalance page allocation** — §4 → 0.75p; §5 → 3.35p; §6+§7 merge → 0.55p
- [ ] **Revise one-sentence contribution** — use template from Reviewer Feedback section (fill numbers after results)

### P3 — Polish (after P1 experiments complete)
- [ ] `/paper-figure` — regenerate figures with final experiment outputs
- [ ] `/paper-write` — draft LaTeX from this plan
- [ ] `/paper-compile` — build PDF
- [ ] `/auto-paper-improvement-loop` — iterative polishing
- [ ] `/citation-audit` — verify all references before submission
- [ ] `/paper-claim-audit` — verify claims against final results

---

## Appendix Plan

Content suitable for appendix (not counted in 9-page budget):
- Full per-distortion SRCC breakdown tables for all benchmarked methods (R029)
- Per-task SRCC matrices (R024a: 4×4 train→test)
- Leave-one-task-out detailed results (R024b: 4 held-out tasks)
- Distortion intensity robustness curves (R033: per-level SRCC for all methods)
- VLA ensemble ICC(3,k) tables per task × distortion (R037a)
- Human MOS scatter plots and per-task Moravec correlation tables (R039)
- Compound distortion SRCC breakdown per pair (R040c)
- Hyperparameter sensitivity analysis
- Training convergence curves for all 21 experiment configs
- Additional backbone comparisons (EfficientNet-B0, ShuffleNetV2) — if run
- Full VLA evaluation protocol details (prompt templates, trajectory configuration)
