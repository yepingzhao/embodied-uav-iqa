# Experiment Plan

**Problem**: In low-altitude UAV embodied intelligence, visual inputs suffer from 6 UAV-specific degradations (propeller vibration blur, atmospheric scattering, 6DoF viewpoint change blur, communication packet-loss blocks, low-res+SR artifacts, propeller shadow) that differ fundamentally from both human-centric and ground-robot distortions. No IQA framework exists to predict whether a visual input is adequate for a specific UAV downstream task.

**Method Thesis**: The UAV-Embodied-IQA database — assembled from CARLA-Air + AirCopBench + Embodied-IQA methodology, with 6 mathematically-modeled UAV-specific distortion types — is the first resource enabling training and evaluation of task-conditioned IQA for aerial embodied intelligence.

**Date**: 2026-06-19

---

## Claim Map

| # | Claim | Type | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|-------|------|----------------|-----------------------------|---------------|
| C1 | UAV-specific distortions produce distinct VLA performance degradation patterns vs. generic distortions | Primary | Core differentiator — without this, the database is just another IQA dataset | ≥4/6 UAV distortion curves are separable from the generic distortion pool (mean shift > 1σ); at least 2 distortions show task-dependent degradation ordering | B1 (DQUS), B5 (QUFD) |
| C2 | Synthetic UAV distortions correlate with real-world UAV distortion effects | Primary | If synthetic distortions don't match reality, the whole database is invalid | Overall SRCC > 0.6 between synthetic and real distortion impact on VLA behavior; gate-check SRCC > 0.5 at 20 flights | B2 (D2RB) |
| C3 | Existing IQA methods fail to predict UAV task performance; database enables training effective UAV-IQA models | Primary | Justifies the need for the database — a benchmark result that defines the problem space | All 15+ existing methods SRCC < 0.5; UAV-IQANet trained on this database achieves SRCC > 0.65; statistically significant gap (p < 0.01) | B3 (B4H), B4 (ABL) |
| C4 | Multi-task training enables cross-task quality generalization | Supporting | Broader impact — task-conditioned IQA transfers beyond single-task training | Cross-task SRCC within 0.05 of single-task; task embedding removal drops > 0.03 SRCC | B4 (XTG) |
| C5 | The Moravec paradox holds for UAV embodied IQA — human quality ratings are misaligned with UAV task performance | Supporting | Justifies the entire task-performance-based annotation paradigm; without this gap, human-centric IQA would suffice | Human MOS vs. VLA decision score SRCC < 0.3; human MOS vs. Execution score SRCC < 0.25; statistically distinct from VLA-Execution SRCC | B6 (MVRX) |
| C6 | Causal assessment (identifying distortion causes) provides annotation signal beyond simple quality scoring | Supporting | AirCopBench showed causal assessment is the cognitive ability most correlated with MLLM overall performance; establishing that this signal complements quality scores strengthens the annotation methodology | Causal assessment accuracy predicts VLA degradation patterns (SRCC > 0.4 between causal score and ΔVLA); causal score adds incremental value beyond quality score in predicting ΔVLA (partial correlation test) | B6 (CASL) |

### Anti-Claims to Rule Out

| # | Anti-Claim | How Ruled Out | Linked Block |
|---|------------|---------------|--------------|
| AC1 | "Gain comes from database size, not UAV distortion types" | Train on 18 generic distortions only vs. 6 UAV + 18 generic; if generic-only also works, the UAV distortion taxonomy is decorative | B4-ABL (R021) |
| AC2 | "VLA pseudo-labels are unreliable; gains come from VLM labels alone" | Compare VLM-only vs. full curriculum; report VLA inter-model SRCC per task; show execution-layer calibration improves VLA-only | B4-ABL (R022, R023) |
| AC3 | "Frequency branch is unnecessary; a deeper CNN achieves the same" | W/o FAB ablation vs. full model; param-matched deeper baseline | B4-ABL (R016) |
| AC4 | "Task-conditioning doesn't matter; a single generic IQA model is sufficient" | Task-agnostic variant vs. task-conditioned; cross-task generalization matrix | B4-ABL (R017), B4-XTG (R024) |
| AC5 | "ViT backbones are the real gain; the CNN choice is incidental" | MobileViT-S and EfficientViT-B0 backbones; if ViT variants dominate, claim of CNN efficiency is false | B4-ABL (R019, R020) |
| AC6 | "UAV distortions are just renamed ground-robot distortions; a model trained on Embodied-IQA/EPD would work on UAV data" | Cross-database evaluation: MA-EIQA (EPD-trained) zero-shot on UAV data; UAV-IQANet zero-shot on Embodied-IQA/EPD. If cross-database SRCC > 0.5, the UAV distortion taxonomy is not genuinely distinct | B6-CDBV (R036a, R036b) |
| AC7 | "VLA ensemble labels are too noisy to serve as training targets; a single strong VLA is sufficient" | Compare VLA ensemble (weighted mean with calibration) vs. single-best-VLA labels for UAV-IQANet training; report ICC(3,k) per task/distortion | B6-VLAC (R037) |

---

## Paper Storyline

**Main paper must prove**:
1. The 6 UAV-specific distortion types produce task-dependent degradation patterns that existing IQA cannot predict (C1 → B1, supporting visuals B5)
2. Synthetic distortions faithfully reproduce real UAV phenomenology (C2 → B2)
3. All existing IQA methods fail on UAV data; the database enables effective UAV-IQA (C3 → B3, B4)
4. Key design choices (frequency branch, task embedding, 6 distortion taxonomy) are individually justified (C3-support → B4)

**Appendix can support**:
- Per-distortion SRCC breakdowns for all benchmarked methods
- Robustness to distortion intensity level (per-level curves)
- Full VLA ensemble agreement statistics (ICC per task + distortion)
- Curriculum training convergence curves
- Additional backbone comparisons (EfficientNet-B0, ShuffleNetV2)
- Human MOS vs. VLA/Execution scatter plots (C5 Moravec paradox evidence)
- Causal assessment accuracy per distortion type and correlation with VLA degradation (C6 evidence)
- Cross-database validation tables (UAV-IQANet on Embodied-IQA/EPD; MA-EIQA on UAV data)
- Compound distortion analysis (per-pair SRCC breakdown)

**Experiments intentionally cut**:
- Full VLA fine-tuning comparison (deferred to Phase 3 model paper)
- Human subject study (not needed for database paper)
- Cross-embodiment transfer to ground robots (future work)
- Multi-UAV collaboration quality assessment (future work)
- SC³ closed-loop integration experiments (future work, Phase 4)
- Real-time inference benchmarking (relevant to Phase 3 model paper, not database paper)

---

## Experiment Blocks

### Block B1 — Distortion Quality-Utility Spectrum (DQUS)

**Claim tested**: C1 — UAV-specific distortions produce distinct VLA performance degradation patterns vs. generic distortions.

**Why this block exists**: This is the foundational evidence that the 6 UAV distortion types are not just cosmetic variations but constitute a genuinely distinct degradation space. Without this, reviewers will ask: "Why not just use the Embodied-IQA database with a few extra images?"

**Dataset / task**: Full 3,000 reference images; all 24 distortion types (6 UAV + 18 generic) at 5 intensity levels; all 4 UAV tasks (inspection, tracking, delivery, SAR). VLA evaluation on a 10% stratified random subset (300 images × 120 distortion levels = 36,000 VLA evaluations).

**Compared systems**: VLA ensemble (UAV-Track VLA, CognitiveDrone-R1, Qwen-VLA; fallback: OpenVLA-7B × 3 seeds) evaluated on clean reference vs. each distorted variant.

**Metrics**:
- **Primary**: ΔVLA-success (success rate drop from reference, per distortion per task), task degradation rank correlation (Kendall's τ between distortion severity rankings across tasks)
- **Secondary**: VLM cognitive score change, per-distortion mean ΔVLA and standard deviation, inter-task pattern divergence (normalized L1 distance between degradation vectors)

**Setup details**:
- VLA evaluation: sliding-window single-frame replacement protocol (replace 1 frame in a clean trajectory with distorted frame, measure success rate change)
- 5 intensity levels per distortion, mapped to perceptually equidistant steps
- 3 VLA seeds per evaluation to control stochasticity
- Normalize ΔVLA by per-task baseline success rate

**Success criterion**: ≥4/6 UAV distortion types show task-dependent degradation ranking (Kendall's τ < 0.8 across at least 2 task pairs); mean ΔVLA of UAV pool separates from generic pool with effect size d > 0.8 (Cohen's d).

**Failure interpretation**: If only 1-2 UAV distortions show separable behavior, or if 5+ overlap with generic pool distribution → the UAV distortion taxonomy is too narrow or the distortions are functionally equivalent to existing types. Consider: add more UAV-specific distortion types, tighten mathematical models, or narrow the claim to the 2-3 that are truly distinct.

**Table / figure target**: Figure 2 (6-panel per-distortion degradation curves, 4 tasks as line series); Table 1 (pool separation statistics, effect sizes, task interaction ANOVA).

**Priority**: MUST-RUN

---

### Block B2 — Database-to-Reality Bridge (D2RB)

**Claim tested**: C2 — Synthetic UAV distortions correlate with real-world UAV distortion effects.

**Why this block exists**: The entire database uses synthetic distortions. If they don't correlate with real UAV flight footage, the database has zero practical value. This is the make-or-break validation block.

**Dataset / task**: 50 real UAV flights capturing the 6 distortion types under controlled conditions. Each flight produces: (a) real distorted frames, (b) matched synthetic frames using the same distortion parameters estimated from telemetry. VLA evaluation on both real and synthetic frames.

**Compared systems**: Real footage vs. synthetic counterpart, per distortion type.

**Metrics**:
- **Primary**: SRCC(synthetic ΔVLA, real ΔVLA) across all distortion types and levels; per-distortion PLCC
- **Secondary**: Qualitative side-by-side visual comparison; per-frame VLA decision agreement rate

**Setup details**:
- Platform: DJI Matrice 300 RTK or Parrot ANAFI Ai (subject to availability)
- Sequential validation protocol:
  - **Gate (20 flights)**: Scattering/haze + propeller vibration only. 10 flights each, varying altitude β and vibration frequency.
  - **Gate criterion**: SRCC > 0.5 for first 20 flights → proceed to full validation. SRCC < 0.4 → recalibrate distortion model parameters, repeat gate.
  - **Full (30 flights)**: Remaining 4 distortion types (6DoF blur, packet loss, SR artifacts, propeller shadow). 7-8 flights each.
- Distortion parameter estimation: IMU for vibration frequency/amplitude; visibility sensor for β; GPS/IMU for 6DoF velocity; onboard packet loss logs; camera metadata for resolution.
- VLA evaluation offline, post-flight

**Success criterion**: Overall SRCC > 0.6; per-distortion SRCC > 0.5 for ≥5/6 types.

**Failure interpretation**: SRCC < 0.4 → synthetic distortion models are fundamentally wrong; major recalibration needed. SRCC 0.4-0.6 → usable but weak claim; paper must acknowledge gap and position as "first approximation." Consider: recalibrate distortion intensity mapping, add multi-distortion compound cases, or increase flight count.

**Table / figure target**: Figure 3 (scatter plot with marginal distributions, per-distortion coloring); Table 2 (per-distortion SRCC/PLCC/RMSE).

**Priority**: MUST-RUN (gate at 20 flights is a hard stop/go decision)

---

### Block B3 — Baseline-for-Humanity Benchmark (B4H)

**Claim tested**: C3 — Existing IQA methods fail to predict UAV task performance; the database enables training effective UAV-IQA.

**Why this block exists**: The main result table. Establishes the problem (existing methods fail) and the solution (database-trained models work). This is the paper's central empirical contribution.

**Dataset / split / task**: UAV-Embodied-IQA database; 80/20 train/test stratified by task category and distortion type; 3 random seeds.

**Compared systems**:

| Category | Methods | FR/NR |
|----------|---------|-------|
| Pixel-reference | PSNR, SSIM, MS-SSIM | FR |
| Deep perceptual | LPIPS (AlexNet), LPIPS (VGG), DISTS | FR |
| Deep FR | AHIQ, TOPIQ | FR |
| Handcrafted NR | BRISQUE, NIQE, ILNIQE | NR |
| Deep NR | MANIQA, Q-Align, CLIP-IQA, LIQE | NR |
| Frequency-aware | BRISQUE (DCT features), DeepFIQA | NR |
| Embodied baseline | MA-EIQA (zero-shot), MA-EIQA (fine-tuned) | NR |
| **UAV baseline (ours)** | UAV-IQANet (task-conditioned), UAV-IQANet (task-agnostic) | NR |

Total: 19 configs (15 distinct methods + 4 variants).

**Metrics**:
- **Primary**: SRCC, PLCC (vs. VLA decision score ground truth)
- **Secondary**: RMSE, per-task SRCC breakdown, per-distortion SRCC breakdown

**Setup details**:
- Ground truth: VLA ensemble mean decision score (3-model average)
- UAV-IQANet training: 3-stage curriculum (VLM epochs 1-20 → VLA epochs 21-40 → Execution epochs 41-50), AdamW, lr=3e-4, batch=64, cosine schedule
- MA-EIQA fine-tuning: same data split, 50 epochs, default hyperparameters from paper
- Zero-shot methods evaluated directly on test set
- Statistical testing: pairwise Wilcoxon signed-rank with Holm-Bonferroni correction (p < 0.01)

**Success criterion**: All existing methods SRCC < 0.5; UAV-IQANet (task-conditioned) SRCC > 0.65; gap between best existing and UAV-IQANet > 0.15 SRCC and statistically significant.

**Failure interpretation**: If any existing method > 0.5 SRCC → database is not hard enough for the benchmark claim. If UAV-IQANet < 0.55 → database may be too noisy for supervised training. If gap < 0.10 → insufficient practical improvement over existing methods. Mitigation: increase distortion intensity range, add compound distortions, or curate harder test split.

**Table / figure target**: Table 3 (main results, SRCC/PLCC for all methods on all tasks + average); Figure 4 (scatter: predicted vs. ground-truth for top-3 methods + UAV-IQANet).

**Priority**: MUST-RUN

---

### Block B4 — Ablation & Cross-Task Generalization (ABL + XTG)

**Claim tested**: C3 (ablation isolation), C4 (cross-task generalization). Anti-claims AC1-AC5.

**Why this block exists**: After showing that the database + model works (B3), this block isolates why it works. Each ablation defends a specific design decision. Cross-task experiments test whether task-conditioned IQA generalizes.

**Dataset / split / task**: Same 80/20 stratified split as B3; all 4 tasks.

**Ablation variants**:

| Run | Variant | Anti-Claim | What It Tests |
|-----|---------|------------|---------------|
| R016 | W/o FAB (frequency branch removed) | AC3 | Value of log-polar FFT features (~30K params) |
| R017 | W/o task embedding (shared MLP only) | AC4 | Value of task-conditioning |
| R018 | W/o CBAM attention | — | Value of spatial-channel attention |
| R019 | MobileViT-S backbone | AC5 | CNN vs. ViT architecture (param-matched) |
| R020 | EfficientViT-B0 backbone | AC5 | CNN vs. ViT architecture |
| R021 | Train on 18 generic distortions only | AC1 | Value of UAV-specific distortion types |
| R021b | Train on 6 UAV distortions only | AC1 | Sufficiency of UAV pool alone |
| R022 | VLM-only curriculum | AC2 | Value of VLA labels |
| R022b | VLA-only curriculum | AC2 | Value of VLM labels |
| R023 | W/o execution layer (5% SITL subset) | AC2 | Value of SITL execution labels for calibration |

**Cross-task variants**:

| Run | Setup | What It Tests |
|-----|-------|---------------|
| R024a | Train on task A, test on B/C/D (4×3 matrix) | Cross-task zero-shot generalization |
| R024b | Train on 3 tasks, test on 4th (leave-one-out × 4) | Multi-task → unseen task transfer |
| R024c | Train on all tasks jointly (multi-task) | Upper bound for multi-task training |

**Metrics**: SRCC, PLCC, per-task breakdown.

**Setup details**: 3 seeds per ablation; fixed hyperparameters except the ablated component. For backbone swaps, match parameter count within ±5% (adjust width multiplier). For curriculum ablations, match total epochs (50 each).

**Success criterion**:
- FAB removal drops > 0.05 SRCC (justifies frequency branch)
- Task embedding removal drops > 0.03 SRCC (justifies task-conditioning)
- 6-UAV-only achieves > 0.90 of 24-distortion SRCC (UAV pool is sufficient)
- 18-generic-only drops > 0.08 below full (UAV distortions are essential)
- ViT backbones not significantly better than CNN (+/- 0.02 SRCC) (justifies CNN efficiency choice)
- Cross-task SRCC within 0.05 of single-task for ≥3/4 task pairs

**Failure interpretation**: If no ablation drops > 0.02 → model may be overparameterized; try smaller variant. If ViT backbones significantly outperform CNN → architecture choice was wrong; reconsider ViT for baseline. If cross-task SRCC drops > 0.10 → task-conditioning is essential but fragile; paper must emphasize this limitation.

**Table / figure target**: Table 4 (ablation results); Table 5 (cross-task generalization matrix); Figure 5 (cross-task heatmap).

**Priority**: MUST-RUN (R016-R021, R024a-c); NICE-TO-HAVE (R022, R022b, R023)

---

### Block B5 — Qualitative Failure Diagnosis (QUFD)

**Claim tested**: C1 (supporting visual evidence), C3 (failure characterization).

**Why this block exists**: Quantitative tables alone don't tell the full story. Qualitative analysis shows *why* frequency-aware methods work and *where* existing methods fail, which strengthens the paper's narrative and provides diagnostic value for future work.

**Dataset / task**: Test set results from B3.

**Sub-blocks**:

**B5a — Frequency Signatures**: Compute per-distortion power spectra averaged over 100 random images. Show log-polar FFT magnitude for each UAV distortion vs. nearest generic counterpart. Expected: UAV distortions show structured frequency patterns (e.g., propeller vibration → directional streak in spectrum; packet loss → block-boundary harmonics).

**B5b — Per-Distortion Error Breakdown**: SRCC per distortion type for top-5 existing methods + UAV-IQANet. Bar chart identifying which distortions cause the largest gap. Expected: Methods without frequency features fail worst on vibration and 6DoF blur (frequency-domain distortions); handcrafted methods fail worst on scattering (requires learned features).

**B5c — Hard Case Gallery**: Select 10-15 images with extreme prediction error (|predicted - ground truth| > 2σ). Show image + distortion label + all method predictions. Identify failure patterns (e.g., all methods fail on extreme haze; human-centric metrics overestimate low-res quality).

**B5d — VLA Ensemble Agreement**: Compute ICC(3,k) for VLA ensemble per task and per distortion. Show which tasks/distortions have highest/lowest VLA agreement. Useful for interpreting label reliability in B3.

**Metrics**: Qualitative (visual evidence) + quantitative (per-distortion SRCC for B5b, ICC for B5d).

**Success criterion**: Clear, interpretable visual patterns; at least 2-3 actionable insights for future work (e.g., "haze + low-res compound degradation is the hardest case").

**Failure interpretation**: No clear pattern → doesn't block claims (C1 and C3 are defended quantitatively in B1 and B3), but weakens the discussion section. Consider: sample more extreme cases, or expand failure taxonomy.

**Table / figure target**: Figure 6 (frequency signatures, 6-panel + 2 generic comparison); Figure 7 (per-distortion error bar chart); Figure 8 (hard case gallery); Table A1 in appendix (per-distortion full breakdown).

**Priority**: MUST-RUN (B5a, B5b, B5c for figures); NICE-TO-HAVE (B5d, full appendix table)

---

### Block B6 — Causal Assessment, Cross-Database & VLA Calibration (CACV)

**Claim tested**: C5 (Moravec paradox), C6 (causal assessment value), supporting C1 and C3. Anti-claims AC6, AC7.

**Why this block exists**: Four literature-motivated sub-blocks that strengthen the paper's foundational claims:

- **CASL**: AirCopBench (2511.11025) found causal assessment is the cognitive ability most correlated with MLLM performance. Testing whether VLM causal assessment accuracy predicts VLA degradation patterns validates annotating this dimension.
- **MVRX**: EPD (2412.18774) demonstrated the Moravec paradox (PLCC < 0.22) for ground robots; quantifying this gap for UAV tasks justifies our task-performance-based annotation paradigm.
- **CDBV**: Cross-database validation proves UAV-specific distortions are genuinely distinct from ground-robot distortions — if models trained on Embodied-IQA/EPD transfer well to UAV data, our taxonomy is decorative.
- **VLAC**: VLA ensemble calibration addresses the low inter-model agreement (SRCC ≈ 0.25) revealed by Embodied-IQA (2505.16815).

**Sub-blocks**:

**B6a — Causal Assessment Signal (CASL)**:

| Aspect | Detail |
|--------|--------|
| Dataset | 10% stratified subset (300 refs × 24 distortions × 5 levels = 36,000 pairs) |
| VLM task | AirCopBench causal assessment VQA: "What type of visual degradation affects this image?" (6 UAV + 18 generic options) |
| Metrics | Causal classification accuracy per distortion type; SRCC between causal confidence and ΔVLA; partial correlation (causal score → ΔVLA controlling for quality score) |
| Success criterion | Causal accuracy > 60% for ≥4/6 UAV types; partial correlation significant (p < 0.05); SRCC(causal, ΔVLA) > 0.4 |
| Runs | R034a (causal VLM annotation), R034b (causal-ΔVLA correlation analysis) |

**B6b — Moravec Paradox Quantification (MVRX)**:

| Aspect | Detail |
|--------|--------|
| Dataset | 5% stratified subset (~9,000 pairs), balanced across tasks and distortion types |
| Human MOS | 15+ subjects, single-stimulus continuous quality scale (0-100), crowdsourced |
| Compared signals | Human MOS vs. VLM quality score vs. VLA decision score vs. Execution score |
| Metrics | SRCC/PLCC between all pairwise signal combinations; per-task and per-distortion breakdown |
| Success criterion | SRCC(human MOS, VLA decision) < 0.3 AND SRCC(human MOS, Execution) < 0.25 AND SRCC(VLA, Execution) ≥ 0.5 (gap between human and robot-aligned signals is statistically significant) |
| Failure interpretation | If human MOS correlates with VLA > 0.4 → Moravec paradox may not hold for UAV tasks; if VLA-Execution < 0.4 → annotation pipeline fundamentally unreliable |
| Runs | R038 (human MOS collection), R039 (correlation analysis) |

**B6c — Cross-Database Validation (CDBV)**:

| Aspect | Detail |
|--------|--------|
| Dataset | UAV-Embodied-IQA test split (4,304 images); Embodied-IQA test split (~7,400 images); EPD test split (~2,500 images) |
| Compared systems | MA-EIQA (EPD-trained, zero-shot on UAV data); UAV-IQANet (zero-shot on Embodied-IQA and EPD); MA-EIQA fine-tuned on UAV data; UAV-IQANet fine-tuned on Embodied-IQA |
| Metrics | SRCC/PLCC on each target database; cross-database SRCC drop vs. in-domain SRCC |
| Success criterion | MA-EIQA zero-shot on UAV SRCC < 0.4; UAV-IQANet zero-shot on Embodied-IQA SRCC < 0.4; cross-database SRCC drop > 0.15 vs. in-domain |
| Failure interpretation | If cross-database SRCC > 0.5 → UAV distortion taxonomy is not meaningfully distinct from ground-robot distortions |
| Runs | R036a (MA-EIQA on UAV data, zero-shot + fine-tuned); R036b (UAV-IQANet on Embodied-IQA/EPD, zero-shot); R036c (cross-database significance tests) |

**B6d — VLA Ensemble Calibration (VLAC)**:

| Aspect | Detail |
|--------|--------|
| Dataset | Full VLA annotation set (from M1 R007) |
| Analysis | ICC(3,k) per task per distortion; compare single-best-VLA, simple ensemble mean, vs. weighted mean with SITL-calibrated confidence weights |
| Metric | SRCC(ensemble label, execution score) for each calibration strategy |
| Success criterion | Weighted ensemble SRCC > simple mean SRCC + 0.03; ICC(3,k) reported transparently for all task×distortion cells |
| Runs | R037a (ICC computation), R037b (calibration weight learning on 5% SITL subset), R037c (calibrated vs. uncalibrated ensemble comparison) |

**Priority**: MUST-RUN (B6a CASL, B6c CDBV, B6d VLAC); NICE-TO-HAVE (B6b MVRX — depends on human subject availability)

---

### Block B7 — Compound Distortion Analysis (CPDA)

**Claim tested**: C1 (supporting), C3 (supporting). Anti-claim: "Single-distortion analysis is sufficient; compound effects are linear combinations."

**Why this block exists**: AirCopBench images contain multiple simultaneous degradations (small targets + complex backgrounds + environmental interference). Embodied-IQA JND analysis showed compound effects can be fatal at low individual intensities — verifying non-linear compound degradation is essential for real-world relevance.

**Dataset / task**: 20% stratified reference subset (~600 images) × 10 compound distortion pairs × 3 intensity level pairs = 18,000 compound-distorted pairs. VLA evaluation on a 5% subset (900 pairs).

**Compound pairs** (10 combinations):
1. Propeller vibration + atmospheric haze
2. Propeller vibration + packet-loss blocks
3. Atmospheric haze + low-res SR artifacts
4. Atmospheric haze + 6DoF viewpoint change blur
5. 6DoF blur + packet-loss blocks
6. Low-res SR + propeller shadow
7. Propeller vibration + propeller shadow
8. Atmospheric haze + propeller shadow
9. Packet-loss blocks + propeller shadow
10. Low-res SR + packet-loss blocks

**Metrics**:
- **Primary**: ΔVLA for compound vs. sum of individual ΔVLA (non-linearity ratio); SRCC(compound quality score predicted by single-distortion IQA, actual compound ΔVLA)
- **Secondary**: Per-pair degradation ranking, task interaction effects

**Setup details**: 3 intensity level pairs: (low, low), (mid, mid), (high, low). Compare compound ΔVLA against additive model: ΔVLA(A+B) vs. ΔVLA(A) + ΔVLA(B) - baseline.

**Success criterion**: Non-linearity ratio > 1.2 for ≥3/10 pairs (compound effect worse than additive model); existing single-distortion IQA methods fail to predict compound ΔVLA (SRCC < 0.4).

**Failure interpretation**: If compound effects are approximately linear → compound distortion analysis is not necessary for the database paper; can be deferred to future work.

**Table / figure target**: Table A2 in appendix (per-pair ΔVLA, non-linearity ratios); Figure A1 (compound vs. additive scatter).

**Priority**: NICE-TO-HAVE (provides strong realism argument but C1/C3 are defended by B1/B3)

---

## Run Order and Milestones

| Milestone | Goal | Runs | Weeks | GPU-hrs | Decision Gate | Risk |
|-----------|------|------|-------|---------|---------------|------|
| M0: Sanity | Data pipeline + quick validation | R001-R004 | 1-3 | ~20 | All data downloadable; all 6 distortions produce plausible outputs; annotation pipeline outputs valid scores; overfit converges | Data access changes, VLA weights unavailable |
| M1: Database | Full database construction + annotation | R005-R008 | 3-8 | ~500 | ≥170K annotated pairs pass QC; VLA inter-model SRCC documented; execution layer complete on 5% subset | VLA models unavailable (mitigation: OpenVLA backup); SITL fidelity low |
| M2: Baselines | Run all existing IQA methods | R009-R012 | 6-10 | ~300 | All baselines run; MA-EIQA fine-tuning converges | Some methods require per-image computation (mitigation: batch preprocessing) |
| M3: Main Model | Train UAV-IQANet, produce benchmark table | R013-R015 | 8-12 | ~150 | UAV-IQANet SRCC > 0.65; main table complete | SRCC below target (mitigation: hyperparameter sweep R015A) |
| M4: Ablations | Ablation studies + cross-task | R016-R024 | 10-14 | ~250 | ≥2 ablations show meaningful SRCC drop; cross-task generalization confirmed or characterized | Ablations inconclusive (mitigation: train smaller variant to amplify differences, R023A) |
| M5: Real UAV | Real-world validation flights | R025-R028 | 8-18 (async) | ~0 | Gate (20 flights): SRCC > 0.5 → proceed; Full (50): SRCC > 0.6 → claim validated | Weather, equipment, permits (mitigation: 2× flight buffer, indoor calibration) |
| M6: Polish | Qualitative analysis + paper figures | R029-R033 | 14-18 | ~30 | All figures ready for paper | None (post-hoc analysis) |
| M7: CACV+CPDA | Causal assessment, cross-database, VLA calibration, compound distortion | R034-R037 | 10-18 | ~120 | Causal accuracy > 60%; cross-database SRCC drop > 0.15; VLA calibration improves ensemble SRCC; compound non-linearity confirmed or characterized | Cross-database SRCC unexpectedly high (mitigation: add more distortion intensity levels); human subject recruitment for MVRX fails (mitigation: skip B6b, keep NICE) |
| M8: Human MOS | Moravec paradox quantification (async, parallel) | R038-R039 | 12-22 (async) | ~0 | Human MOS vs. VLA SRCC < 0.3 → Moravec paradox confirmed for UAV | Low subject recruitment (mitigation: reduce subset to 3,000 pairs) |
| M9: CPDA | Compound distortion VLA evaluation | R040a-R040c | 16-22 | ~40 | Compound non-linearity confirmed for ≥3/10 pairs; existing IQA fails on compound pairs | Compound distortion VLA throughput bottleneck (mitigation: reduce pairs × intensity levels) |

### Dependency Graph

```
M0 (Sanity) ──→ M1 (Database) ──┬──→ M2 (Baselines) ──→ M3 (Main) ──→ M4 (Ablations) ──┬──→ M6 (Polish)
                                 │                                                        │
                                 ├──→ M7 (CACV) ──→ M9 (CPDA) ─────────────────────────────┤
                                 │                                                        │
                                 └──→ M5 (Real UAV, parallel) ────────────────────────────┘
                                 
M8 (Human MOS, async parallel) ───────────────────────────────────────────────────────────→ M6
```

M5 and M8 are fully parallel to M1-M4. M1 is on the critical path. M7 can start once M1 produces VLA annotations (Week 5-6, overlapping with M1 tail). M9 (CPDA) starts after M7 VLA calibration confirms reliable ensemble scoring. M2 can start once M1 produces test images.

### Stop/Go Decision Gates Summary

| Gate | When | Criterion | If Fail |
|------|------|-----------|---------|
| Pre-start VLA gate | Before M0 | ≥2 target VLA models have public weights | Activate OpenVLA-7B × 3 backup; add 2 weeks for fine-tuning |
| M0 completion | End of Week 3 | All 6 distortions visually plausible; annotation pipeline valid | Fix distortion models or annotation code before proceeding |
| M5 gate check | After 20 flights | SRCC > 0.5 | Recalibrate distortion models; repeat 10-flight gate |
| M3 completion | End of Week 12 | UAV-IQANet SRCC > 0.55 | Abandon database-only paper; reconsider model novelty contribution |

---

## Compute and Data Budget

| Category | GPU-hours | Details |
|----------|-----------|---------|
| Distortion synthesis | ~60 | Image processing (mostly CPU); GPU for SR upscaling (Real-ESRGAN inference); +10 for compound distortion synthesis |
| VLM annotation | ~350 | 180K pairs × 3 VLMs × ~2s per inference; batch inference on 4×A100; +50 for causal assessment VQA |
| VLA annotation | ~220 | 180K pairs × 3 VLAs × sliding-window protocol; trajectory-level evaluation; +20 for compound distortion subset |
| Execution annotation | ~50 | 5% subset (~9K pairs) in CARLA-Air SITL; 2 tasks (tracking, inspection) |
| Baseline benchmarking | ~300 | 15+ methods inference; some require per-image optimization (BRISQUE, NIQE) |
| UAV-IQANet training | ~120 | Full 3-stage curriculum, 50 epochs/stage; 3 seeds; +20 for cross-database fine-tuning |
| Ablation runs | ~250 | 10+ variants × 3 seeds × 50 epochs each |
| Cross-database evaluation | ~80 | MA-EIQA zero-shot + fine-tuned on UAV data; UAV-IQANet zero-shot on Embodied-IQA/EPD; significance tests |
| Qualitative analysis | ~50 | Post-hoc; FFT computation, VLA ensemble analysis, causal correlation analysis, compound distortion analysis |
| Compound distortion VLA | ~40 | 900 compound pairs × 3 VLAs × sliding-window protocol |
| **Total** | **~1,520** | Within 2,000 budget; ~480 GPU-hours margin for retries and sweeps |

**Data preparation needs**:
- AirCopBench: download sim + real splits; verify frame counts and task labels
- CARLA-Air: install CARLA 0.9.15, verify tracking/inspection scenarios run
- MotionScape: download and extract frames; verify optical flow distribution matches paper
- MDMT: download real urban scenes; verify license for redistribution in derived database

**Human evaluation needs**: None (database paper; VLM/VLA/Execution automated annotation)

**Biggest bottleneck**: VLA annotation throughput — each evaluation requires running a full trajectory with 1 distorted frame inserted, × 180K pairs × 3 VLAs. Consider multi-GPU parallelization (one VLA per GPU).

---

## Risks and Mitigations

| Risk | Severity | Probability | Mitigation |
|------|----------|-------------|------------|
| Target VLA models (UAV-Track VLA, CognitiveDrone-R1) unavailable | High | Medium | Pre-start gate verifies availability; OpenVLA-7B × 3 seeds as documented backup; add 2 weeks for AirCopBench fine-tuning |
| VLA inter-model agreement too low (SRCC < 0.2) for reliable labels | High | Low | Increase VLA ensemble to 5 models (add Qwen2.5-VL-7B converted to agent); use execution-layer SITL scores as primary ground truth for the 5% subset; report agreement transparently |
| CARLA-Air SITL integration broken or too slow | Medium | Medium | Test SITL integration during M0 sanity; if unreliable, fall back to trajectory-level VLA evaluation (no per-frame SITL) and document limitation |
| Real-UAV validation SRCC < 0.4 | High | Low-Medium | Gate design prevents wasting all 50 flights; recalibrate distortion intensity mapping from telemetry; add multi-distortion compound cases to match real complexity |
| UAV-IQANet SRCC below target (< 0.55) | Critical | Low-Medium | Hyperparameter sweep (lr, weight decay, CBAM reduction ratio); try deeper MobileNetV4 backbone; consider knowledge distillation from VLM features (deferred from reviewer suggestion) |
| Weather prevents real flights within timeline | Medium | Medium | Buffer 2× flight window; schedule during historically stable season; indoor controlled-environment flights as fallback for vibration + 6DoF types |
| Existing method achieves SRCC > 0.5 on our database | Medium | Low | Increase distortion intensity ceiling; add compound distortion types (e.g., haze + vibration); curate harder test split with only highest-intensity samples |
| Database scale insufficient for training (> 180K pairs needed) | Low | Low | Current 180K pairs with data augmentation is adequate for < 5.5M param models; if not, add AirSim drone scenarios or Waymo/UAV cross-domain data |
| Cross-database SRCC unexpectedly high (>0.5) | Medium | Low-Medium | Add more UAV-specific distortions or increase intensity ceiling; verify distortion implementation produces perceptually distinct artifacts |
| Human MOS collection infeasible (recruitment/cost) | Medium | Medium | B6b MVRX is NICE-TO-HAVE; if infeasible, cite EPD's PLCC < 0.22 as prior evidence and skip this block |
| Compound distortion VLA throughput bottleneck | Low | Medium | Compound analysis uses only 900 VLA pairs (manageable); if still too slow, reduce to 5 pairs × 2 intensity levels |
| Causal assessment VQA accuracy too low (< 40%) | Medium | Low | VLMs may struggle with UAV-specific distortion identification; if so, narrow claim to generic-vs-UAV binary classification instead of fine-grained type identification |

---

## Final Checklist

- [ ] Main paper tables are covered (Tables 1-5)
- [ ] Claims C1-C3 have dedicated experiment blocks with success criteria
- [ ] C4 (cross-task generalization) is properly scoped as supporting
- [ ] C5 (Moravec paradox for UAV) has B6b MVRX block; NICE-TO-HAVE contingency documented
- [ ] C6 (causal assessment signal) has B6a CASL block with partial correlation test
- [ ] Novelty is isolated (B4 ablations: frequency branch, task embedding, UAV distortion pool)
- [ ] Simplicity is defended (CNN justified against ViT; shared MLP justified against per-task heads)
- [ ] Frontier contribution is justified (VLM/VLA annotation oracles are essential to methodology, not decorative; frequency branch is justified against deeper-CNN alternative)
- [ ] Real-world validation bridges synthetic → real gap (B2, gate design)
- [ ] Cross-database validation proves domain specificity (B6c CDBV, AC6)
- [ ] VLA ensemble calibration addresses annotation reliability (B6d VLAC, AC7)
- [ ] Compound distortion analysis provides real-world relevance (B7 CPDA)
- [ ] Anti-claims AC1-AC7 are each addressed by at least one experiment
- [ ] Nice-to-have runs are separated from must-run (curriculum ablations, appendix tables, B6b MVRX, B7 CPDA)
- [ ] Compute budget has margin (~480 GPU-hours for retries and troubleshooting)
- [ ] Stop/go gates are defined at critical decision points
- [ ] Dependency graph respects M1 → M2 → M3 → M4 chain; M5, M7, M8 are parallel
