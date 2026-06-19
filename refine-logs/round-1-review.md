# Round 1 External Review (GPT-5.5, xhigh)

## Dimension Scores

### 1. Problem Fidelity — 8/10

The Problem Anchor is well-formulated and the method directly targets it. The 4 UAV-specific constraints (6 distortion types, SWaP, multi-task conditioning, real-time inference) are all addressed in the method design. No drift detected.

**Weakness**: The proposal defines two distinct contributions (database + model) but the Problem Anchor focuses primarily on the model's capability ("predict task-specific visual input adequacy"). A database construction (even if necessary infrastructure) is not itself a solution to the bottleneck. The proposal should make clearer whether the primary intended paper contribution is the database (benchmark paper) or the model (method paper) — the anchor currently implies the latter but the Contribution Focus section hedges toward both.

**Fix**: Either (A) reframe the Problem Anchor to make the database the primary contribution ("No dataset exists to train or evaluate UAV-specific IQA models") and the model as a baseline, or (B) narrow to the model contribution only, treating the database as infrastructure described in an appendix. Priority: IMPORTANT.

### 2. Method Specificity — 9/10

Exceptional level of detail. The 6 UAV distortion types have formal mathematical models (integral, Koschmieder, convolution with PSF, macroblock masks, SR pipeline, periodic modulation). The architecture diagram is precise — layer dimensions, tensor shapes (f_s ∈ R^256, f_f ∈ R^64, fused ∈ R^320), FiLM equations with γ/β shapes. The training recipe specifies loss functions with exact hyperparameters (λ_rank=0.3, λ_task=0.1), three-stage curriculum, and data split ratios.

**Minor concern**: The FAB processes "32×32 patches with stride 16" but the log-polar transform output grid size and the tiny CNN's exact channel progression aren't specified. An engineer would need to decide these. Also, the cross-attention gating mechanism `α = σ(W_g · [f_s, f_f])` is underspecified — is α a scalar or a 64-dim vector? From context (element-wise product with f_f ∈ R^64), it should be 64-dim, but W_g mapping from [256+64]=320 → 64 should be stated explicitly.

These are MINOR documentation issues only.

### 3. Contribution Quality — 6/10

This is the critical dimension. The proposal has **two parallel contributions** that dilute focus:

**(a) UAV-Embodied-IQA Database**: Genuinely novel. First UAV-specific IQA database. The 6 UAV distortion types with mathematical models are a real contribution. But database papers (even good ones like Embodied-IQA 2505.16815) rely on scale, thoroughness, and community adoption — the proposal's 1,500 reference images is ~22× smaller than Embodied-IQA's 36,900 pairs. The database alone may not be sufficient for a top venue without significantly larger scale or a more compelling cross-embodiment story.

**(b) UAV-QANet Model**: The Frequency-Aware Branch (FAB) is positioned as the main novelty, but frequency-domain IQA features have been used since at least BRISQUE (2012, NSS features in DCT domain) and more recently in deep IQA (e.g., DeepFIQA uses DCT, MANIQA uses Swin Transformer which implicitly captures frequency). The proposal does not cite ANY prior frequency-aware IQA work. The FAB may genuinely add value for UAV-specific distortions, but the claim that it's a novel mechanism is undermined by not positioning against prior frequency-domain IQA methods.

The TCRH with FiLM modulation is a standard technique (FiLM, Perez et al. 2018). Applying it to task-conditioned IQA is reasonable but incremental.

**Two-contribution sprawl**: The current proposal tries to be both a database paper AND a method paper. A top-venue reviewer will ask: "Is the main contribution the database or the model?" If the answer is "both," the paper will likely be rejected for lack of focus.

**Fix**: Choose ONE dominant contribution. If database: make the model a baseline (no novelty claim), expand database scale significantly (more reference images, more tasks). If model: treat the database as infrastructure, but the method must have stronger novelty — either (a) position FAB against prior frequency-domain IQA explicitly and show why UAV distortions require frequency features that prior frequency IQA methods don't capture, or (b) replace FAB with a stronger mechanism (e.g., learn a distortion-specific prompt for a frozen VLM). Priority: CRITICAL.

### 4. Frontier Leverage — 7/10

Good use of VLM/VLA as annotation oracles (Qwen2.5-VL, UAV-Track VLA, CognitiveDrone). This is a legitimate foundation-model-era approach — using strong pre-trained models as labeling infrastructure. Using MobileNetV4 as a pre-trained backbone is appropriate.

**Weakness**: The method architecture (MobileNetV4 + PANet + CBAM + MLP regression) is entirely CNN-based. In 2025-2026, the frontier for IQA has moved toward Vision Transformer (MANIQA, IQT) and VLM-based scoring (Q-Align, CLIP-IQA). The proposal does not justify why a CNN is preferred over a ViT/VLM for this problem. The SWaP constraint (<10M params, <10ms) is a valid justification, but the proposal should explicitly compare against lightweight ViT alternatives (e.g., MobileViT, EfficientViT) and explain why CNN is superior for frequency-domain distortions.

**Modernization opportunity**: Instead of hand-designing FFT features, consider using a frozen lightweight VLM (e.g., a small CLIP variant) to extract features and learn a lightweight task-conditioned adapter on top. This would be more frontier-native while potentially providing richer features. However, this might violate the <10M constraint. The proposal should at minimum justify the CNN choice against ViT alternatives. Priority: IMPORTANT.

### 5. Feasibility — 7/10

The compute estimate (1,550 GPU-hours on A100) is reasonable. Data sources are all open-source. The 14-16 month timeline is realistic for a PhD project but aggressive for a single paper.

**Concerns**:
1. **VLA annotation bottleneck**: 800 GPU-hours for VLA inference is significant. VLA models (UAV-Track VLA, CognitiveDrone) may not have publicly available inference code. The proposal assumes these models are accessible — this should be verified before committing to the pipeline.
2. **VLA label quality**: Embodied-IQA (2505.16815) showed VLA inter-model SRCC ≈ 0.25. Using VLA outputs as ground-truth labels when models disagree this strongly is methodologically questionable. The "use best VLA per task" mitigation helps but doesn't solve the fundamental problem: there is no ground truth for "what a UAV should do" without real flight experiments.
3. **Sim-to-real gap**: All 6 UAV-specific distortions are modeled mathematically — but real propeller vibration, real atmospheric scattering, and real packet loss may not match these models. The proposal includes MotionScape real data which helps, but the distortion injection is entirely synthetic.

**Fix**: Add a small-scale real UAV validation experiment (even 50-100 flights with a DJI Mini) to validate that synthetic distortions produce similar VLA behavior to real distorted UAV images. Without this, a reviewer can reject the entire database as "simulation-only." Priority: IMPORTANT.

### 6. Validation Focus — 8/10

Three clean claims with well-designed ablation experiments. The per-distortion ablation for FAB (showing FAB improves frequency-domain distortions specifically) is a particularly good design — it directly tests the mechanism's justification. No unnecessary experimental bloat.

**Missing**: No comparison against frequency-aware NR-IQA baselines (e.g., BRISQUE's DCT features, deep frequency-domain IQA methods). Without this, the claim that "FAB is necessary" could be challenged by "existing frequency IQA already does this." Priority: IMPORTANT for model contribution, MINOR for database contribution.

### 7. Venue Readiness — 6/10

**If positioned as a database paper**: CVPR/ICCV is reasonable IF the database scale is competitive (need >5,000 reference images, >250K annotated pairs) and the 6 UAV-specific distortions are rigorously validated against real UAV data. The current 1,500 reference images is too small for a top-venue database paper (Embodied-IQA has 36,900; EPD has 12,500).

**If positioned as a method paper**: ECCV/NeurIPS requires stronger novelty than FAB + TCRH. The method would need either (a) a novel theoretical insight about why UAV distortions require frequency-domain processing, (b) a stronger architectural contribution, or (c) compelling real-world UAV deployment results showing the model enables new capabilities.

**Fix**: Choose a single paper type and meet its bar. A focused database paper with expanded scale has higher chance at CVPR/ECCV than a method paper with incremental architecture. Priority: IMPORTANT.

---

## Overall Score: 7.3 / 10

Weighted calculation:
- Problem Fidelity: 8 × 0.15 = 1.20
- Method Specificity: 9 × 0.25 = 2.25
- Contribution Quality: 6 × 0.25 = 1.50
- Frontier Leverage: 7 × 0.15 = 1.05
- Feasibility: 7 × 0.10 = 0.70
- Validation Focus: 8 × 0.05 = 0.40
- Venue Readiness: 6 × 0.05 = 0.30
- **Total: 7.40** (rounded to 7.4)

---

## Simplification Opportunities

1. **Choose ONE contribution type**: Either (A) make it purely a database paper — remove the model novelty claims, treat UAV-QANet as a strong baseline, and expand database scale; or (B) make it purely a method paper — treat the database construction as infrastructure, and strengthen the model's novelty. Having both contributions in one paper dilutes focus and makes neither strong enough.

2. **Merge or remove the cross-attention gate**: The gating mechanism `α = σ(W_g · [f_s, f_f])` adds complexity. A simpler concatenation + MLP would likely perform similarly. Ablate this before including it.

3. **Reduce to 3 UAV distortion types**: 6 novel distortions with mathematical models is thorough but may be over-engineered. Pick the 3 most impactful (propeller vibration, atmospheric scattering, packet-loss blocks) and focus depth on those. The other 3 can be added later.

4. **Drop the FAB and replace with learned frequency features**: Instead of explicit FFT + log-polar + tiny CNN, use a learnable patch-wise DCT or a simple learnable high-pass filter bank as the frequency branch. This reduces implementation complexity while preserving the frequency-aware inductive bias. (Or: make this an ablation — test whether explicit FFT is better than learned filters.)

---

## Modernization Opportunities

1. **Compare against ViT-based lightweight IQA**: Before committing to a pure CNN architecture, benchmark MobileViT-S and EfficientViT-B0 as alternatives to MobileNetV4. If MobileNetV4 is genuinely better for frequency-domain distortions, this becomes a finding rather than an assumption.

2. **Consider VLM feature distillation**: Instead of training from scratch, distill a frozen Qwen2.5-VL's feature space into the lightweight model. The VLM already encodes semantic quality understanding; distilling this into a <10M model could improve generalization significantly with no additional annotation cost.

3. **Use a learned quality token instead of hand-designed FAB**: Add a small number (4-8) of learnable "frequency query tokens" that cross-attend to patch-level FFT features, similar to how DETR uses object queries. This is more modern and parameter-efficient than the tiny CNN approach.

---

## Drift Warning

**PARTIAL drift detected.** The Problem Anchor states the bottleneck is predicting visual input adequacy for UAV tasks under SWaP constraints. However, the proposal's dominant contribution becomes the database construction (annotating 180K image pairs with VLM/VLA labels), which is a prerequisite infrastructure activity — not the solution itself. A database enables future solutions but does not itself solve the bottleneck.

The anchor check: "Can a UAV use this work to decide whether its current visual input is adequate?" Only the model (UAV-QANet) answers this. The database alone cannot. If the paper's main contribution is the database, the Problem Anchor should be reframed as "No training/evaluation resource exists for UAV IQA" rather than "No method exists to predict visual input adequacy."

---

## Verdict: REVISE

The direction is promising. The UAV-specific distortion models are genuinely novel and well-specified. The assembly-first approach (reusing Embodied-IQA annotation pipeline + AirCopBench data + CARLA-Air simulation) is smart and practical. However, three issues must be resolved before this reaches READY:

1. **CRITICAL: Choose one contribution type** (database paper OR method paper, not both).
2. **IMPORTANT: Position FAB against prior frequency-domain IQA methods** if keeping the model contribution.
3. **IMPORTANT: Add real-UAV validation** (even small-scale) to ground the synthetic distortions.

If these are addressed, the proposal has a clear path to a focused, defensible paper.

---

## Action Items Summary

| Priority | Issue | Suggested Fix |
|----------|-------|---------------|
| CRITICAL | Contribution sprawl (database + model) | Choose one: make it a database paper with baseline model, OR a method paper with infrastructure appendix |
| IMPORTANT | No citation of prior frequency-domain IQA | Survey and cite frequency-aware IQA (BRISQUE, DeepFIQA, etc.); position FAB against them |
| IMPORTANT | No real-UAV validation | Add small-scale real drone experiment (50-100 frames) to validate synthetic distortion models |
| IMPORTANT | No ViT/architecture justification | Benchmark MobileViT/EfficientViT as backbone alternatives; justify CNN choice |
| IMPORTANT | VLA label reliability concerns | Add execution-layer validation on a held-out subset; discuss label noise handling |
| MINOR | FAB implementation details underspecified | Specify log-polar grid, CNN channels, cross-attention dimensions |
| MINOR | Cross-attention gate may be unnecessary | Ablate concatenation vs. gating |
