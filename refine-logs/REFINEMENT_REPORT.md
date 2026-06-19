# Refinement Report

**Problem**: In low-altitude UAV embodied intelligence, how to assess visual input quality to reliably predict downstream task success?

**Initial Approach**: A 4-phase research plan — cross-validation (Phase 1) → database construction (Phase 2) → lightweight NR-IQA model (Phase 3) → SC³ closed loop (Phase 4). Core contribution: Phases 2+3 as an integrated "database + model" paper.

**Date**: 2026-06-18 — 2026-06-19

**Rounds**: 2 / 5

**Final Score**: 8.7 → ~9.0 (with Round 2 minor clarifications) / 10

**Final Verdict**: READY (approaching)

## Problem Anchor
[Verbatim from all rounds — preserved throughout]

## Output Files
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Round 0 initial proposal: `refine-logs/round-0-initial-proposal.md`
- Round 1 review (external): `refine-logs/round-1-review.md`
- Round 1 refinement: `refine-logs/round-1-refinement.md`
- Round 2 review (external): `refine-logs/round-2-review.md`
- Round 2 refinement: `refine-logs/round-2-refinement.md`

## Score Evolution

| Round | PF | MS | CQ | FL | Feas | VF | VR | Overall | Verdict |
|-------|----|----|----|----|------|----|----|---------|---------|
| 1     | 8  | 9  | 6  | 7  | 7    | 8  | 6  | 7.4     | REVISE  |
| 2     | 9  | 9  | 9  | 8  | 8    | 9  | 8  | 8.7     | REVISE (minor) |

*PF=Problem Fidelity, MS=Method Specificity, CQ=Contribution Quality, FL=Frontier Leverage, Feas=Feasibility, VF=Validation Focus, VR=Venue Readiness*

## Round-by-Round Review Record

| Round | Main Reviewer Concerns | What Was Changed | Result |
|-------|-------------------------|------------------|--------|
| 1 | **CRITICAL**: Two-contribution sprawl. **IMPORTANT**: Missing frequency IQA citations, no real-UAV validation, no ViT justification, VLA label reliability | Chose database-only focus. Added frequency IQA positioning. Added real-UAV validation. Added ViT baselines. Mandatory execution layer. | All CRITICAL and IMPORTANT resolved |
| 2 | **MINOR**: VLA accessibility, SITL task scope, statistical power | Added pre-start VLA gate + OpenVLA backup. Specified SITL per task. Sequential validation protocol. | All MINOR resolved |

## Final Proposal Snapshot
- **Paper type**: Database + benchmark paper (CVPR/ICCV/ECCV target)
- **Sole contribution**: UAV-Embodied-IQA database — first IQA resource for aerial embodied intelligence
- **Key differentiator**: 6 UAV-specific distortion types with physically-motivated mathematical models
- **Scale**: ~3,000 reference images, ~180,000 annotated pairs, 4 UAV task categories
- **Annotation**: VLM (cognitive) + VLA (decision) + CARLA-Air Execution, 3-stage curriculum
- **Baseline model**: UAV-IQANet, <5.5M params, frequency-aware + task-conditioned (no novelty claimed)
- **Validation**: 50-flight real-UAV sequential validation protocol
- **Timeline**: 11 months, ~1,700 GPU-hours

## Method Evolution Highlights
1. **Most important focusing move**: Split the original "database + model" dual contribution into a focused database-only paper. The model is now a strong baseline. The Phase 3 model paper is deferred to a separate submission.
2. **Most important mechanism upgrade**: Elevated the execution layer from optional to mandatory on a 5% stratified subset (~9,000 pairs), providing a ground-truth anchor for VLA pseudo-label calibration.
3. **Most important modernization**: Added ViT backbone baselines (MobileViT-S, EfficientViT-B0) and justified CNN architecture choice with explicit frequency-domain reasoning against prior frequency IQA (BRISQUE/DCT, DeepFIQA).

## Pushback / Drift Log

| Round | Reviewer Said | Author Response | Outcome |
|-------|---------------|-----------------|---------|
| 1 | "Reduce to 3 UAV distortion types" | REJECTED. The 6 types form a taxonomy covering mechanical/environmental/ego-motion/communication/computational/illumination failure modes. Reducing loses contribution. | Accepted by reviewer Round 2 |
| 1 | "Drop FAB, replace with learned frequency features" | PARTIALLY REJECTED. Kept explicit FFT + log-polar for rotation/scale invariance needed by 6DoF UAV motion. Downgraded novelty claim to "design choice." | Accepted by reviewer Round 2 |
| 1 | "Consider VLM feature distillation" | DEFERRED to Phase 3 model paper. Would add a contribution dimension to a database paper. | Accepted by reviewer Round 2 |
| 2 | "50-flight per-distortion correlation may have low statistical power" | ACCEPTED. Added sequential validation protocol: 20 flights → gate check → 30 flights. | Resolved |

## Remaining Weaknesses
1. VLA model availability (UAV-Track VLA, CognitiveDrone-R1) — managed by pre-start gate and OpenVLA backup plan
2. Database scale competitive but not dominant — the 6 UAV-specific distortions are the key differentiator
3. Real-UAV validation is small-scale (50 flights) — sufficient for validation, not for primary experiment
4. Execution-dependent risks: CARLA-Air SITL fidelity for tracking/inspection, weather conditions for real flights

## Next Steps
- **READY**: Proceed to `/experiment-plan` for a detailed claim-driven experiment roadmap
- Then: Execute Phase 1 (cross-validation study, WACV/ICRA workshop) to validate the assembly approach
- Then: Execute Phase 2 (UAV-Embodied-IQA database construction, CVPR/ICCV/ECCV target) following this refined proposal
