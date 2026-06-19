# Round 2 Refinement

## Problem Anchor
[Verbatim from Round 1 — preserved]

## Anchor Check
- **Original bottleneck**: No quality assessment framework exists for UAV embodied IQA.
- **Why the revised method still addresses it**: Unchanged from Round 1. The database fills the resource gap. The two clarifications below are implementation details that improve feasibility without changing the problem scope.
- **No drift**: These are pre-flight checks, not design changes.

## Simplicity Check
- **Dominant contribution**: Unchanged — the UAV-Embodied-IQA database.
- **No new components added**: The two clarifications below specify verification gates and task scoping, not new architectural pieces.

## Changes Made

### 1. VLA Accessibility Pre-Start Gate (MINOR)
- **Reviewer said**: "Add VLA accessibility gate. UAV-Track VLA and CognitiveDrone-R1 may not have public inference code."
- **Action**: Added an explicit **pre-start verification step** before Phase 2 begins:
  - **Week 0 gate**: Verify that at least 2 of 3 target VLAs (UAV-Track VLA, CognitiveDrone-R1, Qwen-VLA) have publicly available model weights and inference code.
  - **If ≥2 VLAs available**: Proceed with primary annotation plan (ensemble of available VLAs).
  - **If <2 VLAs available**: Activate backup plan: Fine-tune OpenVLA-7B on AirCopBench task demonstrations. AirCopBench paper already validated sim-to-real transfer for this setup (+19.64% real-world accuracy after fine-tuning). Use multi-seed fine-tuning (3 seeds) to create a 3-model ensemble for inter-annotator agreement analysis.
  - **Gate owner**: Researcher; verification takes 1-2 days (download models, run test inference on 10 AirCopBench images).
- **Reasoning**: This prevents the annotation pipeline from stalling mid-project due to model unavailability. The backup plan (OpenVLA + AirCopBench fine-tuning) is validated by prior work and produces comparable annotation quality.
- **Impact**: Feasibility risk reduced. No change to database design or contribution.

### 2. Execution Layer Scope Per Task (MINOR)
- **Reviewer said**: "Specify which tasks use SITL vs. VLA-only. SAR/delivery may be challenging for SITL."
- **Action**: Specified task-by-task execution layer scope:

| Task | Execution Layer | Rationale |
|------|----------------|-----------|
| Tracking | CARLA-Air SITL | Well-supported: CARLA-Air has built-in tracking scenarios, UAV-Track VLA protocol is SITL-compatible |
| Inspection | CARLA-Air SITL | Well-supported: waypoint navigation + object inspection can be scripted in CARLA-Air |
| Delivery | VLA-only (no SITL) | Challenging: package pickup/drop-off requires gripper simulation not available in CARLA-Air. Use VLA ensemble with increased weight (3 VLAs × 3 seeds = 9 annotators) |
| SAR (Search & Rescue) | VLA-only (no SITL) | Challenging: complex navigation + victim detection in dynamic environments exceeds CARLA-Air's scenario capabilities. Use VLA ensemble as above. |

- **Execution layer coverage**: SITL on tracking + inspection = ~60% of the 5% execution subset (~5,400 pairs). VLA-only on delivery + SAR = ~40% (~3,600 pairs, with enhanced ensemble).
- **Calibration**: Learn separate calibration mappings for SITL-validated tasks (direct execution score → quality label) and VLA-only tasks (VLA ensemble consensus → estimated quality label). Report SRCC between VLA consensus and SITL execution on tracking/inspection to validate VLA reliability.
- **Reasoning**: Being honest about SITL limitations is better than claiming all tasks use SITL and failing during execution. The split is principled: tasks with well-defined spatial trajectories (tracking, inspection) use SITL; tasks requiring complex physical interaction (delivery, SAR) use enhanced VLA ensembles.
- **Impact**: Feasibility improved; no change to contribution scope.

### 3. Statistical Power for 50-Flight Validation (MINOR)
- **Reviewer said**: "50-flight per-distortion correlation may have low statistical power."
- **Action**: Added a **sequential validation protocol**:
  - **Phase A (20 flights)**: Focus on 2 distortion types with strongest expected real-world signal (atmospheric scattering + vibration blur). 10 flights per type, varied altitudes/maneuvers.
    - Compute SRCC(real distortion, synthetic match) for each type.
    - If SRCC > 0.6 for both: proceed to Phase B with confidence.
    - If SRCC < 0.6 for either: adjust synthetic model parameters and re-test before expanding.
  - **Phase B (30 flights)**: Extend to remaining 4 distortion types. 5 flights per type (some types like packet-loss blocks may not occur naturally; use controlled induction where possible).
  - **Fallback**: If natural distortion capture is insufficient for some types, supplement with controlled distortion induction (e.g., attach neutral-density filter for haze simulation, induce rapid yaw for vibration blur).
- **Reasoning**: Sequential design maximizes statistical power by concentrating early flights on the highest-signal distortion types. If Phase A fails, we catch the problem early and adjust, rather than discovering at the end that all 50 flights have insufficient power.
- **Impact**: Validation design is more robust. No change to total flight count or contribution.

## Revised Sections (incremental)

### Insert into Database Construction section:

**Pre-Start Verification Gate (Week 0)**:
Before any annotation work begins, verify VLA model accessibility:
1. Download UAV-Track VLA, CognitiveDrone-R1, Qwen-VLA model weights and inference code
2. Run test inference on 10 AirCopBench images per model
3. If ≥2 models are operational: proceed with primary multi-VLA annotation
4. If <2 models are operational: switch to backup plan (fine-tune OpenVLA-7B × 3 seeds on AirCopBench task demos)
5. Document model versions, inference configurations, and any adaptation needed

### Insert into Annotation Pipeline → Execution Layer:

**Execution layer scope by task type**:

| Task Type | Execution Validation | Method |
|-----------|---------------------|--------|
| Tracking | Yes (SITL) | CARLA-Air tracking scenario, single-frame distortion injection at track-loss waypoint |
| Inspection | Yes (SITL) | CARLA-Air waypoint navigation + object inspection, distortion injection at inspection waypoint |
| Delivery | No (VLA ensemble) | 3 VLAs × 3 seeds = 9 annotators; enhanced ensemble aggregation with outlier rejection |
| SAR | No (VLA ensemble) | Same enhanced ensemble as delivery |

For SITL-validated tasks: execution score e ∈ [0,1] from binary success + trajectory smoothness.
For VLA-only tasks: estimated quality score from calibrated VLA ensemble (calibration learned from SITL tasks).

### Insert into Real-UAV Validation:

**Sequential validation protocol**:
- **Phase A (20 flights)**: Atmospheric scattering (10 flights) + Vibration blur (10 flights). Gate: SRCC > 0.6 per type.
- **Phase B (30 flights)**: Remaining 4 distortion types (5 flights each, supplemented by controlled induction if needed).
- **Fallback**: ND filter for haze, rapid yaw for vibration, manual frame dropping for packet loss.

## Remaining Weaknesses
- None at the proposal level. Remaining risks are execution-dependent (VLA model availability, SITL scenario fidelity, real-UAV weather conditions) and are managed by the gates and fallbacks specified above.
