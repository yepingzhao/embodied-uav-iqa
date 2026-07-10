# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| **M0: Sanity** |
| R001 | M0 | Data verification | AirCopBench + CARLA-Air + MotionScape + MDMT download | — | Download success, frame count, task label integrity | MUST | TODO | Pre-start gate: verify ≥2 VLA models available |
| R002 | M0 | Distortion implementation | 6 UAV distortion models (vibration, scattering, 6DoF, pkt-loss, SR, shadow) | 10 test images each | Visual plausibility, parameter range coverage | MUST | DONE | 6 UAV + 18 generic implemented in src/uav_iqa/distortion.py; 10/10 tests pass |
| R003 | M0 | Annotation pipeline check | VLM/VLA/Execution annotation | 100-image subset | Score range validity | MUST | TODO | Requires VLM/VLA models; synthetic scores used for pipeline testing |
| R004 | M0 | Overfit test | UAV-IQANet (full) | 100 random images, 50 epochs | Training loss → 0 | MUST | DONE | min_loss=0.000099 < 0.001 → PASSED |
| **M1: Database Construction** |
| R005 | M1 | Distorted dataset generation | 36 distortion types × 1 random level × full AirCopBench | Full reference set (494,352 pairs) | Coverage | MUST | DONE | 6 UAV + 30 generic; VLM scoring complete (R006) |
| R006 | M1 | VLM annotation | Qwen2.5-VL-7B, InternVL2-8B, LLaVA-NeXT-13B | — | Cognitive score | MUST | TODO | Requires VLM models + GPU servers |
| R007 | M1 | VLA annotation | VLA ensemble (3 models) | — | Decision score | MUST | TODO | Requires VLA models + CARLA-Air |
| R008 | M1 | Execution annotation | CARLA-Air SITL | — | Execution success rate | MUST | TODO | Requires CARLA-Air deployment |
| **M2: Baseline Benchmark** |
| R009-012 | M2 | All baseline methods | PSNR, SSIM, BRISQUE, NIQE, CLIP-IQA, etc. | 4,304 test | SRCC, PLCC, RMSE | MUST | RUNNING | CPU benchmark; 12 methods x 4304 imgs; pyiqa downloads may need HF_ENDPOINT fix |
| **M3: Main Model Training** |
| R013 | M3 | UAV-IQANet full training (seed 42,100,200) | UAV-IQANet (task-conditioned, 3-stage) | 34,425/4,303/4,304 | SRCC, PLCC, RMSE | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r013_task_cond_seed<N>; SwanLab project=uav-iqa |
| R014 | M3 | UAV-IQANet task-agnostic (3 seeds) | UAV-IQANet (w/o task embedding) | Same split, 3 seeds | SRCC, PLCC, RMSE | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r014_task_agnostic_seed<N> |
| R015 | M3 | Main benchmark compilation | Merge R009-R014 results | — | All methods SRCC/PLCC/RMSE, significance tests | MUST | TODO | Only after R009-R014 complete |
| R015A | M3 | Hyperparameter sweep | UAV-IQANet: lr ∈ {1e-4, 3e-4, 1e-3}, wd ∈ {0, 1e-5, 1e-4} | 80/20, 1 seed | Validation SRCC | NICE | TODO | Only if R013 SRCC < 0.55 |
| **M4: Ablation Studies** |
| R016 | M4 | W/o frequency branch (FAB) | UAV-IQANet minus FAB (patch FFT + tiny CNN) | Same split, 3 seeds | SRCC, PLCC (drop from R013) | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r016_no_fab_seed<N> |
| R017 | M4 | W/o task embedding | UAV-IQANet minus task embedding concatenation | Same split, 3 seeds | SRCC, PLCC (drop from R013) | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r017_no_task_cond_seed<N> |
| R018 | M4 | W/o CBAM attention | UAV-IQANet minus CBAM spatial-channel attention | Same split, 3 seeds | SRCC, PLCC (drop from R013) | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r018_no_cbam_seed<N> |
| R019 | M4 | MobileViT-S backbone | UAV-IQANet with MobileViT-S backbone (param-matched) | Same split, 3 seeds | SRCC, PLCC (vs. R013) | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r019_mobilevit_s_seed<N> |
| R020 | M4 | EfficientViT-B0 backbone | UAV-IQANet with EfficientViT-B0 backbone (param-matched) | Same split, 3 seeds | SRCC, PLCC (vs. R013) | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r020_efficientvit_b0_seed<N> |
| R021 | M4 | Train on 18 generic only | UAV-IQANet trained on 18 generic distortions only | Same split, 3 seeds | SRCC, PLCC (drop from R013) | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r021_generic_only_seed<N> |
| R021b | M4 | Train on 6 UAV only | UAV-IQANet trained on 6 UAV distortions only | Same split, 3 seeds | SRCC, PLCC (ratio to R013) | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r021b_uav_only_seed<N> |
| R022 | M4 | VLM-only curriculum | UAV-IQANet trained with VLM labels only (50 epochs) | Same split, 3 seeds | SRCC, PLCC (vs. R013) | NICE | DEPLOYED | --all mode; outputs/r022_vlm_only_seed<N> |
| R022b | M4 | VLA-only curriculum | UAV-IQANet trained with VLA labels only (50 epochs) | Same split, 3 seeds | SRCC, PLCC (vs. R013) | NICE | DEPLOYED | --all mode; outputs/r022b_vla_only_seed<N> |
| R023 | M4 | W/o execution layer | UAV-IQANet trained without SITL execution stage | Same split, 3 seeds | SRCC, PLCC (drop from R013) | NICE | DEPLOYED | --all mode; outputs/r023_no_exec_seed<N> |
| R024a | M4 | Cross-task zero-shot gen. | Train task A → test B/C/D (4×3 matrix, 3 seeds) | Per-task splits | SRCC matrix | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r024a_<task>_seed<N> |
| R024b | M4 | Leave-one-task-out | Train 3 tasks → test 4th (×4, 3 seeds each) | Per-config split | SRCC per held-out task | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r024b_leave_<task>_seed<N> |
| R024c | M4 | Multi-task joint training | Train all 4 tasks jointly | 80/20 stratified, 3 seeds | Average SRCC | MUST | DEPLOYED | Full rerun via run_all_experiments.sh; outputs/r024c_multitask_seed<N> |
| **M5: Real-UAV Validation (parallel)** |
| R025 | M5 | Gate: 20 flights | Scattering (10 flights) + vibration (10 flights) | Real vs. synthetic pairs | SRCC (synthetic vs. real ΔVLA) | MUST | TODO | Gate criterion: SRCC > 0.5 to proceed |
| R026 | M5 | Full: 30 flights | 6DoF blur, packet loss, SR artifacts, propeller shadow | Real vs. synthetic pairs | SRCC per distortion, overall SRCC | MUST | TODO | Only if R025 gate passes |
| R027 | M5 | Synthetic replication | Generate matched synthetic frames for all 50 flights | Per-flight parameter estimation | Frame matching accuracy | MUST | TODO | Use telemetry to estimate distortion parameters |
| R028 | M5 | VLA evaluation on real | VLA ensemble evaluation on real + synthetic frames | All flight frames | ΔVLA per frame, per distortion | MUST | TODO | Offline evaluation after flights |
| **M6: Qualitative + Polish** |
| R029 | M6 | Per-distortion SRCC breakdown | All methods from B3 | Test set (from B3) | SRCC per distortion type × method | MUST | TODO | For Figure 7 bar chart |
| R030 | M6 | Frequency signature visualization | 6 UAV + 2 nearest generic distortions | 100 random images each | Log-polar FFT magnitude maps | MUST | TODO | For Figure 6 frequency panel |
| R031 | M6 | Hard case gallery | Select top 15 worst-prediction images from B3 | Test set | Qualitative patterns | MUST | TODO | Manual selection with quantitative criterion |
| R032 | M6 | VLA ensemble agreement | ICC(3,k) per task, per distortion | Full VLA annotation set | ICC, confidence intervals | NICE | TODO | For appendix or supplementary |
| R033 | M6 | Distortion intensity robustness | All methods × 5 intensity levels | Test set | Per-level SRCC curves | NICE | TODO | For appendix robustness analysis |
| **M7: Cross-Database & VLA Calibration (CDVC)** |
| R036a | M7 | Cross-database: MA-EIQA on UAV | MA-EIQA (EPD-trained) zero-shot + fine-tuned on UAV data | UAV test split | SRCC, PLCC vs. in-domain | MUST | TODO | B6b CDBV; tests AC6 — can ground-robot IQA transfer to UAV? |
| R036b | M7 | Cross-database: UAV-IQANet on Embodied-IQA/EPD | UAV-IQANet (UAV-trained) zero-shot on Embodied-IQA + EPD test splits | Embodied-IQA (~7.4K) + EPD (~2.5K) | SRCC, PLCC drop vs. in-domain | MUST | TODO | B6c CDBV; tests AC6 reverse direction on both databases |
| R036c | M7 | Cross-database significance tests | Statistical comparison of cross-database vs. in-domain SRCC | R036a/R036b results | Wilcoxon signed-rank, Holm-Bonferroni p-values | MUST | TODO | B6c CDBV; formal significance testing for domain specificity |
| R037a | M7 | VLA ensemble ICC computation | ICC(3,k) per task per distortion for full VLA annotation set | Full VLA annotation set | ICC, 95% CI per task×distortion cell | MUST | TODO | B6d VLAC; quantifies annotation reliability (Embodied-IQA found SRCC≈0.25) |
| R037b | M7 | VLA calibration weight learning | Per-model confidence weights via SITL execution score regression | 5% SITL subset | Weighted ensemble SRCC vs. simple mean SRCC | MUST | TODO | B6d VLAC; calibrates VLA ensemble using execution ground truth |
| R037c | M7 | Calibrated vs. uncalibrated ensemble | Compare single-best-VLA, simple mean, weighted mean labels | Full VLA set | SRCC(ensemble label, execution score) | MUST | TODO | B6d VLAC; validates calibration benefit |
| **M8: Human MOS (Moravec Paradox) — Async Parallel** |
| R038 | M8 | Human MOS collection | 15+ subjects, single-stimulus continuous quality scale | 5% subset (~9K pairs) | MOS mean, std per image | NICE | TODO | B6b MVRX; crowdsourced; validates C5 Moravec paradox for UAV |
| R039 | M8 | Moravec paradox quantification | Human MOS vs. VLM quality vs. VLA decision vs. Execution | R038 subset | SRCC/PLCC all pairwise; per-task breakdown | NICE | TODO | B6b MVRX; success: SRCC(human,VLA)<0.3, SRCC(VLA,Exec)≥0.5 |
| **M9: Compound Distortion (CPDA)** |
| R040a | M9 | Compound distortion synthesis | 10 compound pairs × 3 intensity levels × 600 refs | 20% stratified subset | Coverage, visual plausibility | NICE | TODO | B7 CPDA; generates ~18K compound-distorted pairs |
| R040b | M9 | Compound VLA evaluation | VLA ensemble on compound-distorted frames | 5% compound subset (900 pairs) | ΔVLA, non-linearity ratio | NICE | TODO | B7 CPDA; tests whether compound effect > additive |
| R040c | M9 | Compound IQA prediction | Existing IQA methods predict compound ΔVLA | 900 compound pairs | SRCC(predicted, actual compound ΔVLA) | NICE | TODO | B7 CPDA; tests whether single-distortion IQA generalizes to compounds |
| **Deferred / Future Work** |
| — | Phase 3 | Model novelty paper | UAV-IQANet with architectural novelties (VLM distillation, etc.) | — | — | FUTURE | — | Separate paper; not in database paper scope |
| — | Phase 4 | SC³ closed-loop | IQA-driven UAV control adaptation | — | — | FUTURE | — | Deferred from original 4-phase roadmap |
| — | — | Cross-embodiment | Transfer to ground robot IQA | — | — | FUTURE | — | Reviewer suggestion, deferred |
