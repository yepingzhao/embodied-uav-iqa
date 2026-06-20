# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| **M0: Sanity** |
| R001 | M0 | Data verification | AirCopBench + CARLA-Air + MotionScape + MDMT download | — | Download success, frame count, task label integrity | MUST | TODO | Pre-start gate: verify ≥2 VLA models available |
| R002 | M0 | Distortion implementation | 6 UAV distortion models (vibration, scattering, 6DoF, pkt-loss, SR, shadow) | 10 test images each | Visual plausibility, parameter range coverage | MUST | DONE | 6 UAV + 18 generic implemented in src/uav_iqa/distortion.py; 10/10 tests pass |
| R003 | M0 | Annotation pipeline check | VLM/VLA/Execution annotation | 100-image subset | Score range validity | MUST | TODO | Requires VLM/VLA models; synthetic scores used for pipeline testing |
| R004 | M0 | Overfit test | UAV-IQANet (full) | 100 random images, 50 epochs | Training loss → 0 | MUST | DONE | min_loss=0.000099 < 0.001 → PASSED |
| **M1: Database Construction** |
| R005 | M1 | Distorted dataset generation | 33 distortion types × 5 levels × 261 refs | Full reference set (43,032 pairs) | Coverage | MUST | DONE | 34,425/4,303/4,304 train/val/test; manifests with scores |
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
| **Deferred / Future Work** |
| — | Phase 3 | Model novelty paper | UAV-IQANet with architectural novelties (VLM distillation, etc.) | — | — | FUTURE | — | Separate paper; not in database paper scope |
| — | Phase 4 | SC³ closed-loop | IQA-driven UAV control adaptation | — | — | FUTURE | — | Deferred from original 4-phase roadmap |
| — | — | Cross-embodiment | Transfer to ground robot IQA | — | — | FUTURE | — | Reviewer suggestion, deferred |
