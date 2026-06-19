# Experiment Tracker

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|---------|----------|--------|-------|
| **M0: Sanity** |
| R001 | M0 | Data verification | AirCopBench + CARLA-Air + MotionScape + MDMT download | — | Download success, frame count, task label integrity | MUST | TODO | Pre-start gate: verify ≥2 VLA models available |
| R002 | M0 | Distortion implementation | 6 UAV distortion models (vibration, scattering, 6DoF, pkt-loss, SR, shadow) | 10 test images each | Visual plausibility, parameter range coverage | MUST | DONE | 6 UAV + 18 generic implemented in src/uav_iqa/distortion.py; 10/10 tests pass |
| R003 | M0 | Annotation pipeline check | VLM (Qwen2.5-VL-7B) + VLA (UAV-Track) + Execution (CARLA-Air) | 100-image subset | Output format validity, score range | MUST | TODO | Verify all 3 annotation levels produce valid scores |
| R004 | M0 | Overfit test | UAV-QANet (full) | 100 random images, 50 epochs | Training loss → 0 | MUST | DONE | min_loss=0.000099 < 0.001 → PASSED; model gradients flow correctly |
| **M1: Database Construction** |
| R005 | M1 | Distorted dataset generation | 24 distortion types × 5 levels × 3,000 images | Full reference set | Coverage (≥170K task-relevant pairs) | MUST | CODE | scripts/run_m1_inject.py + scripts/run_m1_manifest.py |
| R006 | M1 | VLM annotation | Qwen2.5-VL-7B, InternVL2-8B, LLaVA-NeXT-13B | Full 180K pairs | Cognitive score (per-model, per-image) | MUST | TODO | Needs VLM models + GPU servers |
| R006A | M1 | Pre-start VLA gate | Check UAV-Track VLA, CognitiveDrone-R1, Qwen-VLA weights | — | Public weight availability | MUST | TODO | Needs network + model access |
| R007 | M1 | VLA annotation | VLA ensemble (3 models), sliding-window protocol | Full 180K pairs | Decision score (per-model, per-image, per-task) | MUST | TODO | Needs VLA models + CARLA-Air |
| R008 | M1 | Execution annotation | CARLA-Air SITL (tracking + inspection) | 5% stratified subset (~9K pairs) | Execution success rate, trajectory metrics | MUST | TODO | Needs CARLA-Air deployment |
| **M2: Baseline Benchmark** |
| R009 | M2 | FR-IQA baseline | PSNR, SSIM, MS-SSIM, LPIPS (AlexNet), LPIPS (VGG), DISTS | Full 80/20 split | SRCC, PLCC, RMSE | MUST | CODE | scripts/run_m2_benchmark.py (graceful fallback if deps missing) |
| R010 | M2 | FR-IQA deep + NR handcrafted | AHIQ, TOPIQ, BRISQUE, NIQE, ILNIQE | Full 80/20 split | SRCC, PLCC, RMSE | MUST | CODE | scripts/run_m2_benchmark.py |
| R011 | M2 | NR-IQA deep | MANIQA, Q-Align, CLIP-IQA, LIQE | Full 80/20 split | SRCC, PLCC, RMSE | MUST | CODE | scripts/run_m2_benchmark.py |
| R012 | M2 | Frequency + embodied baselines | BRISQUE/DCT, DeepFIQA, MA-EIQA (zero-shot + fine-tuned) | Full 80/20 split | SRCC, PLCC, RMSE | MUST | CODE | scripts/run_m2_benchmark.py |
| **M3: Main Model Training** |
| R013 | M3 | UAV-QANet full training | UAV-QANet (task-conditioned, 3-stage curriculum) | 80/20 stratified, 3 seeds | SRCC, PLCC, RMSE (per-task, overall) | MUST | CODE | scripts/run_m3_train.py (dry-run verified: 10-sample 3-epoch passes) |
| R014 | M3 | UAV-QANet task-agnostic | UAV-QANet (w/o task embedding) | Same split, 3 seeds | SRCC, PLCC, RMSE | MUST | CODE | --no-task-cond flag |
| R015 | M3 | Main benchmark compilation | Merge R009-R014 results | — | All methods SRCC/PLCC/RMSE, significance tests | MUST | TODO | Only after R009-R014 complete |
| R015A | M3 | Hyperparameter sweep | UAV-QANet: lr ∈ {1e-4, 3e-4, 1e-3}, wd ∈ {0, 1e-5, 1e-4} | 80/20, 1 seed | Validation SRCC | NICE | CODE | Cli args support; TODO only if R013 SRCC < 0.55 |
| **M4: Ablation Studies** |
| R016 | M4 | W/o frequency branch (FAB) | UAV-QANet minus FAB (patch FFT + tiny CNN) | Same split, 3 seeds | SRCC, PLCC (drop from R013) | MUST | CODE | --no-fab flag |
| R017 | M4 | W/o task embedding | UAV-QANet minus task embedding concatenation | Same split, 3 seeds | SRCC, PLCC (drop from R013) | MUST | CODE | --no-task-cond flag |
| R018 | M4 | W/o CBAM attention | UAV-QANet minus CBAM spatial-channel attention | Same split, 3 seeds | SRCC, PLCC (drop from R013) | MUST | CODE | --no-cbam flag |
| R019 | M4 | MobileViT-S backbone | UAV-QANet with MobileViT-S backbone (param-matched) | Same split, 3 seeds | SRCC, PLCC (vs. R013) | MUST | CODE | --backbone mobilevit_s |
| R020 | M4 | EfficientViT-B0 backbone | UAV-QANet with EfficientViT-B0 backbone (param-matched) | Same split, 3 seeds | SRCC, PLCC (vs. R013) | MUST | CODE | --backbone efficientvit_b0 |
| R021 | M4 | Train on 18 generic only | UAV-QANet trained on 18 generic distortions only | Same split, 3 seeds | SRCC, PLCC (drop from R013) | MUST | CODE | --distortion-filter generic |
| R021b | M4 | Train on 6 UAV only | UAV-QANet trained on 6 UAV distortions only | Same split, 3 seeds | SRCC, PLCC (ratio to R013) | MUST | CODE | --distortion-filter uav_only |
| R022 | M4 | VLM-only curriculum | UAV-QANet trained with VLM labels only (50 epochs) | Same split, 3 seeds | SRCC, PLCC (vs. R013) | NICE | CODE | --annotator-stage VLM only (skip 3-stage) |
| R022b | M4 | VLA-only curriculum | UAV-QANet trained with VLA labels only (50 epochs) | Same split, 3 seeds | SRCC, PLCC (vs. R013) | NICE | CODE | --annotator-stage VLA only |
| R023 | M4 | W/o execution layer | UAV-QANet trained without SITL execution stage | Same split, 3 seeds | SRCC, PLCC (drop from R013) | NICE | CODE | Shorter 3-stage curriculum without exec stage |
| R024a | M4 | Cross-task zero-shot gen. | Train task A → test B/C/D (4×3 matrix, 3 seeds) | Per-task splits | SRCC matrix | MUST | CODE | --task <name> --val-task <name> |
| R024b | M4 | Leave-one-task-out | Train 3 tasks → test 4th (×4, 3 seeds each) | Per-config split | SRCC per held-out task | MUST | CODE | --leave-out <task> |
| R024c | M4 | Multi-task joint training | Train all 4 tasks jointly | 80/20 stratified, 3 seeds | Average SRCC | MUST | CODE | Default mode (no --task flag) |
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
| — | Phase 3 | Model novelty paper | UAV-QANet with architectural novelties (VLM distillation, etc.) | — | — | FUTURE | — | Separate paper; not in database paper scope |
| — | Phase 4 | SC³ closed-loop | IQA-driven UAV control adaptation | — | — | FUTURE | — | Deferred from original 4-phase roadmap |
| — | — | Cross-embodiment | Transfer to ground robot IQA | — | — | FUTURE | — | Reviewer suggestion, deferred |
