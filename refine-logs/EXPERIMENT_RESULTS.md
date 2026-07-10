# Initial Experiment Results

**Date**: 2026-06-19
**Plan**: refine-logs/EXPERIMENT_PLAN.md
**Status**: IN PROGRESS — experiments still running

## M0: Sanity — PASSED
- R002 (distortion implementation): DONE — 6 UAV + 27 generic distortions implemented; 10/10 tests pass
- R004 (overfit test): DONE — loss → 0.000099 < 0.001 threshold

## M1: Database Construction — PARTIALLY DONE
- R005 (distorted dataset): DONE — 494,352 pairs generated (36 distortions × 1 random level × full AirCopBench)
  - Train: 6,889 / Val: 861 / Test: 863
  - 36 distortion types, 4 task categories
  - Scores: synthetic
- R006-R008 (VLM/VLA/Execution annotation): TODO — requires external models

## M2: Baseline Benchmark — RUNNING
- R009-R012: Full benchmark running on CPU: 12 methods × 863 test images
- Output: outputs/benchmark_v2/ (pending results)

## M3: Main Model Training — RUNNING
| Run | System | Config | Status | GPU | Notes |
|-----|--------|--------|--------|-----|-------|
| R013 (seed 42,100,200) | UAV-IQANet task-conditioned | 50ep, batch=64, lr=3e-4 | RUNNING | GPU 1 | Epoch 1, SRCC=-0.086 (VLM curriculum stage) |
| R014 (3 seeds) | UAV-IQANet task-agnostic | --use_task_conditioning false | QUEUED | — | After R013 |

**Critical fix applied**: Pretrained backbone loading repaired.
- Root cause: `HF_ENDPOINT=https://hf-mirror.com` (broken mirror) caused timm to fall back to random init
- Fix: Override to `https://huggingface.co` in `scripts/run_m3_train.py`
- First attempt SRCC = -0.165 (random init) → now training with proper pretrained weights

## M4: Ablations — QUEUED
- Script prepared: `scripts/run_m4_ablations.sh` (R016-R024c)
- Waiting for M3 seed42 to validate approach

## Summary
- [2/33] must-run experiments completed
- [2/33] currently running (R013, R009-R012)
- Main result: PENDING
- Ready for /auto-review-loop: NO — waiting for experiment completion
