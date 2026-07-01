# Loop Plan: Pipeline Verification & Validation

**Created:** 2026-06-29
**Pattern:** sequential
**Mode:** safe (strict gates)

## Stop Conditions

1. All phases complete with verified outputs
2. OR any phase fails irrecoverably (stop with error report)
3. OR user interrupts

## Prerequisites (confirmed)

- [x] Git: clean, branch `feat/dev`
- [x] Tests: 47 pass (data_synthesis, text_metrics, distortion)
- [x] Lint: ruff clean (`src/`, `scripts/`, `tests/`)

## Loop Phases

### Phase 1: Pipeline Smoke Test (extract + inject dry-run)
- Goal: Verify the new grouped-JSON pipeline works end-to-end with real data
- Steps:
  1. `python scripts/data_synthesis.py extract --dataset aircopbench --input-root data/raw/AirCopBench --output-dir /tmp/uav_processed --max-refs "Sim_3_UAVs=2,Sim_5_UAVs=2" --copy --seed 42`
  2. `python scripts/data_synthesis.py inject --dataset aircopbench --input-root data/raw/AirCopBench --output-dir /tmp/uav_processed --dry-run`
  3. Verify output: `data/processed/{train,test}/` has grouped JSONs, `ref_images/` has images, `distorted/` has distortions
  4. Verify JSON structure: correct fields (dataset, sequence_frame, uav_paths, distortions, vqa_entries)
- Verify: All steps complete without error, JSON schema valid
- Gate: Must pass before Phase 2

### Phase 2: Full Inject (one dataset subset)
- Goal: Run full inject on a small subset (e.g., Sim5 only) to validate at scale
- Steps:
  1. `python scripts/data_synthesis.py extract --dataset aircopbench --input-root data/raw/AirCopBench --output-dir data/processed --copy --seed 42`
  2. Modify/extract to only process Sim5 files
  3. Run inject without dry-run: `python scripts/data_synthesis.py inject --dataset aircopbench --input-root data/raw/AirCopBench --output-dir data/processed --workers 8`
- Verify: All groups processed, no failed distortions, distorted images loadable
- Gate: Must pass before Phase 3

### Phase 3: Dataset Loader Verification
- Goal: Verify the new `UAVIQADataset` loads data correctly from grouped JSONs
- Steps:
  1. Write a verification script that loads the dataset
  2. Check image tensors shape, subtask_id range, cognitive_score format
  3. Verify collate_fn handles variable UAV counts
- Verify: Dataset loads without error, correct shapes
- Gate: Must pass before Phase 4

### Phase 4: Training Smoke Test
- Goal: Verify model can train with new data format (overfit sanity check)
- Steps:
  1. `python scripts/overfit_sanity_check.py` (update if needed)
  2. Check loss decreases on 100-image subset
- Verify: Loss decreases over several iterations
- Gate: Final checkpoint

## Quality Gates (per phase)

- Phase output files exist and are valid
- Tests pass (at minimum: test_data_synthesis.py)
- No new lint errors
- No regression in pass rate

## Error Recovery

- If extract fails: check input data structure, path permissions
- If inject fails: check distortion dependencies (cv2, PIL), disk space
- If dataset fails: check JSON schema, image path resolution
- If training fails: check model config, device availability

## Commands

```bash
# Start Phase 1
python scripts/data_synthesis.py extract --dataset aircopbench --input-root data/raw/AirCopBench --output-dir /tmp/uav_processed --max-refs "Sim_3_UAVs=2,Sim_5_UAVs=2" --copy --seed 42

# Monitor
# Check /tmp/uav_processed/ for output files
```
