#!/usr/bin/env bash
# M4 Ablation Experiments Runner
# Runs after M3 main training validates the approach.
# Usage: bash scripts/run_m4_ablations.sh

set -euo pipefail

export HF_ENDPOINT=https://huggingface.co
GPU=1
OUTDIR="outputs/ablations"
SEEDS="42 100 200"
EPOCHS=50
BS=64

echo "============================================"
echo "M4 Ablation Experiments"
echo "Output: $OUTDIR | Seeds: $SEEDS | GPU: $GPU"
echo "============================================"

run_ablation() {
    local name="$1"
    shift
    echo ""
    echo "===== $name ====="
    HF_ENDPOINT=$HF_ENDPOINT uv run python scripts/run_m3_train.py \
        --seeds $SEEDS \
        --output_dir "$OUTDIR/$name" \
        --trainer.accelerator gpu \
        --trainer.devices "[$GPU]" \
        --trainer.max_epochs $EPOCHS \
        --data.batch_size $BS \
        --data.num_workers 4 \
        "$@" 2>&1 | tee "/tmp/m4_${name}.log"
    echo "===== $name DONE ====="
}

# R016: Without Frequency-Aware Branch
run_ablation "R016_no_fab" \
    --model.init_args.use_fab false

# R017: Without Task Conditioning
run_ablation "R017_no_task_cond" \
    --model.init_args.use_task_conditioning false

# R018: Without CBAM Attention
run_ablation "R018_no_cbam" \
    --model.init_args.use_cbam false

# R021: Train on 18 generic distortions only
run_ablation "R021_generic_only" \
    --data.init_args.distortion_filter generic

# R021b: Train on 6 UAV distortions only
run_ablation "R021b_uav_only" \
    --data.init_args.distortion_filter uav_only

# R024c: Multi-task joint training
run_ablation "R024c_multitask" \
    --data.init_args.task null

# R024a: Cross-task zero-shot (4 tasks)
for task in tracking inspection delivery sar; do
    run_ablation "R024a_${task}" \
        --seeds 42 \
        --data.init_args.task "$task"
done

# R024b: Leave-one-task-out (4 tasks)
for leave_out in tracking inspection delivery sar; do
    run_ablation "R024b_leave_${leave_out}" \
        --seeds 42 \
        --data.init_args.leave_out_task "$leave_out"
done

echo ""
echo "All M4 ablations complete. Results in $OUTDIR/"
