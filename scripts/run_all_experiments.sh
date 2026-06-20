#!/usr/bin/env bash
# =============================================================================
# UAV-Embodied-IQA: Full Experiment Deployment Script
# =============================================================================
# Runs all 21 experiment configs × 3 seeds = 63 training runs on 2 GPUs.
#
# Usage:
#   chmod +x scripts/run_all_experiments.sh
#   ./scripts/run_all_experiments.sh                # run all MUST-RUN
#   ./scripts/run_all_experiments.sh --all           # run MUST-RUN + NICE-TO-HAVE
#   ./scripts/run_all_experiments.sh --dry-run       # print jobs without running
#
# Prerequisites:
#   uv sync --extra swanlab          # SwanLab logger
#   export SWANLAB_API_KEY=...       # or create .env file
#
# Output:
#   outputs/<experiment>_seed<N>/    # per-run checkpoints + results.json
#   SwanLab project: uav-iqa         # cloud tracking
#   scripts/run_all_experiments.log  # job log
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LOG_FILE="$SCRIPT_DIR/run_all_experiments.log"
STATUS_FILE="$SCRIPT_DIR/run_all_experiments.status"

SEEDS=(42 100 200)
GPU_COUNT=2  # 2× RTX 4090

# ---------------------------------------------------------------------------
# Experiment definitions: "config_name:priority"
# priority: must=MUST-RUN, nice=NICE-TO-HAVE
# ---------------------------------------------------------------------------
declare -a EXPERIMENTS=(
    # M3: Main Model (MUST-RUN)
    "r013_task_cond:must"
    "r014_task_agnostic:must"
    # M4: Ablations (MUST-RUN)
    "r016_no_fab:must"
    "r017_no_task_cond:must"
    "r018_no_cbam:must"
    "r019_mobilevit_s:must"
    "r020_efficientvit_b0:must"
    "r021_generic_only:must"
    "r021b_uav_only:must"
    # M4: Cross-task single-task training (MUST-RUN)
    "r024a_tracking:must"
    "r024a_inspection:must"
    "r024a_delivery:must"
    "r024a_sar:must"
    # M4: Leave-one-task-out (MUST-RUN)
    "r024b_leave_tracking:must"
    "r024b_leave_inspection:must"
    "r024b_leave_delivery:must"
    "r024b_leave_sar:must"
    # M4: Multi-task (MUST-RUN)
    "r024c_multitask:must"
    # M4: Curriculum ablations (NICE-TO-HAVE)
    "r022_vlm_only:nice"
    "r022b_vla_only:nice"
    "r023_no_exec:nice"
)

RUN_ALL=false
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --all)      RUN_ALL=true ;;
        --dry-run)  DRY_RUN=true ;;
    esac
done

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

build_jobs() {
    # Build flat job list: "config_name seed priority output_dir"
    for entry in "${EXPERIMENTS[@]}"; do
        local cfg="${entry%%:*}"
        local pri="${entry##*:}"
        if [ "$pri" = "nice" ] && [ "$RUN_ALL" = false ]; then
            continue
        fi
        for seed in "${SEEDS[@]}"; do
            echo "$cfg $seed $pri outputs/${cfg}_seed${seed}"
        done
    done
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
main() {
    echo "" > "$LOG_FILE"

    local total_must=0 total_nice=0
    for entry in "${EXPERIMENTS[@]}"; do
        local pri="${entry##*:}"
        if [ "$pri" = "nice" ]; then
            total_nice=$((total_nice + 1))
        else
            total_must=$((total_must + 1))
        fi
    done

    local must_runs=$((total_must * ${#SEEDS[@]}))
    local nice_runs=$((total_nice * ${#SEEDS[@]}))

    echo "=============================================="
    echo "  UAV-Embodied-IQA: Full Experiment Runner"
    echo "=============================================="
    echo "  Config base dir: configs/experiments/"
    echo "  Seeds: ${SEEDS[*]}"
    echo "  GPUs: $GPU_COUNT"
    echo "  MUST-RUN: $total_must configs × ${#SEEDS[@]} seeds = $must_runs runs"
    echo "  NICE-TO-HAVE: $total_nice configs × ${#SEEDS[@]} seeds = $nice_runs runs"
    echo ""
    if [ "$RUN_ALL" = true ]; then
        echo "  Mode: --all (MUST-RUN + NICE-TO-HAVE)"
    else
        echo "  Mode: MUST-RUN only (add --all for NICE-TO-HAVE)"
    fi
    echo "  Log: $LOG_FILE"
    echo "=============================================="
    echo ""

    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN -- printing jobs without executing:"
        echo ""
        local i=1
        while IFS=' ' read -r cfg seed pri outdir; do
            echo "  [$i] $cfg seed=$seed [$pri] -> $outdir"
            i=$((i + 1))
        done < <(build_jobs)
        exit 0
    fi

    log "Starting full experiment deployment"

    # Verify SwanLab
    if ! uv run python -c "import swanlab" 2>/dev/null; then
        log "ERROR: SwanLab not installed. Run: uv sync --extra swanlab"
        exit 1
    fi
    if [ -z "${SWANLAB_API_KEY:-}" ]; then
        # Check .env file
        if [ -f .env ] && grep -q "SWANLAB_API_KEY" .env 2>/dev/null; then
            log "SwanLab API key found in .env file"
        else
            log "WARNING: SWANLAB_API_KEY not set. SwanLab cloud logging may fail."
        fi
    fi

    # Verify data
    if [ ! -d "data/processed/train" ]; then
        log "ERROR: data/processed/train not found. Run data synthesis first."
        exit 1
    fi

    # Build sorted job list
    local jobs_sorted
    jobs_sorted=$(build_jobs | sort -t' ' -k3,3 -k1,1)

    # Write all jobs to status file
    : > "$STATUS_FILE"
    local i=1
    while IFS=' ' read -r cfg seed pri outdir; do
        echo "$cfg seed=$seed pri=$pri outdir=$outdir status=pending" >> "$STATUS_FILE"
        i=$((i + 1))
    done < <(echo "$jobs_sorted")

    local total=$(wc -l < "$STATUS_FILE")
    log "Total jobs: $total"

    # Simple sequential dispatcher: one job per GPU, use lock file to claim jobs
    local LOCK_DIR="/tmp/run_all_exp_locks"
    mkdir -p "$LOCK_DIR"

    claim_job() {
        # Atomically claim the first pending job. Returns job info on success.
        local lockfile="$LOCK_DIR/dispatch.lock"
        (
            flock -x 200
            local line
            line=$(grep "status=pending" "$STATUS_FILE" | head -1)
            if [ -z "$line" ]; then
                echo ""
                return
            fi
            local cfg
            cfg=$(echo "$line" | awk '{print $1}')
            # Mark as running
            sed -i "s|^$cfg .*status=pending|$cfg status=running|" "$STATUS_FILE"
            echo "$line" | sed 's/status=pending/status=running/'
        ) 200>"$lockfile"
    }

    run_on_gpu() {
        local gpu_id=$1
        while true; do
            local job
            job=$(claim_job)
            if [ -z "$job" ]; then
                break
            fi
            local cfg seed outdir
            cfg=$(echo "$job" | awk '{print $1}')
            seed=$(echo "$job" | grep -oP 'seed=\K[0-9]+')
            outdir=$(echo "$job" | grep -oP 'outdir=\K\S+')

            log "START [GPU $gpu_id] $cfg seed=$seed"
            local start_ts
            start_ts=$(date +%s)

            if CUDA_VISIBLE_DEVICES="$gpu_id" uv run python main.py fit \
                --config "configs/experiments/${cfg}.yaml" \
                --seed_everything "$seed" \
                --trainer.default_root_dir "$outdir" \
                >> "$LOG_FILE" 2>&1; then
                local elapsed
                elapsed=$(($(date +%s) - start_ts))
                log "DONE  [GPU $gpu_id] $cfg seed=$seed (${elapsed}s)"
                sed -i "s|^$cfg .*status=running|$cfg status=done outdir=$outdir|" "$STATUS_FILE"
            else
                local elapsed
                elapsed=$(($(date +%s) - start_ts))
                log "FAIL  [GPU $gpu_id] $cfg seed=$seed (${elapsed}s)"
                sed -i "s|^$cfg .*status=running|$cfg status=failed outdir=$outdir|" "$STATUS_FILE"
            fi
        done
    }

    # Launch one dispatcher per GPU
    run_on_gpu 0 &
    local pid0=$!
    run_on_gpu 1 &
    local pid1=$!

    wait $pid0 $pid1 2>/dev/null || true

    # Final summary
    local done_count failed_count
    done_count=$(grep -c "status=done" "$STATUS_FILE" 2>/dev/null || echo 0)
    failed_count=$(grep -c "status=failed" "$STATUS_FILE" 2>/dev/null || echo 0)

    log "============================================"
    log "ALL JOBS COMPLETED"
    log "  Done: $done_count"
    log "  Failed: $failed_count"
    log "  Total: $total"
    log "============================================"

    if [ "$failed_count" -gt 0 ]; then
        log "Failed jobs:"
        grep "status=failed" "$STATUS_FILE" | while read -r line; do
            log "  $line"
        done
    fi
}

main "$@"
