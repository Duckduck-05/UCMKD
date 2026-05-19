#!/bin/bash
# =============================================================================
# RAVDESS Training Script
#
# Usage:
#   bash run.sh                              # single run (default)
#   RUN_MODE=all_metrics bash run.sh         # all cost metrics (parallel)
#   RUN_MODE=challenging bash run.sh         # challenging splits
#
# Required env vars:
#   DATA_ROOT   path to ravdess data directory
#
# Optional env vars (all have sensible defaults):
#   CKPT_DIR, GPU, STU_TYPE, IMAGE_ARCH, AUDIO_ARCH, METHOD, ...
# =============================================================================

set -euo pipefail

RUN_MODE=${RUN_MODE:-${1:-single}}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}
DATA_ROOT=${DATA_ROOT:?'ERROR: set DATA_ROOT to your ravdess data directory'}
CKPT_DIR=${CKPT_DIR:-"$WORKDIR/ckpts"}
PYTHON=${PYTHON:-python}

cd "$WORKDIR"
mkdir -p logs logs_challenging

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------
GPU=${GPU:-0}
STU_TYPE=${STU_TYPE:-0}              # 0 = image student, 1 = audio student
IMAGE_ARCH=${IMAGE_ARCH:-vit_l_16}
AUDIO_ARCH=${AUDIO_ARCH:-vit_s_16}

# Training schedule
LR=${LR:-1e-4}
MIN_LR=${MIN_LR:-1e-6}
NUM_EPOCHS=${NUM_EPOCHS:-100}
WARMUP_EPOCH=${WARMUP_EPOCH:-10}
OPTIMIZER=${OPTIMIZER:-adamw}
WEIGHT_DECAY=${WEIGHT_DECAY:-5e-2}

# Data
NUM_WORKERS=${NUM_WORKERS:-4}
NUM_FRAME=${NUM_FRAME:-1}
NUM_RUNS=${NUM_RUNS:-1}

# Method defaults (may be overridden by RUN_MODE)
DEFAULT_METHOD=ce
DEFAULT_BATCH_SIZE=64
[[ "$RUN_MODE" == "all_metrics" ]] && DEFAULT_BATCH_SIZE=32
[[ "$RUN_MODE" == "challenging" ]] && DEFAULT_METHOD=bilevel

METHOD=${METHOD:-$DEFAULT_METHOD}    # bilevel | ce | sumall | vanillaKD | feadistill | reviewkd | norm | cost_metric
BATCH_SIZE=${BATCH_SIZE:-$DEFAULT_BATCH_SIZE}
GROUP=${GROUP:-UCMKD}                        # experiment group name for logging/ckpt organization
PRE_TRAIN=${PRE_TRAIN:-1}
PRE_TRAIN_EPOCHS=${PRE_TRAIN_EPOCHS:-20}
METRIC=${METRIC:-cosine}             # l1 | l2 | cosine | chordal

# Loss / KD hyperparameters
LA_WEIGHT=${LA_WEIGHT:-1.0}
FA_WEIGHT=${FA_WEIGHT:-1.0}
NORM_N=${NORM_N:-4}
OT_REG=${OT_REG:-0.1}
OT_ITER=${OT_ITER:-100}
KD_TEMP=${KD_TEMP:-4.0}
KD_ALPHA=${KD_ALPHA:-0.9}
MAX_NORM=${MAX_NORM:-5.0}
N1_STEPS=${N1_STEPS:-1}
N2_STEPS=${N2_STEPS:-1}
INNER_WD=${INNER_WD:-0.05}
INNER_LR=${INNER_LR:-5e-5}

BASE_ARGS=(
    --stu-type        "$STU_TYPE"
    --image_arch      "$IMAGE_ARCH"
    --audio_arch      "$AUDIO_ARCH"
    --lr              "$LR"
    --batch-size      "$BATCH_SIZE"
    --num-workers     "$NUM_WORKERS"
    --num-epochs      "$NUM_EPOCHS"
    --warmup-epoch    "$WARMUP_EPOCH"
    --min-lr          "$MIN_LR"
    --optimizer       "$OPTIMIZER"
    --weight-decay    "$WEIGHT_DECAY"
    --num_frame       "$NUM_FRAME"
    --num-runs        "$NUM_RUNS"
    --cmkd            1
    --data-root       "$DATA_ROOT"
    --ckpt-dir        "$CKPT_DIR"
    --la-weight       "$LA_WEIGHT"
    --fa-weight       "$FA_WEIGHT"
    --ot-reg          "$OT_REG"
    --ot-iter         "$OT_ITER"
    --kd-temp         "$KD_TEMP"
    --kd-alpha        "$KD_ALPHA"
    --max-norm        "$MAX_NORM"
    --n1-steps        "$N1_STEPS"
    --n2-steps        "$N2_STEPS"
    --inner-wd        "$INNER_WD"
    --inner-lr        "$INNER_LR"
    --norm-n          "$NORM_N"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
print_header() {
    echo "=========================================="
    printf '  %s\n' "$1"
    echo "  Mode: $RUN_MODE"
    echo "  GPU: $GPU | Epochs: $NUM_EPOCHS | BS: $BATCH_SIZE"
    echo "  Method: $METHOD | Student: $STU_TYPE"
    echo "  Started at: $(date)"
    echo "=========================================="
}

main_cmd() {
    local gpu=$1; shift
    PYTHONUNBUFFERED=1 "$PYTHON" -u main_overlap_tag.py \
        --gpu "$gpu" \
        "${BASE_ARGS[@]}" \
        "$@"
}

run_with_log() {
    local logfile=$1; shift
    echo "  Log: $logfile"
    local status
    set +e
    "$@" 2>&1 | tee "$logfile"
    status=${PIPESTATUS[0]}
    set -e
    return "$status"
}

# ---------------------------------------------------------------------------
# run_single
# ---------------------------------------------------------------------------
run_single() {
    print_header "RAVDESS single experiment"
    main_cmd "$GPU" \
        --method_type      "$METHOD" \
        --group            "$GROUP" \
        --pre_train        "$PRE_TRAIN" \
        --pre-train-epochs "$PRE_TRAIN_EPOCHS" \
        --metric           "$METRIC"
}

# ---------------------------------------------------------------------------
# run_all_metrics  (parallel, per-metric GPU assignment)
# ---------------------------------------------------------------------------
metric_gpu() {
    case "$1" in
        l2)      echo "${METRIC_GPU_L2:-$GPU}"      ;;
        l1)      echo "${METRIC_GPU_L1:-$GPU}"      ;;
        chordal) echo "${METRIC_GPU_CHORDAL:-$GPU}" ;;
        cosine)  echo "${METRIC_GPU_COSINE:-$GPU}"  ;;
        *)       echo "$GPU"                        ;;
    esac
}

run_all_metrics() {
    local logdir=${ALL_METRICS_LOGDIR:-$WORKDIR/logs}
    mkdir -p "$logdir"

    print_header "RAVDESS all cost metrics"
    echo "  Metrics: l2, l1, chordal, cosine (parallel)"

    local pids=() names=()
    for metric in l2 l1 chordal cosine; do
        local gpu logfile
        gpu=$(metric_gpu "$metric")
        logfile="$logdir/cost_metric_${metric}.log"
        echo "  Launching metric=$metric on GPU $gpu"
        main_cmd "$gpu" \
            --pre_train   0 \
            --method_type cost_metric \
            --metric      "$metric" \
            --group       "${ALL_METRICS_GROUP:-ravdess_cost_metric_comparison}" \
            > "$logfile" 2>&1 &
        pids+=($!) names+=("$metric")
    done

    echo "  PIDs: ${pids[*]} — waiting..."

    while true; do
        local running=0
        for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && running=$((running+1)); done
        [ "$running" -eq 0 ] && break
        echo "[$(date +%H:%M:%S)] $running experiment(s) still running..."
        for i in "${!pids[@]}"; do
            if kill -0 "${pids[$i]}" 2>/dev/null; then
                local last
                last=$(grep -oP "Epoch \d+" "$logdir/cost_metric_${names[$i]}.log" 2>/dev/null | tail -1)
                echo "  ${names[$i]} (PID ${pids[$i]}): ${last:-starting...}"
            fi
        done
        sleep 60
    done

    local failed=0
    set +e
    for i in "${!pids[@]}"; do
        wait "${pids[$i]}"
        if [ $? -ne 0 ]; then
            echo "  ✗ metric=${names[$i]} FAILED"
            failed=$((failed+1))
        else
            echo "  ✓ metric=${names[$i]} completed"
        fi
    done
    set -e

    echo ""
    echo "=========================================="
    echo "  Metric summary"
    echo "=========================================="
    for metric in l2 l1 chordal cosine; do
        local logfile="$logdir/cost_metric_${metric}.log"
        echo ""
        echo "--- Metric: $metric (GPU $(metric_gpu "$metric")) ---"
        if [ -f "$logfile" ]; then
            grep -E "Training finish|Best.*Val|Best.*Test|Saving best" "$logfile" | tail -5 || true
        else
            echo "  Log file not found."
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "=== $failed experiment(s) FAILED at $(date) ===" && return 1
    fi
    echo "=== ALL METRIC EXPERIMENTS COMPLETED at $(date) ==="
}

# ---------------------------------------------------------------------------
# run_challenging
# ---------------------------------------------------------------------------
run_challenging_case() {
    local title=$1 group=$2 logfile=$3; shift 3
    echo ""
    echo "--- $title ---"
    run_with_log "$logfile" main_cmd "$GPU" \
        --method_type "$METHOD" \
        --pre_train   0 \
        --group       "$group" \
        "$@"
}

run_challenging() {
    local logdir=${CHALLENGING_LOGDIR:-$WORKDIR/logs_challenging}
    mkdir -p "$logdir"

    print_header "RAVDESS challenging dataloader"

    run_challenging_case \
        "[1/3] Clean baseline"   "ravdess_baseline_clean" \
        "$logdir/baseline_clean_${METHOD}.log"

    run_challenging_case \
        "[2/3] Preset: moderate" "ravdess_challenging_moderate" \
        "$logdir/preset_moderate_${METHOD}.log" \
        --challenge --challenge_preset moderate

    run_challenging_case \
        "[3/3] Preset: hard"     "ravdess_challenging_hard" \
        "$logdir/preset_hard_${METHOD}.log" \
        --challenge --challenge_preset hard

    echo ""
    echo "=== ALL CHALLENGING EXPERIMENTS COMPLETED at $(date) ==="
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$RUN_MODE" in
    single)      run_single      ;;
    all_metrics) run_all_metrics ;;
    challenging) run_challenging ;;
    *)
        echo "Unknown RUN_MODE: '$RUN_MODE'" >&2
        echo "Expected one of: single, all_metrics, challenging" >&2
        exit 2
        ;;
esac
