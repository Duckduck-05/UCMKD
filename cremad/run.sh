#!/bin/bash
# =============================================================================
# CREMA-D Training Script
#
# Usage:
#   bash run.sh
#
# Required env vars:
#   DATA_ROOT   path to CREMA-D data directory
#
# Optional env vars (all have sensible defaults):
#   CKPT_DIR, LOG_DIR, GPU, STU_TYPE, METHOD, ...
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}
DATA_ROOT=${DATA_ROOT:?'ERROR: set DATA_ROOT to your CREMA-D data directory'}
CKPT_DIR=${CKPT_DIR:-"$WORKDIR/ckpts"}
LOG_DIR=${LOG_DIR:-"$WORKDIR/logs"}
PYTHON=${PYTHON:-python}

cd "$WORKDIR"
mkdir -p logs

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------
GPU=${GPU:-0}
STU_TYPE=${STU_TYPE:-1}           # 0 = image student, 1 = audio student
METHOD=${METHOD:-reviewkd}        # bilevel | ce | sumall | vanillaKD | feadistill | reviewkd | norm
IMAGE_ARCH=${IMAGE_ARCH:-resnet18}
AUDIO_ARCH=${AUDIO_ARCH:-resnet18}
LR=${LR:-1e-2}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_EPOCHS=${NUM_EPOCHS:-100}
WARMUP_EPOCH=${WARMUP_EPOCH:-5}
MIN_LR=${MIN_LR:-1e-5}
OPTIMIZER=${OPTIMIZER:-sgd}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
NUM_FRAME=${NUM_FRAME:-1}
NUM_RUNS=${NUM_RUNS:-1}
GROUP=${GROUP:-UCMKD}

PRE_TRAIN=${PRE_TRAIN:-1}
PRE_TRAIN_EPOCHS=${PRE_TRAIN_EPOCHS:-100}

LA_WEIGHT=${LA_WEIGHT:-1.0}
FA_WEIGHT=${FA_WEIGHT:-1.0}
NORM_N=${NORM_N:-4}
OT_REG=${OT_REG:-0.1}
OT_ITER=${OT_ITER:-100}
METRIC=${METRIC:-cosine}
KD_TEMP=${KD_TEMP:-4.0}
KD_ALPHA=${KD_ALPHA:-0.9}
MAX_NORM=${MAX_NORM:-5.0}
N1_STEPS=${N1_STEPS:-1}
N2_STEPS=${N2_STEPS:-1}
INNER_WD=${INNER_WD:-0.05}
INNER_LR=${INNER_LR:-1e-5}

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
PYTHONUNBUFFERED=1 "$PYTHON" -u main_overlap_tag.py \
    --gpu              "$GPU" \
    --stu-type         "$STU_TYPE" \
    --method_type      "$METHOD" \
    --image_arch       "$IMAGE_ARCH" \
    --audio_arch       "$AUDIO_ARCH" \
    --lr               "$LR" \
    --batch-size       "$BATCH_SIZE" \
    --num-epochs       "$NUM_EPOCHS" \
    --warmup-epoch     "$WARMUP_EPOCH" \
    --min-lr           "$MIN_LR" \
    --optimizer        "$OPTIMIZER" \
    --weight-decay     "$WEIGHT_DECAY" \
    --num_frame        "$NUM_FRAME" \
    --num-runs         "$NUM_RUNS" \
    --group            "$GROUP" \
    --pre_train        "$PRE_TRAIN" \
    --pre-train-epochs "$PRE_TRAIN_EPOCHS" \
    --cmkd             1 \
    --data-root        "$DATA_ROOT" \
    --ckpt-dir         "$CKPT_DIR" \
    --log-dir          "$LOG_DIR" \
    --la-weight        "$LA_WEIGHT" \
    --fa-weight        "$FA_WEIGHT" \
    --ot-reg           "$OT_REG" \
    --ot-iter          "$OT_ITER" \
    --metric           "$METRIC" \
    --kd-temp          "$KD_TEMP" \
    --kd-alpha         "$KD_ALPHA" \
    --max-norm         "$MAX_NORM" \
    --n1-steps         "$N1_STEPS" \
    --n2-steps         "$N2_STEPS" \
    --inner-wd         "$INNER_WD" \
    --inner-lr         "$INNER_LR" \
    --norm-n           "$NORM_N"
