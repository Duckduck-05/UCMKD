#!/bin/bash
# =============================================================================
# AVE Training Script
#
# Usage:
#   bash run.sh
#
# Required env vars:
#   DATA_ROOT   path to AVE dataset directory
# =============================================================================

set -euo pipefail

WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}
DATA_ROOT=${DATA_ROOT:?'ERROR: set DATA_ROOT to your AVE dataset directory'}
CKPT_DIR=${CKPT_DIR:-"$WORKDIR/ckpts"}
PYTHON=${PYTHON:-python}

cd "$WORKDIR"
mkdir -p logs

GPU=${GPU:-0}
STU_TYPE=${STU_TYPE:-0}
METHOD=${METHOD:-ce}
IMAGE_ARCH=${IMAGE_ARCH:-vit_s_16}
AUDIO_ARCH=${AUDIO_ARCH:-vit_l_16}
LR=${LR:-1e-4}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_EPOCHS=${NUM_EPOCHS:-30}
WARMUP_EPOCH=${WARMUP_EPOCH:-5}
MIN_LR=${MIN_LR:-1e-5}
OPTIMIZER=${OPTIMIZER:-adamw}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-5}
NUM_FRAME=${NUM_FRAME:-1}
NUM_RUNS=${NUM_RUNS:-1}
GROUP=${GROUP:-UCMKD}
PRE_TRAIN=${PRE_TRAIN:-1}
PRE_TRAIN_EPOCHS=${PRE_TRAIN_EPOCHS:-20}
LA_WEIGHT=${LA_WEIGHT:-1.0}
FA_WEIGHT=${FA_WEIGHT:-1.0}
NORM_N=${NORM_N:-4}
OT_REG=${OT_REG:-0.1}
OT_ITER=${OT_ITER:-100}
METRIC=${METRIC:-l2}
KD_TEMP=${KD_TEMP:-4.0}
KD_ALPHA=${KD_ALPHA:-0.9}
MAX_NORM=${MAX_NORM:-5.0}
N1_STEPS=${N1_STEPS:-1}
N2_STEPS=${N2_STEPS:-1}
INNER_WD=${INNER_WD:-0.05}
INNER_LR=${INNER_LR:-1e-5}

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
