#!/bin/bash
# =============================================================================
# VGGSound Training Script
#
# Usage:
#   bash run.sh
#
# Required env vars:
#   DATA_ROOT   path to VGGSound dataset directory
# =============================================================================

set -euo pipefail

WORKDIR=${WORKDIR:-$(cd "$(dirname "$0")" && pwd)}
DATA_ROOT=${DATA_ROOT:?'ERROR: set DATA_ROOT to your VGGSound dataset directory'}
CKPT_DIR=${CKPT_DIR:-"$WORKDIR/ckpts"}
PYTHON=${PYTHON:-python}

cd "$WORKDIR"
mkdir -p logs

GPU=${GPU:-0}
STU_TYPE=${STU_TYPE:-0}           # 0 = image student, 1 = audio student
IMAGE_ARCH=${IMAGE_ARCH:-resnet18}
AUDIO_ARCH=${AUDIO_ARCH:-resnet18}
LR=${LR:-1e-2}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_EPOCHS=${NUM_EPOCHS:-100}
NUM_FRAME=${NUM_FRAME:-3}
NUM_RUNS=${NUM_RUNS:-1}
GROUP=${GROUP:-UCMKD}
PRE_TRAIN=${PRE_TRAIN:-1}

PYTHONUNBUFFERED=1 "$PYTHON" -u main_overlap_tag.py \
    --gpu          "$GPU" \
    --stu-type     "$STU_TYPE" \
    --image_arch   "$IMAGE_ARCH" \
    --audio_arch   "$AUDIO_ARCH" \
    --lr           "$LR" \
    --batch-size   "$BATCH_SIZE" \
    --num-epochs   "$NUM_EPOCHS" \
    --num_frame    "$NUM_FRAME" \
    --num-runs     "$NUM_RUNS" \
    --group        "$GROUP" \
    --pre_train    "$PRE_TRAIN" \
    --cmkd         1 \
    --data-root    "$DATA_ROOT" \
    --ckpt-dir     "$CKPT_DIR"
