#!/usr/bin/env bash
# Train the LEGACY DA-Fusion build (Mask2Former-fork "CFS" late-fusion).
# With no extra args this reproduces the best RGB-D run:
#   DI_AGF_rgbd_none_NOW0.4_BS2_LR1e-05  ->  data/checkpoints/<exp_name>/
#
# Config overrides are passed as trailing "KEY VALUE" pairs (detectron2 style; no --opts flag).
# Examples:
#   bash legacy/scripts/train.sh                                 # best RGB-D, 1 GPU
#   NUM_GPUS=4 bash legacy/scripts/train.sh                      # 4 GPUs
#   bash legacy/scripts/train.sh INPUT.INPUT_TYPE rgb            # RGB-only ablation
#   bash legacy/scripts/train.sh SOLVER.MAX_ITER 20 SOLVER.IMS_PER_BATCH 1   # quick smoke test
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$REPO/legacy/.venv/bin/python"; [ -x "$PYTHON" ] || PYTHON="python"

NUM_GPUS="${NUM_GPUS:-1}"
CONFIG="${CONFIG:-configs/legacy_mask2former_swin_uoais.yaml}"
export WANDB_MODE="${WANDB_MODE:-offline}"   # avoid interactive wandb login; unset/override for online logging

cd "$REPO/legacy"
exec "$PYTHON" uoais_train_net.py --num-gpus "$NUM_GPUS" --config-file "$CONFIG" "$@"
