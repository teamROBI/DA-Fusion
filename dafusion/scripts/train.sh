#!/usr/bin/env bash
# Train DA-Fusion (the real DS/DC reimplementation) on UOAIS-Sim.
# No-arg default reproduces the canonical RGB-D run; overrides are trailing "KEY VALUE"
# pairs (detectron2 style). Outputs go to data/checkpoints/dafusion/<exp>/ (on /data1).
#
# Examples:
#   bash dafusion/scripts/train.sh                                  # RGB-D, 6 GPUs
#   NUM_GPUS=1 bash dafusion/scripts/train.sh SOLVER.MAX_ITER 20    # smoke test
#   bash dafusion/scripts/train.sh MODEL.DAFUSION.INPUT_TYPE rgb INPUT.INPUT_TYPE rgb   # RGB ablation
#   bash dafusion/scripts/train.sh --eval-after SOLVER.MAX_ITER 125000   # train, then sweep-eval all ckpts
#
# Recommended batch-16 run (90k images = 2 epochs; ~11 ckpts for the sweep) -- copy the block below:
: <<'USAGE'
NUM_GPUS=4 bash dafusion/scripts/train.sh --eval-after \
  SOLVER.IMS_PER_BATCH 20 \
  SOLVER.MAX_ITER 45000 \
  SOLVER.WARMUP_ITERS 1000 \
  SOLVER.CHECKPOINT_PERIOD 3000
USAGE
#
# Pass --eval-after (anywhere in the args) to auto-run the checkpoint sweep
# (dafusion/scripts/eval_sweep.sh) once training exits successfully. A crashed/non-zero
# training run skips the sweep. For --use_cgnet / --no-keep-final, run eval_sweep.sh directly.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DAFU="$REPO/dafusion"
PYTHON="$DAFU/.venv/bin/python"; [ -x "$PYTHON" ] || PYTHON="python"

NUM_GPUS="${NUM_GPUS:-6}"
CONFIG="${CONFIG:-configs/dafusion_rgbd_uoais.yaml}"
# W&B logging (rank-0): "online" uploads, "offline" logs locally, "disabled" turns it off.
# Set WANDB_PROJECT to change the project (default: da-fusion); run name = OUTPUT_DIR basename.
export WANDB_MODE="${WANDB_MODE:-online}"

# Pull out our own --eval-after flag; forward everything else as detectron2 opts. Also note
# an OUTPUT_DIR override so the chained sweep targets the right run dir.
EVAL_AFTER=0
OUTPUT_DIR_OVERRIDE=""
OPTS=()
args=("$@")
i=0
while [ "$i" -lt "${#args[@]}" ]; do
  a="${args[$i]}"
  if [ "$a" = "--eval-after" ]; then
    EVAL_AFTER=1; i=$((i + 1)); continue
  fi
  if [ "$a" = "OUTPUT_DIR" ] && [ "$((i + 1))" -lt "${#args[@]}" ]; then
    OUTPUT_DIR_OVERRIDE="${args[$((i + 1))]}"
    OPTS+=("$a" "${args[$((i + 1))]}"); i=$((i + 2)); continue
  fi
  OPTS+=("$a"); i=$((i + 1))
done

cd "$DAFU"
"$PYTHON" -m dafusion.train_net --num-gpus "$NUM_GPUS" --config-file "$CONFIG" \
  ${OPTS[@]+"${OPTS[@]}"}

# Reached only on a successful (exit 0) training run thanks to `set -e`.
if [ "$EVAL_AFTER" = "1" ]; then
  echo ">>> training finished; chaining checkpoint eval sweep"
  # eval is CPU-bound, so oversubscribe each GPU (override with SWEEP_WORKERS_PER_GPU).
  # SWEEP_SAVE_VIZ>0 saves that many good/medium/bad prediction panels per checkpoint/dataset.
  SWEEP_ARGS=(--config "$CONFIG" --workers-per-gpu "${SWEEP_WORKERS_PER_GPU:-4}")
  [ "${SWEEP_SAVE_VIZ:-0}" != "0" ] && SWEEP_ARGS+=(--save_viz "${SWEEP_SAVE_VIZ}")
  [ -n "$OUTPUT_DIR_OVERRIDE" ] && SWEEP_ARGS+=(--output-dir "$OUTPUT_DIR_OVERRIDE")
  bash "$DAFU/scripts/eval_sweep.sh" "${SWEEP_ARGS[@]}"
fi
