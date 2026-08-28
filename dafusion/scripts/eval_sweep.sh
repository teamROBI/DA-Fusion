#!/usr/bin/env bash
# Sweep every checkpoint of a DA-Fusion run: eval on OSD / OCID / OCBD across all GPUs,
# plot metrics vs training iteration, then prune the run to the winners (best per benchmark
# + best 3-way average Overlap-F). Outputs go to <OUTPUT_DIR>/sweep/.
#
# Examples:
#   bash dafusion/scripts/eval_sweep.sh --config configs/dafusion_rgbd_uoais.yaml
#   bash dafusion/scripts/eval_sweep.sh --config configs/dafusion_rgbd_uoais.yaml \
#       --output-dir ../data/checkpoints/dafusion/<run> --use_cgnet
#   bash dafusion/scripts/eval_sweep.sh --config configs/dafusion_rgbd_uoais.yaml --dry-run
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DAFU="$REPO/dafusion"
PYTHON="$DAFU/.venv/bin/python"; [ -x "$PYTHON" ] || PYTHON="python"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "$DAFU"
exec "$PYTHON" -m dafusion.eval.sweep "$@"
