#!/usr/bin/env bash
# Evaluate the LEGACY DA-Fusion build on the UOIS benchmarks (OSD / OCID / BOSD).
# With no extra args this evaluates the best RGB-D checkpoint on OCID
#   (target: OCID Overlap-F ~91.9 / Boundary-F ~89.6 / %75 ~93.8).
#
# Examples:
#   bash legacy/scripts/eval.sh                                          # best RGB-D on OCID
#   bash legacy/scripts/eval.sh --input_type rgb  --dataset osd --use_cgnet True
#   bash legacy/scripts/eval.sh --input_type rgbd --dataset bosd
#   bash legacy/scripts/eval.sh --exp_dir ../data/checkpoints/legacy_train_ckpt/DI_depth_none_NOW0.4_BS4_LR1e-05
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
PYTHON="$REPO/legacy/.venv/bin/python"; [ -x "$PYTHON" ] || PYTHON="python"

cd "$REPO/legacy"
exec "$PYTHON" benchmark.py "$@"
