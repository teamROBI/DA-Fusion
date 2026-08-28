#!/usr/bin/env bash
# Evaluate a trained DA-Fusion checkpoint on the UOIS benchmarks (OSD / OCID / OCBD).
# OCBD (dataset dir; formerly BOSD) set (paper renamed BOSD -> OCBD).
#
# Examples:
#   bash dafusion/scripts/eval.sh --dataset ocid --weights ../data/checkpoints/dafusion/<run>/model_final.pth
#   bash dafusion/scripts/eval.sh --dataset osd  --use_cgnet          # UOAIS foreground-filter protocol
#   bash dafusion/scripts/eval.sh --dataset ocbd --input_type rgbd
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DAFU="$REPO/dafusion"
PYTHON="$DAFU/.venv/bin/python"; [ -x "$PYTHON" ] || PYTHON="python"

cd "$DAFU"
exec "$PYTHON" -m dafusion.eval.benchmark --config configs/dafusion_rgbd_uoais.yaml "$@"
