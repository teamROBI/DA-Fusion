#!/usr/bin/env bash
# =============================================================================
# DA-Fusion — one-shot setup for a fresh server (trains the dafusion/ build).
#
# Quick start on a new machine (e.g. 4x A6000):
#   curl -LsSf https://raw.githubusercontent.com/teamROBI/DA-Fusion/main/install.sh | bash
# or, if you already cloned:
#   git clone https://github.com/teamROBI/DA-Fusion.git && cd DA-Fusion && bash install.sh
#
# Requirements on the target server:
#   - an NVIDIA GPU + a CUDA 12.x toolkit (nvcc) for compiling the ops
#   - git, curl. (uv is auto-installed.)
#
# Env overrides:
#   STORE       per-user storage root for shared datasets/ + weights/ (mirrors hinton's
#               /data1/<user>). If unset, the script PROMPTS with the writable disks it finds
#               (or type a custom path). Non-interactive (curl|bash) uses the top candidate.
#   DATASETS_ROOT / WEIGHTS_ROOT / DATA_ROOT  fine-grained overrides (default:
#               $STORE/datasets, $STORE/weights, $STORE/projects/DA-Fusion).
#   SRC         if set, auto-copy datasets + weights from a source STORE (one go). Point it at
#               the source's /data1/<user> — e.g. SRC=jokim@hinton:/data1/jokim. Pulls
#               datasets/UOIS + weights/{mask2former,uoais} with rsync -L. Else prints manual cmds.
#   GIT_URL     repo to clone if not already inside one (default: teamROBI/DA-Fusion).
#   BRANCH      branch to clone (default: main).
#   SKIP_ENV=1  skip the python env build (just scaffold data).
# =============================================================================
set -euo pipefail

GIT_URL="${GIT_URL:-https://github.com/teamROBI/DA-Fusion.git}"
BRANCH="${BRANCH:-main}"
MF_URL="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_swin_large_IN21k_384_bs16_100ep/model_final_e5f453.pkl"

# ---- locate or clone the repo ----
if [ -f "dafusion/scripts/setup_env.sh" ]; then
  REPO="$(pwd)"
elif [ -f "$(dirname "$0")/dafusion/scripts/setup_env.sh" ]; then
  REPO="$(cd "$(dirname "$0")" && pwd)"
else
  echo ">>> cloning $GIT_URL (branch $BRANCH)"
  git clone -b "$BRANCH" "$GIT_URL" DA-Fusion
  REPO="$(pwd)/DA-Fusion"
fi
cd "$REPO"
echo ">>> repo: $REPO"

# ---- storage layout (mirrors hinton: shared datasets/ + weights/, symlinked per project) ----
# STORE      per-user storage root holding datasets/ + weights/, shared across projects
#            (hinton: /data1/jokim). Prompted if unset.
# DATA_ROOT  this project's data dir: symlinks UOIS + checkpoints/{mask2former,uoais} into
#            STORE, plus a REAL checkpoints/dafusion/ for training outputs. repo/data -> here.
me="${USER:-$(whoami)}"
if [ -z "${STORE:-}" ]; then
  cands=() ; labels=()
  for c in /data1 /data2 /data3 /data; do
    [ -d "$c" ] || continue
    if [ -d "$c/$me" ] && [ -w "$c/$me" ]; then base="$c/$me"
    elif [ -w "$c" ]; then base="$c"
    else continue; fi
    avail="$(df -h "$c" 2>/dev/null | awk 'NR==2{print $4}')"
    cands+=("$base") ; labels+=("$c  (${avail:-?} free)")
  done
  cands+=("$REPO/_store") ; labels+=("in-repo fallback")
  if [ -t 0 ]; then
    echo ""
    echo "Storage root for shared datasets/ + weights/ (~57GB; mirrors hinton's /data1/<user>):"
    for i in "${!cands[@]}"; do printf "  %d) %-24s %s\n" "$((i+1))" "${cands[$i]}" "${labels[$i]}"; done
    echo "  (or type a full custom path)"
    read -rp "choice [1]: " ans ; ans="${ans:-1}"
    if [[ "$ans" =~ ^[0-9]+$ ]] && [ "$ans" -ge 1 ] && [ "$ans" -le "${#cands[@]}" ]; then
      STORE="${cands[$((ans-1))]}"
    else STORE="$ans"; fi
  else
    STORE="${cands[0]}"
    echo ">>> non-interactive: STORE=$STORE (set STORE= to override)"
  fi
fi
DATASETS_ROOT="${DATASETS_ROOT:-$STORE/datasets}"
WEIGHTS_ROOT="${WEIGHTS_ROOT:-$STORE/weights}"
DATA_ROOT="${DATA_ROOT:-$STORE/projects/DA-Fusion}"
mkdir -p "$DATASETS_ROOT/UOIS" "$WEIGHTS_ROOT/mask2former" "$WEIGHTS_ROOT/uoais" \
         "$DATA_ROOT/checkpoints/dafusion"
# project data dir: symlink shared stores in (mirrors hinton)
ln -sfn "$DATASETS_ROOT/UOIS"       "$DATA_ROOT/UOIS"
ln -sfn "$WEIGHTS_ROOT/mask2former" "$DATA_ROOT/checkpoints/mask2former"
ln -sfn "$WEIGHTS_ROOT/uoais"       "$DATA_ROOT/checkpoints/uoais"
if [ -L "$REPO/data" ] || [ ! -e "$REPO/data" ]; then ln -sfn "$DATA_ROOT" "$REPO/data"; fi
echo ">>> store: $STORE   data dir: $DATA_ROOT  (symlinked at $REPO/data)"

# ---- copy datasets + weights from a source store (set SRC=user@host:/data1/<user>) ----
if [ -n "${SRC:-}" ]; then
  echo ">>> copying datasets + weights from $SRC  (large — may take a while)"
  rsync -aL --info=progress2 "$SRC/datasets/UOIS/" "$DATASETS_ROOT/UOIS/"
  rsync -aL "$SRC/weights/mask2former/" "$WEIGHTS_ROOT/mask2former/" 2>/dev/null || true
  rsync -aL "$SRC/weights/uoais/"       "$WEIGHTS_ROOT/uoais/"       2>/dev/null || true
else
  echo ">>> SRC not set — datasets/weights will need a manual copy (see end)."
fi

# ---- Mask2Former Swin-L COCO init (into shared weights; auto-download if missing) ----
MF_PKL="$WEIGHTS_ROOT/mask2former/model_final_e5f453.pkl"
if [ ! -f "$MF_PKL" ]; then
  echo ">>> downloading Mask2Former Swin-L COCO init (~1GB)..."
  curl -L "$MF_URL" -o "$MF_PKL" || echo "!! download failed — place it at $MF_PKL manually"
fi

# ---- python env (torch2.4+cu124, detectron2 from source, MSDeformAttn compiled fresh) ----
if [ "${SKIP_ENV:-0}" != "1" ]; then
  bash "$REPO/dafusion/scripts/setup_env.sh"
fi

# ---- build the dual-branch COCO init (both Swin branches) ----
DUAL="$WEIGHTS_ROOT/mask2former/dafusion_swinL_dualinit.pkl"
if [ -f "$MF_PKL" ] && [ ! -f "$DUAL" ] && [ -x "$REPO/dafusion/.venv/bin/python" ]; then
  ( cd "$REPO/dafusion" && .venv/bin/python tools/remap_coco_init.py --src "$MF_PKL" --dst "$DUAL" )
fi

cat <<EOF

=============================================================================
DA-Fusion setup complete.

Layout (mirrors hinton — shared stores, symlinked into the project data dir):
  $DATASETS_ROOT/UOIS/         <- $DATA_ROOT/UOIS
  $WEIGHTS_ROOT/{mask2former,uoais}/   <- $DATA_ROOT/checkpoints/{mask2former,uoais}
  $DATA_ROOT/checkpoints/dafusion/     (training outputs, real dir)

Datasets are NOT in git. If you set SRC they were just copied above; otherwise copy them:
  re-run with SRC set (one go):  SRC=jokim@hinton:/data1/jokim  bash install.sh
  or rsync manually into:
    $DATASETS_ROOT/UOIS/{UOAIS-Sim, OSD-0.20-depth, OCID-dataset, OCBD}   (train + eval sets)
    $WEIGHTS_ROOT/uoais/rgbd_fg.pth                                       (CG-Net filter, --use_cgnet)

Train (4x A6000, 48GB — batch 2/GPU fits; ~20 epochs):
  cd $REPO
  NUM_GPUS=4 bash dafusion/scripts/train.sh SOLVER.IMS_PER_BATCH 8 SOLVER.MAX_ITER 125000
  # closer to the paper's batch-2: SOLVER.IMS_PER_BATCH 4 SOLVER.MAX_ITER 250000

Evaluate:
  bash dafusion/scripts/eval.sh --dataset ocid --weights data/checkpoints/dafusion/<run>/model_final.pth --use_cgnet
=============================================================================
EOF
