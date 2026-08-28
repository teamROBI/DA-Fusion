#!/usr/bin/env bash
# Set up the modern DA-Fusion (reimplementation) environment with uv.
# Stack: Python 3.10 / torch 2.4+cu124 / detectron2 (from source) — compiled
# against this box's CUDA 12.4 toolkit for the RTX 3090s (sm_86).
#
# Unlike the legacy build, this compiles CUDA extensions FRESH (no prebuilt .so reuse):
#   - detectron2 _C           (built by pip from a source clone)
#   - MSDeformAttn op         (dafusion/modeling/pixel_decoder/ops, built here)
#
# Portable: run on any server with an NVIDIA GPU (CUDA 12.x toolkit for the compiles).
# Installs uv if missing and auto-detects the GPU compute capability. Datasets/weights are
# separate — point DAFUSION_DATA at your data dir (default: <repo>/data) after install.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
DAFU="$REPO/dafusion"
cd "$DAFU"

# uv (auto-install if missing)
if ! command -v uv >/dev/null 2>&1; then
  echo ">>> installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v nvcc >/dev/null 2>&1 && echo ">>> nvcc: $(nvcc --version | tail -1)" || echo ">>> WARNING: nvcc not on PATH; CUDA compiles may fail (need a CUDA 12.x toolkit)"

PY=3.10
VENV="$DAFU/.venv"
# auto-detect GPU arch (e.g. 8.6 for RTX 3090, 8.9 for 4090, 9.0 for H100); override with TORCH_CUDA_ARCH_LIST
if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
  CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
  export TORCH_CUDA_ARCH_LIST="${CAP:-8.6}"
fi
echo ">>> building for CUDA arch: $TORCH_CUDA_ARCH_LIST"
export FORCE_CUDA="1"

# 1. venv (idempotent — reuse if present)
[ -d "$VENV" ] || uv venv --python "$PY" "$VENV"
uv pip install --python "$VENV" "setuptools>=64" wheel ninja

# 2. torch + torchvision (CUDA 12.4 wheels; ship sm_86)
uv pip install --python "$VENV" torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu124

# 3. detectron2 from source (EDITABLE clone so detectron2.projects.point_rend resolves,
#    which mask2former/criterion.py needs). Clone kept under third_party/ (gitignored).
mkdir -p "$DAFU/third_party"
D2="$DAFU/third_party/detectron2"
if [ ! -d "$D2/.git" ]; then
  git clone https://github.com/facebookresearch/detectron2.git "$D2"
fi
# detectron2's setup.py imports torch at build time -> disable build isolation so it
# sees the torch we just installed (setuptools/wheel/ninja are already in the venv).
uv pip install --python "$VENV" --no-build-isolation -e "$D2"
# Patch the fresh clone to torch.amp autocast (torch>=2.4) so AMPTrainer doesn't emit a
# per-iteration FutureWarning. Editable install -> patched source is used at runtime.
bash "$DAFU/scripts/patch_detectron2.sh" "$D2"

# 4. project deps
uv pip install --python "$VENV" -e "$DAFU"
uv pip install --python "$VENV" "panopticapi @ git+https://github.com/cocodataset/panopticapi.git"

# 5. compile MSDeformAttn fresh against CUDA 12.4
cd "$DAFU/dafusion/modeling/pixel_decoder/ops"
"$VENV/bin/python" setup.py build install
cd "$DAFU"

echo ">>> Sanity check:"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
import detectron2; print("detectron2", detectron2.__version__)
from detectron2.projects.point_rend.point_features import point_sample  # noqa: F401
print("point_rend: OK")
import MultiScaleDeformableAttention as MSDA; print("MSDeformAttn op: OK", MSDA.__file__)
PY
echo ">>> DA-Fusion (modern) environment ready in $VENV"
