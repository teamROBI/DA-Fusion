#!/usr/bin/env bash
# Set up the LEGACY DA-Fusion environment with uv.
# Reproduces the original stack: Python 3.8 / torch 1.9.0+cu111 / detectron2 0.6.
#
# IMPORTANT: this box has only a CUDA 12.x toolkit (nvcc 12.4), which cannot
# compile torch-1.9 CUDA extensions. Instead we REUSE the prebuilt .so files
# that shipped with the repo (built for py3.8 + torch1.9 + cudart 11.0):
#   - legacy/detectron2/detectron2/_C*.so
#   - legacy/mask2former/modeling/pixel_decoder/ops/build/.../MultiScaleDeformableAttention*.so
# The NVIDIA driver (580, CUDA 13-capable) runs cu111 wheels fine on the 3090s.
# To rebuild from source instead, install a CUDA 11.x toolkit and run
# `python setup.py build install` in the ops dir + `pip install -e legacy/detectron2`.
#
# NOTE: this is the LEGACY env only. The reimplementation lives in dafusion/ with a
# separate modern env (dafusion/scripts/setup_env.sh, py3.10 / torch2.x).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
LEGACY="$REPO/legacy"
cd "$LEGACY"

if ! command -v uv >/dev/null 2>&1; then
  echo "ERROR: 'uv' not found. Install:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

PY=3.8
VENV="$LEGACY/.venv"

# 1. venv
uv venv --python "$PY" "$VENV"

# 2. Runtime setuptools pinned to 59.5.0 — torch 1.9's tensorboard integration
#    breaks on setuptools>=60 ("module 'distutils' has no attribute 'version'").
#    The editable build below uses an isolated env (build-system.requires in
#    pyproject.toml pins a modern setuptools there), so this pin is runtime-only.
uv pip install --python "$VENV" "setuptools==59.5.0" wheel

# 3. torch + torchvision (CUDA 11.1; wheels bundle the 11.0 runtime, driver 580 runs them)
uv pip install --python "$VENV" torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  --index-url https://download.pytorch.org/whl/cu111

# 4. project deps (legacy/pyproject.toml) + panopticapi (pulls detectron2's transitive deps too)
uv pip install --python "$VENV" -e .
uv pip install --python "$VENV" "panopticapi @ git+https://github.com/cocodataset/panopticapi.git"

# 5. Make the vendored detectron2 (with its in-place _C.so) and the prebuilt
#    MSDeformAttn op importable, WITHOUT recompiling — via a .pth file.
SITE="$("$VENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
MSDA_DIR="$LEGACY/mask2former/modeling/pixel_decoder/ops/build/lib.linux-x86_64-cpython-38"
{
  echo "$LEGACY/detectron2"
  echo "$MSDA_DIR"
} > "$SITE/dafusion_legacy.pth"
echo ">>> wrote $SITE/dafusion_legacy.pth (detectron2 + MSDeformAttn on path)"

echo ">>> Sanity check:"
"$VENV/bin/python" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
import detectron2; print("detectron2", detectron2.__version__)
import MultiScaleDeformableAttention as MSDA; print("MSDeformAttn op: OK", MSDA.__file__)
PY
echo ">>> Legacy DA-Fusion environment ready in $VENV"
