#!/usr/bin/env bash
# Patch the freshly-cloned (gitignored) vendored detectron2 to the non-deprecated
# torch.amp autocast API (torch >= 2.4). Upstream's AMPTrainer still calls
# torch.cuda.amp.autocast(...), which emits a FutureWarning EVERY training iteration and
# floods the log. Since third_party/detectron2 is git-cloned fresh on each setup (and
# gitignored), this repo-tracked patch must re-apply the fix on every new install.
#
# Idempotent + version-tolerant: matches by pattern (not line number) and is a no-op if
# the clone already uses torch.amp. Called from setup_env.sh after the detectron2 clone.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
D2="${1:-$REPO/dafusion/third_party/detectron2}"
TL="$D2/detectron2/engine/train_loop.py"
if [ ! -f "$TL" ]; then
  echo ">>> patch_detectron2: $TL not found; skipping"
  exit 0
fi

changed=0
if grep -q "from torch\.cuda\.amp import autocast" "$TL"; then
  sed -i 's/from torch\.cuda\.amp import autocast/from torch.amp import autocast/' "$TL"
  changed=1
fi
if grep -q "with autocast(dtype=self\.precision):" "$TL"; then
  sed -i 's/with autocast(dtype=self\.precision):/with autocast("cuda", dtype=self.precision):/' "$TL"
  changed=1
fi

if [ "$changed" = "1" ]; then
  echo ">>> patch_detectron2: applied torch.amp autocast fix to train_loop.py"
else
  echo ">>> patch_detectron2: train_loop.py already on torch.amp (no change)"
fi
