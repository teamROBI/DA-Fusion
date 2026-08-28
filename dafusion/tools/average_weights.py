"""Average model weights across checkpoints (SWA-style) or across runs (model soup).

Why this is worth trying when 14 single-model experiments have all plateaued at ~88.2: every
experiment so far tried to make ONE model better, and four unrelated mechanisms each bought ~+0.5
while refusing to STACK when combined in training (docs/EXPERIMENTS.md 10e). Combining trained
models is a different operation from combining mechanisms during training, and it has never been
tried here. Two flavours:

  * within-run averaging -- average the last N checkpoints of one run. Late checkpoints of a single
    run sit in the same loss basin, so the average is well-defined and usually lands in a flatter
    minimum than any individual point (the standard SWA result). Free: no training, no GPU.
  * cross-run soup -- average final weights of runs fine-tuned from the SAME initialisation with
    different recipes. Our runs all start from the same COCO dualinit, which is exactly the
    precondition model soups require. Only valid within a weight-compatible group: e.g.
    gridB_adapter_only + support_aug_adapter (identical 1302-tensor signature); geomprior (1359,
    extra geometry-prior params) and normals (1295) are separate architectures.

Only floating-point tensors are averaged; integer/bookkeeping tensors are copied from the first
checkpoint (averaging a counter is meaningless).

Usage:
    python tools/average_weights.py --out /tmp/avg.pth \
        ../data/checkpoints/dafusion/<run>/model_0017999.pth \
        ../data/checkpoints/dafusion/<run>/model_0020999.pth
"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("ckpts", nargs="+", help="two or more .pth checkpoints (same architecture)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert len(args.ckpts) >= 2, "need >= 2 checkpoints to average"

    acc, n_float, n_copied, meta = None, 0, 0, None
    for i, p in enumerate(args.ckpts):
        sd = torch.load(p, map_location="cpu")
        sd = sd.get("model", sd)
        # `_metadata` is an ATTRIBUTE on the state-dict OrderedDict, not a key, so rebuilding the
        # dict silently drops it -- and it carries the per-module `version` that suppresses
        # Mask2Former's legacy MaskFormerHead key-renaming converter (sem_seg_head version: 2).
        # Without it the converter fires on load, renames ~60 pixel-decoder params, and the model
        # comes up PARTLY RANDOM while still "loading successfully": eval then reports
        # Precision 100 / Recall 3.8 (it predicts almost nothing) rather than crashing.
        if meta is None:
            meta = getattr(sd, "_metadata", None)
        if acc is None:
            acc = {}
            for k, v in sd.items():
                if torch.is_tensor(v) and v.is_floating_point():
                    acc[k] = v.clone().double()
                    n_float += 1
                else:
                    acc[k] = v            # counters / non-float: take the first, don't average
                    n_copied += 1
        else:
            missing = set(acc) ^ set(sd)
            if missing:
                raise SystemExit(f"checkpoint {p} key mismatch ({len(missing)} differing keys) — "
                                 "these models are not weight-compatible")
            for k, v in sd.items():
                if k in acc and torch.is_tensor(acc[k]) and acc[k].is_floating_point():
                    if tuple(v.shape) != tuple(acc[k].shape):
                        raise SystemExit(f"shape mismatch on {k}: {v.shape} vs {acc[k].shape}")
                    acc[k] += v.double()

    from collections import OrderedDict
    out = OrderedDict()
    for k, v in acc.items():
        if torch.is_tensor(v) and v.is_floating_point():
            out[k] = (v / len(args.ckpts)).float()
        else:
            out[k] = v
    if meta is not None:
        out._metadata = meta          # REQUIRED -- see the note above
    torch.save({"model": out, "__author__": "dafusion-weight-average"}, args.out)
    print(f">>> averaged {len(args.ckpts)} checkpoints ({n_float} float tensors averaged, "
          f"{n_copied} copied, _metadata {'preserved' if meta is not None else 'ABSENT'}) "
          f"-> {args.out}")


if __name__ == "__main__":
    main()
