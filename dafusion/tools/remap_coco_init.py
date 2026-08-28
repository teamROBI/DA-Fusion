"""Duplicate the Mask2Former Swin-L COCO backbone weights into BOTH DA-Fusion branches.

The COCO checkpoint (model_final_e5f453.pkl) has single-branch keys ``backbone.*``.
DA-Fusion's DualSwinFusionBackbone expects ``backbone.swin_rgb.*`` and ``backbone.swin_d.*``.
This writes a new .pkl that seeds both branches from the COCO backbone and leaves the
Mask2Former head keys untouched (the 80-class COCO class head shape-mismatches our 1-class
head and is skipped at load time, which is correct).

Usage:
    python tools/remap_coco_init.py \
        --src ../data/checkpoints/mask2former/model_final_e5f453.pkl \
        --dst ../data/checkpoints/mask2former/dafusion_swinL_dualinit.pkl

With INPUT.DEPTH_VALIDITY_CHANNEL the depth stem takes 4 channels, so its patch-embed weight
must be inflated to match or the checkpointer SILENTLY drops it (see --validity-channel):
    python tools/remap_coco_init.py \
        --src ../data/checkpoints/mask2former/model_final_e5f453.pkl \
        --dst ../data/checkpoints/mask2former/dafusion_swinL_dualinit_v4.pkl \
        --validity-channel
"""
import argparse
import pickle

import numpy as np

# The depth branch's input stem. With a validity channel this becomes (embed_dim, 4, k, k).
DEPTH_STEM = "backbone.swin_d.patch_embed.proj.weight"


def _inflate_stem(w, init="zeros"):
    """(C_out, 3, k, k) -> (C_out, 4, k, k), adding an input slice for the validity channel.

    Zero-init is the default and is the point of the whole exercise: the new slice contributes
    exactly nothing on the first forward pass, so training starts numerically identical to the
    3-channel model and *learns* whether validity is useful. A random slice would instead inject
    noise into a pretrained stem, and 'mean' would assert up-front that validity looks like a
    colour channel, which it does not.
    """
    if w.ndim != 4 or w.shape[1] != 3:
        raise ValueError(f"unexpected depth stem shape {w.shape}; expected (C_out, 3, k, k)")
    extra = np.zeros_like(w[:, :1])
    if init == "mean":
        extra = w.mean(axis=1, keepdims=True)
    return np.concatenate([w, extra], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--validity-channel", action="store_true",
                    help="inflate the DEPTH patch-embed stem from 3 to 4 input channels, for "
                         "INPUT.DEPTH_VALIDITY_CHANNEL runs. Without this the model's 4-channel "
                         "stem shape-mismatches the 3-channel checkpoint tensor and detectron2 "
                         "WARN-AND-SKIPS it (c2_model_loading.py), leaving the depth stem at "
                         "PyTorch default init -- which is the from-scratch configuration that "
                         "scored 24.7 mean. Easy to miss: it is a log warning, not an error.")
    ap.add_argument("--stem-init", default="zeros", choices=["zeros", "mean"],
                    help="how to initialize the new 4th input slice (default zeros)")
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        ck = pickle.load(f)
    sd = ck["model"] if "model" in ck else ck

    # head tensors whose shape depends on #queries / #classes -> must reinit (drop them)
    DROP = ("class_embed", "query_feat", "query_embed", "static_query")

    new_sd = {}
    n_dup = n_drop = 0
    for k, v in sd.items():
        if any(d in k for d in DROP):
            n_drop += 1
            continue
        if k.startswith("backbone."):
            suffix = k[len("backbone."):]
            new_sd[f"backbone.swin_rgb.{suffix}"] = v
            new_sd[f"backbone.swin_d.{suffix}"] = v
            n_dup += 1
        else:
            new_sd[k] = v  # sem_seg_head.pixel_decoder.* + decoder layers unchanged

    if args.validity_channel:
        if DEPTH_STEM not in new_sd:
            raise KeyError(f"{DEPTH_STEM} missing — cannot inflate the depth stem")
        before = np.asarray(new_sd[DEPTH_STEM])
        after = _inflate_stem(before, args.stem_init)
        new_sd[DEPTH_STEM] = after
        print(f">>> inflated {DEPTH_STEM}: {before.shape} -> {after.shape} "
              f"(4th slice init={args.stem_init})")

    out = {"model": new_sd, "__author__": "dafusion-remap", "matching_heuristics": True}
    with open(args.dst, "wb") as f:
        pickle.dump(out, f)
    print(f">>> duplicated {n_dup} backbone tensors into swin_rgb + swin_d; dropped {n_drop} query/class head tensors")
    print(f">>> wrote {args.dst}  ({len(new_sd)} keys)")


if __name__ == "__main__":
    main()
