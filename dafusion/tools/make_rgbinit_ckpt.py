"""Make an init checkpoint with an UNINITIALIZED depth branch (for the xyz encoding).

The dualinit checkpoint (tools/remap_coco_init.py) seeds BOTH DA-Fusion branches
(backbone.swin_rgb.* and backbone.swin_d.*) from ImageNet/COCO Swin-L. For the metric-XYZ
depth encoding the depth branch should be trained FROM SCRATCH (XYZ != natural RGB), so we
drop every backbone.swin_d.* key: at load time detectron2 reports them as newly-initialized
and the backbone's from-scratch init (MODEL.DAFUSION.DEPTH_FROM_SCRATCH=True) stands.
The RGB branch + pixel decoder + transformer decoder weights are kept as-is.

Usage:
    python tools/make_rgbinit_ckpt.py \
        --src ../data/checkpoints/mask2former/dafusion_swinL_dualinit.pkl \
        --dst ../data/checkpoints/mask2former/dafusion_swinL_rgbinit.pkl
"""
import argparse
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../data/checkpoints/mask2former/dafusion_swinL_dualinit.pkl")
    ap.add_argument("--dst", default="../data/checkpoints/mask2former/dafusion_swinL_rgbinit.pkl")
    args = ap.parse_args()

    with open(args.src, "rb") as f:
        ck = pickle.load(f)
    sd = ck["model"] if "model" in ck else ck

    new_sd = {}
    n_drop = 0
    for k, v in sd.items():
        if k.startswith("backbone.swin_d."):
            n_drop += 1
            continue
        new_sd[k] = v

    out = {"model": new_sd, "__author__": "dafusion-rgbinit", "matching_heuristics": True}
    with open(args.dst, "wb") as f:
        pickle.dump(out, f)
    print(f">>> dropped {n_drop} backbone.swin_d.* tensors (depth branch -> from scratch)")
    print(f">>> wrote {args.dst}  ({len(new_sd)} keys)")


if __name__ == "__main__":
    main()
