"""RGB-D copy-paste augmentation (Simple Copy-Paste, Ghiasi et al. 2021), adapted for depth.

Pastes object instances (RGB + depth + mask) from a random source training image into the
target image to synthesize denser, more touching/occluded clutter than UOAIS-Sim naturally
has. This directly teaches the model to SEPARATE adjacent instances (our OCID/OSD
under-segmentation failure mode) and to handle occlusion — all as data augmentation, with NO
change to the model architecture.

Depth handling (the RGB-D-specific part): a pasted object is placed as a FOREGROUND occluder
by shifting its source depth so it sits just in front of the local target surface, then its
depth/RGB overwrite the target only within the object silhouette. Existing target masks have
the pasted footprint subtracted (they are now occluded), keeping instance masks consistent.
"""
import numpy as np

try:
    from pycocotools import mask as mask_utils
except Exception:  # pragma: no cover
    mask_utils = None


def decode_masks(annotations):
    """Decode each annotation's visible_mask (COCO RLE) -> list of (H,W) bool bitmasks."""
    masks = []
    for a in annotations:
        if a.get("iscrowd", 0) != 0 or a.get("visible_mask") is None:
            continue
        masks.append(mask_utils.decode(a["visible_mask"]).astype(bool))
    return masks


def copy_paste_rgbd(rgb, depth_mm, masks, src_rgb, src_depth_mm, src_masks,
                    rng=np.random, n_paste=3, p_flip=0.5, min_area=200, front_offset_mm=50.0):
    """Paste up to n_paste source instances into (rgb, depth_mm). Returns new
    (rgb, depth_mm, masks) where masks is the updated list of (H,W) bool instance masks
    (existing masks occluded by pastes are trimmed; each paste adds one new instance)."""
    if not src_masks:
        return rgb, depth_mm, masks
    H, W = depth_mm.shape
    out_rgb = rgb.copy()
    out_depth = depth_mm.astype(np.float32).copy()
    masks = [m.copy() for m in masks]

    order = rng.permutation(len(src_masks))[:n_paste]
    for si in order:
        sm = src_masks[si]
        ys, xs = np.where(sm)
        if len(ys) < min_area:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cm = sm[y0:y1 + 1, x0:x1 + 1]
        crgb = src_rgb[y0:y1 + 1, x0:x1 + 1]
        cd = src_depth_mm[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
        if rng.rand() < p_flip:
            cm, crgb, cd = cm[:, ::-1], crgb[:, ::-1], cd[:, ::-1]
        ch, cw = cm.shape
        if H - ch <= 0 or W - cw <= 0:
            continue
        ty, tx = rng.randint(0, H - ch), rng.randint(0, W - cw)

        src_valid = cm & (cd > 0)
        if src_valid.sum() < min_area:
            continue
        # place as a foreground occluder: shift obj depth just in front of local target surface
        tgt_local = out_depth[ty:ty + ch, tx:tx + cw]
        tl = tgt_local[tgt_local > 0]
        ref = np.percentile(tl, 30) if tl.size else (
            np.percentile(out_depth[out_depth > 0], 30) if (out_depth > 0).any() else 1000.0)
        obj_ref = np.percentile(cd[src_valid], 50)
        cd_shifted = cd + (ref - obj_ref - front_offset_mm)

        region = np.zeros((H, W), bool)
        region[ty:ty + ch, tx:tx + cw] = cm                       # full object silhouette
        dvalid = np.zeros((H, W), bool)
        dvalid[ty:ty + ch, tx:tx + cw] = src_valid                # where we have depth
        # write RGB over silhouette; depth only where source depth is valid (else leave a hole=0)
        prgb = np.zeros_like(out_rgb); prgb[ty:ty + ch, tx:tx + cw] = crgb
        pd = np.zeros((H, W), np.float32); pd[ty:ty + ch, tx:tx + cw] = np.clip(cd_shifted, 1.0, None)
        out_rgb[region] = prgb[region]
        out_depth[dvalid] = pd[dvalid]
        out_depth[region & ~dvalid] = 0.0                         # silhouette hole where no src depth

        masks = [m & ~region for m in masks]                      # occlude existing instances
        masks = [m for m in masks if m.sum() >= min_area]
        masks.append(region)                                      # new pasted instance
    return out_rgb, out_depth, masks
