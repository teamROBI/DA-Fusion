"""DA-Fusion evaluation on the UOIS benchmarks: OSD / OCID / OCBD.

Reuses the exact UOIS PRF metric from the legacy build (compute_PRF) for comparability.
OCBD (dataset dir; formerly BOSD) directory (paper renamed BOSD -> OCBD). Optional CG-Net
foreground filter (`--use_cgnet`) reproduces the standard UOAIS eval protocol.

Usage (run from dafusion/):
    python -m dafusion.eval.benchmark --dataset ocid --input_type rgbd \
        --config configs/dafusion_rgbd_uoais.yaml \
        --weights ../data/checkpoints/dafusion/<run>/model_final.pth --use_cgnet
"""
import argparse
import json
import os

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

import dafusion  # noqa: F401  register arch/backbone/datasets
from dafusion.config import add_dafusion_config
from dafusion.data.datasets.intrinsics import get_intrinsics
from dafusion.engine.predictor import DAFusionPredictor
from dafusion.eval import compute_PRF
from dafusion.eval import post_process as pp
from dafusion.eval.model import Context_Guided_Network
from dafusion.paths import DATASET_PATHS, CGNET_WEIGHTS, BENCHMARK_RESULT_ROOT

BACKGROUND_LABEL = 0
SCORE_THRESHOLD = float(os.environ.get("DAFUSION_SCORE_THRESH", "0.5"))  # global default / sweep override
# Per-dataset confidence threshold overrides (fall back to SCORE_THRESHOLD). OCID's low-confidence
# background FPs benefit from a higher bar; OCBD lone objects need a lower one to keep recall.
DATASET_SCORE_THRESH = {"ocid": 0.9}   # OCID over-predicts low-conf FPs: 0.9 -> Overlap-F 83.1->85.8
                                       # (OSD/OCBD flat, keep 0.5). Confirmed on xyz+noise model_0020999.
W, H = 640, 480                                    # default (OSD/OCID); OCBD overrides below
# Per-dataset eval resolution, matching the legacy build (OCBD/BOSD is 600x400 native).
DATASET_WH = {"osd": (640, 480), "ocid": (640, 480), "ocbd": (600, 400)}
OCID_BG = {"floor": [0, 1], "table": [0, 1, 2]}   # non-object labels per OCID scene type
# Native capture size per benchmark, for rescaling intrinsics to the eval resolution. OCID/OSD are
# 640x480 natively AND at eval; OCBD is natively 600x400 (its annotations and organized cloud are
# both 600x400), contradicting intrinsics.py's "all benchmarks here are 640x480" docstring.
NATIVE_WH = {"ocid": (640, 480), "osd": (640, 480), "ocbd": (600, 400)}
# Per-dataset horizontal depth shift (px) applied to align depth onto the RGB frame, env-gated by
# DAFUSION_ALIGN_DEPTH=1 (default OFF = original behavior). Measured over the full sets with
# scripts/measure_rgbd_alignment.py (GT-mask boundaries vs depth discontinuities): OCID dx=0
# (99.6% of 2287 frames exactly 0) and OSD dx=-1 are already aligned and are NOT listed; only
# OCBD carries a real offset (median -5, IQR [-7,-3]), from its physical RGB/IR baseline. Positive
# = shift depth right. Vacated columns become invalid (0), not wrapped.
DATASET_DEPTH_SHIFT = {"ocbd": -5}
# Depth-validity foreground filter thresholds (UCN / MSMFormer protocol): drop a predicted
# mask if the fraction of its pixels with valid (non-zero) depth is below this.
DEPTH_FILTER_THRESH = {"ocid": 0.5, "osd": 0.8, "ocbd": 0.8}
# Env override for sweeping the depth-validity filter. This filter DELETES any predicted
# mask whose valid-depth fraction is below threshold, so it is a recall knob: lowering it
# keeps masks the model found but whose depth is mostly holes. Motivated by the floor/table
# split (Track 10i) -- floor scenes are both the holed ones AND the ones carrying 55% of
# the loss, so this filter may be discarding real objects precisely there.
_DF_ENV = os.environ.get("DAFUSION_DEPTH_FILTER_THRESH")
if _DF_ENV is not None:
    DEPTH_FILTER_THRESH = {k: float(_DF_ENV) for k in DEPTH_FILTER_THRESH}


def array_to_tensor(a):
    if a.ndim == 3:
        return torch.from_numpy(a).permute(2, 0, 1).float()
    return torch.from_numpy(a).float()


def standardize_image(image):
    out = np.zeros_like(image).astype(np.float32)
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    for i in range(3):
        out[..., i] = (image[..., i] / 255.0 - mean[i]) / std[i]
    return out


def normalize_depth_cgnet(depth, min_val=300.0, max_val=1800.0):
    min_val = max(np.percentile(depth, 5), min_val)
    max_val = min(np.percentile(depth, 95), max_val)
    depth = np.clip(depth, min_val, max_val)
    depth = (depth - min_val) / (max_val - min_val) * 255
    return np.uint8(np.repeat(np.expand_dims(depth, -1), 3, -1))


# ----------------------------- dataset enumeration -----------------------------
def list_osd(root):
    import glob
    # Default to the SAM-corrected, RGB-aligned masks ("annotation_fixed") when present, since
    # the original OSD GT is registered to the depth frame and is offset on the RGB. Falls back
    # to the original "annotation" if the fixed set is absent. Override via env DAFUSION_OSD_ANN
    # (e.g. DAFUSION_OSD_ANN=annotation to force the original).
    ann_sub = os.environ.get("DAFUSION_OSD_ANN")
    if not ann_sub:
        ann_sub = "annotation_fixed" if os.path.isdir(f"{root}/annotation_fixed") else "annotation"
    rgb = sorted(glob.glob(f"{root}/image_color/*.png"))
    dep = sorted(glob.glob(f"{root}/disparity/*.png"))
    ann = sorted(glob.glob(f"{root}/{ann_sub}/*.png"))
    return list(zip(rgb, dep, ann, ["osd"] * len(rgb)))


def _walk(root, subdirs_levels):
    """Collect (rgb,depth,label,scene) under root/<fixed level combos>/<seq>/{rgb,depth,label}.

    subdirs_levels is a list of allowed dir names per fixed level (e.g. [["floor","table"],
    ["bottom","top"]]); after those, each remaining dir is a sequence holding rgb/depth/label.
    """
    items = []

    def recurse(base, levels):
        if not os.path.isdir(base):
            return
        if not levels:
            for seq in sorted(os.listdir(base)):
                data_dir = os.path.join(base, seq)
                if not os.path.isdir(os.path.join(data_dir, "rgb")):
                    continue
                scene = "floor" if f"{os.sep}floor{os.sep}" in data_dir + os.sep else "table"
                for name in sorted(os.listdir(os.path.join(data_dir, "rgb"))):
                    items.append((os.path.join(data_dir, "rgb", name),
                                  os.path.join(data_dir, "depth", name),
                                  os.path.join(data_dir, "label", name), scene))
            return
        for d in levels[0]:
            recurse(os.path.join(base, d), levels[1:])

    recurse(root, subdirs_levels)
    return items


def list_ocid(root):
    items = []
    items += _walk(f"{root}/ARID20", [["floor", "table"], ["bottom", "top"]])
    items += _walk(f"{root}/YCB10", [["floor", "table"], ["bottom", "top"], ["cuboid", "curved", "mixed"]])
    items += _walk(f"{root}/ARID10", [["floor", "table"], ["bottom", "top"],
                                      ["box", "curved", "fruits", "mixed", "non-fruits"]])
    return items


def list_ocbd(root):
    items = []
    for group in ["YCB", "Non-YCB"]:
        for bin_name in ["scene_hole_gray_bin", "scene_large_yellow_bin", "scene_small_white_bin"]:
            data_dir = os.path.join(root, group, bin_name)
            if not os.path.isdir(os.path.join(data_dir, "rgb")):
                continue
            for name in sorted(os.listdir(os.path.join(data_dir, "rgb"))):
                items.append((os.path.join(data_dir, "rgb", name),
                              os.path.join(data_dir, "depth", name),
                              os.path.join(data_dir, "label", name), "bin"))
    return items


LISTERS = {"osd": list_osd, "ocid": list_ocid, "ocbd": list_ocbd}


# ----------------------------- eval -----------------------------
def load_cgnet(weight_path):
    ck = torch.load(weight_path, map_location="cpu")
    m = Context_Guided_Network(classes=2, in_channel=4)
    m.load_state_dict(ck["model"])
    return m.cuda().eval()


def build_cfg(args):
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_dafusion_config(cfg)
    cfg.merge_from_file(args.config)
    if args.weights:
        cfg.MODEL.WEIGHTS = args.weights
    cfg.INPUT.INPUT_TYPE = args.input_type
    cfg.MODEL.DAFUSION.INPUT_TYPE = args.input_type
    cfg.freeze()
    return cfg


def _cgnet_filter(pred_masks, rgb_img, depth_raw, w, h, fg_model):
    """Keep predicted masks overlapping CG-Net's predicted foreground (UOAIS protocol)."""
    fg_rgb = array_to_tensor(standardize_image(cv2.resize(rgb_img, (320, 240)))).unsqueeze(0)
    d = cv2.resize(depth_raw, (320, 240), interpolation=cv2.INTER_NEAREST)
    # CG-Net wants a single depth channel in [0,1]. OCBD depth is already a 0-255 3-channel
    # image (take one channel); OSD/OCID raw mm needs normalizing first.
    d1 = d[..., 0].astype(np.float32) if d.ndim == 3 else normalize_depth_cgnet(d)[..., 0].astype(np.float32)
    fg_d = array_to_tensor(d1[..., None]).unsqueeze(0) / 255
    fg_out = fg_model(torch.cat([fg_rgb, fg_d], 1).cuda())
    fg_out = np.argmax(fg_out.cpu().data[0].numpy().transpose(1, 2, 0), axis=2).astype(np.uint8)
    fg_out = cv2.resize(fg_out, (w, h), interpolation=cv2.INTER_NEAREST) > 0
    kept = [m for m in pred_masks if np.sum(m) > 0 and
            np.sum(np.bitwise_and(m > 0, fg_out)) / np.sum(m > 0) >= 0.5]
    return np.array(kept) if kept else np.zeros((0, h, w), np.uint8)


OCBD_XYZ_HW = (400, 600)   # native organized point-cloud grid (H, W)


def _read_pcd_xyz(path):
    """Minimal binary-PCD reader for OCBD clouds (FIELDS x y z, float32, DATA binary)."""
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line or line.startswith(b"DATA"):
                break
        buf = np.frombuffer(f.read(), dtype=np.float32)
    return buf.reshape(-1, 3)


def _load_ocbd_xyz(depth_path, w, h):
    """OCBD's depth/*.png is a colorized (non-metric) image; the real geometry is the sibling
    pcd/img_N cloud. Most scenes ship an organized .npy (H,W,3, mm); some ship only the
    binary .pcd (flat float32 XYZ). Return metric XYZ in meters at (w,h)."""
    base = os.path.splitext(depth_path.replace(f"{os.sep}depth{os.sep}", f"{os.sep}pcd{os.sep}"))[0]
    if os.path.exists(base + ".npy"):
        pts = np.load(base + ".npy").astype(np.float32)
    else:
        pts = _read_pcd_xyz(base + ".pcd").reshape(*OCBD_XYZ_HW, 3)
    pts = np.nan_to_num(pts, nan=0.0)
    if np.abs(pts).max() > 100.0:                             # values in mm -> meters
        pts = pts / 1000.0
    if pts.shape[1::-1] != (w, h):                            # (W,H) vs cv2 (w,h)
        pts = cv2.resize(pts, (w, h), interpolation=cv2.INTER_NEAREST)
    return pts.astype(np.float32)


def _drop_gt_slivers(anno, min_pixels=None, min_cm2=None, area_cm2_map=None):
    """Zero out GT instance labels too small to be objects, i.e. annotation artifacts.

    Two criteria; `min_cm2` is the better one and needs `area_cm2_map`. Pixel count conflates size
    with distance, so a pixel cut cannot separate the populations cleanly: OCBD's GT area histogram
    is sharply bimodal in cm^2 -- a spike of 225 labels under 1 cm^2, a valley of 64 across 1-10,
    then the real object mode from ~10 cm^2 up -- whereas a 20 px cut leaves 143 sub-10 cm^2 labels
    behind (110 of them 20-22 px at 0.2-0.3 cm^2, i.e. 2-3 mm patches).

    Visual verification at 10-13x drove the threshold: a 0.88 cm^2 / 87 px label is a 1-2 px strip
    along a lid seam (artifact), while a 9.89 cm^2 / 973 px label is a genuine partially-occluded
    object. So 2 cm^2 sits inside the valley, clear of both.

    OCBD ships 152 labels of <=20 px, 21 of them exactly 1 px (identical at its native 600x400, so
    not a resize artifact). Rendered at 13x (`scripts/viz_ocbd_slivers.py`) every one inspected sits
    ON A SEAM between two adjacent objects -- leftover pixels from a neighbour's mask that were
    given their own label id, e.g. a 1-px dot on a banana/box edge and a 1-px-wide 15-px line along
    a ball/bottle contact. Each counts as an undetectable GT object, so it is a guaranteed
    false negative for any model.

    OCID and OSD contain ZERO sub-20px GT labels (and their smallest objects are 18 / 39 cm^2), so
    either criterion is a no-op there and cannot flatter them. OCBD is the only set affected.

    CAVEAT: this edits the benchmark's ground truth. Scores computed with it are NOT comparable to
    the paper or to prior work, nor to earlier rows in docs/EXPERIMENTS.md. Hence opt-in via
    DAFUSION_MIN_GT_PIXELS / DAFUSION_MIN_GT_CM2, never on by default."""
    out = anno.copy()
    n = 0
    for v in np.unique(anno):
        if v == 0:
            continue
        m = anno == v
        npix = int(m.sum())
        if min_pixels is not None and npix < min_pixels:
            out[m] = 0
            n += 1
            continue
        if min_cm2 is not None and area_cm2_map is not None:
            good = m & (area_cm2_map > 0)
            nv = int(good.sum())
            # no depth anywhere in the label -> physical size unknown, keep it
            if nv and float(area_cm2_map[good].sum()) * npix / nv < min_cm2:
                out[m] = 0
                n += 1
    return out, n


def pixel_area_cm2(dataset, depth_raw, w, h):
    """Per-pixel frontal physical area in cm^2 (0 where depth is missing), for the metric size
    gate in `post_process.refined_mask` (DAFUSION_METRIC_SIZE_GATE).

    A pixel on a fronto-parallel patch subtends (z/fx)(z/fy), which is what this returns for every
    dataset, using that dataset's intrinsics rescaled to the eval resolution.

    HISTORY / WHY NOT LOCAL SPACING. This originally measured OCBD's footprint directly from the
    cloud's neighbouring X/Y spacing (|diff(X)|, |diff(Y)|) to sidestep intrinsics.py's then-wrong
    OCBD entry. That estimator is **biased upward by sensor noise**: |diff| of a noisy signal has
    positive expectation even where the true spacing is zero, so areas came out 3-5x too large --
    a 29.1 x 25.0 cm plate (ellipse ~571 cm^2) measured as 1627 cm^2, with mean per-pixel area
    0.02421 cm^2 against an analytic 0.00518. Now that the OCBD intrinsics are recovered and
    validated to 1.02 mm (see intrinsics.py), the analytic form is both unbiased and available, so
    all datasets use it. Cross-checked against direct 3D bbox extent on OCBD.

    OCBD still needs its cloud for Z: on a `normalized` config `depth_raw` is the COLORIZED 8-bit
    depth PNG, 3-channel but non-metric, and reading it as depth would be silent nonsense."""
    is_metric_cloud = (depth_raw.ndim == 3 and np.issubdtype(depth_raw.dtype, np.floating)
                       and float(np.abs(depth_raw).max()) < 100.0)
    if dataset == "ocbd" and not is_metric_cloud:
        raise ValueError(
            "the metric size gate needs OCBD's metric point cloud, but depth_raw looks like the "
            "colorized depth PNG (non-metric). Use an xyz / depth_normals config, which routes "
            "OCBD through _load_ocbd_xyz.")
    if is_metric_cloud:
        z = depth_raw[..., 2].astype(np.float32)              # already metres
    else:
        z = depth_raw.astype(np.float32) / 1000.0             # raw uint16 mm -> metres
    fx, fy, _cx, _cy = get_intrinsics(dataset)
    nw, nh = NATIVE_WH.get(dataset, (640, 480))
    fxs, fys = fx * w / nw, fy * h / nh
    return ((z / fxs) * (z / fys) * 1e4).astype(np.float32)


def valid_depth_bbox(depth_raw, min_cover=0.5):
    """Bounding box of the region where depth actually exists, as (y0, y1, x0, x1).

    OCID and OSD depth is registered to a different sensor than the RGB, so a border band carries no
    depth at all: measured over 60 images/set, OCID loses its rightmost ~33 columns in EVERY frame
    (11.6% of the frame invalid) and OSD its rightmost ~66 columns plus all top rows (24.3%). The
    model, however, is fed the FULL RGB frame and so is asked to segment regions it has no depth for
    -- a train/test mismatch, since UOAIS-Sim training depth is 100% valid.

    A row/column is kept when at least `min_cover` of it has valid depth; using "any valid pixel"
    would be defeated by a few scattered survivors in an otherwise dead band.

    The box is snapped OUTWARD to a multiple of 32. This is not cosmetic: 640x480 is exactly divisible
    by 32, so the full frame needs no padding, but an arbitrary crop (e.g. OSD's 566x425) gets padded
    up by `ImageList.from_tensors(..., size_divisibility)` -- with ZEROS. In standardized `xyz` space
    zero is the scene's MEDIAN depth, so padding fabricates a surface along the border and the model
    predicts on it. Measured cost of not snapping: OSD precision 96.0 -> 89.7, F 94.1 -> 90.5.
    """
    valid = np.any(depth_raw != 0, axis=-1) if depth_raw.ndim == 3 else depth_raw > 0
    H, W = valid.shape
    rows = np.nonzero(valid.mean(axis=1) >= min_cover)[0]
    cols = np.nonzero(valid.mean(axis=0) >= min_cover)[0]
    if rows.size == 0 or cols.size == 0:
        return 0, H, 0, W
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    def snap(a, b, limit):
        need = -(-(b - a) // 32) * 32                     # round the extent up to a multiple of 32
        grow = need - (b - a)
        a2 = max(0, a - grow // 2)
        b2 = min(limit, a2 + need)
        a2 = max(0, b2 - need)                            # if we hit the far edge, shift back
        return a2, b2

    y0, y1 = snap(y0, y1, H)
    x0, x1 = snap(x0, x1, W)
    return y0, y1, x0, x1


def _normalize_ocid_labels(anno):
    """Rebase OCID label ids that start at 256 instead of 1.

    `ARID10/floor/top/curved/seq05` (10 of OCID's 2390 images) numbers its uint16 label map
    {0, 256, 257, 258, ...} where every other sequence uses {0, 1, 2, 3, ...}. Since OCID_BG filters
    the floor by literal value ([0, 1] for floor scenes), 256 slips through and **the entire wooden
    floor becomes a GT "object"** -- 272844 px, 89% of the frame, 1.36 m^2. No model will ever
    segment that, so each of those 10 images carries a guaranteed false negative.

    Verified as the only affected sequence: the smallest nonzero label is 1 in 2380 images and 256 in
    exactly these 10. Rebasing by 255 maps 256->1 (floor, i.e. background) and 257->2 (first object),
    which is precisely the convention the other sequences use; label counts in these files also grow
    one per frame as objects are added, consistent with 256 being the surface and 257+ the objects."""
    nz = anno[anno > 0]
    if nz.size and int(nz.min()) >= 256:
        anno = np.where(anno > 0, anno.astype(np.int32) - 255, 0)
    return anno


def _depth_validity_filter(pred_masks, dataset, depth_raw, h, w):
    """Drop masks whose fraction of valid (non-zero) depth pixels < per-dataset threshold
    (UCN / MSMFormer protocol: OCID 0.5, OSD/OCBD 0.8)."""
    thr = DEPTH_FILTER_THRESH.get(dataset, 0.8)
    # 3-channel depth may be a colorized image OR a metric XYZ map (invalid = all-zero point),
    # so treat a pixel valid iff any channel is non-zero; single-channel raw mm is >0.
    valid = np.any(depth_raw != 0, axis=-1) if depth_raw.ndim == 3 else depth_raw > 0
    kept = [m for m in pred_masks if np.sum(m) > 0 and
            np.sum(np.bitwise_and(m > 0, valid)) / np.sum(m > 0) >= thr]
    return np.array(kept) if kept else np.zeros((0, h, w), np.uint8)


def _shift_depth(depth_raw, dx, dy=0):
    """Translate depth by (dx, dy) px to register it onto the RGB frame, filling the vacated
    rows/columns with 0 (= invalid) rather than wrapping. dx<0 moves depth left, dy<0 moves it up.

    dy exists because the adopted OCBD shift was measured horizontally only (median dx=-5 over an
    IQR of [-7,-3]) and a vertical component was never checked -- a stereo RGB/IR baseline is mostly
    horizontal but rarely exactly so.
    """
    if dx == 0 and dy == 0:
        return depth_raw
    out = np.zeros_like(depth_raw)
    ys = slice(dy, None) if dy > 0 else slice(None, depth_raw.shape[0] + dy if dy else None)
    ysrc = slice(None, -dy if dy > 0 else None) if dy > 0 else slice(-dy, None)
    if dx > 0:
        out[ys, dx:] = depth_raw[ysrc, :-dx]
    elif dx < 0:
        out[ys, :dx] = depth_raw[ysrc, -dx:]
    else:
        out[ys, :] = depth_raw[ysrc, :]
    return out


def _depth_shift_for(dataset):
    """(dx, dy) for this dataset. DAFUSION_DEPTH_SHIFT="dx,dy" (or "dx") overrides the table so the
    registration can be swept without editing code; unset keeps the adopted value."""
    env = os.environ.get("DAFUSION_DEPTH_SHIFT")
    if env:
        parts = [int(p) for p in env.split(",")]
        return (parts[0], parts[1] if len(parts) > 1 else 0)
    return (DATASET_DEPTH_SHIFT.get(dataset, 0), 0)


def _predict_raw(predictor, dataset, item, w, h, input_type):
    """Load one benchmark image + run inference ONCE; return the untouched raw pieces
    (rgb_img, depth_raw, anno, instances) so callers can threshold/post-process/score
    without re-running the model. Split out of `_eval_one` so a threshold sweep (see
    `scripts/analyze_ocid_fp.py`) can reuse a single forward pass across many thresholds."""
    rgb_path, depth_path, anno_path, scene = item
    rgb_img = cv2.resize(cv2.imread(rgb_path), (w, h))
    # xyz / depth_normals on OCBD need the metric point cloud (.npy), not the colorized depth PNG.
    if getattr(predictor, "depth_encoding", None) in ("xyz", "depth_normals") and dataset == "ocbd":
        depth_raw = _load_ocbd_xyz(depth_path, w, h)
    else:
        depth_raw = cv2.resize(imageio.imread(depth_path).astype(np.float32), (w, h),
                               interpolation=cv2.INTER_NEAREST)
    if os.environ.get("DAFUSION_ALIGN_DEPTH", "0") == "1":
        depth_raw = _shift_depth(depth_raw, *_depth_shift_for(dataset))
    anno = cv2.resize(imageio.imread(anno_path), (w, h), interpolation=cv2.INTER_NEAREST)
    if dataset == "ocid":
        anno = _normalize_ocid_labels(anno)
    bg = OCID_BG[scene] if dataset == "ocid" else [BACKGROUND_LABEL]
    anno = np.where(np.isin(anno, bg), 0, anno)
    _mgp, _mgc = os.environ.get("DAFUSION_MIN_GT_PIXELS"), os.environ.get("DAFUSION_MIN_GT_CM2")
    if _mgp or _mgc:
        amap = None
        if _mgc:
            try:
                amap = pixel_area_cm2(dataset, depth_raw, w, h)
            except ValueError as e:      # e.g. OCBD on a non-metric `normalized` config
                if not _predict_raw._warned:
                    print(colored(f"DAFUSION_MIN_GT_CM2 ignored: {e}", "yellow"))
                    _predict_raw._warned = True
        anno, _ = _drop_gt_slivers(anno, int(_mgp) if _mgp else None,
                                   float(_mgc) if _mgc and amap is not None else None, amap)

    # DAFUSION_CROP_TO_VALID=1: run inference on only the region where RGB **and** depth both exist,
    # so the two modalities are one-to-one at the input, then paste the masks back into full-frame
    # coordinates so scoring against the untouched GT stays comparable. Without this the model is fed
    # the whole RGB frame including a border band with no depth (OCID ~33 dead columns, OSD ~66 plus
    # all top rows) and predicts into it, where OCID's GT never is -- unmatched by construction.
    if os.environ.get("DAFUSION_CROP_TO_VALID") == "1" and input_type != "rgb":
        y0, y1, x0, x1 = valid_depth_bbox(depth_raw)
        if (y1 - y0) > 32 and (x1 - x0) > 32 and (y1 - y0, x1 - x0) != depth_raw.shape[:2]:
            inst = predictor(rgb_img[y0:y1, x0:x1], depth_raw[y0:y1, x0:x1])
            full_hw = (rgb_img.shape[0], rgb_img.shape[1])
            for fld in ("pred_masks", "soft_masks"):
                if inst.has(fld):
                    sub = inst.get(fld)
                    out = torch.zeros((sub.shape[0],) + full_hw, dtype=sub.dtype)
                    out[:, y0:y1, x0:x1] = sub
                    inst.set(fld, out)
            if inst.has("pred_boxes"):
                inst.pred_boxes.tensor[:, [0, 2]] += x0
                inst.pred_boxes.tensor[:, [1, 3]] += y0
            inst._image_size = full_hw
            return rgb_img, depth_raw, anno, inst

    instances = predictor(rgb_img, depth_raw if input_type != "rgb" else None)
    return rgb_img, depth_raw, anno, instances


_predict_raw._warned = False


def _finalize(dataset, instances, threshold, w, h, rgb_img, depth_raw, anno,
              fg_filter="none", fg_model=None):
    """Apply a score threshold + post-processing + foreground filter to already-computed
    raw `instances`, then score the resulting labelmap against GT. Returns (pred, metrics)."""
    # Optional searched pipeline (Track 11). DAFUSION_POSTPROC_V2 points at a JSON config; it needs
    # DAFUSION_KEEP_SOFT_MASKS=1 because it thresholds the raw probabilities itself rather than
    # inheriting instance_inference's hardcoded sigmoid 0.5. Unset -> the legacy path below, byte
    # identical.
    _v2 = os.environ.get("DAFUSION_POSTPROC_V2")
    if _v2:
        from dafusion.eval import postproc_v2 as pv2
        cfg_v2 = json.load(open(_v2)) if os.path.exists(_v2) else json.loads(_v2)
        if isinstance(cfg_v2.get(dataset), dict):      # optional per-dataset override block
            cfg_v2 = dict(cfg_v2[dataset])
        cfg_v2 = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg_v2.items()}
        if not instances.has("soft_masks"):
            raise RuntimeError("DAFUSION_POSTPROC_V2 needs soft masks; set DAFUSION_KEEP_SOFT_MASKS=1")
        soft = instances.soft_masks.numpy().astype(np.float32)
        z = (depth_raw[..., 2] if depth_raw.ndim == 3 else depth_raw.astype(np.float32) / 1000.0)
        z = z.astype(np.float32)
        valid = np.any(depth_raw != 0, axis=-1) if depth_raw.ndim == 3 else depth_raw > 0
        try:
            amap = pixel_area_cm2(dataset, depth_raw, w, h)
        except ValueError:
            amap = None
        plane = pv2.fit_support_plane(z) if cfg_v2.get("plane_reject") else None
        pred = pv2.run(soft, instances.scores.numpy(), dataset, cfg_v2, area_cm2_map=amap,
                       depth_valid=valid, shape=anno.shape, depth_z=z, plane=plane)
        pred = pred.astype(anno.dtype)
        metrics, _ = compute_PRF.multilabel_metrics(pred, anno, rgb_img, rgb_img, return_assign=True)
        return pred, metrics

    keep = instances.scores > threshold
    pred_masks = instances.pred_masks[keep].numpy().astype(np.uint8)
    bboxes = instances.pred_boxes.tensor[keep].numpy() if instances.has("pred_boxes") else \
        np.zeros((len(pred_masks), 4))
    area_map = (pixel_area_cm2(dataset, depth_raw, w, h)
                if os.environ.get("DAFUSION_METRIC_SIZE_GATE") else None)
    pred_masks, bboxes = pp.post_image_process(pred_masks, bboxes, w, h, rgb_img,
                                               area_cm2_map=area_map)

    if fg_filter == "cgnet" and fg_model is not None:
        pred_masks = _cgnet_filter(pred_masks, rgb_img, depth_raw, w, h, fg_model)
    elif fg_filter == "depth":
        pred_masks = _depth_validity_filter(pred_masks, dataset, depth_raw, h, w)

    pred = np.zeros_like(anno)
    for idx, mask in enumerate(pred_masks):
        pred[mask > 0] = idx + 1

    # NOTE: multilabel_metrics returns a (metrics, assignments) tuple on every path
    # (its edge-case branches always append `, []`), so unpack unconditionally.
    metrics, _ = compute_PRF.multilabel_metrics(pred, anno, rgb_img, rgb_img, return_assign=True)
    return pred, metrics


def _eval_one(predictor, dataset, item, w, h, input_type, fg_filter="none", fg_model=None):
    """Run the full per-image pipeline for one benchmark image; return
    (rgb_img, pred_labelmap, gt_labelmap, metrics). fg_filter in {none, cgnet, depth}."""
    rgb_img, depth_raw, anno, instances = _predict_raw(predictor, dataset, item, w, h, input_type)
    threshold = DATASET_SCORE_THRESH.get(dataset, SCORE_THRESHOLD)
    pred, metrics = _finalize(dataset, instances, threshold, w, h, rgb_img, depth_raw, anno,
                              fg_filter, fg_model)
    return rgb_img, pred, anno, metrics


def _mask_color(i):
    """Deterministic BGR color for instance id i."""
    hsv = np.uint8([[[(i * 47) % 180, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _overlay(base_bgr, labelmap):
    vis = base_bgr.copy()
    for i in np.unique(labelmap):
        if i == 0:
            continue
        m = labelmap == i
        color = _mask_color(int(i))
        vis[m] = (0.5 * vis[m] + 0.5 * np.array(color)).astype(np.uint8)
        cnts, _ = cv2.findContours((m * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 1)
    return vis


def _render_viz(rgb, pred, anno, title, out_path):
    """Save a side-by-side [GT | Prediction] overlay panel (both drawn on the RGB image, so
    the raw RGB itself is omitted to save space). Written as JPEG."""
    h, w = rgb.shape[:2]
    panel = np.hstack([_overlay(rgb, anno), _overlay(rgb, pred)])
    for x, txt in ((0, "GT"), (w, "Pred")):
        cv2.putText(panel, txt, (x + 6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(panel, title, (6, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.imwrite(out_path, panel, [cv2.IMWRITE_JPEG_QUALITY, 90])


def _save_representative_viz(predictor, dataset, records, w, h, input_type, fg_filter, fg_model, viz_dir, k):
    """Pick k good / k medium / k bad images by Overlap-F and save prediction panels.
    `records` is a list of (overlap_f, item). Re-runs inference only for the ~3k selected."""
    records = sorted(records, key=lambda r: r[0])          # ascending Overlap-F
    n = len(records)
    if n == 0:
        return
    k = min(k, n)
    mid = max(0, n // 2 - k // 2)
    groups = {"good": records[-k:][::-1], "bad": records[:k], "medium": records[mid:mid + k]}
    out = os.path.join(viz_dir, dataset)
    os.makedirs(out, exist_ok=True)
    for label, grp in groups.items():
        for rank, (score, item) in enumerate(grp, 1):
            rgb, pred, anno, _ = _eval_one(predictor, dataset, item, w, h, input_type, fg_filter, fg_model)
            name = os.path.splitext(os.path.basename(item[0]))[0]
            _render_viz(rgb, pred, anno, f"{dataset.upper()} {label} OverlapF={score:.3f}",
                        os.path.join(out, f"{label}_{rank}_F{score:.3f}_{name}.jpg"))
    print(colored(f"[viz] {dataset.upper()}: wrote good/medium/bad samples -> {out}/", "cyan"))


def run_benchmark(predictor, dataset, input_type, fg_filter="none", fg_model=None,
                  save_viz=0, viz_dir=None):
    """Evaluate an already-loaded predictor on one benchmark; return per-image metrics.

    Reused by both the CLI (`evaluate`) and the multi-checkpoint sweep so the exact same
    per-image pipeline and UOIS PRF metric back every number. `predictor` is a
    DAFusionPredictor; `fg_filter` in {none, cgnet, depth} (cgnet needs `fg_model`).
    If save_viz>0, also saves that many good/medium/bad prediction panels to viz_dir.
    Returns a list of per-image metric dicts (average with `average_metrics`).
    """
    items = LISTERS[dataset](DATASET_PATHS[dataset])
    w, h = DATASET_WH.get(dataset, (W, H))   # per-dataset eval resolution (OCBD=600x400)
    print(colored(f"Evaluation on {dataset.upper()}: {len(items)} images @ {w}x{h} "
                  f"(fg_filter={fg_filter})", "green"))

    metrics_all, viz_records = [], []
    # mininterval=5 keeps refreshes sparse so parallel sweep workers don't spam a shared log.
    for item in tqdm(items, desc=dataset.upper(), unit="img", mininterval=5.0):
        _, _, _, metrics = _eval_one(predictor, dataset, item, w, h, input_type, fg_filter, fg_model)
        metrics_all.append(metrics)
        if save_viz:
            viz_records.append((metrics.get("Objects F-measure", 0.0), item))

    if save_viz and viz_dir:
        try:
            _save_representative_viz(predictor, dataset, viz_records, w, h, input_type,
                                     fg_filter, fg_model, viz_dir, save_viz)
        except Exception as e:
            print(colored(f"[viz] WARNING failed to save samples: {e}", "yellow"))

    return metrics_all


def average_metrics(metrics_all):
    """Mean each metric key over the per-image list. Empty list -> empty dict."""
    keys = list(metrics_all[0].keys()) if metrics_all else []
    return {k: float(np.mean([m[k] for m in metrics_all])) for k in keys}


def evaluate(args):
    cfg = build_cfg(args)
    predictor = DAFusionPredictor(cfg, dataset=args.dataset)
    fg_filter = "cgnet" if args.use_cgnet else args.fg_filter   # --use_cgnet is legacy alias
    fg_model = load_cgnet(args.cgnet_weight) if fg_filter == "cgnet" else None
    viz_dir = None
    if args.save_viz:
        run_tag = os.path.basename(os.path.dirname(args.weights)) if args.weights else "model"
        viz_dir = args.viz_dir or os.path.join(BENCHMARK_RESULT_ROOT, "viz", run_tag)
    metrics_all = run_benchmark(predictor, args.dataset, args.input_type, fg_filter=fg_filter,
                                fg_model=fg_model, save_viz=args.save_viz, viz_dir=viz_dir)
    report(args, metrics_all)


def report(args, metrics_all):
    keys = list(metrics_all[0].keys()) if metrics_all else []
    avg = average_metrics(metrics_all)
    op, orr, of = avg.get("Objects Precision", 0), avg.get("Objects Recall", 0), avg.get("Objects F-measure", 0)
    bp, br, bf = avg.get("Boundary Precision", 0), avg.get("Boundary Recall", 0), avg.get("Boundary F-measure", 0)
    p75 = avg.get("obj_detected_075_percentage", 0)
    print(colored(f"\n=== DA-Fusion on {args.dataset.upper()} ({len(metrics_all)} images) ===", "green", attrs=["bold"]))
    print("    Overlap    |    Boundary")
    print("  P    R    F  |   P    R    F  |  %75")
    print(f"{op*100:.1f} {orr*100:.1f} {of*100:.1f} | {bp*100:.1f} {br*100:.1f} {bf*100:.1f} | {p75*100:.1f}")
    os.makedirs(BENCHMARK_RESULT_ROOT, exist_ok=True)
    out = os.path.join(BENCHMARK_RESULT_ROOT, f"{args.dataset}_{os.path.basename(os.path.dirname(args.weights or 'model'))}.txt")
    with open(out, "w") as f:
        f.write(f"DA-Fusion {args.dataset.upper()}  OverlapF {of*100:.2f}  BoundaryF {bf*100:.2f}  %75 {p75*100:.2f}\n")
        for k in keys:
            f.write(f"{k}: {avg[k]:.4f}\n")
    print(">>> wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser("DA-Fusion UOIS benchmark eval")
    ap.add_argument("--dataset", required=True, choices=["osd", "ocid", "ocbd"])
    ap.add_argument("--input_type", default="rgbd", choices=["rgb", "depth", "rgbd"])
    ap.add_argument("--config", default="configs/dafusion_rgbd_uoais.yaml")
    ap.add_argument("--weights", default=None, help="checkpoint .pth (default: config MODEL.WEIGHTS)")
    ap.add_argument("--fg_filter", default="depth", choices=["none", "cgnet", "depth"],
                    help="foreground filter: none | cgnet (UOAIS) | depth (UCN/MSMFormer depth-validity)")
    ap.add_argument("--use_cgnet", action="store_true", help="legacy alias for --fg_filter cgnet")
    ap.add_argument("--cgnet_weight", default=CGNET_WEIGHTS)
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--save_viz", type=int, default=0, metavar="K",
                    help="save K good + K medium + K bad prediction panels (RGB|GT|Pred)")
    ap.add_argument("--viz_dir", default=None, help="where to write viz (default: benchmark_result/viz/<run>)")
    args = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    evaluate(args)
