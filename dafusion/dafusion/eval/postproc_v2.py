"""Parameterised post-processing pipeline, replacing the legacy fixed algorithm in `post_process.py`.

The legacy chain is worth +4 F but its constants are untuned and two of its design choices are
actively wrong for the metric being optimised:

  * `instance_inference` binarises masks at sigmoid 0.5, hardcoded. The UOIS metric is PIXEL-weighted
    (`recall = TP_px / GT_px`), so this single threshold is the most direct P/R knob in the pipeline.
  * `benchmark._finalize` resolves overlaps with `pred[mask>0] = idx+1`, i.e. LAST MASK WINS in
    whatever order `topk(..., sorted=False)` returned. The winner of an overlap is arbitrary rather
    than the more confident mask.

Every stage here is a named, swappable option so the whole pipeline can be searched offline against
cached raw predictions (see `scripts/dump_raw_preds.py`, `scripts/postproc_sweep.py`).

Stage order matches the live pipeline: threshold -> dedup -> cleanup -> size gate -> foreground
filter -> label assignment. `CONFIG_LEGACY` reproduces today's behaviour exactly and is the fidelity
guard for the offline harness.
"""
import cv2
import numpy as np

# Today's pipeline, expressed in this parameterisation. Used to prove the offline harness reproduces
# the live numbers (OCID 86.4 / OSD 93.2 / OCBD 85.3) before any sweep result is believed.
CONFIG_LEGACY = dict(
    score_thresh=None,        # None -> per-dataset default (OCID 0.9, else 0.5)
    mask_thresh=0.5,          # sigmoid; the hardcoded value in instance_inference
    min_mask_px=10,
    dedup="union0.7",         # legacy merge_overlaps
    cleanup="legacy",         # refined_mask's fragment pruning (>150 px, centre within 100 px)
    size_gate="pixel",        # [500, 40000] px on the largest contour
    fg_filter="depth",        # per-dataset depth-validity threshold
    depth_valid_thresh=None,  # None -> per-dataset default (OCID 0.5, OSD/OCBD 0.8)
    assign="last_wins",       # arbitrary-order overwrite
)

# The metric size gate validated in Track 10s: +0.50 mean, removes 0 real GT on OCID.
CONFIG_BASE = dict(CONFIG_LEGACY, size_gate="cm2", size_cm2=(2.0, 1000.0))

DEFAULT_SCORE_THRESH = {"ocid": 0.9, "osd": 0.5, "ocbd": 0.5}
DEFAULT_DEPTH_VALID = {"ocid": 0.5, "osd": 0.8, "ocbd": 0.8}
LEGACY_PIXEL_BAND = (500.0, 40000.0)


# ---------------------------------------------------------------- dedup / suppression

def _mask_ious(masks, areas):
    n = len(masks)
    iou = np.zeros((n, n), np.float32)
    for i in range(n):
        for j in range(i + 1, n):
            inter = np.logical_and(masks[i], masks[j]).sum()
            if inter:
                iou[i, j] = iou[j, i] = inter / (areas[i] + areas[j] - inter)
    return iou


def dedup_masks(masks, scores, mode, iou_thresh=0.7, contain_thresh=0.7):
    """Return indices to keep, plus optional unions. Modes:
      none        - keep everything
      union0.7    - LEGACY: if intersection/min(area) > 0.7, union j into i and drop j
      nms         - classic mask NMS: drop the lower-scoring of any pair with IoU > iou_thresh
      nms_contain - NMS, but also drop a mask mostly CONTAINED in a higher-scoring one. Duplicate
                    queries in Mask2Former often differ in extent rather than position, so IoU alone
                    misses a small mask sitting inside a large one.
    """
    n = len(masks)
    if n == 0 or mode == "none":
        return list(range(n)), masks
    areas = np.array([int(m.sum()) for m in masks], np.int64)

    if mode == "union0.7":
        # Replicates merge_overlaps: greedy in array order (NOT score order), unions into the earlier
        # index. Order-dependence is part of the legacy behaviour and is reproduced deliberately.
        out = [m.copy() for m in masks]
        cur = areas.copy()
        merged = set()
        for i in range(n):
            if i in merged:
                continue
            for j in range(i + 1, n):
                if j in merged:
                    continue
                inter = np.logical_and(out[i], out[j]).sum()
                if inter and inter / min(cur[i], cur[j]) > contain_thresh:
                    out[i] |= out[j]
                    cur[i] = out[i].sum()
                    merged.add(j)
        keep = [i for i in range(n) if i not in merged]
        return keep, [out[i] for i in keep]

    order = np.argsort(-scores)
    iou = _mask_ious(masks, areas)
    dead = set()
    for a_idx, i in enumerate(order):
        if i in dead:
            continue
        for j in order[a_idx + 1:]:
            if j in dead:
                continue
            if iou[i, j] > iou_thresh:
                dead.add(j)
            elif mode == "nms_contain":
                inter = np.logical_and(masks[i], masks[j]).sum()
                if inter and inter / max(areas[j], 1) > contain_thresh:
                    dead.add(j)
    keep = [i for i in order if i not in dead]
    return keep, [masks[i] for i in keep]


# ---------------------------------------------------------------- per-mask cleanup

def cleanup_mask(m, mode, min_frag_px=150, max_centre_dist=100.0, fill_holes=False):
    """Prune a mask to a coherent object. Modes:
      none     - unchanged
      legacy   - refined_mask's rule: keep contours >min_frag_px whose bbox centre is within
                 max_centre_dist of the largest contour's centre
      largest  - keep only the largest connected component
    """
    if mode == "none" and not fill_holes:
        return m
    mm = np.ascontiguousarray(m, np.uint8)

    if mode == "largest":
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mm, connectivity=8)
        if n > 1:
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            mm = (lab == k).astype(np.uint8)
    elif mode == "legacy":
        cnts = cv2.findContours(mm, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
        if cnts:
            mx = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(mx)
            c0 = np.array([x + w / 2.0, y + h / 2.0])
            keep = np.zeros_like(mm)
            for c in cnts:
                if cv2.contourArea(c) <= min_frag_px:
                    continue
                bx, by, bw, bh = cv2.boundingRect(c)
                if np.linalg.norm(c0 - np.array([bx + bw / 2.0, by + bh / 2.0])) < max_centre_dist:
                    cv2.drawContours(keep, [c], 0, 1, -1)
            mm = mm & keep

    if fill_holes:
        # Fill interior holes: flood the complement from the border with 2, so background reachable
        # from outside becomes 2 and only enclosed holes keep the value 1. The mask is then
        # (original OR holes) -- note `inv == 1`, not `!= 1`: the inverted test marks everything
        # EXCEPT the holes and blows the mask up to nearly the whole frame (measured mean F 1.35).
        h_, w_ = mm.shape
        ff = np.zeros((h_ + 2, w_ + 2), np.uint8)
        inv = (1 - mm).astype(np.uint8)
        cv2.floodFill(inv, ff, (0, 0), 2)
        mm = ((inv == 1) | (mm > 0)).astype(np.uint8)
    return mm.astype(bool)


# ---------------------------------------------------------------- size gate

def size_ok(m, mode, area_cm2_map=None, pixel_band=LEGACY_PIXEL_BAND, cm2_band=(2.0, 1000.0)):
    """Is this mask within the plausible-object size band? See Track 10p/10r/10s for why cm^2 beats
    pixels: pixel area falls as 1/z^2 and scales with resolution, so [500,40000] px is 17-1382 cm^2
    on OCID but only 6-465 on OCBD, deleting real 28 cm dinner plates there."""
    if mode == "none":
        return True
    mm = np.ascontiguousarray(m, np.uint8)
    cnts = cv2.findContours(mm, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)[0]
    if not cnts:
        return False
    mx = max(cnts, key=cv2.contourArea)
    if mode == "pixel" or area_cm2_map is None:
        a = cv2.contourArea(mx)
        return pixel_band[0] <= a <= pixel_band[1]

    comp = np.zeros(mm.shape, np.uint8)
    cv2.drawContours(comp, [mx], 0, 1, -1)
    comp = (comp > 0) & (mm > 0)
    npix = int(comp.sum())
    good = comp & (area_cm2_map > 0)
    nv = int(good.sum())
    if nv == 0:                                    # no depth -> fall back to pixels
        a = cv2.contourArea(mx)
        return pixel_band[0] <= a <= pixel_band[1]
    a_cm2 = float(area_cm2_map[good].sum()) * npix / nv
    return cm2_band[0] <= a_cm2 <= cm2_band[1]


# ---------------------------------------------------------------- support-plane rejection

def fit_support_plane(z, iters=120, tol_m=0.012, min_frac=0.15, rng_seed=0):
    """RANSAC the dominant plane (table / floor / bin bottom) from a depth map.

    Trick that avoids needing intrinsics: a 3D plane aX+bY+cZ=d with X=(u-cx)Z/fx becomes
    **1/Z = alpha*u + beta*v + gamma**, exactly linear in pixel coordinates. So fitting inverse depth
    against (u,v) fits a true 3D plane, which matters because intrinsics were wrong for OCBD until
    tonight and are only "documented best-effort" elsewhere.

    Returns (z_plane map, inlier mask) or (None, None) if no plane covers >= min_frac of valid pixels.
    """
    valid = z > 0
    n_valid = int(valid.sum())
    if n_valid < 2000:
        return None, None
    h, w = z.shape
    v_idx, u_idx = np.nonzero(valid)
    inv = 1.0 / z[valid]
    # subsample for speed; a support surface is large so sampling does not hide it
    if n_valid > 20000:
        sub = np.random.RandomState(rng_seed).choice(n_valid, 20000, replace=False)
        u_s, v_s, inv_s = u_idx[sub], v_idx[sub], inv[sub]
    else:
        u_s, v_s, inv_s = u_idx, v_idx, inv
    A = np.stack([u_s, v_s, np.ones_like(u_s)], 1).astype(np.float64)
    rng = np.random.RandomState(rng_seed)
    best_cnt, best_p = 0, None
    for _ in range(iters):
        i = rng.choice(len(A), 3, replace=False)
        try:
            p = np.linalg.solve(A[i], inv_s[i])
        except np.linalg.LinAlgError:
            continue
        zp = A @ p
        with np.errstate(divide="ignore", invalid="ignore"):
            zz = 1.0 / zp
        ok = np.isfinite(zz) & (np.abs(zz - 1.0 / inv_s) < tol_m)
        c = int(ok.sum())
        if c > best_cnt:
            best_cnt, best_p = c, p
    if best_p is None or best_cnt < min_frac * len(A):
        return None, None
    # refit on inliers for stability
    zp = A @ best_p
    with np.errstate(divide="ignore", invalid="ignore"):
        ok = np.isfinite(1.0 / zp) & (np.abs(1.0 / zp - 1.0 / inv_s) < tol_m)
    if ok.sum() >= 3:
        best_p = np.linalg.lstsq(A[ok], inv_s[ok], rcond=None)[0]
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    full = best_p[0] * uu + best_p[1] * vv + best_p[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z_plane = np.where(full > 1e-6, 1.0 / full, 0.0)
    inliers = valid & (z_plane > 0) & (np.abs(z - z_plane) < tol_m)
    return z_plane.astype(np.float32), inliers


def on_plane_fraction(m, z, z_plane, tol_m):
    """Fraction of a mask's valid-depth pixels lying on the support plane."""
    g = m & (z > 0) & (z_plane > 0)
    n = int(g.sum())
    if n == 0:
        return 0.0
    return float((np.abs(z[g] - z_plane[g]) < tol_m).sum()) / n


# ---------------------------------------------------------------- depth-aware boundary snap

def snap_to_depth(m, z, band_px=3, tol_m=0.02):
    """Re-decide only the pixels in a thin band around the mask boundary, by depth agreement with the
    mask's interior.

    Mask2Former predicts masks at 1/4 resolution and upsamples, so boundaries are smooth and bleed
    across depth steps. Object silhouettes ARE depth discontinuities, so a pixel within the band joins
    the mask iff its depth is within tol of the interior's median. This both trims bleed onto the
    background and recovers pixels the upsampled mask missed. Only the band is touched, so a
    well-placed boundary is left alone.
    """
    mm = np.ascontiguousarray(m, np.uint8)
    k = np.ones((3, 3), np.uint8)
    core = cv2.erode(mm, k, iterations=band_px)
    g = (core > 0) & (z > 0)
    if g.sum() < 30:
        return m

    # LOCAL depth reference, not the mask's global median. An extended or tilted object spans a wide
    # depth range (an OCBD plate covers 0.25-0.56 m), so a global median rejects most of its own
    # surface and the mask erodes by band_px everywhere -- measured as Boundary-F 76.6 -> 43.5.
    # Instead compare each band pixel to the mean depth of nearby CORE pixels, via a box filter over
    # the core normalised by core coverage.
    win = 4 * band_px + 1
    zc = np.where(g, z, 0.0).astype(np.float32)
    num = cv2.boxFilter(zc, -1, (win, win), normalize=False)
    den = cv2.boxFilter(g.astype(np.float32), -1, (win, win), normalize=False)
    z_ref = np.where(den > 0, num / np.maximum(den, 1e-6), 0.0)

    band = (cv2.dilate(mm, k, iterations=band_px) > 0) & (core == 0)
    out = core.astype(bool)
    out |= band & (z > 0) & (z_ref > 0) & (np.abs(z - z_ref) < tol_m)
    # where depth or a local reference is unavailable, keep the original assignment rather than
    # silently eroding the mask
    out |= band & ((z <= 0) | (z_ref <= 0)) & (mm > 0)
    return out


# ---------------------------------------------------------------- depth-discontinuity split

def split_on_depth(m, z, tol_m=0.03, min_part_px=400):
    """Split one mask into several instances when its pixels form depth-separated clusters.

    Targets under-segmentation: two touching objects at different depths merged into one query. The
    mask's depth values are clustered by a 1-D gap search (sorted depths, cut at any gap > tol), then
    each cluster's largest connected component becomes an instance. Returns [m] unchanged if no clean
    split exists.
    """
    g = m & (z > 0)
    n = int(g.sum())
    if n < 2 * min_part_px:
        return [m]
    zv = np.sort(z[g])
    gaps = np.diff(zv)
    if gaps.size == 0 or gaps.max() < tol_m:
        return [m]
    cut = zv[int(np.argmax(gaps))] + gaps.max() / 2.0
    lo, hi = m & (z > 0) & (z <= cut), m & (z > 0) & (z > cut)
    parts = []
    for part in (lo, hi):
        if part.sum() < min_part_px:
            return [m]
        n_cc, lab, stats, _ = cv2.connectedComponentsWithStats(
            np.ascontiguousarray(part, np.uint8), connectivity=8)
        if n_cc <= 1:
            return [m]
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[k, cv2.CC_STAT_AREA] < min_part_px:
            return [m]
        parts.append(lab == k)
    return parts


# ---------------------------------------------------------------- label assignment

def assign_labels(masks, scores, soft, shape, mode):
    """Build the integer labelmap scored by the metric. Modes:
      last_wins  - LEGACY: pred[mask>0] = idx+1 in array order; an overlapped pixel goes to whichever
                   mask happens to come last. Arbitrary, since topk() is unsorted.
      score_sort - same overwrite but ascending by score, so the MOST confident mask wins overlaps
      argmax     - assign each contested pixel to argmax_q score_q * prob_q(pixel). Principled for a
                   pixel-weighted metric: it splits an overlap along the probability ridge instead of
                   giving all of it to one mask.
    """
    lab = np.zeros(shape, np.int32)
    if not len(masks):
        return lab
    if mode == "last_wins":
        for i, m in enumerate(masks):
            lab[m] = i + 1
        return lab
    if mode == "score_sort":
        for i in np.argsort(scores):             # ascending -> highest score written last
            lab[masks[i]] = int(i) + 1
        return lab
    # argmax
    best = np.zeros(shape, np.float32)
    for i, m in enumerate(masks):
        w = (soft[i] if soft is not None else np.ones(shape, np.float32)) * float(scores[i])
        upd = m & (w > best)
        best[upd] = w[upd]
        lab[upd] = i + 1
    return lab


# ---------------------------------------------------------------- full pipeline

def run_traced(soft_masks, scores, dataset, cfg, **kw):
    """Same as `run` but also returns the surviving masks after each stage, for visualising how raw
    predictions become the final labelmap (`scripts/viz_postproc_stages.py`)."""
    trace = []
    lab = run(soft_masks, scores, dataset, cfg, _trace=trace, **kw)
    return lab, trace


def run(soft_masks, scores, dataset, cfg, area_cm2_map=None, depth_valid=None, shape=None,
        depth_z=None, plane=None, _trace=None):
    """soft_masks: (Q,H,W) float32 probabilities. Returns an integer labelmap.

    `depth_valid` / `area_cm2_map` / `depth_z` / `plane` are config-independent per-image quantities
    passed in so a sweep computes them once rather than per config. `plane` is
    (z_plane, inliers) from `fit_support_plane`.
    """
    shape = shape or soft_masks.shape[-2:]
    st = cfg.get("score_thresh")
    st = DEFAULT_SCORE_THRESH.get(dataset, 0.5) if st is None else st
    if _trace is not None:
        _trace.append((f"raw ({len(scores)} queries cached)",
                       [soft_masks[i] > 0.5 for i in range(len(scores))]))
    sel = np.nonzero(scores > st)[0]
    if len(sel) == 0:
        if _trace is not None:
            _trace.append((f"score > {st}", []))
        return np.zeros(shape, np.int32)

    mt = cfg.get("mask_thresh", 0.5)
    masks = [soft_masks[i] > mt for i in sel]
    sc = scores[sel]
    if _trace is not None:
        _trace.append((f"score > {st}  +  mask prob > {mt}", [m.copy() for m in masks]))

    # prune specks (legacy does this first, before dedup)
    keep = [i for i, m in enumerate(masks) if m.sum() > cfg.get("min_mask_px", 10)]
    masks = [masks[i] for i in keep]
    sc = sc[keep]
    sel = sel[keep]
    if _trace is not None:
        _trace.append((f"prune < {cfg.get('min_mask_px', 10)} px", [m.copy() for m in masks]))
    if not masks:
        return np.zeros(shape, np.int32)

    kept, masks = dedup_masks(masks, sc, cfg.get("dedup", "union0.7"),
                              cfg.get("nms_iou", 0.7), cfg.get("contain_thresh", 0.7))
    sc = sc[kept]
    sel = sel[kept]
    if _trace is not None:
        _trace.append((f"dedup = {cfg.get('dedup', 'union0.7')}", [m.copy() for m in masks]))

    # optional: split under-segmented masks along a depth gap (before cleanup, so each part is then
    # cleaned and size-gated on its own)
    if cfg.get("split_depth") and depth_z is not None:
        s_m, s_s, s_i = [], [], []
        for m, s, i in zip(masks, sc, sel):
            for p in split_on_depth(m, depth_z, cfg.get("split_tol_m", 0.03),
                                    cfg.get("split_min_px", 400)):
                s_m.append(p); s_s.append(s); s_i.append(i)
        masks, sc, sel = s_m, np.array(s_s, np.float32), np.array(s_i, np.int32)

    out_m, out_s, out_i = [], [], []
    for m, s, i in zip(masks, sc, sel):
        if cfg.get("snap_depth") and depth_z is not None:
            m = snap_to_depth(m, depth_z, cfg.get("snap_band_px", 3), cfg.get("snap_tol_m", 0.02))
        m = cleanup_mask(m, cfg.get("cleanup", "legacy"), cfg.get("min_frag_px", 150),
                         cfg.get("max_centre_dist", 100.0), cfg.get("fill_holes", False))
        if m.sum() == 0:
            continue
        if not size_ok(m, cfg.get("size_gate", "pixel"), area_cm2_map,
                       cfg.get("size_pixel", LEGACY_PIXEL_BAND), cfg.get("size_cm2", (2.0, 1000.0))):
            continue
        # optional: drop masks that are mostly the support surface (table / floor / bin bottom)
        if cfg.get("plane_reject") and plane is not None and plane[0] is not None and depth_z is not None:
            if on_plane_fraction(m, depth_z, plane[0], cfg.get("plane_tol_m", 0.012)) \
                    > cfg.get("plane_max_frac", 0.7):
                continue
        out_m.append(m)
        out_s.append(s)
        out_i.append(i)

    if _trace is not None:
        _trace.append((f"cleanup={cfg.get('cleanup','legacy')} + size gate"
                       f"{' + plane reject' if cfg.get('plane_reject') else ''}",
                       [m.copy() for m in out_m]))

    if cfg.get("fg_filter", "depth") == "depth" and depth_valid is not None and out_m:
        thr = cfg.get("depth_valid_thresh")
        thr = DEFAULT_DEPTH_VALID.get(dataset, 0.8) if thr is None else thr
        f_m, f_s, f_i = [], [], []
        for m, s, i in zip(out_m, out_s, out_i):
            tot = int(m.sum())
            if tot and np.logical_and(m, depth_valid).sum() / tot >= thr:
                f_m.append(m); f_s.append(s); f_i.append(i)
        out_m, out_s, out_i = f_m, f_s, f_i
        if _trace is not None:
            _trace.append((f"depth-validity >= {thr}", [m.copy() for m in out_m]))

    # Optional: TRIM each mask to the valid-depth region (not the same as the validity FILTER, which
    # drops whole masks). ORDER MATTERS AND COST 1.74 F WHEN WRONG: trimming BEFORE the validity
    # filter makes every surviving mask 100% valid by construction, so the filter can no longer drop
    # anything and spurious masks in holed regions survive (OCID 88.77 -> 87.03). It must run AFTER.
    # Justified per dataset by where GT actually lives, measured over
    # 150 images/set -- fraction of GT pixels falling outside valid depth:
    #     OCID 0.00%   OSD 3.06%   OCBD 0.00% (OCBD has no invalid depth at all)
    # OCID's labels were annotated on the depth-registered frame, so GT never leaves the valid region
    # and any predicted pixel outside it is unmatched by construction (measured 0.57% of predicted
    # pixels). OSD's default `annotation_fixed` masks are SAM-corrected and RGB-aligned, so its GT
    # does extend past the depth border and trimming there would delete real object area.
    if cfg.get("trim_to_valid") and depth_valid is not None and out_m:
        t_m, t_s, t_i = [], [], []
        for m, s, i in zip(out_m, out_s, out_i):
            mm = m & depth_valid
            if mm.sum() > 0:
                t_m.append(mm); t_s.append(s); t_i.append(i)
        out_m, out_s, out_i = t_m, t_s, t_i
        if _trace is not None:
            _trace.append(("trim to valid depth", [m.copy() for m in out_m]))


    if not out_m:
        return np.zeros(shape, np.int32)
    soft_sel = soft_masks[np.array(out_i)] if cfg.get("assign") == "argmax" else None
    return assign_labels(out_m, np.array(out_s, np.float32), soft_sel, shape,
                         cfg.get("assign", "last_wins"))
