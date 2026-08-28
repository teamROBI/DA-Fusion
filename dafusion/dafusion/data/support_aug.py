"""Support-surface appearance randomization (train-only, RGB-only).

THE MEASURED PROBLEM (docs/EXPERIMENTS.md Track 10i). UOAIS-Sim contains only `bin` and
`tabletop` scenes. OCID is half `floor` scenes. So **48% of the eval benchmark is a scene type
absent from training**, and it carries **54.9% of all OCID loss**, with the damage specifically in
precision -- 82.0 on floor vs 90.8 on table. The model never learned that a large patterned plane
filling the frame is background, so it invents objects on floor/wall texture (the same pathology as
the wall-trim slivers in the empty-scene viz, but over 1140 images instead of 90). It also explains
why the *most* cluttered OCID scenes score BEST (92.7): objects cover the distracting support
surface.

THE FIX. Re-texture the support plane during training with varied patterns -- tiles, grids, stripes,
multi-scale noise, gradients -- so "large textured plane" stops correlating with "object".

Two invariants that make this safe, and that the whole idea depends on:
  1. **RGB only. Depth is never touched.** The plane must stay geometrically a plane; we are
     teaching "texture on a flat surface is not an object", so corrupting the geometry would teach
     the opposite. This is also why the augmentation cannot simply be generic color jitter (already
     tried and failed, Track 4 color-aug -1.9): the point is *spatial* pattern on a *flat* region.
  2. **GT object pixels are excluded via the masks, not guessed.** A plane-residual test alone
     would classify a flat object lying on the surface (a book, a phone) as support and paint over
     it, destroying the very instance the model must find.
"""
import cv2
import numpy as np


SUPPORT = dict(
    p_apply=0.5,          # fraction of training frames re-textured (keep originals in the mix)
    plane_tol_rel=0.04,   # |z - plane| < tol * median_depth counts as ON the support surface
                          # (0.04 ~= 20 cm at UOAIS-Sim's ~5 m; 0.02 was too tight even with
                          #  an object-excluded fit, leaving most of the table unlabelled)
    min_frac=0.15,        # skip if the detected plane covers < this fraction (bad fit / bin scene)
    dilate_obj=5,         # grow object masks before exclusion, so we never paint an object's edge
    alpha=(0.55, 1.0),    # blend strength range: 1.0 = full replace, <1 keeps some original shading
)


def _fit_plane(depth_mm, rng, exclude=None):
    """Least-squares z ~ a*u + b*v + c (same form as upright.estimate_roll).

    `exclude` (bool mask, e.g. the union of GT object masks) is omitted from the fit. This matters
    enormously: UOAIS-Sim scenes carry 17-21 instances covering 26-50% of pixels, and objects stand
    ABOVE the support surface, so an all-pixel fit is dragged off the plane and only 6-13% of pixels
    then land within tolerance -- the augmentation silently no-ops. Fitting on background pixels
    only recovers the real support surface.
    """
    usable = depth_mm > 0
    if exclude is not None:
        usable = usable & ~exclude
    vv, uu = np.nonzero(usable)
    if len(uu) < 500:
        return None
    z = depth_mm[vv, uu].astype(np.float64)
    if len(z) > 20000:
        sel = rng.choice(len(z), 20000, replace=False)
        uu, vv, z = uu[sel], vv[sel], z[sel]
    A = np.stack([uu, vv, np.ones_like(uu)], axis=1).astype(np.float64)
    try:
        (a, b, c), *_ = np.linalg.lstsq(A, z, rcond=None)
    except np.linalg.LinAlgError:
        return None
    return float(a), float(b), float(c)


def _random_texture(h, w, rng):
    """A random background-ish pattern. The tile/grid modes matter most: OCID floors are tiled and
    carpeted, which is exactly the appearance the tabletop-only training set never shows."""
    kind = rng.randint(0, 5)
    base = rng.randint(40, 215, size=3).astype(np.float32)
    img = np.tile(base, (h, w, 1))
    if kind == 0:                                    # tiles / grout grid
        period = int(rng.randint(24, 90)); lw = int(rng.randint(1, 5))
        off = rng.randint(0, period, size=2)
        gy = ((np.arange(h) + off[0]) % period) < lw
        gx = ((np.arange(w) + off[1]) % period) < lw
        grout = np.logical_or(gy[:, None], gx[None, :])
        img[grout] = np.clip(base + rng.randint(-70, 70), 0, 255)
    elif kind == 1:                                  # stripes (wood / carpet nap), random angle
        period = int(rng.randint(8, 40))
        ang = float(rng.uniform(0, np.pi))
        yy, xx = np.mgrid[0:h, 0:w]
        s = ((xx * np.cos(ang) + yy * np.sin(ang)) % period) < period / 2
        img[s] = np.clip(base + rng.randint(-45, 45), 0, 255)
    elif kind == 2:                                  # multi-scale noise (speckled carpet / stone)
        acc = np.zeros((h, w), np.float32)
        for sc in (4, 8, 16, 32):
            small = rng.normal(0, 1, size=(max(2, h // sc), max(2, w // sc))).astype(np.float32)
            acc += cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC) / sc ** 0.5
        acc = acc / (np.abs(acc).max() + 1e-6) * rng.randint(15, 60)
        img = np.clip(img + acc[..., None], 0, 255)
    elif kind == 3:                                  # smooth gradient (lighting falloff)
        yy, xx = np.mgrid[0:h, 0:w]
        g = (xx / max(w - 1, 1)) * rng.uniform(-1, 1) + (yy / max(h - 1, 1)) * rng.uniform(-1, 1)
        img = np.clip(img + (g * rng.randint(20, 70))[..., None], 0, 255)
    # kind == 4 -> flat colour, already set
    return img.astype(np.uint8)


def randomize_support_surface(rgb, depth_mm, masks, rng=np.random):
    """Re-texture the support plane in `rgb`. Returns a new RGB; `depth_mm` is NOT modified.

    rgb:      (H,W,3) uint8
    depth_mm: (H,W) float raw mm (0 = invalid)
    masks:    list of (H,W) bool GT instance masks (may be empty); excluded from re-texturing.
    """
    if rng.rand() > SUPPORT["p_apply"]:
        return rgb
    H, W = depth_mm.shape
    # Dilated union of GT objects: excluded from the plane FIT (so objects don't bias it) and from
    # the painting (so no instance is ever overwritten).
    obj = None
    if masks:
        obj = np.zeros((H, W), bool)
        for m in masks:
            obj |= m
        k = SUPPORT["dilate_obj"]
        if k > 0:
            obj = cv2.dilate(obj.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    plane = _fit_plane(depth_mm, rng, exclude=obj)
    if plane is None:
        return rgb
    a, b, c = plane
    yy, xx = np.mgrid[0:H, 0:W]
    fit = a * xx + b * yy + c
    valid = depth_mm > 0
    med = float(np.median(depth_mm[valid])) if valid.any() else 0.0
    if med <= 0:
        return rgb
    on_plane = valid & (np.abs(depth_mm - fit) < SUPPORT["plane_tol_rel"] * med)

    # Never paint over an object (the plane residual alone cannot tell a flat object lying on the
    # surface from the surface itself).
    if obj is not None:
        on_plane &= ~obj

    if on_plane.mean() < SUPPORT["min_frac"]:
        return rgb                                   # weak/absent support plane -> leave alone
    tex = _random_texture(H, W, rng)
    alpha = float(rng.uniform(*SUPPORT["alpha"]))
    out = rgb.copy()
    out[on_plane] = (alpha * tex[on_plane] + (1 - alpha) * rgb[on_plane]).astype(np.uint8)
    return out
