"""Train-time depth sensor-noise augmentation (UCN / DexNet 2.0 recipe).

UOAIS-Sim depth is clean BlenderProc renders, but the real benchmarks (OCID/OSD/OCBD) have
Kinect/Xtion noise: holes, edge fringing, quantization. Every strong baseline corrupts its
synthetic training depth (UCN/MSMFormer: multiplicative gamma + GP noise + ellipse dropout;
UOAIS: Perlin distortion); without it the model never sees invalid/noisy regions it will
meet at test time. Ported from baselines/ucn/lib/utils/augmentation.py (DexNet 2.0 params
via UCN's tabletop_object.py), with ONE deliberate change: the additive GP-noise sigma is
RELATIVE (0.5% of the per-image median valid depth) instead of an absolute 5 mm — UCN's
scenes are ~1 m where 5 mm == 0.5%, while UOAIS-Sim is ~2.5-9 m where absolute 5 mm would
be invisible. Applied to raw (H,W) mm depth BEFORE encoding, so every encoding
(normalized / xyz / hha) sees the same corruption; gated by cfg.INPUT.DEPTH_NOISE.
"""
import cv2
import numpy as np

NOISE = dict(
    p_clean=0.1,                 # fraction of frames left clean (UCN: rand > 0.1)
    gamma_shape=1000.0,          # multiplicative per-image factor ~ Gamma(1000, 0.001):
    gamma_scale=0.001,           #   mean 1.0, std ~3%
    gp_rescale_factor=4,         # GP field sampled at 1/4 res, bicubic-upsampled
    gp_rel_sigma=0.005,          # GP sigma = 0.5% of median valid depth (UCN: 5 mm @ ~1 m)
    # Hole dropout tuned to match REAL sensor stats (OCID ~13%, OSD ~21% missing depth); the
    # original UCN params gave only ~0.3%. Two mechanisms: (1) larger/more random ellipses,
    # (2) edge dropout at depth discontinuities (where real sensors actually fail).
    ellipse_dropout_mean=14.0,   # Poisson mean #ellipse holes
    ellipse_gamma_shape=3.0,     # radii ~ Gamma(3, 6) px  (mean ~18px, bigger holes)
    ellipse_gamma_scale=6.0,
    edge_dropout_frac=0.55,      # fraction of high-gradient (edge) pixels dropped
    edge_grad_pct=88.0,          # pixels above this depth-gradient percentile count as edges
    edge_dilate=2,               # dilate dropped edges (kernel size)
)

# --- Pattern-matched hole augmentation (INPUT.DEPTH_HOLE_PATTERN) ---------------------------
# The params above produce holes that are content-driven (gradient) or uniform-random. Direct
# measurement of the real benchmarks (docs/EXPERIMENTS.md Track 9b) shows the dominant real
# artifact is neither: it is a *systematic dead band at the frame edge*, from warping depth into
# the RGB frame across a physical RGB/IR baseline. Measured over 60 images/dataset:
#     OCID  11.6% invalid, rightmost 33 columns dead in EVERY frame, top rows 69% dead
#     OSD   23.8% invalid, rightmost 66 columns dead in EVERY frame, top rows 100% dead
# Nothing in the original recipe is position-dependent, so it could not reproduce this at all.
# Additionally, real structured-light shadow is DIRECTIONAL (falls on one side of an occluder,
# widening with the depth step) whereas `_edge_dropout` is isotropic and fires on both sides.
PATTERN = dict(
    # NOTE the goal is COVERAGE, not marginal-statistic matching. The three benchmarks disagree
    # (OCID right-33, OSD right-66, OCBD none), so the model must be robust to any of them rather
    # than memorize one; hence a stochastic side/width instead of a fixed band. Widths span 10-80
    # so both 33 and 66 sit comfortably inside the sampled range, and p_apply<1 (plus NOISE's
    # p_clean) keeps band-free frames in the mix for OCBD.
    p_apply=0.9,                 # fraction of frames that get a dead band at all
    band_side_probs=(0.6, 0.05, 0.22, 0.13),    # right, left, top, bottom (right dominates)
    band_cols=(10, 80),          # dead-band width in px, ~U(lo,hi): spans OCID 33 / OSD 66
    band_rows=(4, 45),           # top/bottom band height in px
    band_ragged=6,               # per-column jitter so the edge isn't a perfectly straight line
    shadow_frac=0.6,             # fraction of frames that also get directional shadows
    shadow_grad_pct=90.0,        # depth-gradient percentile counting as an occlusion edge
    shadow_max_px=7,             # max shadow width (px) at the largest depth step
)


def _edge_dropout(depth, rng):
    """Drop depth at depth-discontinuity edges (where real sensors miss), the dominant source
    of real holes. Threshold the gradient magnitude, randomly drop a fraction, dilate."""
    valid = depth > 0
    if valid.sum() < 100:
        return depth
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx * gx + gy * gy)
    thr = np.percentile(grad[valid], NOISE["edge_grad_pct"])
    edges = (grad >= thr) & valid
    drop = edges & (rng.random(depth.shape) < NOISE["edge_dropout_frac"])
    k = NOISE["edge_dilate"]
    if k > 0:
        drop = cv2.dilate(drop.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool) & valid
    depth = depth.copy()
    depth[drop] = 0.0
    return depth


def _gp_noise(depth, rng):
    """Additive approximate-Gaussian-process noise on valid pixels (low-res field,
    bicubic upsample), sigma relative to the per-image median valid depth."""
    valid = depth > 0
    if not valid.any():
        return depth
    h, w = depth.shape
    sigma = NOISE["gp_rel_sigma"] * float(np.median(depth[valid]))
    small = (max(1, h // NOISE["gp_rescale_factor"]), max(1, w // NOISE["gp_rescale_factor"]))
    field = rng.normal(0.0, sigma, size=small).astype(np.float32)
    field = cv2.resize(field, (w, h), interpolation=cv2.INTER_CUBIC)
    depth = depth.copy()
    depth[valid] += field[valid]
    return depth


def _dropout_ellipses(depth, rng):
    """Zero out a Poisson number of random ellipses (missing-depth holes), as in DexNet/UCN."""
    depth = depth.copy()
    n = rng.poisson(NOISE["ellipse_dropout_mean"])
    if n == 0:
        return depth
    nz = np.array(np.nonzero(depth > 0)).T          # (N,2) row,col of valid pixels
    if len(nz) == 0:
        return depth
    centers = nz[rng.choice(len(nz), size=n)]
    x_r = rng.gamma(NOISE["ellipse_gamma_shape"], NOISE["ellipse_gamma_scale"], size=n)
    y_r = rng.gamma(NOISE["ellipse_gamma_shape"], NOISE["ellipse_gamma_scale"], size=n)
    angles = rng.randint(0, 360, size=n)
    for (cy, cx), xr, yr, a in zip(centers, x_r, y_r, angles):
        cv2.ellipse(depth, (int(cx), int(cy)), (int(round(xr)), int(round(yr))),
                    angle=int(a), startAngle=0, endAngle=360, color=0, thickness=-1)
    return depth


def _dead_band(depth, rng):
    """Zero a band along ONE frame edge, reproducing the RGB/IR-baseline dead band that every
    real OCID/OSD frame carries (Track 9b). Width is sampled per frame and the boundary is made
    slightly ragged, since the measured P(invalid) maps show a jittered rather than ruler-straight
    edge (the band follows a warp, not a crop)."""
    if rng.rand() > PATTERN["p_apply"]:
        return depth
    h, w = depth.shape
    depth = depth.copy()
    side = rng.choice(4, p=PATTERN["band_side_probs"])          # 0=right 1=left 2=top 3=bottom
    if side in (0, 1):
        lo, hi = PATTERN["band_cols"]
        base = rng.randint(lo, hi + 1)
        jitter = rng.randint(0, PATTERN["band_ragged"] + 1, size=h)
        for y in range(h):
            n = min(w, base + int(jitter[y]))
            if side == 0:
                depth[y, w - n:] = 0.0
            else:
                depth[y, :n] = 0.0
    else:
        lo, hi = PATTERN["band_rows"]
        base = rng.randint(lo, hi + 1)
        jitter = rng.randint(0, PATTERN["band_ragged"] + 1, size=w)
        for x in range(w):
            n = min(h, base + int(jitter[x]))
            if side == 2:
                depth[:n, x] = 0.0
            else:
                depth[h - n:, x] = 0.0
    return depth


def _directional_shadow(depth, rng):
    """Directional occlusion shadow: for a structured-light sensor the projector and camera are
    offset horizontally, so the surface immediately to ONE side of a depth step is unobserved.
    Shadow width scales with the depth step, matching the real geometry (bigger step -> wider
    shadow). Contrast with `_edge_dropout`, which drops both sides symmetrically."""
    if rng.rand() > PATTERN["shadow_frac"]:
        return depth
    valid = depth > 0
    if valid.sum() < 100:
        return depth
    gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)           # signed horizontal step
    mag = np.abs(gx)
    thr = np.percentile(mag[valid], PATTERN["shadow_grad_pct"])
    step = np.where((mag >= thr) & valid, mag, 0.0)
    if not step.any():
        return depth
    # normalize the step to [0,1] and turn it into a per-pixel shadow width
    width = (step / (step.max() + 1e-6) * PATTERN["shadow_max_px"]).astype(np.int32)
    direction = 1 if rng.rand() < 0.5 else -1                  # which side the baseline shadows
    depth = depth.copy()
    for k in range(1, PATTERN["shadow_max_px"] + 1):
        # pixels whose shadow reaches at least k px get zeroed k px to `direction`
        src = width >= k
        if not src.any():
            break
        depth[np.roll(src, direction * k, axis=1)] = 0.0
    return depth


def inpaint_holes_u8(depth3, kernel_size=3):
    """Telea-inpaint zero (hole) pixels of a 3-ch uint8 depth image — the same treatment
    the eval predictor applies to real sensor holes, so train/test artifacts match."""
    mask = np.all(depth3 == 0, axis=2).astype(np.uint8)
    if not mask.any():
        return depth3
    inpainted = cv2.inpaint(depth3, mask, kernel_size, cv2.INPAINT_TELEA)
    return np.where(depth3 == 0, inpainted, depth3)


def augment_depth_mm(depth_mm, rng=np.random, hole_pattern=False):
    """Corrupt a raw (H,W) float mm depth map. ~10% of frames pass through clean.

    `rng` must expose the LEGACY numpy API (`rand`, `randint`) as well as `random` — i.e. pass
    the `np.random` module, not a `np.random.Generator` (Generator has no `.rand`/`.randint`).

    hole_pattern=True (INPUT.DEPTH_HOLE_PATTERN) additionally reproduces the *measured* real hole
    geometry: a systematic frame-edge dead band and directional occlusion shadows. See PATTERN.
    """
    if rng.rand() < NOISE["p_clean"]:
        return depth_mm
    d = depth_mm * rng.gamma(NOISE["gamma_shape"], NOISE["gamma_scale"])
    d = _gp_noise(d, rng)
    if hole_pattern:
        d = _directional_shadow(d, rng)   # one-sided shadow (real structured-light geometry)
    else:
        d = _edge_dropout(d, rng)         # isotropic gradient dropout (original recipe)
    d = _dropout_ellipses(d, rng)         # plus random ellipse holes
    if hole_pattern:
        d = _dead_band(d, rng)            # the dominant real artifact; apply last so it wins
    return d
