"""Depth -> metric XYZ organized point cloud (UCN / MSMFormer-style depth encoding).

Instead of a normalized depth *image*, back-project depth to a per-pixel (x, y, z) point
map in camera coordinates, then per-image standardize (center + isotropic scale) so the
representation is invariant to absolute depth scale. This is important here because the
UOAIS-Sim training depth is ~2.5-9 m while the real eval benchmarks are ~0.6-2.5 m: raw
metric XYZ would be badly out-of-distribution, so we standardize each frame to zero-mean,
unit-scale over its valid points. Isotropic (single-scalar) scaling preserves the cloud's
geometric shape and also cancels the (assumed) UOAIS-Sim focal length / principal point.
"""
import numpy as np

from .hha import _backproject


def depth_to_xyz(depth_mm, intrinsics):
    """depth_mm: (H,W) depth in millimeters (0 = invalid). intrinsics: (fx,fy,cx,cy).
    Returns (H,W,3) float32 metric XYZ in meters (invalid pixels = 0)."""
    depth_m = depth_mm.astype(np.float32) / 1000.0
    pts = _backproject(depth_m, *intrinsics)
    pts[depth_mm == 0] = 0.0
    return pts.astype(np.float32)


def standardize_xyz(pts, valid, radius_pct=95.0):
    """Robust, scale-invariant per-image normalization over valid points.

    pts: (H,W,3) float metric XYZ. valid: (H,W) bool mask of usable points.
    Center on the per-channel MEDIAN (robust to holes/outliers), then divide by a single
    scalar = the `radius_pct` percentile of the point-to-center distance (isotropic, so the
    cloud's geometric shape is preserved). This makes the encoding invariant to each frame's
    absolute depth scale AND range — critical because UOAIS-Sim (~2.5-9 m, with far-background
    outliers up to ~30 m), OCID, OSD and OCBD all differ in scale and spread. Using a
    percentile radius rather than std keeps the normalization from being dominated by a few
    far pixels, so the scene body maps to a consistent scale across datasets.

    Invalid pixels are left at 0 in the output. NOTE that 0 is *not* a neutral value here: the
    transform is (v - median)/scale, so 0 is exactly the scene's median centre — an invalid
    pixel is indistinguishable from a real surface at mid-scene depth. Callers that need the
    network to tell "missing" from "measured" must pass the mask on separately (see
    INPUT.DEPTH_VALIDITY_CHANNEL). This also means `valid` must exclude augmentation padding:
    train-time padding enters as 128.0 m, and if marked valid it dominates the p95 radius and
    flattens the real scene (the reason build_augmentation pins seg_pad_value=0).
    """
    out = np.zeros_like(pts, dtype=np.float32)
    if valid is None or not np.any(valid):
        return out
    v = pts[valid]
    mu = np.median(v, axis=0)                        # (3,) robust center
    r = np.linalg.norm(v - mu, axis=1)               # distance of each valid point to center
    scale = float(np.percentile(r, radius_pct)) + 1e-6   # outlier-robust isotropic radius
    out[valid] = (v - mu) / scale
    return out
