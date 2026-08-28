"""Depth -> HHA encoding (Gupta et al., "Learning Rich Features from RGB-D Images").

Three channels: Horizontal disparity, Height above ground, Angle between the local
surface normal and the inferred gravity direction. The paper feeds the depth branch
in HHA format ("[26]").

Notes / documented approximations (see dafusion/README.md):
  * Camera intrinsics come from data/datasets/intrinsics.py (best-effort; a known risk).
  * Gravity is approximated as camera-down (+y in camera coords, i.e. the image points
    downward). The full Gupta iterative gravity estimation is a TODO; for the roughly
    upright / top-down captures in these benchmarks the approximation is reasonable.
Output is a uint8 (H, W, 3) image, ready to normalize like an RGB image.
"""
import cv2
import numpy as np


def _backproject(depth_m, fx, fy, cx, cy):
    """depth_m (H,W) meters -> points (H,W,3) in camera coords (x right, y down, z fwd)."""
    h, w = depth_m.shape
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    z = depth_m
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    return np.stack((x, y, z), axis=-1)


def _surface_normals(points):
    """Estimate per-pixel unit normals from the point map via image-space gradients."""
    dzdx = cv2.Sobel(points, cv2.CV_32F, 1, 0, ksize=3)  # (H,W,3)
    dzdy = cv2.Sobel(points, cv2.CV_32F, 0, 1, ksize=3)
    normals = np.cross(dzdx, dzdy)                       # (H,W,3)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    return normals / norm


def depth_to_hha(depth_m, intrinsics):
    """depth_m: (H,W) float32 depth in METERS (0 = invalid). intrinsics: (fx,fy,cx,cy).
    Returns uint8 (H,W,3) HHA."""
    fx, fy, cx, cy = intrinsics
    valid = depth_m > 0
    pts = _backproject(depth_m, fx, fy, cx, cy)

    # gravity ~ camera-down (+y). height above ground = distance below the highest point
    # along gravity (floor/ground has the largest +y).
    gravity = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    # --- H1: horizontal disparity (inverse depth), higher = closer ---
    disp = np.zeros_like(depth_m)
    disp[valid] = 1.0 / depth_m[valid]

    # --- H2: height above ground ---
    y = pts[..., 1]
    if valid.any():
        ground = np.percentile(y[valid], 95)     # lowest point (max +y) ~ ground
        height = ground - y                      # >= 0 above ground
    else:
        height = np.zeros_like(y)

    # --- A: angle between surface normal and gravity (radians) ---
    normals = _surface_normals(pts)
    cos_a = np.clip((normals * gravity).sum(-1), -1.0, 1.0)
    angle = np.arccos(cos_a)                     # [0, pi]

    def norm01(a, mask):
        out = np.zeros_like(a)
        if mask.any():
            lo, hi = np.percentile(a[mask], 1), np.percentile(a[mask], 99)
            if hi <= lo:
                hi = lo + 1e-6
            out = np.clip((a - lo) / (hi - lo), 0, 1)
        return out

    hha = np.stack([
        norm01(disp, valid),
        norm01(height, valid),
        angle / np.pi,
    ], axis=-1)
    return (hha * 255.0).astype(np.uint8)
