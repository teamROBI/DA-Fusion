"""Depth -> [normalized-depth | normal_x | normal_y] hybrid encoding.

Keeps absolute distance (channel 0, percentile-normalized so it's scale-robust across
UOAIS-Sim / OCID / OSD / OCBD) AND adds surface-normal orientation (channels 1-2), which
sharpens boundaries between touching objects at different orientations — the OCID/OCBD clutter
failure. Unlike HHA it needs no gravity estimate; unlike pure normals it doesn't discard depth.
Output is uint8 (H,W,3), fed to the depth branch like an RGB image.

Note: normals are direction-only, so uniform scale (mm vs m) is irrelevant. On a horizontal
flip the caller must negate normal_x (channel 1 -> 255 - channel1), analogous to the xyz X-sign
flip; normal_y is unchanged by horizontal flipping.
"""
import numpy as np

from .hha import _backproject, _surface_normals


def _norm_depth01(depth, valid, inverted=True):
    d = np.zeros_like(depth, dtype=np.float32)
    if valid.any():
        lo, hi = np.percentile(depth[valid], 5), np.percentile(depth[valid], 95)
        if hi <= lo:
            hi = lo + 1e-6
        d = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    if inverted:
        d = 1.0 - d
    d[~valid] = 0.0
    return d


def _encode_from_points(pts, depth_scalar, valid, inverted):
    normals = _surface_normals(pts)                       # (H,W,3) unit, components in [-1,1]
    nx = (normals[..., 0] + 1.0) / 2.0
    ny = (normals[..., 1] + 1.0) / 2.0
    nx[~valid] = 0.0
    ny[~valid] = 0.0
    d = _norm_depth01(depth_scalar, valid, inverted)
    return (np.stack([d, nx, ny], axis=-1) * 255.0).astype(np.uint8)


def depth_to_normals(depth_mm, intrinsics, inverted=True):
    """(H,W) raw depth (mm; 0=invalid) + intrinsics (fx,fy,cx,cy) -> uint8 (H,W,3)."""
    fx, fy, cx, cy = intrinsics
    valid = depth_mm > 0
    pts = _backproject(depth_mm, fx, fy, cx, cy)
    return _encode_from_points(pts, depth_mm, valid, inverted)


def points_to_normals(pts, inverted=True):
    """(H,W,3) metric XYZ point map (0=invalid) -> uint8 (H,W,3). For OCBD organized clouds."""
    valid = np.any(pts != 0, axis=-1)
    return _encode_from_points(pts.astype(np.float32), pts[..., 2], valid, inverted)
