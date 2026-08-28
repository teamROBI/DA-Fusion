"""Estimate + correct in-image camera ROLL so sim scenes are upright, matching the
always-upright real benchmarks (OCID/OSD/OCBD). UOAIS-Sim renders random camera poses
(incl. large roll — e.g. tabletop/33.png is ~90° rolled); for the metric-XYZ encoding this
is a train/test mismatch (per-image standardization removes translation/scale but NOT
rotation). Un-rolling the training data removes that gap.

Roll is estimated from the ground/support-plane depth gradient: fit z ~ a*u + b*v + c over
valid pixels; (a,b) points toward increasing depth (FAR). In an upright scene the far region
is at the TOP of the image, so we rotate to bring (a,b) to point up. A confidence (tilt
strength) gates it — fronto-parallel / top-down views (e.g. OCBD bins) have no reliable roll
and are left untouched. Data-only preprocessing; model architecture unchanged.
"""
import cv2
import numpy as np


def estimate_roll(depth_mm, conf_thresh=0.12, rng=np.random):
    """Return (angle_deg, confidence). angle_deg is the rotation to apply to make the scene
    upright; 0 when confidence < conf_thresh (unreliable -> leave as-is)."""
    vv, uu = np.nonzero(depth_mm > 0)
    if len(uu) < 500:
        return 0.0, 0.0
    z = depth_mm[vv, uu].astype(np.float64)
    if len(z) > 20000:
        sel = rng.choice(len(z), 20000, replace=False)
        uu, vv, z = uu[sel], vv[sel], z[sel]
    A = np.stack([uu, vv, np.ones_like(uu)], axis=1).astype(np.float64)
    (a, b, _c), *_ = np.linalg.lstsq(A, z, rcond=None)
    H, W = depth_mm.shape
    conf = float(np.hypot(a, b) * np.hypot(H, W) / (np.median(z) + 1e-6))  # relative depth span
    if conf < conf_thresh:
        return 0.0, conf
    phi = np.degrees(np.arctan2(b, a))       # gradient (far) direction, image coords (row down)
    theta = (-90.0 - phi + 180) % 360 - 180  # rotate so FAR points up (-row)
    return theta, conf


def apply_roll(img, angle_deg, interp=cv2.INTER_LINEAR, border_value=0):
    """Rotate img in-plane by angle_deg about its center (border filled with border_value)."""
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, M, (W, H), flags=interp, borderValue=border_value)
