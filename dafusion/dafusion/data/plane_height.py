"""Height above the fitted support plane, as a unit-invariant [0,1] map.

The user's mechanism: a floor or table is flat in depth; an object on it rises out of that plane. The
pipeline already exploits this at post-process time -- `plane_reject` is worth +4.0 F on OCID -- but only
to delete masks after the fact. As an input channel the model can use it while segmenting.

Unit invariance is essential and deliberate. `fit_support_plane`'s tolerance is expressed in metres for
~1 m scenes (1.2%), but UOAIS-Sim depth is stored in 0.1 mm units and reads ~7 "m" while the benchmarks
read ~1 m. Dividing by the per-image median depth makes every scene ~1.0, so the same relative tolerance
applies everywhere and the 10x training/eval unit discrepancy cannot affect this channel.
"""
import numpy as np

from dafusion.eval.postproc_v2 import fit_support_plane

MAX_REL_HEIGHT = 0.30      # objects taller than 30% of scene depth saturate (36 cm at 1.2 m)


def plane_height_map(depth_raw):
    """(H,W) raw depth (any consistent unit) -> (H,W) float32 in [0,1]; 0 on the plane and at holes."""
    z = depth_raw.astype(np.float32)
    if z.ndim == 3:
        z = z[..., 2]
    valid = z > 0
    out = np.zeros(z.shape, np.float32)
    if valid.sum() < 1000:
        return out
    med = float(np.median(z[valid]))
    if med <= 0:
        return out
    zn = np.where(valid, z / med, 0.0).astype(np.float32)   # scene normalised to ~1.0
    zp, _ = fit_support_plane(zn)
    if zp is None:
        return out
    # a pixel in FRONT of the plane (closer to the camera) stands above the surface
    h = np.where(valid & (zp > 0), zp - zn, 0.0)
    np.clip(h, 0.0, MAX_REL_HEIGHT, out=h)
    return (h / MAX_REL_HEIGHT).astype(np.float32)
