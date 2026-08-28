"""Bilinear deformable sampling — the paper's function phi.

phi(z; (px, py)) = sum_{rx,ry} g(px, rx) g(py, ry) z[ry, rx, :],  g(a,b) = max(0, 1 - |a - b|)

This is exactly ``F.grid_sample(mode="bilinear")`` with an align-corners grid, so we
use it directly rather than reimplementing the bilinear kernel.
"""
import torch
import torch.nn.functional as F


def make_reference_grid(h, w, device, dtype):
    """Uniform reference grid R over an ``h x w`` feature map, normalized to [-1, 1].

    Returns a tensor of shape ``(h*w, 2)`` with (x, y) order (grid_sample convention).
    """
    ys, xs = torch.meshgrid(
        torch.linspace(0.5, h - 0.5, h, device=device, dtype=dtype),
        torch.linspace(0.5, w - 0.5, w, device=device, dtype=dtype),
        indexing="ij",
    )
    # normalize pixel-centre coords to [-1, 1]
    xs = xs / w * 2 - 1
    ys = ys / h * 2 - 1
    ref = torch.stack((xs, ys), dim=-1)  # (h, w, 2), (x, y)
    return ref.reshape(h * w, 2)


def sample_features(feat, points):
    """Sample ``feat`` at normalized ``points`` via bilinear interpolation (phi).

    Args:
        feat:   (B, C, H, W)
        points: (B, N, 2) in [-1, 1], (x, y) order
    Returns:
        (B, C, N) sampled features.
    """
    b, c, _, _ = feat.shape
    n = points.shape[1]
    grid = points.reshape(b, n, 1, 2)  # (B, N, 1, 2)
    sampled = F.grid_sample(
        feat, grid, mode="bilinear", padding_mode="border", align_corners=True
    )  # (B, C, N, 1)
    return sampled.reshape(b, c, n)
