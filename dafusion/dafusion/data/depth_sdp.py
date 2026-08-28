"""Sinusoidal Depth Preprocessing (SDP) — Vanishing Depth (Koch & Krüger, arXiv:2503.19947).

Maps each scalar depth value onto many sine/cosine channels at different frequencies, so a
single-channel measurement becomes a rich multi-channel signal an image encoder can consume,
while absolute metric depth stays recoverable:

    l          = 2*pi * d / max(d)
    SDP(d, 2i) = sin( l / T^(2i/c) )
    SDP(d,2i+1)= cos( l / T^(2i/c) )

with c encoding channels and temperature T < 1 setting the frequency spread. Low-frequency
channels capture coarse distance, high-frequency ones fine depth differences.

GSDP vs LSDP: `max(d)` is either a fixed global constant (GSDP — the model learns one scale) or
the per-image max (LSDP — needs max(d) fed in separately to decode metric depth). We use **GSDP**,
which the paper adopts as the default for downstream tasks and which avoids per-image scale
ambiguity; DEPTH_SDP_MAX is that constant in millimetres.

Expectation, stated up front so the result is interpretable either way: the paper's large SDP wins
are on **depth completion** (NYU RMSE 129.8 -> 94.4) and **6D pose** (+0.7..+8.1) -- tasks needing
metric precision. On **semantic segmentation** its own Table 4 shows SDP ~= plain normalization
(54.8/56.1/77.7/53.4/49.9 vs 55.1/55.8/77.1/53.6/49.5). Ours is a segmentation task, so the prior
expectation is a null result; this module exists to test that rather than assume it.

Holes are handled the way the paper insists: sin(0)=0 / cos(0)=1 would make a missing pixel look
like a *specific valid depth*, so invalid pixels are zeroed in ALL encoded channels and flagged
through the separate validity channel (INPUT.DEPTH_VALIDITY_CHANNEL).
"""
import numpy as np


def depth_to_sdp(depth_mm, channels=32, temperature=0.1, max_depth_mm=15000.0, local=False):
    """(H,W) raw mm depth (0 = invalid) -> (H,W,channels) float32 sinusoidal encoding in [-1,1].

    `channels` must be even (sin/cos pairs). Invalid pixels are exactly 0 in every channel.

    local=False (GSDP): normalize by the fixed `max_depth_mm`. WARNING for sim->real transfer --
    this makes the encoding absolute-scale-dependent, and our train and test ranges differ by ~5x
    (UOAIS-Sim 2.5-9 m vs OCID/OSD 0.3-1.8 m). With max=15 m, training occupies l in [1.05, 3.77]
    while real eval collapses into l in [0.25, 0.75]: the two never overlap in frequency space, so
    the network is tested on a region of the encoding it never saw. This measurably wrecked
    Grid D (-3.47 vs the same config with xyz; docs/EXPERIMENTS.md Track 10d).

    local=True (LSDP): normalize by each frame's own p99 valid depth, so every image maps onto the
    same [0, 2pi] span regardless of absolute range. This is the direct analogue of what
    standardize_xyz does for the xyz encoding -- and the reason xyz survives the sim->real scale
    gap at all. Absolute metric scale is discarded, which is the trade: for *segmentation* that is
    probably fine (relative surface structure is what matters), whereas depth completion / pose
    would need the scale back via the paper's max(d) side-channel.
    """
    assert channels % 2 == 0, "SDP channel count must be even (sin/cos pairs)"
    d = depth_mm.astype(np.float32)
    valid = d > 0
    if local:
        # p99 rather than max: a single hot pixel would otherwise compress the whole scene.
        scale = float(np.percentile(d[valid], 99.0)) if valid.any() else 1.0
        scale = max(scale, 1e-6)
        l = 2.0 * np.pi * np.clip(d, 0.0, scale) / scale
    else:
        l = 2.0 * np.pi * np.clip(d, 0.0, max_depth_mm) / max_depth_mm  # GSDP: fixed global scale
    half = channels // 2
    i = np.arange(half, dtype=np.float32)
    # T^(2i/c) with T<1 gives an increasing sequence of divisors -> decreasing frequencies.
    freqs = 1.0 / np.power(temperature, (2.0 * i) / channels)
    ang = l[..., None] * freqs[None, None, :]                            # (H,W,half)
    out = np.empty((*d.shape, channels), np.float32)
    out[..., 0::2] = np.sin(ang)
    out[..., 1::2] = np.cos(ang)
    out[~valid] = 0.0            # NOT sin(0)/cos(0)=(0,1), which would read as a real depth
    return out
