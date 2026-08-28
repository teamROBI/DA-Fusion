"""Deformable Cross-Attention (DC) — paper eqs. (8)-(13).

Cross-modal fusion on the DS outputs f'_rgb, f'_d. Each modality samples its own
key/value embeddings at deformed points (eqs. 9-11); then the RGB query attends to
the DEPTH keys/values and the depth query attends to the RGB keys/values (eqs. 12-13).
The two attended results are concatenated and projected to form the fused feature f_i:

    z'_rgb = softmax(Q'_rgb Ktilde'_d^T / sqrt(d_h) + phi(B'_rgb;R'_rgb)) Vtilde'_d
    z'_d   = softmax(Q'_d Ktilde'_rgb^T / sqrt(d_h) + phi(B'_d;R'_d))     Vtilde'_rgb
    f_i    = W_proj [ z'_rgb ; z'_d ]
"""
import torch
import torch.nn as nn

from .deformable_self_attention import DeformableUnit, multihead_attend
from .relative_position_bias import ContinuousPositionBias


class DeformableCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_groups=4, stride=2, range_factor=2.0, use_rpb=True,
                 geom_prior=None):
        super().__init__()
        self.num_heads = num_heads
        self.unit_rgb = DeformableUnit(dim, num_heads, n_groups, stride, range_factor)
        self.unit_d = DeformableUnit(dim, num_heads, n_groups, stride, range_factor)
        self.rpb_rgb = ContinuousPositionBias(num_heads) if use_rpb else None
        self.rpb_d = ContinuousPositionBias(num_heads) if use_rpb else None
        self.geom = geom_prior              # shared across both cross directions
        self.out_proj = nn.Conv2d(2 * dim, dim, 1)   # concat(z_rgb, z_d) -> C_i
        self.norm = nn.GroupNorm(1, dim)

    def _add_geom(self, bias, ref_q, ref_k, depth_tokens):
        if self.geom is None or depth_tokens is None:
            return bias
        z_q, z_k, w_q, w_k = depth_tokens
        g = self.geom(ref_q, ref_k, z_q, z_k, w_k, w_q)
        if g is None:
            return bias
        return g if bias is None else g + bias.unsqueeze(0)

    def forward(self, f_rgb, f_d, depth_tokens=None):
        """f_rgb, f_d (B,C,H,W) -> fused f_i (B,C,H,W).

        The geometry prior applies to BOTH cross directions: whichever modality is querying, the
        pair of locations being mixed is the same pair of 3D points, so the same depth-distance
        decay is the correct prior."""
        b, c, h, w = f_rgb.shape
        # queries (full grid) per modality
        q_rgb, ref_q_rgb = self.unit_rgb.query(f_rgb)
        q_d, ref_q_d = self.unit_d.query(f_d)
        # each modality samples its own K/V at its own deformed points
        k_rgb, v_rgb, ref_k_rgb = self.unit_rgb.sample_kv(f_rgb)
        k_d, v_d, ref_k_d = self.unit_d.sample_kv(f_d)
        # cross: RGB query -> depth K/V ; depth query -> RGB K/V
        bias_rgb = self.rpb_rgb(ref_q_rgb, ref_k_d) if self.rpb_rgb is not None else None
        bias_d = self.rpb_d(ref_q_d, ref_k_rgb) if self.rpb_d is not None else None
        bias_rgb = self._add_geom(bias_rgb, ref_q_rgb, ref_k_d, depth_tokens)
        bias_d = self._add_geom(bias_d, ref_q_d, ref_k_rgb, depth_tokens)
        z_rgb = multihead_attend(q_rgb, k_d, v_d, self.num_heads, bias_rgb)
        z_d = multihead_attend(q_d, k_rgb, v_rgb, self.num_heads, bias_d)
        fused = torch.cat([z_rgb.reshape(b, c, h, w), z_d.reshape(b, c, h, w)], dim=1)
        fused = self.out_proj(fused)
        # residual on the RGB stream keeps the fused feature well-conditioned for the decoder
        return self.norm(f_rgb + fused)
