"""Deformable Self-Attention (DS) — paper eqs. (1)-(7).

Per modality: project the feature map to a query Q, predict sampling offsets
Delta_p = theta_offset(Q) over a (downsampled) uniform reference grid R, bilinearly
sample keys/values at the deformed points R + Delta_p, and run multi-head attention
with an (optional) relative-position bias phi(B; R):

    f' = Concat_m( softmax(Q^m Ktilde^m^T / sqrt(d_h) + phi(B;R)) Vtilde^m ) W_o

Sampling offsets are per-*group* (n_groups fields shared across heads within a group,
as in the Deformable Attention Transformer). The key/value grid is downsampled by
``stride`` so the attention key count stays bounded at high-resolution stages.

``DeformableUnit`` (the projections + offset sampling for one feature map) and
``multihead_attend`` are reused by the cross-attention module (DC).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .offset_network import OffsetNetwork
from .relative_position_bias import ContinuousPositionBias
from .sampling import make_reference_grid, sample_features


def multihead_attend(q, k, v, num_heads, bias=None):
    """q (B,C,Nq), k/v (B,C,Nk) -> out (B,C,Nq).

    bias may be:
      * (num_heads, Nq, Nk)      -- batch-independent, e.g. ContinuousPositionBias (2D position)
      * (B, num_heads, Nq, Nk)   -- batch-dependent, e.g. GeometryPrior (depth differs per image)
      * None
    """
    b, c, nq = q.shape
    nk = k.shape[2]
    dh = c // num_heads
    q = q.reshape(b, num_heads, dh, nq).transpose(-1, -2)   # (B,h,Nq,dh)
    k = k.reshape(b, num_heads, dh, nk).transpose(-1, -2)   # (B,h,Nk,dh)
    v = v.reshape(b, num_heads, dh, nk).transpose(-1, -2)   # (B,h,Nk,dh)
    if bias is not None:
        attn = torch.matmul(q, k.transpose(-1, -2)) / (dh ** 0.5)   # (B,h,Nq,Nk)
        attn = attn + (bias if bias.dim() == 4 else bias.unsqueeze(0))
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)                                 # (B,h,Nq,dh)
    else:
        out = F.scaled_dot_product_attention(q, k, v)               # memory-efficient
    out = out.transpose(-1, -2).reshape(b, c, nq)
    return out


class DeformableUnit(nn.Module):
    """Projections + offset-driven K/V sampling for one feature map.

    Produces the query tokens (over the full grid) and the key/value tokens
    (sampled at deformed points on a downsampled grid), plus the reference grids
    needed for the relative-position bias.
    """

    def __init__(self, dim, num_heads, n_groups, stride, range_factor):
        super().__init__()
        assert dim % n_groups == 0 and dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.n_groups = n_groups
        self.group_dim = dim // n_groups
        self.stride = stride
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.offset_net = OffsetNetwork(self.group_dim, stride=stride, range_factor=range_factor)

    def query(self, feat):
        """feat (B,C,H,W) -> Q tokens (B,C,Nq), ref_q (Nq,2)."""
        b, c, h, w = feat.shape
        q = self.q_proj(feat).reshape(b, c, h * w)
        ref_q = make_reference_grid(h, w, feat.device, feat.dtype)
        return q, ref_q

    def sample_kv(self, feat):
        """feat (B,C,H,W) -> K,V tokens (B,C,Nk) sampled at deformed points, ref_k (Nk,2)."""
        b, c, h, w = feat.shape
        g, gd = self.n_groups, self.group_dim
        # offsets are predicted from the (grouped) query of this feature map
        q_grouped = self.q_proj(feat).reshape(b * g, gd, h, w)
        offsets = self.offset_net(q_grouped)                # (B*g, Hk, Wk, 2)
        hk, wk = offsets.shape[1], offsets.shape[2]
        ref_k = make_reference_grid(hk, wk, feat.device, feat.dtype)   # (Nk, 2)
        deformed = ref_k.reshape(1, hk * wk, 2) + offsets.reshape(b * g, hk * wk, 2)
        feat_grouped = feat.reshape(b * g, gd, h, w)
        sampled = sample_features(feat_grouped, deformed)   # (B*g, gd, Nk)
        sampled = sampled.reshape(b, c, hk * wk)            # (B, C, Nk)
        k = self.k_proj(sampled.unsqueeze(-1)).squeeze(-1)
        v = self.v_proj(sampled.unsqueeze(-1)).squeeze(-1)
        return k, v, ref_k


class DeformableSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8, n_groups=4, stride=2, range_factor=2.0, use_rpb=True,
                 geom_prior=None):
        super().__init__()
        self.num_heads = num_heads
        self.unit = DeformableUnit(dim, num_heads, n_groups, stride, range_factor)
        self.o_proj = nn.Conv2d(dim, dim, 1)
        self.rpb = ContinuousPositionBias(num_heads) if use_rpb else None
        self.geom = geom_prior              # optional GeometryPrior (3D depth+spatial decay)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, feat, depth_tokens=None):
        """feat (B,C,H,W) -> refined f' (B,C,H,W), residual + norm.

        depth_tokens: optional (z_q, z_k, w_k) pooled depth for the geometry prior, where z_q is
        at the query grid and z_k at the (downsampled) key grid. The 2D RPB and the 3D geometry
        prior are simply summed -- both are additive pre-softmax biases."""
        b, c, h, w = feat.shape
        q, ref_q = self.unit.query(feat)
        k, v, ref_k = self.unit.sample_kv(feat)
        bias = self.rpb(ref_q, ref_k) if self.rpb is not None else None
        if self.geom is not None and depth_tokens is not None:
            z_q, z_k, w_q, w_k = depth_tokens
            g = self.geom(ref_q, ref_k, z_q, z_k, w_k, w_q)
            if g is not None:
                bias = g if bias is None else g + bias.unsqueeze(0)
        out = multihead_attend(q, k, v, self.num_heads, bias)
        out = self.o_proj(out.reshape(b, c, h, w))
        return self.norm(feat + out)
