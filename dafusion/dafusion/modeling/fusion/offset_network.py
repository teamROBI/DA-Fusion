"""Offset network theta_offset: predicts sampling offsets Delta_p = theta_offset(Q).

Follows the Deformable Attention Transformer (Xia et al., CVPR 2022) design that the
paper's DS/DC modules describe: a small depthwise-conv head over the (grouped) query
feature map produces a 2-channel offset field at a (possibly strided) reference
resolution. Offsets are bounded with tanh * range so sampling stays local and stable.
"""
import torch
import torch.nn as nn


class _ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dim of an (B, C, H, W) tensor."""

    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


class OffsetNetwork(nn.Module):
    """theta_offset for one attention group.

    Args:
        group_dim: channels per group (C // n_groups).
        stride:    downsample factor of the sampling grid vs the query grid.
        range_factor: max |offset| in normalized [-1,1] units per axis (tanh-bounded).
    """

    def __init__(self, group_dim, stride=1, range_factor=2.0):
        super().__init__()
        self.stride = stride
        self.range_factor = range_factor
        k = stride if stride > 1 else 3
        pad = 0 if stride > 1 else 1
        self.net = nn.Sequential(
            nn.Conv2d(group_dim, group_dim, kernel_size=k, stride=stride,
                      padding=pad, groups=group_dim),
            _ChannelLayerNorm(group_dim),
            nn.GELU(),
            nn.Conv2d(group_dim, 2, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, q_group):
        """Args: q_group (Bg, Cg, H, W).  Returns offsets (Bg, Hk, Wk, 2) in norm units."""
        off = self.net(q_group)                    # (Bg, 2, Hk, Wk)
        off = off.permute(0, 2, 3, 1)              # (Bg, Hk, Wk, 2)
        # bound the offset range; normalized-coord units (grid is [-1, 1])
        return torch.tanh(off) * (self.range_factor / max(off.shape[1], off.shape[2]))
