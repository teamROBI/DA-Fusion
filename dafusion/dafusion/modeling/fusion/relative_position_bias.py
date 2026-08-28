"""Relative position bias phi(B; R) for deformable attention.

The paper adds a relative-position-bias term phi(B; R) to the attention logits
(eqs. 6, 12, 13). We realise B as a *continuous* position bias (CPB, as in the
Deformable Attention Transformer): a tiny MLP maps the relative displacement
between each query reference point and each key reference point to a per-head
scalar bias. This is differentiable and resolution-agnostic ("interpolates the
bias table"), unlike a fixed lookup table tied to one grid size.

To bound memory, CPB is only materialised when ``Nq * Nk`` is below a threshold
(the highest-resolution stage would otherwise need a huge (Nq, Nk) tensor); above
it, the bias is skipped (returns None) and attention runs without the term. This
per-stage behaviour is recorded as a deviation in dafusion/README.md.
"""
import torch
import torch.nn as nn


class ContinuousPositionBias(nn.Module):
    def __init__(self, num_heads, hidden_dim=32, max_pairs=4_000_000):
        super().__init__()
        self.num_heads = num_heads
        self.max_pairs = max_pairs
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_heads),
        )

    def forward(self, ref_q, ref_k):
        """Args: ref_q (Nq, 2), ref_k (Nk, 2) in [-1, 1].
        Returns bias (num_heads, Nq, Nk), or None if too large to materialise."""
        nq, nk = ref_q.shape[0], ref_k.shape[0]
        if nq * nk > self.max_pairs:
            return None
        rel = ref_q[:, None, :] - ref_k[None, :, :]      # (Nq, Nk, 2)
        bias = self.mlp(rel)                              # (Nq, Nk, heads)
        return bias.permute(2, 0, 1).contiguous()        # (heads, Nq, Nk)
