"""One fusion stage: DS on each modality, then DC across them -> fused f_i.

    f'_rgb = DS_rgb(f_rgb)
    f'_d   = DS_d(f_d)
    f_i    = DC(f'_rgb, f'_d)

Channel count is preserved (C_i in, C_i out) so the fused multi-scale features drop
straight into the unchanged Mask2Former pixel decoder.
"""
import torch.nn as nn

from .deformable_self_attention import DeformableSelfAttention
from .deformable_cross_attention import DeformableCrossAttention
from .geometry_prior import GeometryPrior


class FusionStage(nn.Module):
    def __init__(self, dim, num_heads=8, n_groups=4, stride=2, range_factor=2.0, use_rpb=True,
                 geom_prior=False, geom_per_head=False, geom_max_pairs=8_000_000):
        super().__init__()
        # ONE GeometryPrior shared by all three attention modules in the stage: the depth-distance
        # decay is a property of the scene, not of which module is looking at it, so sharing keeps
        # the learned decay consistent within a stage (and is 3x fewer params).
        self.geom = GeometryPrior(num_heads, per_head=geom_per_head,
                                  max_pairs=geom_max_pairs) if geom_prior else None
        kw = dict(num_heads=num_heads, n_groups=n_groups, stride=stride,
                  range_factor=range_factor, use_rpb=use_rpb, geom_prior=self.geom)
        self.ds_rgb = DeformableSelfAttention(dim, **kw)
        self.ds_d = DeformableSelfAttention(dim, **kw)
        self.dc = DeformableCrossAttention(dim, **kw)

    def forward(self, f_rgb, f_d, depth_tokens=None):
        f_rgb = self.ds_rgb(f_rgb, depth_tokens)
        f_d = self.ds_d(f_d, depth_tokens)
        return self.dc(f_rgb, f_d, depth_tokens)
