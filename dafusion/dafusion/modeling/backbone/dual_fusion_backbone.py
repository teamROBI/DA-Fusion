"""Dual-branch Swin backbone with per-stage DS/DC fusion.

Two independent Swin-L branches extract features from the RGB and (HHA) depth inputs;
at each of the 4 output stages a ``FusionStage`` (DS on each modality + DC across them)
produces a fused feature. The fused multi-scale dict {res2..res5} has the same channel
dims as a single Swin, so it feeds the unchanged Mask2Former pixel decoder directly.

For RGB-only / depth-only ablations the fusion is bypassed (single branch passthrough),
selected by ``MODEL.DAFUSION.INPUT_TYPE``.
"""
import torch.nn as nn
from timm.layers import trunc_normal_
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec

from .swin import D2SwinTransformer
from ..fusion.fusion_stage import FusionStage
from ..fusion.geometry_prior import pool_depth_scalar
from .depth_adapter import DepthAdapter


def _swin_scratch_init(m):
    """Proper Swin from-scratch init (swin.py's own init_weights is a no-op / never applied)."""
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class DualSwinFusionBackbone(Backbone):
    def __init__(self, cfg, input_shape):
        super().__init__()
        self.input_type = cfg.MODEL.DAFUSION.INPUT_TYPE          # rgbd | rgb | depth
        # Depth carries a 4th validity channel when INPUT.DEPTH_VALIDITY_CHANNEL is set; the RGB
        # branch is always 3. Both branches are built from the same cfg, so the count is passed
        # explicitly (see D2SwinTransformer's in_chans arg).
        # channels actually arriving in depth_image (encoding + optional validity channel)
        enc_chans = cfg.INPUT.DEPTH_SDP_CHANNELS if cfg.INPUT.DEPTH_ENCODING == "sdp" else 3
        in_chans_depth = (enc_chans + (1 if cfg.INPUT.DEPTH_VALIDITY_CHANNEL else 0)
                          + (1 if cfg.INPUT.DEPTH_PLANE_HEIGHT_CHANNEL else 0))
        # With an adapter the depth STEM stays 3-channel (so the COCO-pretrained patch embed is
        # reused verbatim) and the adapter maps in_chans_depth -> 3 in front of it. Without one,
        # the stem itself must widen -- which requires the inflated init weights.
        self.depth_adapter = (DepthAdapter(in_chans_depth, 3, cfg.INPUT.DEPTH_ADAPTER_HIDDEN)
                              if cfg.INPUT.DEPTH_ADAPTER else None)
        depth_chans = 3 if self.depth_adapter is not None else in_chans_depth
        if self.depth_adapter is None and enc_chans != 3:
            raise ValueError(
                f"DEPTH_ENCODING='{cfg.INPUT.DEPTH_ENCODING}' emits {enc_chans} channels; the "
                "Swin stem cannot take that directly. Set INPUT.DEPTH_ADAPTER=True.")
        if self.input_type == "depth" and depth_chans != 3:
            # the "depth" ablation routes depth through swin_rgb, which is a 3-channel stem
            raise ValueError(
                "INPUT.DEPTH_VALIDITY_CHANNEL is unsupported with INPUT_TYPE='depth': that "
                "ablation feeds depth through the 3-channel RGB stem (see forward()). Use "
                "INPUT_TYPE='rgbd', or disable the validity channel."
            )
        self.swin_rgb = D2SwinTransformer(cfg, input_shape)
        self._out_features = list(self.swin_rgb._out_features)
        self._out_feature_channels = dict(self.swin_rgb._out_feature_channels)
        self._out_feature_strides = dict(self.swin_rgb._out_feature_strides)

        if self.input_type == "rgbd":
            self.swin_d = D2SwinTransformer(cfg, input_shape, in_chans=depth_chans)
            # For the xyz encoding the depth branch is trained from scratch (XYZ != RGB).
            # The paired weights file omits backbone.swin_d.* so the checkpointer won't
            # overwrite this init. (swin.py init_weights is never applied, so do it here.)
            if cfg.MODEL.DAFUSION.DEPTH_FROM_SCRATCH:
                self.swin_d.apply(_swin_scratch_init)
            strides = list(cfg.MODEL.DAFUSION.STRIDES)
            num_heads = cfg.MODEL.DAFUSION.NUM_HEADS
            n_groups = cfg.MODEL.DAFUSION.N_GROUPS
            rf = cfg.MODEL.DAFUSION.OFFSET_RANGE_FACTOR
            use_rpb = cfg.MODEL.DAFUSION.USE_RPB
            self.geom_prior = cfg.MODEL.DAFUSION.GEOMETRY_PRIOR
            self.depth_encoding = cfg.INPUT.DEPTH_ENCODING
            self.has_validity = cfg.INPUT.DEPTH_VALIDITY_CHANNEL
            self.fusions = nn.ModuleDict()
            for i, name in enumerate(self._out_features):
                self.fusions[name] = FusionStage(
                    self._out_feature_channels[name],
                    num_heads=num_heads, n_groups=n_groups,
                    stride=strides[i], range_factor=rf, use_rpb=use_rpb,
                    geom_prior=self.geom_prior,
                    geom_per_head=cfg.MODEL.DAFUSION.GEOM_PER_HEAD,
                    geom_max_pairs=cfg.MODEL.DAFUSION.GEOM_MAX_PAIRS,
                )
            # per-stage KV grid size, needed to pool depth onto the key grid
            self._kv_strides = strides
        else:
            self.swin_d = None
            self.fusions = None

        # Freeze the RGB Swin branch (rgbd only): keep COCO-pretrained RGB features fixed so
        # only the depth branch + fusion + decoder learn. requires_grad=False excludes these
        # params from the optimizer (train_net.build_optimizer filters on requires_grad).
        self._freeze_rgb = cfg.MODEL.DAFUSION.FREEZE_RGB_BACKBONE and self.input_type == "rgbd"
        if self._freeze_rgb:
            for p in self.swin_rgb.parameters():
                p.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        # keep the frozen RGB branch in eval mode so drop_path/dropout stay off (deterministic
        # frozen features); survives every trainer .train() call.
        if getattr(self, "_freeze_rgb", False):
            self.swin_rgb.eval()
        return self

    def _depth_tokens(self, depth, feat, stride):
        """Pool the raw encoded depth onto this stage's query grid and (downsampled) key grid.

        The geometry prior needs actual distances, so it reads the DEPTH INPUT rather than depth
        *features*. Sampled at the undeformed reference grid, matching the approximation
        ContinuousPositionBias already makes (the true key locations are per-group deformed, which
        would make the bias per-group and blow up memory)."""
        h, w = feat.shape[-2:]
        kh, kw = max(1, h // stride), max(1, w // stride)
        validity = depth[:, 3:4] if self.has_validity else None
        z_q, w_q = pool_depth_scalar(depth, (h, w), self.depth_encoding, validity)
        z_k, w_k = pool_depth_scalar(depth, (kh, kw), self.depth_encoding, validity)
        return z_q, z_k, w_q, w_k

    def forward(self, rgb, depth=None):
        fr = self.swin_rgb(rgb)
        if self.input_type == "rgb":
            return fr
        if self.input_type == "depth":
            return self.swin_rgb(depth) if depth is not None else fr
        if self.depth_adapter is not None:
            depth_in = self.depth_adapter(depth)
        else:
            depth_in = depth
        fd = self.swin_d(depth_in)
        out = {}
        for i, name in enumerate(self._out_features):
            dt = (self._depth_tokens(depth, fr[name], self._kv_strides[i])
                  if self.geom_prior else None)
            out[name] = self.fusions[name](fr[name], fd[name], dt)
        return out

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32


@BACKBONE_REGISTRY.register()
def build_dafusion_backbone(cfg, input_shape):
    return DualSwinFusionBackbone(cfg, input_shape)
