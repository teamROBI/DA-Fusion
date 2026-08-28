"""Learnable depth modality adapter (Vanishing Depth, arXiv:2503.19947).

Purpose: an RGB-pretrained encoder expects natural-image statistics. Depth -- whether raw,
normalized, XYZ, or sinusoidally encoded -- is not that. The adapter is a small learnable module
that maps an arbitrary-channel depth representation into something the pretrained patch embedding
can consume, instead of forcing the depth branch's first conv to absorb that mismatch.

This is the component with the strongest *segmentation* evidence in the paper (+2.1 mIoU on DINOv2,
+8.0 on EVA-02, Table 4) -- and the one thing this repo had NOT tried; an earlier round conflated
it with patch-embed weight *inflation*, which merely widens the existing conv with a zero-init
input slice and is a different thing entirely.

IMPORTANT CAVEAT ON TRANSFER, so the result is read correctly: the paper's adapter gains are all
measured with the RGB encoder FROZEN ("During training, we freeze the pretrained RGB encoders";
Table 4 is "on top of the frozen non-finetuned backbones"). Its stated thesis is depth
understanding *without* finetuning. We fully fine-tune both Swin branches, and freezing the RGB
backbone was already measured to be catastrophic here (Track 4: 82.6 mean, OCID -7.4). Under full
finetuning the depth branch can learn much of this mapping itself, so the adapter's headroom is
expected to be far smaller for us. That is a hypothesis this module exists to test, not to assume.

Design: depthwise-separable 3x3 -> GELU -> pointwise 1x1 to exactly `out_chans`, plus a residual
path when the channel counts allow. Kept deliberately small (a few hundred K params) so any gain
is attributable to the representation mapping rather than to added capacity. Output is zero-init on
the final projection so training starts as a near-identity perturbation of the existing pipeline.
"""
import torch
import torch.nn as nn


class DepthAdapter(nn.Module):
    def __init__(self, in_chans, out_chans=3, hidden=64, zero_init=True):
        super().__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.dw = nn.Conv2d(in_chans, in_chans, 3, padding=1, groups=in_chans, bias=False)
        self.pw = nn.Conv2d(in_chans, hidden, 1, bias=True)
        self.norm = nn.GroupNorm(1, hidden)
        self.act = nn.GELU()
        self.proj = nn.Conv2d(hidden, out_chans, 1, bias=True)
        # Identity-ish start: with proj zeroed, the adapter initially emits the skip only, so the
        # run begins numerically close to the no-adapter baseline and *learns* its way out. Without
        # this, a randomly-initialised adapter would scramble the input to a pretrained stem.
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        y = self.proj(self.act(self.norm(self.pw(self.dw(x)))))
        # Skip path: reuse the first out_chans input channels when available so the pretrained stem
        # still sees the original signal at init (and the adapter is a learned correction to it).
        if self.in_chans >= self.out_chans:
            y = y + x[:, : self.out_chans]
        return y
