"""Auxiliary instance-boundary supervision.

Why this and not another representation change. By elimination (docs/EXPERIMENTS.md Tracks 10d-10g)
the OCID ceiling is **under-segmentation of touching objects** in the model's own predictions:
    * the model emits FEWER instances than exist -- 8.4 vs 8.9 GT in the bucket holding 81% of OCID
      -- so the precision loss is not spurious detections (Track 10f);
    * ten runs across four unrelated encoder/fusion mechanisms all cap at ~88.2 and refuse to stack
      (Track 10e);
    * the decoder's query count is exonerated (it under-counts, it does not over-count);
    * post-processing is exonerated -- `merge_overlaps` is inert and `refined_mask` already *helps*
      by +4 (Track 10g);
    * eval protocol / annotations are exonerated -- OCID is pixel-aligned (dx=0.0, Track 9c).
What is left is the **training objective**: nothing in it ever asks the model to separate two
adjacent instances. Boundary F-measure runs 4-8 points below Overlap F on every benchmark, which is
the direct symptom.

The head predicts a per-pixel "is this an instance boundary" map from the pixel decoder's
high-resolution mask features, supervised by the morphological edges of the GT instance masks.
Boundaries are free supervision -- derived from masks already loaded -- and the signal is
sensor-invariant in a way appearance is not: a real depth/instance discontinuity looks like a
synthetic one far more than real RGB looks like rendered RGB, so it should survive sim->real.

Deliberately an AUXILIARY loss, not a change to mask prediction: it shapes the shared features so
adjacent instances become separable, while leaving Mask2Former's decoder, matcher and mask losses
exactly as the paper specifies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def instance_boundary_target(gt_masks, out_hw):
    """(N,H,W) bool GT instance masks -> (1,h,w) float boundary map in {0,1}.

    Uses each instance's own morphological gradient, then unions them, so the boundary BETWEEN two
    touching objects is marked twice over and never cancels -- that seam is exactly the signal we
    want. A union-then-edge would miss it entirely, since the union of two touching masks has no
    internal edge.
    """
    if gt_masks is None or len(gt_masks) == 0:
        return torch.zeros((1, *out_hw), dtype=torch.float32, device=(
            gt_masks.device if gt_masks is not None else "cpu"))
    m = gt_masks.float().unsqueeze(1)                                  # (N,1,H,W)
    # 3x3 max-pool minus min-pool == morphological gradient == 1-px boundary ring per instance
    dil = F.max_pool2d(m, 3, stride=1, padding=1)
    ero = -F.max_pool2d(-m, 3, stride=1, padding=1)
    edges = (dil - ero).clamp(0, 1)                                    # (N,1,H,W)
    b = edges.amax(dim=0)                                              # (1,H,W) union over instances
    if b.shape[-2:] != tuple(out_hw):
        b = F.interpolate(b.unsqueeze(0), size=out_hw, mode="nearest").squeeze(0)
    return b


class BoundaryHead(nn.Module):
    """Tiny conv head predicting instance boundaries from mask features."""

    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, hidden, 3, padding=1), nn.GroupNorm(1, hidden), nn.GELU(),
            nn.Conv2d(hidden, 1, 1),
        )
        # Start with a strong negative bias: boundaries are ~2-5% of pixels, so predicting "not a
        # boundary" everywhere is the correct prior and keeps the auxiliary term from dominating
        # early training before the features can support it.
        nn.init.constant_(self.net[-1].bias, -4.0)

    def forward(self, feats):
        return self.net(feats)

    def loss(self, logits, gt_masks_list, pos_weight=5.0):
        """Weighted BCE against instance-edge targets, averaged over the batch.

        pos_weight compensates the ~20:1 background:boundary imbalance; without it the head
        trivially predicts all-zero and the term contributes no gradient.
        """
        h, w = logits.shape[-2:]
        tgt = torch.stack([instance_boundary_target(g, (h, w)) for g in gt_masks_list]).to(
            logits.dtype)                                              # (B,1,h,w)
        pw = torch.as_tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        return F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pw)
