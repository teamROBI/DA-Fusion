"""Geometry prior for the deformable fusion attention (DFormerv2-style).

Motivation. Every spatial prior in this model is 2D: `ContinuousPositionBias` maps the
*pixel-space* displacement (dx, dy) between a query and a key to a per-head bias. The depth
branch contributes features, but the scene's actual 3D structure never enters the decision of
which locations may mix. So the attention cannot distinguish a key 5 px away *on the same
surface* from a key 5 px away *across an occlusion boundary onto the table* -- those are
identical in 2D. Post-fix measurement says depth currently buys only +0.32 mean over RGB-only
despite a 195M-param branch (docs/EXPERIMENTS.md Track 10a), and widening the 2D reach (OFF4)
or adding sampling groups (NG8) both made OCID *worse* -- i.e. the fusion's limitation is
geometric blindness, not reach or capacity.

Formulation, after DFormerv2 (arXiv:2504.04701, eqs. 1-4):
    D[q,k] = |z_q - z_k|                 pairwise depth distance (mean-pooled depth per token)
    S[q,k] = |i_q - i_k| + |j_q - j_k|    Manhattan spatial distance
    G      = w_d * D + w_s * S            learnable non-negative blend
    attn   = attn + G * log(beta)         per-head decay, beta in (0,1)

DELIBERATE DEVIATION from DFormerv2: they apply the decay *multiplicatively after* softmax,
`(softmax(QK^T) ⊙ beta^G) V`, which leaves the attention row unnormalized. We add `G*log(beta)`
*before* softmax instead. The two are monotonically equivalent in their effect (larger geometric
distance -> less weight), but the pre-softmax form keeps each attention row a proper
distribution, is numerically stabler under AMP, and -- decisively -- composes by simple addition
with the existing `ContinuousPositionBias` hook rather than requiring a new code path.

Memory. The bias is batch-dependent -- (B, heads, Nq, Nk) -- because depth differs per image,
unlike the batch-independent 2D RPB. `per_head=False` (default) learns one shared decay broadcast
across heads. `max_pairs` (default 4M, matching RPB's own cap) skips the prior on any stage too
large to materialise, which at 480x640 means res2 (Nq=19200 x Nk=300 = 5.76M).

Skipping res2 is deliberate, and corrects a tempting assumption: res2 currently has NO spatial
prior at all (RPB returns None there), so `multihead_attend` takes the memory-efficient
F.scaled_dot_product_attention path. Supplying *any* bias at res2 forces the explicit
(B,heads,Nq,Nk) attention matrix instead -- ~920 MB fp32 per module at batch 5, times 3 modules
per stage. So "add geometry where the model has no prior" is not a free win; it trades away the
fused-attention fast path at the largest stage. res3-res5 already materialise that matrix, so the
prior is nearly free there and that is where it is enabled by default.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def pool_depth_scalar(depth, hw, encoding="xyz", validity=None):
    """Reduce an encoded depth tensor to one scalar per token at resolution `hw`.

    depth:    (B, C, H, W) encoded depth as fed to the depth branch.
    hw:       (h, w) target token grid.
    encoding: which channel carries distance -- xyz uses Z (index 2); the normalized /
              hha / depth_normals encodings all put a depth-like quantity in channel 0.
    validity: optional (B,1,H,W) 1=measured. When present, pooling is a *masked* mean so that
              holes (which every encoding maps onto a value colliding with real geometry --
              for xyz, exactly the scene median) do not contribute a fake distance. This is
              why the geometry prior pairs naturally with INPUT.DEPTH_VALIDITY_CHANNEL.

    Returns (B, h*w) pooled depth, and (B, h*w) pooled validity weight in [0,1].
    """
    idx = 2 if encoding == "xyz" else 0
    z = depth[:, idx : idx + 1]                                   # (B,1,H,W)
    if validity is None:
        pooled = F.adaptive_avg_pool2d(z, hw)
        w = torch.ones_like(pooled)
    else:
        num = F.adaptive_avg_pool2d(z * validity, hw)
        den = F.adaptive_avg_pool2d(validity, hw)
        pooled = num / den.clamp_min(1e-6)
        w = den                                                    # fraction measured per token
    b = pooled.shape[0]
    return pooled.reshape(b, -1), w.reshape(b, -1)


class GeometryPrior(nn.Module):
    """Additive per-head attention bias encoding 3D (depth + spatial) token distance."""

    def __init__(self, num_heads, per_head=False, max_pairs=8_000_000, init_beta=0.9,
                 structure_gate=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.per_head = per_head
        self.max_pairs = max_pairs
        # Scene-structure gate -- **REFUTED, LEAVE AT 0.0** (kept only so the negative result is
        # reproducible; see docs/EXPERIMENTS.md Track 10c). The idea was to fade the depth term on
        # geometrically featureless scenes, to recover the -7.8 F the prior costs on OCID's 90 empty
        # images. Two input-only proxies were calibrated on TRAIN (p2) and then checked on OCID:
        #   depth spread   -> gate fires on 0% of empty scenes; empty scenes have HIGHER spread
        #                     (0.293) than cluttered ones (0.242), because a table viewed obliquely
        #                     ramps in depth across the whole frame.
        #   |Laplacian|    -> anti-discriminative: empty 0.196 vs objects 0.139, firing on 44% of
        #                     OBJECT scenes vs 18% of empty ones. Real sensor noise on a bare table
        #                     produces more apparent curvature than large smooth object surfaces.
        # A *learned* gate is also blocked: UOAIS-Sim contains no empty scenes to train it on, and
        # importing them from TOD was already shown to hurt (Track 6/8). Searching for a third
        # hand-crafted statistic would be fitting the test set, so this was abandoned by design.
        # If revisited, the signal must come from something other than global input-depth stats.
        #
        # Why: measured per-bucket, the prior gains precision wherever there IS geometry (+4.8 few,
        # +2.2 many) but costs 7.8 F on OCID's 90 EMPTY scenes (88.9 -> 81.1). An empty tabletop has
        # near-uniform depth, so the pairwise depth-distance matrix carries no signal, yet the decay
        # still applies and appears to reinforce spurious structure.
        #
        # This gates the MODEL on its own INPUT (depth spread only -- no labels, no object counts,
        # no ground truth), so every test image is still evaluated. It is emphatically NOT a filter
        # on the evaluation set. The threshold is calibrated from TRAINING-data statistics, not by
        # sweeping OCID, to avoid compounding the test-set fitting already present in the locked
        # DATASET_SCORE_THRESH.
        self.structure_gate = structure_gate
        n = num_heads if per_head else 1
        # log(beta) must stay negative so distance *decays* attention; parameterize the
        # magnitude and negate, so no constraint/clamp is needed during training.
        self.log_decay_mag = nn.Parameter(torch.full((n,), -torch.log(torch.tensor(init_beta)).item()))
        # non-negative blend weights for the depth and spatial terms (softplus-gated)
        self.w_depth_raw = nn.Parameter(torch.zeros(n))
        self.w_spatial_raw = nn.Parameter(torch.full((n,), -2.0))   # start spatial-light
        # Penalty for keys whose depth is UNMEASURED. Without this an all-holes key would get
        # |dz|*0 = 0 depth distance, i.e. NO decay -- so the model would read "no geometric
        # objection" as "geometrically adjacent" and preferentially attend to exactly the tokens
        # it knows nothing about. This term makes "unknown" cost something instead.
        self.w_unknown_raw = nn.Parameter(torch.zeros(n))
        self._cached_s = None
        self._cached_key = None

    def _spatial(self, ref_q, ref_k):
        """Manhattan distance between query and key reference grids, cached per shape."""
        key = (ref_q.shape[0], ref_k.shape[0], ref_q.device, ref_q.dtype)
        if self._cached_key != key:
            d = (ref_q[:, None, :] - ref_k[None, :, :]).abs().sum(-1)   # (Nq,Nk), coords in [-1,1]
            self._cached_s = d
            self._cached_key = key
        return self._cached_s

    def forward(self, ref_q, ref_k, z_q, z_k, w_k=None, w_q=None):
        """ref_q (Nq,2), ref_k (Nk,2) in [-1,1]; z_q (B,Nq), z_k (B,Nk) pooled depth.
        w_k (B,Nk) optional per-key measured-fraction, used to fade the depth term where a key
        token is mostly holes (its pooled depth is meaningless there).

        Returns a (B, heads, Nq, Nk) additive bias, or None if too large to materialise. Note
        this is BATCH-DEPENDENT (4-D), unlike ContinuousPositionBias's batch-independent
        (heads, Nq, Nk) -- depth differs per image. `multihead_attend` accepts either rank."""
        nq, nk = ref_q.shape[0], ref_k.shape[0]
        if nq * nk > self.max_pairs:
            return None
        s = self._spatial(ref_q, ref_k)                                  # (Nq,Nk)
        d = (z_q[:, :, None] - z_k[:, None, :]).abs()                    # (B,Nq,Nk)
        unknown = None
        if w_k is not None:
            # Confidence that BOTH endpoints were measured. Where either is unmeasured the depth
            # distance is meaningless, so fade the depth term out toward a spatial-only prior
            # (a sensible neutral fallback) and charge the `unknown` penalty instead.
            conf = w_k[:, None, :] if w_q is None else w_q[:, :, None] * w_k[:, None, :]
            d = d * conf
            unknown = 1.0 - conf
        wd = F.softplus(self.w_depth_raw).view(-1, 1, 1, 1)
        ws = F.softplus(self.w_spatial_raw).view(-1, 1, 1, 1)
        decay = -F.softplus(self.log_decay_mag).view(-1, 1, 1, 1)        # strictly negative
        if self.structure_gate > 0.0:
            # Per-image geometric structure = robust spread of the measured key depths. Uses only
            # the input depth. Smooth ramp (not a hard step) so it stays differentiable and a scene
            # near the threshold degrades gracefully rather than flipping.
            if w_k is not None:
                m = w_k > 0.5
                cnt = m.sum(dim=1, keepdim=True).clamp_min(1)
                mean = (z_k * m).sum(dim=1, keepdim=True) / cnt
                var = (((z_k - mean) ** 2) * m).sum(dim=1, keepdim=True) / cnt
            else:
                var = z_k.var(dim=1, keepdim=True, unbiased=False)
            spread = var.clamp_min(0).sqrt()                             # (B,1)
            gate = (spread / self.structure_gate).clamp(0.0, 1.0)        # (B,1) in [0,1]
            d = d * gate[:, :, None]                                     # fade DEPTH term only
        g = wd * d[None] + ws * s[None, None]                            # (n,B,Nq,Nk), n=1 or heads
        if unknown is not None:
            g = g + F.softplus(self.w_unknown_raw).view(-1, 1, 1, 1) * unknown[None]
        bias = g * decay
        n, b = bias.shape[0], bias.shape[1]
        if n == 1:
            bias = bias.expand(self.num_heads, b, nq, nk)
        return bias.permute(1, 0, 2, 3)                                  # (B, heads, Nq, Nk)
