"""Dual-input (RGB + depth) predictor for DA-Fusion eval.

Loads a trained checkpoint and, given an RGB image + a raw depth map, applies the SAME
depth encoding as training (HHA by default) and returns detectron2 Instances. Depth is
encoded per the config's INPUT.DEPTH_ENCODING using the benchmark's camera intrinsics.
"""
import os

import cv2
import numpy as np
from dafusion.data.plane_height import plane_height_map
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model
from detectron2.structures import Instances

from ..data.hha import depth_to_hha
from ..data.depth_xyz import depth_to_xyz, standardize_xyz
from ..data.depth_sdp import depth_to_sdp
from ..data.depth_normals import depth_to_normals, points_to_normals
from ..data.datasets.intrinsics import get_intrinsics

# Eval clamp for the "normalized" encoding on the REAL benchmark depth (OSD/OCID), in mm.
# Deliberately different from the training config's DEPTH_MIN/MAX (2500/15000, tuned for the
# UOAIS-Sim depth range ~5-9 m): the percentile normalization maps each image to its own
# p5-p95, and these bounds are chosen non-binding for tabletop depth (~0.3-1.8 m) so real
# frames aren't clipped to a constant. Matches the legacy eval (which scored the paper's 92).
EVAL_DEPTH_MIN = 300.0
EVAL_DEPTH_MAX = 1800.0


def _normalize_depth(depth, min_val=EVAL_DEPTH_MIN, max_val=EVAL_DEPTH_MAX):
    """Percentile-clamped min-max normalize a single-channel depth map to a 3-channel uint8
    image (legacy eval `normalize_depth`). No inversion here; the caller applies it."""
    min_val = max(np.percentile(depth, 5), min_val)
    max_val = min(np.percentile(depth, 95), max_val)
    depth = np.clip(depth, min_val, max_val)
    depth = (depth - min_val) / (max_val - min_val) * 255.0
    return np.repeat(depth[..., None], 3, axis=-1).astype(np.uint8)


def _inpaint_depth(depth, kernel_size=3):
    """Telea-inpaint zero (missing) depth pixels, as in the legacy eval. `depth` is 3-ch uint8."""
    mask = np.all(depth == 0, axis=2).astype(np.uint8)
    inpainted = cv2.inpaint(depth, mask, kernel_size, cv2.INPAINT_TELEA)
    return np.where(depth == 0, inpainted, depth)


class DAFusionPredictor:
    def __init__(self, cfg, dataset="uoais_sim"):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        DetectionCheckpointer(self.model).load(cfg.MODEL.WEIGHTS)
        self.input_type = cfg.INPUT.INPUT_TYPE
        self.depth_encoding = cfg.INPUT.DEPTH_ENCODING
        self.depth_inverted = cfg.INPUT.DEPTH_INVERTED
        self.validity_channel = cfg.INPUT.DEPTH_VALIDITY_CHANNEL
        # must mirror the mapper exactly, including channel ORDER (validity then height): a swapped
        # order would train fine and silently mis-evaluate
        self.plane_height_channel = cfg.INPUT.DEPTH_PLANE_HEIGHT_CHANNEL
        self.sdp_channels = cfg.INPUT.DEPTH_SDP_CHANNELS
        self.sdp_temperature = cfg.INPUT.DEPTH_SDP_TEMPERATURE
        self.sdp_max = cfg.INPUT.DEPTH_SDP_MAX
        self.sdp_local = cfg.INPUT.DEPTH_SDP_LOCAL
        self.intrinsics = get_intrinsics(dataset)

    def _encode_depth(self, depth_raw):
        """Encode raw depth exactly as training does. Returns (H,W,3), or (H,W,4) with channel 3
        = validity when INPUT.DEPTH_VALIDITY_CHANNEL is set (train/eval parity is mandatory —
        the network's 4th input slice must mean the same thing in both paths)."""
        # Validity is derived from the RAW input, before any encoding collapses holes into
        # values that collide with real geometry. Mirrors dafusion_mapper's valid_in.
        valid = (np.any(depth_raw != 0, axis=-1) if depth_raw.ndim == 3 else depth_raw > 0)

        if self.depth_encoding == "sdp":
            # OCBD supplies a metric XYZ cloud, not raw mm -> take its Z as the scalar
            # depth so SDP sees the same quantity it does in training.
            raw_mm = (depth_raw[..., 2] * 1000.0) if depth_raw.ndim == 3 else depth_raw
            enc = depth_to_sdp(raw_mm.astype(np.float32), self.sdp_channels,
                               self.sdp_temperature, self.sdp_max, local=self.sdp_local)
        elif self.depth_encoding == "hha":
            enc = depth_to_hha(depth_raw.astype(np.float32) / 1000.0, self.intrinsics)  # mm -> m
        elif self.depth_encoding == "xyz":
            # depth_raw is either (H,W) raw mm (OCID/OSD -> back-project with this dataset's
            # intrinsics) or an already-metric (H,W,3) XYZ point map (OCBD .npy). Standardize
            # per-image (center + isotropic scale), identical to training.
            pts = depth_raw.astype(np.float32) if depth_raw.ndim == 3 else \
                depth_to_xyz(depth_raw, self.intrinsics)
            enc = standardize_xyz(pts, valid)
        elif self.depth_encoding == "depth_normals":
            # (H,W,3) metric XYZ point map (OCBD) -> normals directly; else back-project raw mm.
            if depth_raw.ndim == 3:
                enc = points_to_normals(depth_raw.astype(np.float32), inverted=self.depth_inverted)
            else:
                enc = depth_to_normals(depth_raw.astype(np.float32), self.intrinsics,
                                       inverted=self.depth_inverted)
        else:
            # Per-dataset handling (matches legacy eval):
            #   3-channel input (OCBD) is already a normalized 8-bit depth image -> use as-is.
            #   single-channel input (OSD/OCID) is raw uint16 mm -> normalize + inpaint holes.
            if depth_raw.ndim == 3:
                enc = depth_raw.astype(np.uint8)
            else:
                enc = _normalize_depth(depth_raw.astype(np.float32))
                enc = _inpaint_depth(enc)
            if self.depth_inverted:
                enc = (255 - enc).astype(np.uint8)

        out = [enc.astype(np.float32)]
        if self.validity_channel:
            out.append(valid.astype(np.float32)[..., None])
        if self.plane_height_channel:
            out.append(plane_height_map(depth_raw)[..., None])
        return np.concatenate(out, axis=-1) if len(out) > 1 else enc

    def _forward(self, rgb_in, depth_enc, h, w):
        """Single forward pass at native (h,w) output resolution. `rgb_in`/`depth_enc` may
        already be a flipped/otherwise-transformed view; this only handles the optional
        shortest-edge upsample and the model call."""
        # Optional legacy-style shortest-edge upsample at eval (env-gated; default off = native
        # resolution, unchanged behavior). The legacy build fed the net ~800 shortest-edge via
        # detectron2's MIN_SIZE_TEST=800; masks are resized back to native (h,w) by the meta-arch
        # (inp height/width), so metrics stay at native resolution.
        minsize = int(os.environ.get("DAFUSION_EVAL_MINSIZE", "0"))
        if minsize > 0:
            maxsize = int(os.environ.get("DAFUSION_EVAL_MAXSIZE", "1333"))
            scale = minsize / min(h, w)
            if round(scale * max(h, w)) > maxsize:
                scale = maxsize / max(h, w)
            nh, nw = int(round(h * scale)), int(round(w * scale))
            rgb_in = cv2.resize(rgb_in, (nw, nh), interpolation=cv2.INTER_LINEAR)
            if depth_enc is not None:
                if self.validity_channel or self.plane_height_channel:
                    # Resample the geometry channels bilinearly. The validity flag must use NEAREST:
                    # interpolating a 0/1 mask would invent fractional "half-valid" pixels along every
                    # hole border, which is the ambiguity the channel exists to remove. Height IS
                    # continuous, so it interpolates like geometry. Channel order here must match the
                    # order appended in _encode_depth (geometry, validity, height).
                    n_extra = int(self.validity_channel) + int(self.plane_height_channel)
                    geo = cv2.resize(depth_enc[..., :-n_extra], (nw, nh), interpolation=cv2.INTER_LINEAR)
                    parts = [geo]
                    idx = depth_enc.shape[-1] - n_extra
                    if self.validity_channel:
                        parts.append(cv2.resize(depth_enc[..., idx], (nw, nh),
                                                interpolation=cv2.INTER_NEAREST)[..., None])
                        idx += 1
                    if self.plane_height_channel:
                        parts.append(cv2.resize(depth_enc[..., idx], (nw, nh),
                                                interpolation=cv2.INTER_LINEAR)[..., None])
                    depth_enc = np.concatenate(parts, axis=-1)
                else:
                    depth_enc = cv2.resize(depth_enc, (nw, nh), interpolation=cv2.INTER_LINEAR)
        image = torch.as_tensor(np.ascontiguousarray(rgb_in.transpose(2, 0, 1))).float()
        inp = {"image": image, "height": h, "width": w}
        if depth_enc is not None:
            inp["depth_image"] = torch.as_tensor(np.ascontiguousarray(depth_enc.transpose(2, 0, 1))).float()
        # DAFUSION_ABLATE_MODALITY={depth,rgb}: blank one modality at inference to measure how much each
        # benchmark actually depends on it. Each channel is filled with its OWN per-image mean, not zero:
        # that removes all spatial information (the point of the ablation) while keeping the magnitude
        # in-distribution, so the model is not additionally penalised for an out-of-range input -- these
        # tensors are normalised inside the meta-arch, where a hard zero would land far from the mean.
        # Diagnostic only; unset (the default) leaves the forward pass untouched.
        _abl = os.environ.get("DAFUSION_ABLATE_MODALITY", "none")
        if _abl == "depth" and "depth_image" in inp:
            d = inp["depth_image"]
            inp["depth_image"] = d.mean(dim=(1, 2), keepdim=True).expand_as(d).contiguous()
        elif _abl == "rgb":
            im = inp["image"]
            inp["image"] = im.mean(dim=(1, 2), keepdim=True).expand_as(im).contiguous()
        outputs = self.model([inp])[0]
        return outputs["instances"].to("cpu")

    def _flip_depth_enc(self, depth_enc):
        """Horizontally flip an already-encoded depth map for TTA, applying the same
        channel-sign fixups the training mapper uses for directional encodings (xyz's X
        channel / depth_normals' normal_x channel) -- see dafusion_mapper._apply_depth_transforms.

        A validity channel (index 3, when enabled) is deliberately NOT fixed up: it is a flag,
        not a direction, so mirroring it is the whole transform. The fixups below index only
        channels 0/1, so it is already left alone -- asserted here so a future channel reorder
        can't silently corrupt it."""
        flipped = np.flip(depth_enc, axis=1).copy()
        if self.depth_encoding == "xyz":
            flipped[..., 0] *= -1.0
        elif self.depth_encoding == "depth_normals":
            flipped[..., 1] = 255 - flipped[..., 1]
        if self.validity_channel:
            assert flipped.shape[-1] == 4, f"expected 4 depth channels, got {flipped.shape[-1]}"
            np.testing.assert_array_equal(flipped[..., 3], np.flip(depth_enc, axis=1)[..., 3])
        return flipped

    @staticmethod
    def _unflip_instances(instances, w):
        """Undo a horizontal flip on predicted masks/boxes so a flipped-view's detections
        align with the original image orientation and can be merged with the unflipped view."""
        if instances.has("pred_masks"):
            instances.pred_masks = torch.flip(instances.pred_masks, dims=[-1])
        if instances.has("pred_boxes"):
            x1, y1, x2, y2 = instances.pred_boxes.tensor.unbind(-1)
            instances.pred_boxes.tensor = torch.stack([w - x2, y1, w - x1, y2], dim=-1)
        return instances

    @torch.no_grad()
    def __call__(self, rgb_img, depth_raw=None):
        """rgb_img: HxWx3 (BGR from cv2 or RGB — matched to cfg.INPUT.FORMAT upstream).
        depth_raw: HxW raw depth (e.g. uint16 mm). Returns Instances (cpu).

        Optional horizontal-flip TTA (env DAFUSION_EVAL_FLIP_TTA=1): also run the
        horizontally-flipped image, un-flip its detections back to the original orientation,
        and concatenate with the unflipped view's detections. This is a union-style ensemble --
        downstream post-processing (`eval.post_process.merge_overlaps`) already merges
        near-duplicate masks with >0.7 overlap, so agreeing detections from both views collapse
        into one, while a detection unique to one view survives as-is (helps recall on objects
        one view missed; does not by itself suppress a hallucination unique to one view)."""
        h, w = rgb_img.shape[:2]
        depth_enc = self._encode_depth(depth_raw) if self.input_type != "rgb" else None
        instances = self._forward(rgb_img, depth_enc, h, w)

        if os.environ.get("DAFUSION_EVAL_FLIP_TTA", "0") == "1":
            rgb_flip = cv2.flip(rgb_img, 1)
            depth_flip = self._flip_depth_enc(depth_enc) if depth_enc is not None else None
            inst_flip = self._forward(rgb_flip, depth_flip, h, w)
            inst_flip = self._unflip_instances(inst_flip, w)
            instances = Instances.cat([instances, inst_flip])

        return instances
