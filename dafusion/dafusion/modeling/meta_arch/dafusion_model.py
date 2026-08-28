# DA-Fusion meta-architecture — dual-input (RGB + HHA depth) variant of Mask2Former's
# MaskFormer. The backbone is the DualSwinFusionBackbone, which takes two tensors and
# returns fused multi-scale features; everything downstream (pixel decoder, transformer
# decoder, criterion, inference) is unchanged Mask2Former.
import os
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom

from ..criterion import SetCriterion
from ..matcher import HungarianMatcher
from ..boundary_head import BoundaryHead


def _depth_norm(values, cfg, fill):
    """Size a per-channel depth mean/std list to the channel count actually arriving.

    depth_image carries: <encoding channels> [+ 1 validity channel]. The encoding is 3 channels
    except SDP (INPUT.DEPTH_SDP_CHANNELS). SDP output is already in [-1,1] and the validity flag is
    already 0/1, so the extra channels are passed through unnormalized (mean 0 / std 1) rather than
    inheriting the ImageNet-style constants meant for the 3-channel pseudo-image encodings.

    Sized here rather than in every config so a channel-count change cannot silently
    broadcast-fail against a 3-element buffer. Idempotent when already the right length.
    """
    values = list(values)
    n = cfg.INPUT.DEPTH_SDP_CHANNELS if cfg.INPUT.DEPTH_ENCODING == "sdp" else 3
    if cfg.INPUT.DEPTH_VALIDITY_CHANNEL:
        n += 1
    # height-above-support-plane is already in [0,1], so like the validity flag it is passed through
    # unnormalized. Missing this is what made the first plane-height run die instantly with
    # "size of tensor a (5) must match tensor b (4)" -- the depth buffers were still sized for 4.
    if getattr(cfg.INPUT, "DEPTH_PLANE_HEIGHT_CHANNEL", False):
        n += 1
    if len(values) == n:
        return values
    if len(values) > n:
        return values[:n]
    return values + [fill] * (n - len(values))


@META_ARCH_REGISTRY.register()
class DAFusion(nn.Module):
    """Deformable Attention-based RGB-D fusion for class-agnostic UOIS."""

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        pixel_mean_depth: Tuple[float],
        pixel_std_depth: Tuple[float],
        input_type: str,
        boundary_head: nn.Module,
        boundary_weight: float,
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.input_type = input_type
        self.boundary_head = boundary_head
        self.boundary_weight = boundary_weight
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        # HHA depth uses its own normalization (it is not natural RGB)
        self.register_buffer("pixel_mean_depth", torch.Tensor(pixel_mean_depth).view(-1, 1, 1), False)
        self.register_buffer("pixel_std_depth", torch.Tensor(pixel_std_depth).view(-1, 1, 1), False)

        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}
        # amodal auxiliary supervision: weighted RELATIVE to the visible mask losses, so a single knob
        # (AMODAL_WEIGHT, a fraction) sets how much the auxiliary task counts without re-tuning the
        # primary balance. 0 = off and nothing is added anywhere.
        amodal_w = getattr(cfg.MODEL.MASK_FORMER, "AMODAL_WEIGHT", 0.0)
        if amodal_w > 0:
            weight_dict["loss_amodal_mask"] = mask_weight * amodal_w
            weight_dict["loss_amodal_dice"] = dice_weight * amodal_w
        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "masks"] + (["amodal"] if amodal_w > 0 else [])
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # The validity channel (INPUT.DEPTH_VALIDITY_CHANNEL) is already 0/1, so it is passed
            # through unnormalized (mean 0, std 1). Appended here rather than in every config so
            # the 3-element depth mean/std stay the single source of truth for the geometry
            # channels -- and so a 4-channel run can't silently broadcast-fail against a
            # 3-element buffer.
            "pixel_mean_depth": _depth_norm(cfg.MODEL.DAFUSION.PIXEL_MEAN_DEPTH, cfg, 0.0),
            "pixel_std_depth": _depth_norm(cfg.MODEL.DAFUSION.PIXEL_STD_DEPTH, cfg, 1.0),
            "input_type": cfg.MODEL.DAFUSION.INPUT_TYPE,
            # Auxiliary boundary head on the FUSED res2 feature (stride 4). Shapes the
            # fused representation so adjacent instances become separable, without
            # touching the Mask2Former decoder/matcher/mask losses.
            "boundary_head": (BoundaryHead(backbone.output_shape()["res2"].channels,
                                          cfg.MODEL.DAFUSION.BOUNDARY_HIDDEN)
                              if cfg.MODEL.DAFUSION.BOUNDARY_WEIGHT > 0 else None),
            "boundary_weight": cfg.MODEL.DAFUSION.BOUNDARY_WEIGHT,
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _normalize(self, batched_inputs, key, mean, std):
        imgs = [x[key].to(self.device) for x in batched_inputs]
        imgs = [(x - mean) / std for x in imgs]
        return ImageList.from_tensors(imgs, self.size_divisibility)

    def forward(self, batched_inputs):
        images = self._normalize(batched_inputs, "image", self.pixel_mean, self.pixel_std)
        if self.input_type == "rgbd":
            depth = self._normalize(batched_inputs, "depth_image", self.pixel_mean_depth, self.pixel_std_depth)
            features = self.backbone(images.tensor, depth.tensor)
        elif self.input_type == "depth":
            depth = self._normalize(batched_inputs, "depth_image", self.pixel_mean_depth, self.pixel_std_depth)
            features = self.backbone(images.tensor, depth.tensor)  # depth-only handled in backbone
        else:  # rgb
            features = self.backbone(images.tensor)
        outputs = self.sem_seg_head(features)

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            losses = self.criterion(outputs, targets)
            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    losses.pop(k)
            if self.boundary_head is not None and targets is not None:
                # Supervised by the morphological edges of the GT instance masks -- free labels,
                # and the seam between two touching objects is what the model is failing on.
                b_logits = self.boundary_head(features["res2"])
                losses["loss_boundary"] = self.boundary_weight * self.boundary_head.loss(
                    b_logits, [t["masks"] for t in targets])
            return losses

        mask_cls_results = outputs["pred_logits"]
        mask_pred_results = outputs["pred_masks"]
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            mode="bilinear",
            align_corners=False,
        )
        del outputs
        processed_results = []
        for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
            mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            processed_results.append({})
            if self.sem_seg_postprocess_before_inference:
                mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                    mask_pred_result, image_size, height, width
                )
                mask_cls_result = mask_cls_result.to(mask_pred_result)
            if self.semantic_on:
                r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                if not self.sem_seg_postprocess_before_inference:
                    r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                processed_results[-1]["sem_seg"] = r
            if self.panoptic_on:
                panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["panoptic_seg"] = panoptic_r
            if self.instance_on:
                instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                processed_results[-1]["instances"] = instance_r
        return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros(
                (gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device
            )
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            tgt = {"labels": targets_per_image.gt_classes, "masks": padded_masks}
            # amodal targets ride along only when the mapper produced them, padded identically so the
            # two heads see pixel-aligned supervision
            if targets_per_image.has("gt_amodal_masks"):
                am = targets_per_image.gt_amodal_masks
                padded_am = torch.zeros((am.shape[0], h_pad, w_pad), dtype=am.dtype, device=am.device)
                padded_am[:, : am.shape[1], : am.shape[2]] = am
                tgt["amodal_masks"] = padded_am
            new_targets.append(tgt)
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid()
        return torch.einsum("qc,qhw->chw", mask_cls, mask_pred)

    def instance_inference(self, mask_cls, mask_pred):
        image_size = mask_pred.shape[-2:]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = (
            torch.arange(self.sem_seg_head.num_classes, device=self.device)
            .unsqueeze(0)
            .repeat(self.num_queries, 1)
            .flatten(0, 1)
        )
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        result = Instances(image_size)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (
            result.pred_masks.flatten(1).sum(1) + 1e-6
        )
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        if os.environ.get("DAFUSION_KEEP_SOFT_MASKS") == "1":
            # `pred_masks` above is binarised at logit>0 i.e. sigmoid 0.5 -- a hardcoded, never-swept
            # threshold that is the most direct precision/recall knob in the pipeline, because the
            # UOIS metric is pixel-weighted (recall = TP_px / GT_px). Keeping the probabilities lets
            # scripts/dump_raw_preds.py cache them once so any binarisation threshold (and any
            # soft-mask-based overlap resolution) can be swept offline on CPU.
            # Also expose the two score factors separately so score recalibration is testable.
            result.soft_masks = mask_pred.sigmoid()
            result.cls_scores = scores_per_image
            result.mask_scores = mask_scores_per_image
        return result
