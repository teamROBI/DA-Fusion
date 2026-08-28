"""DA-Fusion dataset mapper: emits RGB ``image`` + HHA (or normalized) ``depth_image``
with a SHARED geometric augmentation, plus class-agnostic instances from the visible mask.

Unlike the legacy mapper (which re-sampled augmentation independently for depth, risking
rgb/depth misalignment — masked only because augmentation was off), this applies the SAME
sampled transforms to both modalities.
"""
import copy

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image

from detectron2.data import DatasetCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.structures import Instances
from fvcore.transforms.transform import HFlipTransform

from ..color_aug import ColorAugSSDTransform
from ..hha import depth_to_hha
from ..depth_noise import augment_depth_mm, inpaint_holes_u8
from ..depth_sdp import depth_to_sdp
from ..depth_xyz import depth_to_xyz, standardize_xyz
from ..depth_normals import depth_to_normals
from ..copy_paste import decode_masks, copy_paste_rgbd
from ..upright import estimate_roll, apply_roll
from ..plane_height import plane_height_map
from ..support_aug import randomize_support_surface
from ..datasets.intrinsics import get_intrinsics


def _normalize_depth(depth, min_val=2500.0, max_val=15000.0, inverted=True):
    min_val = max(np.percentile(depth, 5), min_val)
    max_val = min(np.percentile(depth, 95), max_val)
    depth = np.clip(depth, min_val, max_val)
    depth = (depth - min_val) / (max_val - min_val) * 255.0
    depth = np.repeat(depth[..., None], 3, axis=-1).astype(np.uint8)
    if inverted:
        depth = 255 - depth
    return depth


def build_augmentation(cfg, is_train):
    h, w = 480, 640
    if is_train and cfg.INPUT.AUGMENTATION:
        return [
            T.ResizeScale(min_scale=cfg.INPUT.MIN_SCALE, max_scale=cfg.INPUT.MAX_SCALE,
                          target_height=h, target_width=w),
            # seg_pad_value=0 is REQUIRED (detectron2 defaults it to 255, the semantic-seg
            # ignore label). Anything "segmentation-shaped" here is read as a boolean via
            # `!= 0` / BitMasks(...).to(bool), so a 255 pad reads as TRUE: every GT instance
            # mask swallows the pad border, and the xyz validity mask marks the pad (128.0 m,
            # from pad_value) as valid, which inflates standardize_xyz's p95 radius and
            # crushes the real scene's geometry toward a constant. Measured before the fix:
            # 27% of UOAIS-Sim samples corrupted. See docs/EXPERIMENTS.md Track 9.
            T.FixedSizeCrop(crop_size=(h, w), seg_pad_value=0),
            T.RandomFlip(horizontal=True),
        ]
    return [T.Resize((h, w))]


class DAFusionDatasetMapper:
    def __init__(self, cfg, is_train=True):
        self.is_train = is_train
        self.image_format = cfg.INPUT.FORMAT
        self.input_type = cfg.INPUT.INPUT_TYPE
        self.depth_encoding = cfg.INPUT.DEPTH_ENCODING
        self.depth_inverted = cfg.INPUT.DEPTH_INVERTED
        self.depth_min = cfg.INPUT.DEPTH_MIN
        self.depth_max = cfg.INPUT.DEPTH_MAX
        self.depth_noise = cfg.INPUT.DEPTH_NOISE
        self.hole_pattern = cfg.INPUT.DEPTH_HOLE_PATTERN        # measured dead-band/shadow holes
        self.keep_empty_neg = cfg.INPUT.KEEP_EMPTY_NEGATIVES   # train empty views as negatives
        self.validity_channel = cfg.INPUT.DEPTH_VALIDITY_CHANNEL   # emit a 4th "measured?" channel
        # height above the fitted support plane: 0 on the floor/table, positive on objects standing on
        # it. Measured AUC for separating GT objects from background on OCID: 0.974 floor / 0.942 table.
        self.plane_height_channel = cfg.INPUT.DEPTH_PLANE_HEIGHT_CHANNEL
        # modality dropout (train only): force each branch to be independently competent. See
        # config.py for the measured motivation (depth AUC 0.95 on OSD/OCID vs 0.64 on OCBD).
        self.amodal_targets = getattr(cfg.MODEL.MASK_FORMER, "AMODAL_WEIGHT", 0.0) > 0 and is_train
        self.mdrop_depth = cfg.INPUT.MODALITY_DROPOUT_DEPTH if is_train else 0.0
        self.mdrop_rgb = cfg.INPUT.MODALITY_DROPOUT_RGB if is_train else 0.0
        self._pending_height = None      # stashed between _encode and _apply_depth_transforms
        self.sdp_channels = cfg.INPUT.DEPTH_SDP_CHANNELS
        self.sdp_temperature = cfg.INPUT.DEPTH_SDP_TEMPERATURE
        self.sdp_max = cfg.INPUT.DEPTH_SDP_MAX
        self.sdp_local = cfg.INPUT.DEPTH_SDP_LOCAL
        # intrinsics for the xyz/normals back-projection follow the TRAINING dataset (TOD has a
        # different camera than UOAIS-Sim). Eval uses per-benchmark intrinsics in the predictor.
        _train_ds = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else "uoais_sim"
        self.intrinsics = get_intrinsics("tabletop" if "tabletop" in _train_ds else "uoais_sim")
        self.augmentations = T.AugmentationList(build_augmentation(cfg, is_train))
        # SSD-style photometric jitter on RGB only (train-only). Applied directly to the rgb
        # array before the shared geometric transforms, so it never touches the depth modality.
        self.color_aug = cfg.INPUT.COLOR_AUG_SSD and is_train
        self._color_aug = ColorAugSSDTransform(img_format=self.image_format) if self.color_aug else None
        # RGB-D copy-paste augmentation (train-only): source pool = the training records
        self.copy_paste = cfg.INPUT.COPY_PASTE and is_train and self.input_type != "rgb"
        self.cp_prob = cfg.INPUT.COPY_PASTE_PROB
        self.cp_n = cfg.INPUT.COPY_PASTE_N
        self.cp_records = DatasetCatalog.get(cfg.DATASETS.TRAIN[0]) if self.copy_paste else None
        # upright/gravity alignment: un-roll the base scene to match always-upright real data
        self.upright = cfg.INPUT.UPRIGHT_ALIGN and is_train and self.input_type != "rgb"
        # support-plane re-texturing: needs depth (to fit the plane) so rgb-only is excluded
        self.support_aug = cfg.INPUT.SUPPORT_AUG and is_train and self.input_type != "rgb"
        # both copy-paste and upright operate on raw depth + bitmasks (the "bitmask path")
        self.bitmask_path = self.copy_paste or self.upright

    def _read_raw(self, path):
        return imageio.imread(path).astype(np.float32)      # raw depth, millimeters

    @staticmethod
    def _read_tabletop_masks(seg_path):
        """TOD palette segmentation -> list of object bitmasks (idx 0=bg, 1=table, >=2 objects)."""
        idx = np.array(Image.open(seg_path))                # palette indices (NOT RGB)
        if idx.ndim == 3:                                   # safety: collapse if RGB-decoded
            idx = idx[..., 0]
        return [idx == i for i in np.unique(idx) if i >= 2]

    def _encode(self, depth_mm):
        """Encode a raw (H,W) mm depth map per the configured encoding (applies depth-noise
        augmentation first, if enabled). Copy-paste modifies the raw depth before this."""
        noisy = self.is_train and self.depth_noise
        if noisy:
            depth_mm = augment_depth_mm(depth_mm, hole_pattern=self.hole_pattern)
        # from RAW depth: the plane fit must see metric-ish depth, not an encoded representation.
        # Quantised to uint8 so it can ride the same nearest-neighbour warp as the validity mask.
        self._pending_height = (np.clip(plane_height_map(depth_mm), 0.0, 1.0) * 255.0).astype(np.uint8) \
            if self.plane_height_channel else None
        if self.depth_encoding == "sdp":
            # multi-frequency sinusoidal encoding (N channels); needs INPUT.DEPTH_ADAPTER to map
            # N -> the pretrained stem's channel count.
            return depth_to_sdp(depth_mm, self.sdp_channels, self.sdp_temperature, self.sdp_max,
                                local=self.sdp_local)
        if self.depth_encoding == "hha":
            return depth_to_hha(depth_mm / 1000.0, self.intrinsics)          # meters
        if self.depth_encoding == "xyz":
            # metric XYZ (meters); holes stay invalid (0) — the validity mask in __call__
            # excludes them. Standardized AFTER geometric augmentation.
            return depth_to_xyz(depth_mm, self.intrinsics)
        if self.depth_encoding == "depth_normals":
            # [normalized-depth | normal_x | normal_y] uint8 image (holes -> 0).
            return depth_to_normals(depth_mm, self.intrinsics, inverted=self.depth_inverted)
        # normalized: match the EVAL pipeline order (normalize -> inpaint holes -> invert).
        depth = _normalize_depth(depth_mm, self.depth_min, self.depth_max, inverted=False)
        if noisy:
            depth = inpaint_holes_u8(depth)
        if self.depth_inverted:
            depth = 255 - depth
        return depth

    def _load_depth(self, path):
        return self._encode(self._read_raw(path))

    def _apply_depth_transforms(self, depth, transforms):
        """Apply the sampled geometric transforms to the encoded depth (xyz needs a validity
        mask carried via apply_segmentation + X-sign flip; other encodings are plain images).

        Returns (H,W,3) normally, or (H,W,4) when INPUT.DEPTH_VALIDITY_CHANNEL is set, with
        channel 3 = 1.0 where the sensor measured a value and 0.0 where it did not. The mask is
        warped with the SAME sampled transforms (via apply_segmentation, nearest-neighbour) so it
        stays pixel-aligned with the geometry channels, and it is *not* subject to any of the
        directional flip fixups below -- a validity flag has no orientation.
        """
        valid = None
        if self.depth_encoding == "xyz":
            valid_in = (np.abs(depth).sum(-1) > 0).astype(np.uint8)
            depth = transforms.apply_image(depth)
            valid = transforms.apply_segmentation(valid_in).astype(bool)
            if any(isinstance(t, HFlipTransform) for t in transforms.transforms):
                depth = depth.copy()
                depth[..., 0] *= -1.0
            depth = standardize_xyz(depth, valid)
        elif self.depth_encoding == "depth_normals":
            # derive validity BEFORE the transform, for the same reason as xyz: after padding /
            # resampling there is no way to tell a padded pixel from an encoded zero.
            valid_in = (depth.sum(-1) > 0).astype(np.uint8) if self.validity_channel else None
            depth = transforms.apply_image(depth)
            if valid_in is not None:
                valid = transforms.apply_segmentation(valid_in).astype(bool)
            if any(isinstance(t, HFlipTransform) for t in transforms.transforms):
                depth = depth.copy()
                depth[..., 1] = 255 - depth[..., 1]      # horizontal flip negates normal_x
        elif self.depth_encoding == "sdp":
            # SDP sets ALL channels to 0 at invalid pixels (see depth_sdp), so "any
            # nonzero channel" is an exact validity test. No directional fixup on flip:
            # the channels encode distance, not orientation.
            valid_in = (np.abs(depth).sum(-1) > 0).astype(np.uint8)
            depth = transforms.apply_image(depth)
            valid = transforms.apply_segmentation(valid_in).astype(bool)
        else:
            # normalized / hha: holes are already inpainted or clamped away, so "validity" here
            # means "was this pixel measured, or fabricated/clamped" -- still worth signalling
            # (see reference_papers/deep_research.md 7.2: keep the mask even after hole filling).
            valid_in = (depth.sum(-1) > 0).astype(np.uint8) if self.validity_channel else None
            depth = transforms.apply_image(depth)
            if valid_in is not None:
                valid = transforms.apply_segmentation(valid_in).astype(bool)
        return self._append_extra(depth, valid, transforms)

    def _append_extra(self, depth, valid, transforms=None):
        """Stack the optional extra channels: validity, then plane height. Order is fixed and mirrored
        in the predictor -- train/eval parity is mandatory, a swapped channel order would train fine and
        silently mis-evaluate. No-op when neither flag is set, so existing runs are byte-identical."""
        out = [depth.astype(np.float32)]
        if self.validity_channel:
            v = valid if valid is not None else (np.abs(depth).sum(-1) > 0)
            out.append(v.astype(np.float32)[..., None])
        if self.plane_height_channel:
            h = self._pending_height
            if h is None:
                h = np.zeros(depth.shape[:2], np.uint8)
            elif transforms is not None:
                # same nearest-neighbour warp as the validity mask: height has no orientation, so no
                # flip fixup is needed, and nearest keeps padded regions at exactly 0 (= on the plane)
                h = transforms.apply_segmentation(h)
            out.append((h.astype(np.float32) / 255.0)[..., None])
        self._pending_height = None
        return np.concatenate(out, axis=-1) if len(out) > 1 else out[0]

    def _load_source_arrays(self):
        rec = self.cp_records[np.random.randint(len(self.cp_records))]
        s_rgb = utils.read_image(rec["file_name"], format=self.image_format)
        s_depth = self._read_raw(rec["depth_file_name"])
        return s_rgb, s_depth, decode_masks(rec["annotations"])

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        rgb = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, rgb)

        # Bitmask path (copy-paste and/or upright): composite raw depth + bitmasks BEFORE
        # encoding. Otherwise the original path (encoded depth + RLE annotation instances).
        cp_masks = None
        depth_mm = None      # raw mm depth, bound in every branch below (support_aug needs it)
        if "seg_file_name" in dataset_dict:
            # tabletop (TOD): instance bitmasks come from the palette segmentation PNG (lazy),
            # not RLE annotations. Routes through the shared bitmask -> Instances path below.
            depth_mm = self._read_raw(dataset_dict["depth_file_name"])
            cp_masks = self._read_tabletop_masks(dataset_dict["seg_file_name"])
            if self.is_train and not cp_masks and not self.keep_empty_neg:
                return None                      # skip background-only views (no negatives);
                                                 # detectron2 MapDataset resamples another index
            depth = self._encode(depth_mm)
        elif self.bitmask_path:
            depth_mm = self._read_raw(dataset_dict["depth_file_name"])
            cp_masks = decode_masks(dataset_dict["annotations"])
            if self.upright:
                # un-roll the base scene to upright (rgb linear, depth/masks nearest, 0-fill)
                theta, conf = estimate_roll(depth_mm)
                if abs(theta) > 1.0:
                    rgb = apply_roll(rgb, theta, cv2.INTER_LINEAR)
                    depth_mm = apply_roll(depth_mm, theta, cv2.INTER_NEAREST, border_value=0)
                    cp_masks = [apply_roll(m.astype(np.uint8), theta, cv2.INTER_NEAREST).astype(bool)
                                for m in cp_masks]
                    cp_masks = [m for m in cp_masks if m.sum() > 0]
            if self.copy_paste and np.random.rand() < self.cp_prob:
                s_rgb, s_depth, s_masks = self._load_source_arrays()
                rgb, depth_mm, cp_masks = copy_paste_rgbd(
                    rgb, depth_mm, cp_masks, s_rgb, s_depth, s_masks, n_paste=self.cp_n)
            depth = self._encode(depth_mm)
        else:
            if self.input_type != "rgb":
                depth_mm = self._read_raw(dataset_dict["depth_file_name"])
                depth = self._encode(depth_mm)      # == _load_depth, but keeps depth_mm
            else:
                depth = None

        # Support-surface re-texturing: RGB only, on the raw arrays, before the geometric
        # transforms. Needs the raw mm depth to fit the plane and the GT masks to protect object
        # pixels; both are available here. cp_masks is already decoded on the bitmask/TOD paths.
        if self.support_aug and depth_mm is not None:
            _m = cp_masks if cp_masks is not None else decode_masks(
                dataset_dict.get("annotations", []))
            rgb = randomize_support_surface(rgb, depth_mm, _m)

        # photometric jitter on RGB only (before geometric transforms; depth untouched)
        if self.color_aug:
            rgb = self._color_aug.apply(rgb)

        # sample transforms ONCE on rgb, apply the SAME to depth (and masks)
        aug_input = T.AugInput(rgb)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]
        if depth is not None:
            depth = self._apply_depth_transforms(depth, transforms)

        # Modality dropout, applied to the FINAL tensors so it is independent of encoding and of every
        # augmentation above it. A dropped modality is replaced by its own per-channel mean: that removes
        # all spatial information while staying in-distribution, matching the eval-time ablation hook
        # (engine/predictor.py DAFUSION_ABLATE_MODALITY) so train and diagnostic agree. Mutually
        # exclusive per sample -- never blank both, which would leave nothing to learn from.
        _drop = None
        if self.mdrop_depth or self.mdrop_rgb:
            r = np.random.rand()
            if r < self.mdrop_depth:
                _drop = "depth"
            elif r < self.mdrop_depth + self.mdrop_rgb:
                _drop = "rgb"
        if _drop == "rgb":
            image = np.broadcast_to(image.reshape(-1, image.shape[-1]).mean(0),
                                    image.shape).astype(image.dtype)
        if _drop == "depth" and depth is not None:
            depth = np.broadcast_to(depth.reshape(-1, depth.shape[-1]).mean(0),
                                    depth.shape).astype(depth.dtype)
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if self.input_type != "rgb":
            dataset_dict["depth_image"] = torch.as_tensor(np.ascontiguousarray(depth.transpose(2, 0, 1)).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict

        gt = Instances(image_shape)
        if cp_masks is not None:
            # copy-paste: instances are already bitmasks; apply the same geometric transform
            tm = [transforms.apply_segmentation(m.astype(np.uint8)).astype(bool) for m in cp_masks]
            tm = [m for m in tm if m.sum() > 0]
            gt.gt_classes = torch.zeros(len(tm), dtype=torch.long)
            gt.gt_masks = (torch.as_tensor(np.ascontiguousarray(np.stack(tm))) if tm
                           else torch.zeros((0, image_shape[0], image_shape[1]), dtype=torch.bool))
            dataset_dict.pop("annotations", None)
            dataset_dict["instances"] = gt
            return dataset_dict

        # training targets from the VISIBLE mask (class-agnostic)
        annos = []
        annos_amodal = []
        for obj in dataset_dict.pop("annotations"):
            if obj.get("iscrowd", 0) != 0 or obj.get("visible_mask") is None:
                continue
            obj = dict(obj)
            amodal_seg = obj.get("segmentation")          # UOAIS-Sim: `segmentation` IS the amodal mask
            obj["segmentation"] = obj["visible_mask"]
            annos.append(utils.transform_instance_annotations(obj, transforms, image_shape))
            # keep the two lists index-aligned: an amodal entry is appended for EVERY kept object, so
            # instance i in one list is instance i in the other. A missing amodal mask falls back to the
            # visible one, which makes the auxiliary loss a no-op for that instance rather than
            # silently shifting the alignment.
            if self.amodal_targets:
                a = dict(obj)
                a["segmentation"] = amodal_seg if amodal_seg is not None else obj["visible_mask"]
                annos_amodal.append(utils.transform_instance_annotations(a, transforms, image_shape))
        instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")
        if self.amodal_targets and annos_amodal:
            am = utils.annotations_to_instances(annos_amodal, image_shape, mask_format="bitmask")
            if am.has("gt_masks"):
                instances.gt_amodal_masks = am.gt_masks
        # filter_empty_instances drops objects whose VISIBLE mask vanished under augmentation; because
        # gt_amodal_masks is now a field of the same Instances, it is filtered by the same mask and the
        # two stay aligned automatically.
        instances = utils.filter_empty_instances(instances)
        gt.gt_classes = instances.gt_classes if instances.has("gt_classes") else torch.zeros(0, dtype=torch.long)
        if instances.has("gt_masks") and len(instances) > 0:
            gt.gt_masks = instances.gt_masks.tensor
        else:
            gt.gt_masks = torch.zeros((0, image_shape[0], image_shape[1]), dtype=torch.uint8)
        # `gt` is a FRESH Instances that only receives the fields copied here -- setting a field on
        # `instances` above is not enough, which is why the first version silently produced no amodal
        # targets at all. Copy it across explicitly, after filter_empty_instances has already aligned it.
        if instances.has("gt_amodal_masks") and len(instances) > 0:
            gt.gt_amodal_masks = instances.gt_amodal_masks.tensor
        dataset_dict["instances"] = gt
        return dataset_dict
