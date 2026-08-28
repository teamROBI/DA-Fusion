import copy
import logging
import os.path as osp

import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from pycocotools import mask as maskUtils

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.data.detection_utils import SizeMismatchError
from detectron2.structures import BoxMode
from detectron2.structures import BitMasks, Instances, polygons_to_bitmask


from mask2former.utils.detection_utils import (annotations_to_instances, transform_instance_annotations)
from mask2former.utils.augmentation import ColorAugSSDTransform, Resize, PerlinDistortion

import cv2
import imageio
import torch.nn.functional as F
import random
import os

"""
This file contains the default mapping that's applied to "dataset dicts".
"""

__all__ = ["DatasetMapperWithBasis"]

logger = logging.getLogger(__name__)


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks




def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.ResizeScale(
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size)),
    ])

    return augmentation

def segmToRLE(segm, img_size):
    h, w = img_size
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm["counts"]) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = segm
    return rle


def segmToMask(segm, img_size):
    rle = segmToRLE(segm, img_size)
    m = maskUtils.decode(rle)
    return m

def normalize_depth(depth, min_val=2500.0, max_val=15000.0):
    min_val = max(np.percentile(depth, 5), min_val)
    max_val = min(np.percentile(depth, 95), max_val)
    depth[depth < min_val] = min_val
    depth[depth > max_val] = max_val
    depth = (depth - min_val) / (max_val - min_val) * 255
    depth = np.expand_dims(depth, -1)
    depth = np.uint8(np.repeat(depth, 3, -1))

    return depth


class DatasetMapperWithBasis(DatasetMapper):
    """
    This caller enables the default Detectron2 mapper to read an additional basis semantic label
    """

    def __init__(self, cfg, is_train=True):
        super().__init__(cfg, is_train)

        # Rebuild augmentations
        logger.info(
            "Rebuilding the augmentations. The previous augmentations will be overridden."
        )
        # self.img_size = (1280,720)  # width, height
        self.img_size = (640, 480)  # width, height
        self.amodal = False
        self.depth = True
        self.rgbd_fusion = cfg.INPUT.INPUT_TYPE
        self.colormap = cfg.INPUT.COLORMAP
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>', self.rgbd_fusion)
        self.depth_inverted = cfg.INPUT.DEPTH_INVERTED
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Depth inverted:', self.depth_inverted)
        self.depth_min, self.depth_max = 2500, 15000
        self.recompute_boxes = True
        self.ann_set = "coco"
        self.boxinst_enabled = False
        self.aug = cfg.INPUT.AUGMENTATION
        self.color_aug = False
        self.depth_only = False
        self.perlin_distortion = False
        self.check_input = 0

        if self.boxinst_enabled:
            self.use_instance_mask = False
            self.recompute_boxes = False
        cr = 0.5

        # if self.color_aug and is_train:
        #     if self.depth_only:
        #        self.augmentation_lists = [
        #         T.RandomApply(T.RandomCrop("relative_range", (cr, cr))),
        #         T.RandomFlip(0.5),
        #         Resize((self.img_size[1], self.img_size[0]))
        #         ]
        #     else:
        #         self.augmentation_lists = [
        #             T.RandomApply(T.RandomCrop("relative_range", (cr, cr))),
        #             ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT),
        #             T.RandomFlip(0.5),
        #             Resize((self.img_size[1], self.img_size[0]))
        #             ]
        # elif not self.color_aug and is_train:
        #     self.augmentation_lists = [
        #         T.RandomApply(T.RandomCrop("relative_range", (cr, cr))),
        #         T.RandomFlip(0.5),
        #         Resize((self.img_size[1], self.img_size[0]))
        #     ]
            
        # else:
        #     self.augmentation_lists = [
        #         Resize((self.img_size[1], self.img_size[0]))
        #     ]
        if self.aug and not self.rgbd_fusion == "rgbd":
            if self.rgbd_fusion == "depth" or self.rgbd_fusion == "depth_colormap":
               self.augmentation_lists = [
                T.RandomApply(T.RandomCrop("relative_range", (cr, cr))),
                T.RandomFlip(0.5),
                Resize((self.img_size[1], self.img_size[0]))
                ]
               print("('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Augment Depth")
            elif self.rgbd_fusion == "rgb":
                self.augmentation_lists = [
                    # T.RandomApply(T.RandomCrop("relative_range", (cr, cr))),
                    # ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT),
                    T.RandomFlip(0.5),
                    Resize((self.img_size[1], self.img_size[0]))
                    ]
                print("('>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Augment RGB")
        else:
            self.augmentation_lists = [
                Resize((self.img_size[1], self.img_size[0]))
            ]
            print("No Augmentation")
        logging.getLogger(__name__).info(
            "Augmentation used in training: {}".format(self.augmentation_lists)
            )
        self.augmentation_lists = T.AugmentationList(self.augmentation_lists)


    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
    
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        if self.rgbd_fusion == 'rgb':
            rgb = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            utils.check_image_size(dataset_dict, rgb)
            image = rgb
        elif self.rgbd_fusion == 'depth_colormap':
            # depth = cv2.imread(dataset_dict["depth_file_name"], cv2.IMREAD_LOAD_GDAL)
            depth = imageio.imread(dataset_dict["depth_file_name"]).astype(np.float32)
            depth = normalize_depth(depth, self.depth_min, self.depth_max)
            if self.depth_inverted:
                depth = 255 - depth

            # normalized_depth_image = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            colormap_attribute = getattr(cv2, self.colormap)
            depth_render = cv2.applyColorMap(depth, colormap_attribute)
            image = depth_render
        elif self.rgbd_fusion == 'depth':
            depth = imageio.imread(dataset_dict["depth_file_name"]).astype(np.float32)
            depth = normalize_depth(depth, self.depth_min, self.depth_max)
            if self.depth_inverted:
                depth = 255 - depth
            
            image = depth
        elif self.rgbd_fusion == 'rgbd':
            rgb = utils.read_image(dataset_dict["file_name"], format=self.image_format)
            utils.check_image_size(dataset_dict, rgb)
            image = rgb
            
            depth = imageio.imread(dataset_dict["depth_file_name"]).astype(np.float32)
            depth = normalize_depth(depth, self.depth_min, self.depth_max)
            if self.depth_inverted:
                depth = 255 - depth
            
            depth_image = depth
        else:
            print(self.rgbd_fusion)
            assert False, "Input type is wrong. Select between[rgb, depth_colormap, depth]"

        masks = []
        boxes = np.asarray([BoxMode.convert(
                    instance["bbox"], instance["bbox_mode"], BoxMode.XYXY_ABS
                    )
                        for instance in dataset_dict["annotations"]])

        # apply the color augmentation
        aug_input = T.AugInput(image, boxes=boxes)
        transforms = self.augmentation_lists(aug_input)
        image = aug_input.image
        image_shape = image.shape[:2]  # h, w
        
        if self.rgbd_fusion == "rgbd":
            # apply the color augmentation
            aug_input = T.AugInput(depth_image, boxes=boxes)
            transforms = self.augmentation_lists(aug_input)
            depth_image = aug_input.image
            depth_image_shape = depth_image.shape[:2]  # h, w
            
        if self.check_input < 100:
            if self.check_input % 10 == 0:
                # Construct the directory path
                from legacy_paths import DEBUG_ROOT
                dir_path = os.path.join(DEBUG_ROOT, "check_input_image", f"{self.rgbd_fusion}_{self.colormap}") + "/"
        
                # Check if the directory exists, create if not (exist_ok: dataloader workers race)
                os.makedirs(dir_path, exist_ok=True)
                
                # Save the image
                # name = os.path.basename(dataset_dict["file_name"]).split('.')[0]
                image_path = f"{dir_path}{self.check_input}.png"
                success = cv2.imwrite(image_path, image)
                if self.rgbd_fusion == 'rgbd':
                    # name = os.path.basename(dataset_dict["depth_file_name"]).split('.')[0]
                    depth_image_path = f"{dir_path}{self.check_input}_depth.png"
                    cv2.imwrite(depth_image_path, depth_image)
                print(f"Check input image: {success}")
            self.check_input += 1


        assert "annotations" in dataset_dict
        for anno in dataset_dict["annotations"]:
            anno.pop("keypoints", None)

        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]

        if len(annos):
            assert "visible_mask" in annos[0]
        segms = [obj["visible_mask"] for obj in annos]
        masks = []
        for segm in segms:
            if isinstance(segm, list):
                # polygon
                masks.append(polygons_to_bitmask(segm, *image.shape[:2]))
            elif isinstance(segm, dict):
                # COCO RLE
                masks.append(maskUtils.decode(segm))
            elif isinstance(segm, np.ndarray):
                assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                    segm.ndim
                )
                # mask array
                masks.append(segm)
            else:
                raise ValueError(
                    "Cannot convert segmentation of type '{}' to BitMasks!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict, or a binary segmentation mask "
                    " in a 2D numpy array of shape HxW.".format(type(segm))
                )

        masks_copy = [torch.from_numpy(np.ascontiguousarray(x)) for x in masks]

        classes = [int(obj["category_id"]) for obj in annos]
        classes = torch.tensor(classes, dtype=torch.int64)
        instances = Instances(image_shape)
        instances.gt_classes = classes
        
        if len(masks) == 0:
            # Some image does not have annotation (all ignored)
            instances.gt_masks = torch.zeros((0, image.shape[-2], image.shape[-1]))
        else:
            masks_bit = BitMasks(torch.stack(masks_copy))
            instances.gt_masks = masks_bit.tensor

        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
            , dtype=torch.float16
        )
        if self.rgbd_fusion == "rgbd":
            dataset_dict["depth_image"] = torch.as_tensor(
            np.ascontiguousarray(depth_image.transpose(2, 0, 1))
            , dtype=torch.float16
            )

        dataset_dict["instances"] = instances

        return dataset_dict