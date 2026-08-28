# Class-agnostic COCO-format loader for UOAIS-Sim (ported from the legacy build).
# Reads visible-mask annotations; forces a single foreground class (category_id=0).
import contextlib
import io
import logging
import os

import pycocotools.mask as mask_utils
from detectron2.data import MetadataCatalog
from detectron2.structures import BoxMode
from fvcore.common.file_io import PathManager
from fvcore.common.timer import Timer

logger = logging.getLogger(__name__)
__all__ = ["load_uois_json"]


def load_segm(anno, key):
    segm = anno.get(key, None)
    if isinstance(segm, dict):
        if isinstance(segm["counts"], list):
            # uncompressed RLE -> compressed RLE
            segm = mask_utils.frPyObjects(segm, *segm["size"])
    else:
        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
        if len(segm) == 0:
            segm = None
    return segm


def load_uois_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
    """Parse a UOAIS COCO json into detectron2 dataset dicts (class-agnostic).

    Each record carries ``file_name`` (RGB), ``depth_file_name`` (16-bit depth, mm),
    ``height``/``width``, and per-object ``annotations`` with amodal ``segmentation``
    plus ``visible_mask`` / ``occluded_mask`` (kept for reference; training uses the
    visible mask as the segmentation target — see the dataset mapper).
    """
    from pycocotools.coco import COCO

    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file, timer.seconds()))

    if dataset_name is not None:
        MetadataCatalog.get(dataset_name)  # ensure metadata exists

    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

    ann_ids = [ann["id"] for anns_per_image in anns for ann in anns_per_image]
    assert len(set(ann_ids)) == len(ann_ids), f"Annotation ids in '{json_file}' are not unique!"

    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"] + (extra_annotation_keys or [])
    dataset_dicts = []
    for img_dict, anno_dict_list in zip(imgs, anns):
        record = {
            "file_name": os.path.join(image_root, img_dict["file_name"]),
            "depth_file_name": os.path.join(image_root, img_dict["depth_file_name"]),
            "height": img_dict["height"],
            "width": img_dict["width"],
            "image_id": img_dict["id"],
        }
        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == record["image_id"]
            assert anno.get("ignore", 0) == 0
            obj = {key: anno[key] for key in ann_keys if key in anno}
            if anno.get("segmentation", None):
                obj["segmentation"] = load_segm(anno, "segmentation")
            if anno.get("visible_mask", None):
                obj["visible_mask"] = load_segm(anno, "visible_mask")
            if anno.get("occluded_mask", None):
                obj["occluded_mask"] = load_segm(anno, "occluded_mask")
            obj["occluded_rate"] = anno.get("occluded_rate", None)
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            obj["category_id"] = 0  # class-agnostic
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
