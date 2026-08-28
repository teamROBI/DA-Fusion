"""Register the Tabletop Object Dataset (TOD, "tabletop_dataset_v5_public") for training.

This is the synthetic tabletop RGB-D dataset used by UCN / MSMFormer — 40k training scenes
(7 rendered views each) with per-pixel palette segmentation. Unlike UOAIS-Sim (COCO JSON), the
labels live in `segmentation_%05d.png` as palette indices: 0 = background, 1 = table, >= 2 =
object instances. To avoid loading 280k label PNGs at registration, records are lightweight —
they carry the rgb / depth / segmentation file paths, and the DA-Fusion mapper reads the
segmentation lazily (per-worker) via the `seg_file_name` key.
"""
import glob
import json
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from ...paths import TABLETOP_PATH

_EMPTY_CACHE = os.path.join(TABLETOP_PATH, "empty_views.json")  # background-only seg paths

_SPLITS = {"tabletop_train": "training_set", "tabletop_test": "test_set"}
_VIEWS = 7   # rgb_00000..rgb_00006 per scene


def load_tabletop(subdir):
    root = os.path.join(TABLETOP_PATH, subdir)
    records = []
    for scene in sorted(glob.glob(os.path.join(root, "scene_*"))):
        for v in range(_VIEWS):
            rgb = os.path.join(scene, f"rgb_{v:05d}.jpeg")
            if not os.path.exists(rgb):
                continue
            records.append({
                "file_name": rgb,
                "depth_file_name": os.path.join(scene, f"depth_{v:05d}.png"),
                "seg_file_name": os.path.join(scene, f"segmentation_{v:05d}.png"),
                "height": 480, "width": 640,
                "image_id": f"{os.path.basename(scene)}_{v:05d}",
            })
    return records


def load_tabletop_empty():
    """Background-only (0-object) TOD views, for use as NEGATIVES mixed into UOAIS-Sim training
    (requires INPUT.KEEP_EMPTY_NEGATIVES=True so the mapper keeps them). Paths from the cached
    scan (empty_views.json); each seg path -> its sibling rgb/depth."""
    if not os.path.exists(_EMPTY_CACHE):
        return []
    records = []
    for seg in json.load(open(_EMPTY_CACHE)):
        d = os.path.dirname(seg)
        v = os.path.splitext(os.path.basename(seg))[0].split("_")[-1]   # NNNNN
        records.append({
            "file_name": os.path.join(d, f"rgb_{v}.jpeg"),
            "depth_file_name": os.path.join(d, f"depth_{v}.png"),
            "seg_file_name": seg,
            "height": 480, "width": 640,
            "image_id": f"{os.path.basename(d)}_{v}_neg",
            # empty annotations so detectron2's dataset stats/filters (which assume every record
            # has this key when mixed with annotated datasets) don't KeyError; the mapper ignores
            # it (the seg_file_name branch handles these records). Needs FILTER_EMPTY_ANNOTATIONS
            # False so these negatives aren't dropped.
            "annotations": [],
        })
    return records


def load_tabletop_empty_lowdose(stride=5):
    """A ~2-3% dose of `load_tabletop_empty`'s negatives (vs the ~12% full set): every `stride`-th
    entry of the cached empty-view list, evenly spread across scenes rather than a prefix slice.
    stride=5 -> ~1200 views -> ~1200/(45000+1200) = 2.6% of the mixed UOAIS-Sim+negatives train
    set. The 12% dose over-suppressed recall (88.1->77.3 on OCID); testing whether a much smaller
    dose nudges empty-scene precision without the recall collapse (see EXPERIMENTS.md Track 6)."""
    return load_tabletop_empty()[::stride]


def register_all_tabletop():
    for name, subdir in _SPLITS.items():
        DatasetCatalog.register(name, lambda s=subdir: load_tabletop(s))
        MetadataCatalog.get(name).set(evaluator_type="coco", thing_classes=["object"])
    DatasetCatalog.register("tabletop_empty", load_tabletop_empty)
    MetadataCatalog.get("tabletop_empty").set(evaluator_type="coco", thing_classes=["object"])
    DatasetCatalog.register("tabletop_empty_lowdose", load_tabletop_empty_lowdose)
    MetadataCatalog.get("tabletop_empty_lowdose").set(evaluator_type="coco", thing_classes=["object"])


register_all_tabletop()
