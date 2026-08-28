"""Register the UOAIS-Sim training/val datasets (class-agnostic, single 'object' class)."""
import os

from detectron2.data import DatasetCatalog, MetadataCatalog

from .uois_json import load_uois_json
from ...paths import UOAIS_SIM_PATH

_SPLITS = {
    "uoais_sim_train": (os.path.join(UOAIS_SIM_PATH, "train"),
                        os.path.join(UOAIS_SIM_PATH, "annotations", "coco_anns_uoais_sim_train.json")),
    "uoais_sim_val": (os.path.join(UOAIS_SIM_PATH, "val"),
                      os.path.join(UOAIS_SIM_PATH, "annotations", "coco_anns_uoais_sim_val.json")),
}


def register_all_uois():
    for name, (image_root, json_file) in _SPLITS.items():
        DatasetCatalog.register(
            name, lambda j=json_file, r=image_root, n=name: load_uois_json(j, r, n)
        )
        MetadataCatalog.get(name).set(
            json_file=json_file, image_root=image_root,
            evaluator_type="coco", thing_classes=["object"],
        )


register_all_uois()
