# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved


"""
This file registers pre-defined datasets at hard-coded paths, and their metadata.

We hard-code metadata for common datasets. This will enable:
1. Consistency check when loading the datasets
2. Use models on these standard datasets directly and run demos,
   without having to download the dataset annotations

We hard-code some paths to the dataset that's assumed to
exist in "./datasets/".

Users SHOULD NOT use this file to create new dataset / metadata for new dataset.
To add new dataset, refer to the tutorial "docs/DATASETS.md".
"""

import os

from detectron2.data import MetadataCatalog, DatasetCatalog
from .builtin_meta import _get_builtin_metadata
from .register_uoais import register_uoais_instances

from legacy_paths import UOAIS_SIM_PATH

__all__ = ["register_all_uoais"]

_PREDEFINED_SPLITS_uoais = {
    "uoais_sim_train_amodal": (
        os.path.join(UOAIS_SIM_PATH, "train"),
        os.path.join(UOAIS_SIM_PATH, "annotations", "coco_anns_uoais_sim_train.json"),
    ),
    "uoais_sim_val_amodal": (
        os.path.join(UOAIS_SIM_PATH, "val"),
        os.path.join(UOAIS_SIM_PATH, "annotations", "coco_anns_uoais_sim_val.json"),
    ),
}


def register_all_uoais(root="./datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_uoais.items():
        # Assume pre-defined datasets live in `./datasets`.
        amodal = "amodal" in key
        if "occ" in key:
            md = "uoais_occ"
        else:
            md = "uoais"
        register_uoais_instances(
            key,
            _get_builtin_metadata(md),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            amodal=amodal
        )


# Register them all under "./datasets"
register_all_uoais()
# register_all_wisdom()
