# Upstream provenance

This project vendors several third-party repositories. Their original `.git`
histories were removed when they were folded into the DA-Fusion monorepo (except
the baselines, which keep their own `.git`). This file records where each came
from so the divergence can be reconstructed if needed.

## `core/` — DA-Fusion implementation (built on Mask2Former)
- **Upstream:** https://github.com/facebookresearch/Mask2Former.git
- **Fork point:** `9b0651c` — "update license" (2022-05-20)
- **Our divergence** (uncommitted working-tree changes on top of the fork):
  - Modified: `mask2former/config.py`, `mask2former/maskformer_model.py`,
    `mask2former/__init__.py`, `mask2former/data/datasets/__init__.py`,
    `configs/coco/instance-segmentation/Base-COCO-InstanceSegmentation.yaml`,
    `configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml`,
    `configs/coco/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml`,
    `demo/demo.py`, `demo/predictor.py`
  - Added (DA-Fusion specific): `uoais_train_net.py`, `benchmark*.py`, `eval/`,
    `mask2former/data/dataset_mappers/uoais_dataset_mapper.py`,
    `mask2former/data/datasets/{builtin,builtin_meta,register_uoais,uoais}.py`,
    `mask2former/utils/{augmentation,detection_utils}.py`,
    `configs/coco/instance-segmentation/swin/mask2former_swin_uoais.yaml`

## `core/detectron2/` — vendored + patched
- **Upstream:** https://github.com/facebookresearch/detectron2.git
- **Base commit:** `3ff5dd1` — "Pass cfg.SEED to dataloader building" (2024-02-07)
- **Patch:** `detectron2/engine/defaults.py::DefaultPredictor` accepts a second
  `original_depth_image` argument and reads `cfg.INPUT.INPUT_TYPE` for dual RGB-D
  inference. Must be installed editable from here — do not replace with a pip wheel.

## `baselines/` — comparison methods (keep their own `.git`)
| dir | method | upstream |
|-----|--------|----------|
| `baselines/uoais` | UOAIS-Net (ICRA'22) | https://github.com/gist-ailab/uoais.git |
| `baselines/ucn`   | UCN (CoRL'20)       | https://github.com/NVlabs/UnseenObjectClustering.git |
| `baselines/msmformer` | MSMFormer (ICRA'24) | https://github.com/YoungSean/UnseenObjectsWithMeanShift.git |
