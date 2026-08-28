# DA-Fusion — legacy build

The **frozen** original build: a fork of [Mask2Former](https://github.com/facebookresearch/Mask2Former)
(+ vendored, lightly-patched detectron2) whose RGB-D "fusion" is **late averaging** of two independent
backbone passes — `CFS_outputs = (rgb_outputs + depth_outputs) / 2` in
[`mask2former/maskformer_model.py`](mask2former/maskformer_model.py). This is **not** the paper's
DS/DC feature-level fusion; the faithful reimplementation lives in [`../dafusion/`](../dafusion/).

We keep this build because it trained a solid checkpoint that is the **baseline to beat**:
`DI_AGF_rgbd_none_NOW0.4_BS2_LR1e-05` → OCID Overlap-F 91.9 / OSD 91.2 / BOSD 88.3.

## Environment

Original stack: **Python 3.8 · torch 1.9.0+cu111 · detectron2 0.6**. This box only has a CUDA 12.x
toolkit (can't compile torch-1.9 extensions), so setup **reuses the repo's prebuilt `.so` files**
(detectron2 `_C` + MSDeformAttn) via a `.pth` file instead of recompiling.

```bash
bash legacy/scripts/setup_env.sh     # creates legacy/.venv
```

## Train

`train.sh` reproduces the best RGB-D run by default (writes to `data/checkpoints/<exp_name>/`):

```bash
bash legacy/scripts/train.sh                                 # best RGB-D, 1 GPU
NUM_GPUS=4 bash legacy/scripts/train.sh                      # multi-GPU
bash legacy/scripts/train.sh INPUT.INPUT_TYPE rgb            # RGB-only ablation
bash legacy/scripts/train.sh SOLVER.MAX_ITER 20 SOLVER.IMS_PER_BATCH 1   # smoke test
```

Overrides are trailing `KEY VALUE` pairs (detectron2 positional style; **no** `--opts`). Knobs:
`INPUT.INPUT_TYPE ∈ {rgb,depth,depth_colormap,rgbd}`, `INPUT.COLORMAP`, `INPUT.DEPTH_INVERTED`,
`MODEL.MASK_FORMER.NO_OBJECT_WEIGHT`, `SOLVER.IMS_PER_BATCH`, `SOLVER.BASE_LR`. Params live in
[`configs/legacy_mask2former_swin_uoais.yaml`](configs/legacy_mask2former_swin_uoais.yaml).

## Evaluate

`eval.sh` evaluates the best RGB-D checkpoint on OCID by default:

```bash
bash legacy/scripts/eval.sh                                          # best RGB-D on OCID
bash legacy/scripts/eval.sh --input_type rgb  --dataset osd --use_cgnet True
bash legacy/scripts/eval.sh --input_type rgbd --dataset bosd
bash legacy/scripts/eval.sh --exp_dir ../data/checkpoints/legacy_train_ckpt/<run> --dataset ocid
```

Results write to `legacy/eval_result/`; visualizations to `data/UOIS/benchmark_result/`.

### Best known configuration (surviving checkpoints)

| modality | checkpoint (`legacy_train_ckpt/…`) | benchmark | Overlap-F | Boundary-F | %75 |
|----------|-----------------------------------|-----------|:---------:|:----------:|:---:|
| **RGB-D** (eval.sh default) | `DI_AGF_rgbd_none_NOW0.4_BS2_LR1e-05/` model_final | OCID | **91.9** | 89.6 | 93.8 |
|          | (same) | OSD | 91.2 | 88.0 | 88.4 |
|          | (same) | BOSD | 88.3 | 84.8 | 76.3 |
| RGB      | `dafusion_rgb_NOW0.5_iter19999/` | OSD | 92.0 | 89.7 | 88.9 |
| depth    | `DI_depth_none_NOW0.4_BS4_LR1e-05/` | — | — | — | — |

> The benchmark numbers above use the **CG-Net foreground filter** (`--use_cgnet True`, the standard
> UOAIS eval protocol). Without it, precision drops sharply (e.g. OSD RGB-D Overlap-F 81.4 vs 91.2).
> `eval.sh` does **not** enable it by default — pass `--use_cgnet True` to reproduce the table.

All numbers re-verified on this build. Winning config: `rgbd · NO_OBJECT_WEIGHT 0.4 · IMS_PER_BATCH 2 ·
BASE_LR 1e-5 · 22.5k iter · DEPTH_INVERTED · class-agnostic`, initialized from the Mask2Former Swin-L COCO
checkpoint. The paper's headline OCID 92.1 (from the `NOW0.5` `model_0019999` iteration) came from a weight
file that did **not** survive; historical numbers for every ablation are archived in
[`../docs/legacy_results/`](../docs/legacy_results/).

All code resolves paths via [`legacy_paths.py`](legacy_paths.py) (override with `DAFUSION_DATA` /
`DAFUSION_DATASETS` / `DAFUSION_CHECKPOINTS`). Upstream provenance is recorded in
[`../docs/UPSTREAM.md`](../docs/UPSTREAM.md).
