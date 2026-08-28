# `dafusion/` — DA-Fusion reimplementation (the real DS/DC model)

Faithful reimplementation of the paper's architecture (ICRA 2025), built from scratch on a
copy of the pristine `baselines/Mask2Former/` clone. Independent of the `legacy/` build
(separate code, config, scripts, env). The legacy build only does late `(rgb+depth)/2`
averaging; this build implements the paper's **feature-level deformable fusion**.

## Architecture (paper Fig. 2, §III)

Two Swin-L branches encode RGB and depth (**normalized** depth by default — matching the
legacy build; HHA is available via `INPUT.DEPTH_ENCODING=hha`). At each of the 4 backbone
stages a `FusionStage` fuses the two feature maps:

```
f'_rgb = DS_rgb(f_rgb)          # Deformable Self-Attention (eqs. 1-7)
f'_d   = DS_d(f_d)
f_i    = DC(f'_rgb, f'_d)       # Deformable Cross-Attention (eqs. 8-13)
```

- **DS** (`modeling/fusion/deformable_self_attention.py`): query Q=W_Q·f, offsets
  Δp=θ_offset(Q) over a uniform grid R, keys/values bilinearly sampled at R+Δp
  (`F.grid_sample` = the paper's φ), multi-head attention + relative-position bias.
- **DC** (`modeling/fusion/deformable_cross_attention.py`): RGB query → depth K/V and
  depth query → RGB K/V; the two results are concatenated and projected to the fused f_i.
- **`DualSwinFusionBackbone`** returns the fused multi-scale dict {res2..res5} with the
  native Swin channel dims, so the **unchanged Mask2Former pixel decoder + transformer
  decoder** produce the class-agnostic mask. Meta-arch: `DAFusion` (`modeling/meta_arch/`).

Loss: `5·CE + 5·Dice`, class weight 2.0 (fg) / 0.1 (bg via NO_OBJECT_WEIGHT), NUM_CLASSES=1.
AdamW lr 1e-4, batch 2, cosine + warmup. Targets (Table I/II): OCID Overlap-F 92.1 / OSD 92.9 / OCBD 91.3.

## Layout

```
dafusion/
  dafusion/                package (import registers config/datasets/mapper/backbone/arch)
    modeling/fusion/       DS, DC, offset net, RPB, sampling, fusion stage        (NEW)
    modeling/backbone/     swin.py (copied) + dual_fusion_backbone.py             (NEW)
    modeling/meta_arch/    dafusion_model.py (NEW) + mask_former_head.py (copied)
    modeling/pixel_decoder/ msdeformattn.py + fpn.py + ops/  (copied from Mask2Former)
    modeling/transformer_decoder/  (copied)   matcher.py criterion.py (copied)
    data/                  hha.py, datasets/{register_uois,uois_json,intrinsics}, dataset_mappers/
    config.py  paths.py  train_net.py
  configs/                 Base-DAFusion-SwinL.yaml + dafusion_rgbd_uoais.yaml
  scripts/                 setup_env.sh · train.sh · eval.sh
  third_party/detectron2/  source clone (editable-installed; gitignored)
```

## Setup / train / eval

**Install (portable — works on any server with an NVIDIA GPU + CUDA 12.x toolkit):**
```bash
bash dafusion/scripts/setup_env.sh        # auto-installs uv, auto-detects GPU arch;
                                          # py3.10 · torch2.4+cu124 · detectron2 (source) · MSDeformAttn (fresh)
```
Datasets/weights are separate from code — point `DAFUSION_DATA` at your data dir (default `<repo>/data`).
One-time: build the dual-branch COCO init `python tools/remap_coco_init.py --src ../data/checkpoints/mask2former/model_final_e5f453.pkl --dst ../data/checkpoints/mask2former/dafusion_swinL_dualinit.pkl`.

**Train** (multi-GPU via detectron2 DDP; outputs to `data/checkpoints/dafusion/<exp>/` on /data1):
```bash
bash dafusion/scripts/train.sh                       # RGB-D, 6 GPUs (NUM_GPUS default)
NUM_GPUS=4 bash dafusion/scripts/train.sh            # pick GPU count
NUM_GPUS=1 bash dafusion/scripts/train.sh SOLVER.MAX_ITER 20   # smoke test
```
On 24 GB 3090s use `SOLVER.IMS_PER_BATCH == NUM_GPUS` (1/GPU; batch-2/GPU exceeds 24 GB — enable
`MODEL.SWIN.USE_CHECKPOINT True` to fit more). AMP is on. DDP uses `find_unused_parameters=True`
(the highest-res stage skips its RPB, so those params idle some steps).

**Evaluate** a trained checkpoint (OCBD dir, formerly BOSD):
```bash
bash dafusion/scripts/eval.sh --dataset ocid --weights ../data/checkpoints/dafusion/<run>/model_final.pth
bash dafusion/scripts/eval.sh --dataset osd  --use_cgnet     # UOAIS foreground-filter protocol
bash dafusion/scripts/eval.sh --dataset ocbd --input_type rgbd
```

## Status

**Validated end-to-end:** portable env; full model builds (461M params); DS/DC fusion is
differentiable (all params get gradient); HHA + datasets + mapper produce correct tensors on real
UOAIS-Sim; real training smoke test (0.7s/it, 15 GB @ batch1); **multi-GPU DDP** (2-GPU run);
**COCO dual-branch init** loads cleanly into both Swin branches; eval harness (OSD/OCID/OCBD +
CG-Net + PRF) runs end-to-end. **Not yet done:** the full 20-epoch training run and the resulting
benchmark numbers vs the paper (needs the multi-day run) — the eval harness is implemented but its
accuracy is only meaningful once a trained checkpoint exists.

## Key decisions & deviations from the paper (revisit if numbers miss)

1. **Offset net** = DAT-style depthwise-conv head (paper says only "small conv/MLP"), `n_groups=4`.
2. **DS/DC on 2D maps** with `F.grid_sample` (matches φ's bilinear definition), not windowed tokens.
3. **KV grid downsample** `STRIDES=[8,4,2,1]` (res2..res5) bounds the sampled-key count so
   high-res-stage attention is tractable.
4. **RPB** is a continuous position-bias MLP; skipped where `Nq·Nk` is too large to materialise
   (the highest-res stage) — a memory/fidelity trade documented in `relative_position_bias.py`.
5. **"Concat f1..f4 → pixel decoder"** = pass the 4 fused maps as the multi-scale dict (channels
   preserved); DC projects Concat(z_rgb,z_d) back to C_i so the pixel decoder is unmodified.
6. **Backbone init**: both branches from the Mask2Former Swin-L COCO checkpoint. NOTE: the COCO
   pkl has single-branch keys; loading into `swin_rgb`/`swin_d` needs a key-remap (TODO) — until
   then the branches start from the pkl only if remapped, else random.
7. **Depth encoding = `normalized`** (default). The legacy build that produced the paper's numbers
   used percentile-clipped grayscale depth, NOT HHA, and the UOAIS-Sim camera intrinsic HHA needs
   was never in the pipeline / isn't published. HHA remains available (`INPUT.DEPTH_ENCODING=hha`)
   with best-effort intrinsics in `data/datasets/intrinsics.py` and a camera-down gravity
   approximation, but it's experimental until a true intrinsic is confirmed.
8. **Depth normalization stats**: the depth branch uses `MODEL.DAFUSION.PIXEL_*_DEPTH` (defaulting to
   the RGB ImageNet-style stats, as the legacy build did for its 3x-grayscale depth).
9. **VRAM**: dual Swin-L is heavy; AMP on, and enable `MODEL.SWIN.USE_CHECKPOINT True` if needed.
