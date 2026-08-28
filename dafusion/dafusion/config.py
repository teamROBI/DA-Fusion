# -*- coding: utf-8 -*-
# DA-Fusion config: Mask2Former + Swin nodes (from the upstream add_maskformer2_config)
# plus the DA-Fusion fusion node (MODEL.DAFUSION) and RGB-D input knobs.
from detectron2.config import CfgNode as CN


def add_maskformer2_config(cfg):
    """Upstream Mask2Former config (verbatim from baselines/Mask2Former/mask2former/config.py)."""
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    cfg.INPUT.COLOR_AUG_SSD = False
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.MODEL.MASK_FORMER = CN()
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


def add_dafusion_config(cfg):
    """DA-Fusion additions on top of the Mask2Former config."""
    add_maskformer2_config(cfg)

    # RGB-D input handling (consumed by the DA-Fusion dataset mapper + meta-arch)
    cfg.INPUT.INPUT_TYPE = "rgbd"          # rgbd | rgb | depth
    # normalized = percentile-clipped depth -> grayscale x3 (what the legacy build actually
    # used, and what produced the paper's numbers). hha needs a camera intrinsic that was
    # never part of the pipeline (see dafusion/README.md); available but experimental.
    # xyz = metric XYZ organized point cloud (UCN/MSMFormer-style), per-image standardized
    # (center + isotropic scale); experimental A/B alternative to normalized.
    # depth_normals = [normalized-depth | normal_x | normal_y]; no gravity assumption needed.
    cfg.INPUT.DEPTH_ENCODING = "normalized"   # normalized | hha | xyz | depth_normals
    cfg.INPUT.DEPTH_INVERTED = True
    cfg.INPUT.DEPTH_MIN = 2500.0           # mm; percentile clamp for normalized depth
    cfg.INPUT.DEPTH_MAX = 15000.0
    cfg.INPUT.AUGMENTATION = True          # random crop / flip / scale (paper)
    # Train-time depth sensor-noise augmentation (UCN/DexNet: gamma scale + GP noise +
    # ellipse-hole dropout; see data/depth_noise.py). Sim depth is clean but real eval
    # depth is not — off by default to preserve the original protocol.
    cfg.INPUT.DEPTH_NOISE = False
    # Reproduce the MEASURED real hole geometry rather than generic random holes: a systematic
    # frame-edge dead band (OCID's rightmost 33 / OSD's 66 columns are invalid in EVERY frame --
    # the RGB/IR-baseline warp) plus directional (one-sided) occlusion shadows instead of the
    # isotropic gradient dropout. Requires DEPTH_NOISE=True; see data/depth_noise.py PATTERN
    # and docs/EXPERIMENTS.md Track 9b.
    cfg.INPUT.DEPTH_HOLE_PATTERN = False
    # RGB-D copy-paste augmentation (Simple Copy-Paste): paste object instances from other
    # training images to synthesize denser touching/occluded clutter -> teaches instance
    # separation (data/depth_xyz.py has no bearing; see data/copy_paste.py). Train-only.
    # Weight EMA. Measured motivation: three runs of one config with different seeds scored OCID
    # 91.40 / 89.93 / 91.00 (spread 1.47), so run-to-run noise exceeds most measured effects. EMA
    # averages the late trajectory, which raises the expected score and shrinks that spread.
    cfg.SOLVER.EMA_ENABLED = False      # off -> training is byte-identical to every prior run
    cfg.SOLVER.EMA_DECAY = 0.9998       # ~5000-iteration effective window at 22.5k total iters
    # Height-above-support-plane channel. The user's observation: a floor/table is FLAT in depth and an
    # object standing on it rises out of that plane, so the model should be handed that distinction
    # explicitly. Evidence it matters: `plane_reject` in post-processing is worth +4.0 F on OCID
    # (91.20 with, 86.63 without) -- plane-vs-object is already the most valuable single signal in the
    # pipeline, but only as a delete-after-the-fact rule. This gives it to the model as an input.
    # Unit-invariant by construction (see _plane_height): depth is divided by its own median before
    # fitting, which matters because UOAIS-Sim depth is in 0.1 mm while the benchmarks are in mm.
    # Modality dropout. Motivation is measured, not generic: depth separates objects from background at
    # AUC 0.95 on OSD/OCID but only 0.64 on OCBD, because a bin's background IS 3D structure. Nothing in
    # the architecture forces the RGB branch to stand on its own, so a model that leans on depth will
    # underperform exactly where depth is uninformative -- our weakest benchmark. Dropping a modality
    # during training forces each branch to be independently competent.
    # Per sample: drop depth with p=DEPTH, else drop RGB with p=RGB, else keep both.
    # AMODAL AUXILIARY SUPERVISION. UOAIS-Sim ships, per instance, the amodal mask (`segmentation`),
    # the visible mask, the occluded mask and an occluded_rate -- and only the visible mask has ever been
    # used. This adds a SECOND mask head predicting amodal extent for the same queries, sharing the
    # Hungarian assignment computed from the visible masks, so the model must learn "this object
    # continues behind that one". That is the representation a merge/split decision needs.
    # Weight 0 (default) does not even build the head, so every existing run is byte-identical.
    # Honest bound: only ~11% of OCBD's merges are the occluding kind (Track 11), so the upside on our
    # worst benchmark is limited by measurement, not by hope.
    cfg.MODEL.MASK_FORMER.AMODAL_WEIGHT = 0.0
    cfg.INPUT.MODALITY_DROPOUT_DEPTH = 0.0
    cfg.INPUT.MODALITY_DROPOUT_RGB = 0.0
    cfg.INPUT.DEPTH_PLANE_HEIGHT_CHANNEL = False
    cfg.INPUT.COPY_PASTE = False
    cfg.INPUT.COPY_PASTE_PROB = 0.5     # fraction of training images that get pastes
    cfg.INPUT.COPY_PASTE_N = 3          # max instances pasted per image
    # Upright/gravity alignment: un-roll sim scenes (random camera roll) to match the
    # always-upright real benchmarks. Estimated from the ground-plane depth gradient with a
    # confidence gate (see data/upright.py). Train-only, data-only; architecture unchanged.
    cfg.INPUT.UPRIGHT_ALIGN = False
    # Support-surface appearance randomization (train-only, RGB-only; depth untouched).
    # UOAIS-Sim has only bin/tabletop scenes but 48% of OCID is FLOOR scenes -- a scene
    # type absent from training that carries 54.9% of all OCID loss, with precision 82.0
    # vs 90.8 on table scenes (Track 10i). Re-textures the fitted support plane with
    # tiles/grids/stripes/noise so 'large patterned plane' stops reading as 'object'.
    # GT object pixels are excluded via masks, so nothing paints over an instance.
    cfg.INPUT.SUPPORT_AUG = False
    # Keep background-only (0-object) views and train them as NEGATIVES (the matcher tolerates
    # empty targets) instead of skipping them. Used to add empty-scene negatives (e.g. TOD's
    # background-only views) that suppress false positives on empty scenes like OCID's.
    cfg.INPUT.KEEP_EMPTY_NEGATIVES = False
    # Append a 4th DEPTH input channel holding the validity mask (1.0 = sensor measured this
    # pixel, 0.0 = missing). Without it the network cannot distinguish a hole from a real
    # surface: every encoding maps invalid pixels to a value that collides with real geometry
    # (for xyz, standardize_xyz leaves them at 0, which IS the scene's median centre). Real
    # eval depth is systematically holed -- OCID 11.6% / OSD 23.8% invalid, with the rightmost
    # 33/66 columns dead in EVERY frame -- while UOAIS-Sim training depth is 100% valid, so
    # this is a large train/test shift. Requires the 4-channel dualinit weights (see
    # tools/remap_coco_init.py --validity-channel) so the depth patch-embed keeps its
    # pretrained prior; the 4th input slice is zero-initialized, so training starts
    # numerically identical to the 3-channel model and learns to use validity.
    cfg.INPUT.DEPTH_VALIDITY_CHANNEL = False
    # Sinusoidal Depth Preprocessing (SDP, Vanishing Depth): encode scalar depth onto many
    # sin/cos channels at different frequencies instead of a 3-channel pseudo-image. Set
    # DEPTH_ENCODING="sdp" to use it. NOTE the paper's SDP wins are on depth *completion* and 6D
    # *pose* (metric-precision tasks); on semantic segmentation its own Table 4 shows SDP ~= plain
    # normalization, so a null result here is the prior expectation. Requires DEPTH_ADAPTER
    # (the encoder stem cannot take DEPTH_SDP_CHANNELS channels directly).
    cfg.INPUT.DEPTH_SDP_CHANNELS = 32       # even; sin/cos pairs
    cfg.INPUT.DEPTH_SDP_TEMPERATURE = 0.1   # T<1, sets the frequency spread
    cfg.INPUT.DEPTH_SDP_MAX = 15000.0       # mm; GSDP fixed global scale
    # LSDP: normalize each frame by its own p99 depth instead of the fixed global max.
    # Needed for sim->real here: train (2.5-9 m) and eval (0.3-1.8 m) otherwise occupy
    # disjoint regions of the sinusoid, which is the leading suspect for Grid D's -3.47.
    cfg.INPUT.DEPTH_SDP_LOCAL = False
    # Learnable depth modality adapter mapping the depth representation into the RGB-pretrained
    # stem's expected input. The piece with the strongest segmentation evidence in Vanishing Depth
    # (+2.1 DINOv2 / +8.0 EVA-02) -- but measured with the encoder FROZEN, whereas we fully
    # fine-tune (and Track 4 showed freezing our RGB backbone is catastrophic), so expected
    # headroom here is much smaller. See modeling/backbone/depth_adapter.py.
    cfg.INPUT.DEPTH_ADAPTER = False
    # 128 rather than a token width: a ~700-param adapter would be too small to fairly test
    # 'does an adapter help', so a null result could just mean 'too small'. Still tiny next to
    # the 461M model, so a gain is attributable to the representation mapping, not capacity.
    # NOTE this is far smaller than Vanishing Depth's largest variant (a full ViT-S depth
    # encoder); a null result here bounds lightweight adapters, not adapters in general.
    cfg.INPUT.DEPTH_ADAPTER_HIDDEN = 128

    # DA-Fusion deformable fusion module
    cfg.MODEL.DAFUSION = CN()
    cfg.MODEL.DAFUSION.INPUT_TYPE = "rgbd"       # mirrors INPUT.INPUT_TYPE for the backbone/meta-arch
    cfg.MODEL.DAFUSION.NUM_HEADS = 8
    cfg.MODEL.DAFUSION.N_GROUPS = 4
    cfg.MODEL.DAFUSION.OFFSET_RANGE_FACTOR = 2.0
    cfg.MODEL.DAFUSION.USE_RPB = True
    # 3D geometry prior in the fusion attention (DFormerv2-style depth+spatial decay). The
    # existing RPB is purely 2D pixel-space, so attention cannot tell "5 px away on the same
    # surface" from "5 px away across an occlusion boundary" -- see modeling/fusion/
    # geometry_prior.py. Post-fix, depth buys only +0.32 mean over RGB-only (Track 10a), and
    # widening 2D reach (OFF4) / adding groups (NG8) both hurt OCID, so the fusion's limit looks
    # like geometric blindness rather than capacity.
    # Auxiliary instance-BOUNDARY supervision (weight 0 = off). By elimination the OCID
    # ceiling is under-segmentation of touching objects in the model's own predictions
    # (pred 8.4 vs gt 8.9 instances; Track 10f), and encoder representation, decoder query
    # count, post-processing and eval protocol are all separately exonerated. The training
    # objective never asks the model to separate adjacent instances -- Boundary F runs 4-8
    # pts below Overlap F. See modeling/boundary_head.py.
    cfg.MODEL.DAFUSION.BOUNDARY_WEIGHT = 0.0
    cfg.MODEL.DAFUSION.BOUNDARY_HIDDEN = 64
    cfg.MODEL.DAFUSION.GEOMETRY_PRIOR = False
    # Per-head decay/blend params instead of one shared across heads. ~8x the bias memory; the
    # highest-resolution stage is 5.76M pairs/image at 480x640, so shared is the default.
    cfg.MODEL.DAFUSION.GEOM_PER_HEAD = False
    # Skip the prior on any stage exceeding this many (query x key) pairs. Default matches
    # ContinuousPositionBias's own 4M cap, which means the geometry prior is active at res3-res5
    # and SKIPPED at res2 (5.76M pairs at 480x640) -- deliberately.
    #
    # Rationale (a correction to an earlier assumption): res2 currently has NO spatial prior,
    # because RPB returns None there, and `multihead_attend` therefore takes the memory-efficient
    # F.scaled_dot_product_attention path. Supplying ANY bias at res2 forces the explicit
    # (B,heads,Nq,Nk) attention matrix instead -- ~920 MB fp32 per module at batch 5, x3 modules
    # -- so "adding geometry where the model has no prior" is not free, it costs the fused-attention
    # fast path. res3-res5 already materialise that matrix (RPB applies), so the prior is nearly
    # free there. Raise to 8M to include res2 only if you have verified the memory headroom.
    cfg.MODEL.DAFUSION.GEOM_MAX_PAIRS = 4_000_000
    # per-stage KV-grid downsample (res2..res5); keeps the sampled-key count bounded
    cfg.MODEL.DAFUSION.STRIDES = [8, 4, 2, 1]
    # HHA depth normalization (not natural RGB); overridable, computed over UOAIS-Sim
    cfg.MODEL.DAFUSION.PIXEL_MEAN_DEPTH = [123.675, 116.280, 103.530]
    cfg.MODEL.DAFUSION.PIXEL_STD_DEPTH = [58.395, 57.120, 57.375]
    # Reinitialize the depth (swin_d) branch from scratch instead of the ImageNet dualinit
    # weights. Used for the xyz encoding (XYZ != natural RGB, so the ImageNet patch-embed
    # prior is a poor fit); pair with a weights file that omits backbone.swin_d.* keys.
    cfg.MODEL.DAFUSION.DEPTH_FROM_SCRATCH = False
    # Freeze the RGB Swin backbone (backbone.swin_rgb) during training: keep the COCO-pretrained
    # RGB features fixed and let only the depth branch + fusion + decoder learn object-vs-bg. Tests
    # whether full fine-tuning overfits sim RGB appearance (capping real-domain transfer). Frozen
    # params are excluded from the optimizer and the branch is eval-locked (drop_path off).
    cfg.MODEL.DAFUSION.FREEZE_RGB_BACKBONE = False
