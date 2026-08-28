"""Central path resolution for the LEGACY DA-Fusion build (the Mask2Former-fork
"CFS" late-averaging model). Replaces the old hard-coded ``/root/Seg2Grasp/...``
container paths.

All paths resolve relative to the repo root by default, but every root can be
overridden with an environment variable so the code stays portable:

    DAFUSION_DATA         -> data dir            (default: <repo>/data)
    DAFUSION_DATASETS     -> benchmark datasets  (default: <DATA>/UOIS)
    DAFUSION_CHECKPOINTS  -> training outputs     (default: <DATA>/checkpoints)

Layout expected under DAFUSION_DATASETS:
    UOAIS-Sim/            (training set + annotations/)
    OSD-0.20-depth/  OCID-dataset/  BOSD/   (eval benchmarks)
    benchmark_result/    (per-run eval visualizations)
"""
import os

LEGACY_ROOT = os.path.dirname(os.path.abspath(__file__))        # <repo>/legacy
REPO_ROOT = os.path.dirname(LEGACY_ROOT)                        # <repo>

DATA_ROOT = os.environ.get("DAFUSION_DATA", os.path.join(REPO_ROOT, "data"))
DATASET_ROOT = os.environ.get("DAFUSION_DATASETS", os.path.join(DATA_ROOT, "UOIS"))
CHECKPOINT_ROOT = os.environ.get("DAFUSION_CHECKPOINTS", os.path.join(DATA_ROOT, "checkpoints"))

EVAL_RESULT_ROOT = os.path.join(LEGACY_ROOT, "eval_result")
DEBUG_ROOT = os.path.join(DATA_ROOT, "debug")

# Weights live in the canonical store (/data1/jokim/weights) and are surfaced here
# as symlinks under data/checkpoints/{mask2former,uoais,ucn,sam,cropformer}.
# Legacy DA-Fusion training runs live under data/checkpoints/legacy_train_ckpt/.
LEGACY_TRAIN_CKPT = os.path.join(CHECKPOINT_ROOT, "legacy_train_ckpt")

CGNET_WEIGHTS = os.path.join(CHECKPOINT_ROOT, "uoais", "rgbd_fg.pth")     # UOAIS CG-Net foreground net
SAM_WEIGHTS = os.path.join(CHECKPOINT_ROOT, "sam", "sam_vit_h_4b8939.pth")
MASK2FORMER_INIT = os.path.join(CHECKPOINT_ROOT, "mask2former", "model_final_e5f453.pkl")  # training init

# Benchmark datasets
OSD_PATH = os.path.join(DATASET_ROOT, "OSD-0.20-depth")
OCID_PATH = os.path.join(DATASET_ROOT, "OCID-dataset")
BOSD_PATH = os.path.join(DATASET_ROOT, "OCBD")   # dir renamed BOSD->OCBD (paper); var kept for frozen build
UOAIS_SIM_PATH = os.path.join(DATASET_ROOT, "UOAIS-Sim")
BENCHMARK_RESULT_ROOT = os.path.join(DATASET_ROOT, "benchmark_result")

# name -> (image/data root, eval-result filename suffix)
DATASET_PATHS = {"osd": OSD_PATH, "ocid": OCID_PATH, "bosd": BOSD_PATH}

# Best trained checkpoints per modality (folders with config.yaml + *.pth).
# rgbd: DI_AGF_rgbd_none_NOW0.4 model_final — the best *verified* RGB-D checkpoint (OSD 91.2 / OCID 91.9).
#       (The paper's OCID 92.1 came from NOW0.5 iter-19999, whose weight file did not survive.)
# rgb:  dafusion_rgb_NOW0.5_iter19999 — best measured RGB checkpoint (OSD 92.0 in rgb mode).
BEST_CHECKPOINT = {
    "rgbd": os.path.join(LEGACY_TRAIN_CKPT, "DI_AGF_rgbd_none_NOW0.4_BS2_LR1e-05"),
    "rgb": os.path.join(LEGACY_TRAIN_CKPT, "dafusion_rgb_NOW0.5_iter19999"),
    "depth": os.path.join(LEGACY_TRAIN_CKPT, "DI_depth_none_NOW0.4_BS4_LR1e-05"),
}
