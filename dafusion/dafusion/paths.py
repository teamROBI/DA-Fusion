"""Path resolution for the DA-Fusion (reimplementation) build.

Resolves repo-relative by default; override with env vars:
    DAFUSION_DATA        -> data dir            (default: <repo>/data)
    DAFUSION_DATASETS    -> benchmark datasets  (default: <DATA>/UOIS)
    DAFUSION_CHECKPOINTS -> checkpoints         (default: <DATA>/checkpoints)

Note: the OCBD benchmark dir was formerly named BOSD (paper renamed it).
"""
import os

PKG_ROOT = os.path.dirname(os.path.abspath(__file__))            # <repo>/dafusion/dafusion
BUILD_ROOT = os.path.dirname(PKG_ROOT)                           # <repo>/dafusion
REPO_ROOT = os.path.dirname(BUILD_ROOT)                          # <repo>

DATA_ROOT = os.environ.get("DAFUSION_DATA", os.path.join(REPO_ROOT, "data"))
DATASET_ROOT = os.environ.get("DAFUSION_DATASETS", os.path.join(DATA_ROOT, "UOIS"))
CHECKPOINT_ROOT = os.environ.get("DAFUSION_CHECKPOINTS", os.path.join(DATA_ROOT, "checkpoints"))

DAFUSION_CKPT_ROOT = os.path.join(CHECKPOINT_ROOT, "dafusion")          # new training runs
MASK2FORMER_INIT = os.path.join(CHECKPOINT_ROOT, "mask2former", "model_final_e5f453.pkl")
CGNET_WEIGHTS = os.path.join(CHECKPOINT_ROOT, "uoais", "rgbd_fg.pth")   # optional fg filter (eval)

# Checkpoint sweep outputs (curves/tables/winners), kept OUT of the checkpoints tree.
# (dafusion/eval/sweep.py recomputes this inline to stay import-light; keep them in sync.)
EVAL_RESULT_ROOT = os.environ.get("DAFUSION_EVAL_RESULTS", os.path.join(DATA_ROOT, "eval_results"))

UOAIS_SIM_PATH = os.path.join(DATASET_ROOT, "UOAIS-Sim")
TABLETOP_PATH = os.path.join(DATASET_ROOT, "tabletop_dataset_v5_public")  # TOD (UCN/MSMFormer)
OSD_PATH = os.path.join(DATASET_ROOT, "OSD-0.20-depth")
OCID_PATH = os.path.join(DATASET_ROOT, "OCID-dataset")
OCBD_PATH = os.path.join(DATASET_ROOT, "OCBD")     # paper name (formerly BOSD)
BENCHMARK_RESULT_ROOT = os.path.join(DATASET_ROOT, "benchmark_result")

# eval benchmark name -> data root
DATASET_PATHS = {"osd": OSD_PATH, "ocid": OCID_PATH, "ocbd": OCBD_PATH}
