import argparse
import os

from eval.eval_utils2 import eval_visible_on_OSD, eval_visible_on_BOSD, eval_visible_on_OCID
from legacy_paths import (
    DATASET_PATHS, CGNET_WEIGHTS, EVAL_RESULT_ROOT, BENCHMARK_RESULT_ROOT, BEST_CHECKPOINT,
)

##### Command example #####
# python benchmark_all.py --input_type rgbd --gpu 0
# Runs the best checkpoint for the given modality across OSD / OCID / BOSD.

# Best surviving checkpoints (only model_final.pth remains per dir).
BEST_MODEL_FILE = {
    "rgb": "model_final.pth",
    "depth": "model_final.pth",
    "rgbd": "model_final.pth",
}


def check_dir(path):
    os.makedirs(path, exist_ok=True)
    print(f"result dir: {path}")


def start(args, dataset, use_cgnet):
    args.dataset_path = DATASET_PATHS[dataset]
    args.save_txt = os.path.join(EVAL_RESULT_ROOT, "Benchmark_All.txt")
    args.save_result_dir = os.path.join(BENCHMARK_RESULT_ROOT, "all", f"{dataset.upper()}_{args.exp_name}")
    args.use_cgnet = use_cgnet
    check_dir(args.save_result_dir)
    {"osd": eval_visible_on_OSD, "ocid": eval_visible_on_OCID, "bosd": eval_visible_on_BOSD}[dataset](args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unseen Object Segmentation Benchmark (all datasets)", add_help=False)
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument("--input_type", type=str, default="rgbd", help="rgb, depth, rgbd")
    parser.add_argument("--save_img", type=bool, default=False)
    parser.add_argument("--datasets", nargs="+", default=["osd"], help="subset of: osd ocid bosd")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    exp_dir = BEST_CHECKPOINT[args.input_type]
    args.exp_name = os.path.basename(os.path.normpath(exp_dir))
    args.config_file = os.path.join(exp_dir, "config.yaml")
    args.cgnet_weight_path = CGNET_WEIGHTS
    args.weight = os.path.join(exp_dir, BEST_MODEL_FILE[args.input_type])
    filename = os.path.splitext(os.path.basename(args.weight))[0]
    args.name = f"{args.exp_name}_{filename}"
    os.makedirs(EVAL_RESULT_ROOT, exist_ok=True)

    # OSD uses the CG-Net foreground filter; OCID/BOSD do not.
    cgnet_by_dataset = {"osd": True, "ocid": False, "bosd": False}
    for ds in args.datasets:
        start(args, ds, cgnet_by_dataset[ds])
