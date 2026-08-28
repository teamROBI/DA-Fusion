import argparse
import os

from eval.eval_utils2 import eval_visible_on_OSD, eval_visible_on_BOSD, eval_visible_on_OCID
from legacy_paths import (
    DATASET_PATHS, CGNET_WEIGHTS, EVAL_RESULT_ROOT, BENCHMARK_RESULT_ROOT, CHECKPOINT_ROOT,
)

##### Command example #####
# python benchmark_single.py --input_type rgbd --dataset ocid --gpu 0 \
#     --exp_name DI_AGF_rgbd_none_NOW0.4_BS2_LR1e-05 --new_ckpts <path1.pth> <path2.pth>
# Evaluates ONLY the explicitly listed checkpoints (used by benchmark_auto.py during training).

EVAL_FUNCS = {"osd": eval_visible_on_OSD, "ocid": eval_visible_on_OCID, "bosd": eval_visible_on_BOSD}
SAVE_SUFFIX = {"osd": "OSD_{}.txt", "ocid": "OCID_{}_final.txt", "bosd": "BOSD_{}.txt"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unseen Object Segmentation Benchmark (single/explicit ckpts)", add_help=False)
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument("--input_type", type=str, default="rgbd", help="rgb, depth, depth_colormap, rgbd")
    parser.add_argument("--dataset", type=str, default="ocid", help="osd, ocid, bosd")
    parser.add_argument("--save_img", type=bool, default=False)
    parser.add_argument("--use_cgnet", type=bool, default=False)
    parser.add_argument("--exp_name", type=str, required=True, help="experiment (checkpoint folder) name")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="checkpoint folder (default: <checkpoints>/<exp_name>)")
    parser.add_argument("--new_ckpts", nargs="+", type=str, required=True,
                        help="checkpoint paths to evaluate, e.g. /path/model_0009999.pth")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    exp_name = args.exp_name
    exp_dir = args.exp_dir or os.path.join(CHECKPOINT_ROOT, exp_name)

    if args.dataset not in EVAL_FUNCS:
        raise SystemExit(f">>> Invalid dataset '{args.dataset}' (choose osd, ocid, bosd)")

    args.config_file = os.path.join(exp_dir, "config.yaml")
    args.cgnet_weight_path = CGNET_WEIGHTS
    args.dataset_path = DATASET_PATHS[args.dataset]
    args.save_result_dir = os.path.join(BENCHMARK_RESULT_ROOT, exp_name)
    os.makedirs(args.save_result_dir, exist_ok=True)
    os.makedirs(EVAL_RESULT_ROOT, exist_ok=True)
    args.save_txt = os.path.join(EVAL_RESULT_ROOT, SAVE_SUFFIX[args.dataset].format(exp_name))

    for pth in sorted(args.new_ckpts):
        args.weight = pth
        filename = os.path.splitext(os.path.basename(pth))[0]
        args.name = f"{exp_name}_{filename}"
        print(f">>> Evaluating {args.name} on {args.dataset.upper()}")
        EVAL_FUNCS[args.dataset](args)
