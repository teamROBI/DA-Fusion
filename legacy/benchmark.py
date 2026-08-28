import argparse
import os

from eval.eval_utils2 import eval_visible_on_OSD, eval_visible_on_BOSD, eval_visible_on_OCID
from legacy_paths import (
    DATASET_PATHS, CGNET_WEIGHTS, EVAL_RESULT_ROOT, BENCHMARK_RESULT_ROOT, BEST_CHECKPOINT,
)

##### Command examples #####
# python benchmark.py --input_type rgbd --dataset ocid --gpu 0
# python benchmark.py --input_type rgb  --dataset osd  --gpu 0 --use_cgnet True --save_img True
# python benchmark.py --input_type rgbd --dataset ocid --exp_dir ../data/checkpoints/legacy_train_ckpt/DI_AGF_rgbd_none_NOW0.4_BS2_LR1e-05

EVAL_FUNCS = {"osd": eval_visible_on_OSD, "ocid": eval_visible_on_OCID, "bosd": eval_visible_on_BOSD}
# ocid keeps the historical "_final" suffix so archived result files line up
SAVE_SUFFIX = {"osd": "OSD_{}.txt", "ocid": "OCID_{}_final.txt", "bosd": "BOSD_{}.txt"}


def get_weights(folder_path):
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pth")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Unseen Object Segmentation Benchmark Datasets", add_help=False)
    parser.add_argument("--gpu", type=str, default="0", help="GPU id")
    parser.add_argument("--input_type", type=str, default="rgbd", help="rgb, depth, depth_colormap, rgbd")
    parser.add_argument("--dataset", type=str, default="ocid", help="osd, ocid, bosd")
    parser.add_argument("--save_img", type=bool, default=False, help="save prediction visualizations")
    parser.add_argument("--use_cgnet", type=bool, default=False, help="apply CG-Net foreground filter")
    parser.add_argument("--exp_dir", type=str, default=None,
                        help="checkpoint folder containing config.yaml + *.pth "
                             "(default: best checkpoint for --input_type)")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="experiment name used in result filenames (default: basename of exp_dir)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    exp_dir = args.exp_dir or BEST_CHECKPOINT.get(args.input_type)
    if exp_dir is None or not os.path.isdir(exp_dir):
        raise SystemExit(f">>> checkpoint folder not found: {exp_dir}")
    exp_name = args.exp_name or os.path.basename(os.path.normpath(exp_dir))

    if args.dataset not in EVAL_FUNCS:
        raise SystemExit(f">>> Invalid dataset '{args.dataset}' (choose osd, ocid, bosd)")

    args.config_file = os.path.join(exp_dir, "config.yaml")
    args.cgnet_weight_path = CGNET_WEIGHTS
    args.dataset_path = DATASET_PATHS[args.dataset]
    args.save_result_dir = os.path.join(BENCHMARK_RESULT_ROOT, exp_name)
    os.makedirs(args.save_result_dir, exist_ok=True)
    os.makedirs(EVAL_RESULT_ROOT, exist_ok=True)
    args.save_txt = os.path.join(EVAL_RESULT_ROOT, SAVE_SUFFIX[args.dataset].format(exp_name))

    for pth in sorted(get_weights(exp_dir), reverse=True):
        args.weight = pth
        filename = os.path.splitext(os.path.basename(pth))[0]
        args.name = f"{exp_name}_{filename}"
        print(f">>> Evaluating {args.name} on {args.dataset.upper()}")
        EVAL_FUNCS[args.dataset](args)
