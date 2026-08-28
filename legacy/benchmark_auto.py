"""Watcher: polls a training checkpoint dir and evaluates each new checkpoint
via benchmark_single.py as it appears. Useful for tracking eval during training.

    python benchmark_auto.py --input_type rgbd --dataset ocid --gpu 0 \
        --exp_name DI_AGF_rgbd_none_NOW0.4_BS2_LR1e-05
"""
import argparse
import os
import subprocess
import time

from legacy_paths import CHECKPOINT_ROOT, LEGACY_ROOT

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Auto-benchmark watcher", add_help=False)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--input_type", type=str, default="rgbd")
    parser.add_argument("--dataset", type=str, default="ocid")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--interval", type=int, default=360, help="poll seconds")
    args = parser.parse_args()

    ckpt_dir = os.path.join(CHECKPOINT_ROOT, args.exp_name)
    benchmark_script = os.path.join(LEGACY_ROOT, "benchmark_single.py")
    print(f"GPU: {args.gpu}, watching: {ckpt_dir}")

    seen_files = set()
    while True:
        current = {f for f in os.listdir(ckpt_dir) if f.endswith(".pth")} if os.path.isdir(ckpt_dir) else set()
        new_files = current - seen_files
        if new_files:
            print("New checkpoint(s):", new_files)
            new_ckpts = [os.path.join(ckpt_dir, f) for f in new_files]
            cmd = [
                "python", benchmark_script,
                "--input_type", args.input_type, "--dataset", args.dataset,
                "--gpu", args.gpu, "--exp_name", args.exp_name,
                "--new_ckpts", *new_ckpts,
            ]
            subprocess.run(cmd)
        else:
            print("No new checkpoints.")
        seen_files = current
        time.sleep(args.interval)
