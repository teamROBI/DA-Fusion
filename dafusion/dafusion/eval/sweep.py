"""Multi-checkpoint DA-Fusion sweep: eval every checkpoint of a run on all three UOIS
benchmarks (OCID / OSD / OCBD) across all GPUs, plot metrics vs training iteration, and
prune the run down to the winners.

Retention: keep the best checkpoint for EACH benchmark (by Overlap-F) plus the best on the
3-way average Overlap-F -> 1-4 distinct files after dedup; everything else is deleted
permanently. `model_final.pth` is kept regardless unless --no-keep-final.

Two roles in one file:
  orchestrator (default): discovers checkpoints, schedules one worker subprocess per
      checkpoint across GPUs, then aggregates/plots/prunes. Import-light (no torch, no
      `dafusion`) so --dry-run works even before the `dafusion.data` package exists.
  worker (--worker): evaluates ONE checkpoint on the requested datasets on ONE GPU and
      writes a JSON result. Imports the model stack (needs `dafusion`).

Usage (orchestrator):
    python -m dafusion.eval.sweep --config configs/dafusion_rgbd_uoais.yaml \
        [--output-dir <run>] [--datasets ocid,osd,ocbd] [--gpus 0,1,2,3] \
        [--use_cgnet] [--no-keep-final] [--dry-run]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

# Metric keys as produced by compute_PRF.multilabel_metrics.
OVERLAP_F = "Objects F-measure"
BOUNDARY_F = "Boundary F-measure"
PCT75 = "obj_detected_075_percentage"
CSV_KEYS = [
    "Objects Precision", "Objects Recall", OVERLAP_F,
    "Boundary Precision", "Boundary Recall", BOUNDARY_F, PCT75,
]
ALL_DATASETS = ["ocid", "osd", "ocbd"]


# ----------------------------- config / discovery (import-light) -----------------------------
def resolve_config_scalars(config_path):
    """Resolve OUTPUT_DIR and SOLVER.MAX_ITER from a config, following _BASE_.

    Uses detectron2's yaml-with-base loader so we don't reimplement inheritance and don't
    need the custom DA-Fusion config keys registered (avoids importing `dafusion`, whose
    __init__ pulls in the not-yet-written `data` package).
    """
    from detectron2.config import CfgNode
    d = CfgNode.load_yaml_with_base(config_path, allow_unsafe=True)
    out = d.get("OUTPUT_DIR")
    max_iter = (d.get("SOLVER") or {}).get("MAX_ITER")
    return out, max_iter


def discover_checkpoints(output_dir, max_iter=None):
    """Return [{path, iter, name, is_final}] sorted by iteration.

    Numbered checkpoints are `model_{iter:07d}.pth`; `model_final.pth` is placed at MAX_ITER
    (or just past the last numbered checkpoint when MAX_ITER is unknown).
    """
    ckpts = []
    for path in glob.glob(os.path.join(output_dir, "model_*.pth")):
        name = os.path.basename(path)
        if name == "model_final.pth":
            ckpts.append({"path": path, "iter": None, "name": name, "is_final": True})
            continue
        m = re.fullmatch(r"model_(\d+)\.pth", name)
        if not m:
            continue
        ckpts.append({"path": path, "iter": int(m.group(1)), "name": name, "is_final": False})

    # Place model_final just after the last numbered checkpoint (its true iteration). Config
    # MAX_ITER is only a fallback -- it can be a stale/base value or overridden on the CLI, so
    # it may not reflect where this run actually stopped.
    numbered = [c["iter"] for c in ckpts if c["iter"] is not None]
    final_iter = (max(numbered) + 1) if numbered else (max_iter or 0)
    for c in ckpts:
        if c["is_final"]:
            c["iter"] = final_iter
    ckpts.sort(key=lambda c: c["iter"])
    return ckpts


def eval_result_root():
    """data/eval_results (or $DAFUSION_EVAL_RESULTS). Computed inline -- mirrors
    dafusion.paths.EVAL_RESULT_ROOT -- so the orchestrator needn't import the model stack."""
    env = os.environ.get("DAFUSION_EVAL_RESULTS")
    if env:
        return env
    data = os.environ.get("DAFUSION_DATA")
    if not data:
        # repo/data ; this file is repo/dafusion/dafusion/eval/sweep.py
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        data = os.path.join(repo, "data")
    return os.path.join(data, "eval_results")


def detect_gpus(gpus_arg):
    if gpus_arg:
        return [int(x) for x in gpus_arg.split(",") if x.strip() != ""]
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [int(x) for x in env.split(",") if x.strip() != ""]
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"]).decode()
        n = len([ln for ln in out.strip().splitlines() if ln.strip()])
        return list(range(max(n, 1)))
    except Exception:
        return [0]


# ----------------------------- worker (imports the model stack) -----------------------------
def run_worker(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # Imported here (not at module top) so the orchestrator stays torch/dafusion-free.
    import argparse as _ap
    from dafusion.eval.benchmark import build_cfg, run_benchmark, average_metrics, load_cgnet
    from dafusion.engine.predictor import DAFusionPredictor
    from dafusion.data.datasets.intrinsics import get_intrinsics

    datasets = [d for d in args.datasets.split(",") if d]
    cfg = build_cfg(_ap.Namespace(config=args.config, weights=args.ckpt, input_type=args.input_type))
    predictor = DAFusionPredictor(cfg, dataset=datasets[0])
    fg_filter = "cgnet" if args.use_cgnet else args.fg_filter   # --use_cgnet is a legacy alias
    fg_model = None
    if fg_filter == "cgnet":
        cgnet_weight = args.cgnet_weight
        if not cgnet_weight:
            from dafusion.paths import CGNET_WEIGHTS
            cgnet_weight = CGNET_WEIGHTS
        fg_model = load_cgnet(cgnet_weight)

    # viz for this checkpoint goes under <sweep_dir>/viz/<iter>/ (sweep_dir = dir of --out)
    viz_dir = os.path.join(os.path.dirname(args.out), "viz", str(args.iter)) if args.save_viz else None
    result = {"ckpt": args.ckpt, "iter": args.iter, "datasets": {}}
    for ds in datasets:
        # Depth encoding (HHA) is intrinsics-dependent, so re-point per benchmark instead of
        # rebuilding/reloading the model for each dataset.
        predictor.intrinsics = get_intrinsics(ds)
        metrics_all = run_benchmark(predictor, ds, args.input_type, fg_filter=fg_filter,
                                    fg_model=fg_model, save_viz=args.save_viz, viz_dir=viz_dir)
        result["datasets"][ds] = average_metrics(metrics_all)

    with open(args.out, "w") as f:
        json.dump(result, f)
    print(f"[worker gpu{args.gpu}] wrote {args.out} for {os.path.basename(args.ckpt)}")


# ----------------------------- orchestrator scheduling -----------------------------
_PROGRESS_COLS = 4   # per-worker progress grid width


def _worker_stage(log_path):
    """Parse a worker's per-process log -> (dataset, pct) for its current benchmark, or None
    if it hasn't started a dataset yet. tqdm writes with '\\r', normalised to newlines first."""
    try:
        with open(log_path, "rb") as f:
            text = f.read().decode("utf-8", "ignore").replace("\r", "\n")
    except OSError:
        return None
    started = re.findall(r"Evaluation on (OCID|OSD|OCBD)", text)
    bar = None
    for line in text.splitlines():
        m = re.search(r"(OCID|OSD|OCBD):\s*(\d+)%", line)
        if m:
            bar = m
    if not started:
        return None
    cur = started[-1]
    pct = int(bar.group(2)) if bar and bar.group(1) == cur else 0
    return cur, pct


def _progress_summary(running, results_n, total):
    """Multi-line progress showing BOTH the whole-sweep rollup (overall checkpoint count +
    per-benchmark worker avg/min/max %) AND each individual worker's own progress line
    (checkpoint iter -> current benchmark + %)."""
    stages = {}
    per_worker = []
    starting = 0
    for w in running:
        it = w["ck"]["iter"]
        st = _worker_stage(w["log"])
        if st is None:
            starting += 1
            per_worker.append((it, "start", 0))
        else:
            stages.setdefault(st[0], []).append(st[1])
            per_worker.append((it, st[0], st[1]))
    # --- whole progress ---
    lines = [f"[progress] checkpoints {results_n}/{total} done | {len(running)} running"]
    for ds in ("OCID", "OSD", "OCBD"):
        if ds in stages:
            p = stages[ds]
            lines.append(f"    [{ds:<4}] {len(p):>2} wk  avg {sum(p)//len(p):>3}%  "
                         f"(min {min(p)} / max {max(p)})")
    if starting:
        lines.append(f"    starting {starting} wk")
    # --- each worker (per checkpoint), laid out in a grid (default 4 columns) ---
    cols = _PROGRESS_COLS
    cells = [f"i{it:<6}{ds:<4}{pct:>3}%" for it, ds, pct in sorted(per_worker)]
    for i in range(0, len(cells), cols):
        lines.append("   " + "  ".join(c.ljust(15) for c in cells[i:i + cols]))
    return "\n".join(lines)


def schedule(ckpts, gpus, work_dir, common_args, workers_per_gpu=1):
    """Evaluate every checkpoint, keeping len(gpus)*workers_per_gpu workers in flight. Each
    worker is pinned to one GPU and writes to its own log (sweep/worker_<iter>.log) so the
    parallel tqdm bars don't clobber a shared pane; the orchestrator prints one aggregated
    progress line periodically. Returns parsed result dicts (failed workers are skipped)."""
    pending = list(ckpts)
    # A "slot" is one concurrent worker bound to a specific GPU; oversubscribe each GPU.
    free = [g for g in gpus for _ in range(workers_per_gpu)]
    running = []  # list of dicts: {proc, ck, out, gpu, log}
    results = []
    last_report = 0.0

    def launch(gpu, ck):
        out_path = os.path.join(work_dir, f"result_{ck['iter']:09d}.json")
        log_path = os.path.join(work_dir, f"worker_{ck['iter']:09d}.log")
        cmd = [
            sys.executable, "-m", "dafusion.eval.sweep", "--worker",
            "--gpu", str(gpu), "--ckpt", ck["path"], "--iter", str(ck["iter"]),
            "--config", common_args["config"], "--datasets", common_args["datasets"],
            "--input_type", common_args["input_type"], "--out", out_path,
        ]
        if common_args["use_cgnet"]:
            cmd.append("--use_cgnet")
        if common_args.get("fg_filter") and common_args["fg_filter"] != "none":
            cmd += ["--fg_filter", common_args["fg_filter"]]
        if common_args["cgnet_weight"]:
            cmd += ["--cgnet_weight", common_args["cgnet_weight"]]
        if common_args.get("save_viz"):
            cmd += ["--save_viz", str(common_args["save_viz"])]
        print(f"[sched] gpu{gpu} <- {ck['name']} (iter {ck['iter']})", flush=True)
        logf = open(log_path, "w")
        proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
        return {"proc": proc, "ck": ck, "out": out_path, "gpu": gpu, "log": log_path, "logf": logf}

    total = len(pending)
    while pending or running:
        while pending and free:
            gpu = free.pop(0)
            running.append(launch(gpu, pending.pop(0)))
        still = []
        for w in running:
            ret = w["proc"].poll()
            if ret is None:
                still.append(w)
                continue
            w["logf"].close()
            free.append(w["gpu"])
            if ret == 0 and os.path.exists(w["out"]):
                with open(w["out"]) as f:
                    results.append(json.load(f))
                print(f"[sched] done {w['ck']['name']} ({len(results)}/{total})", flush=True)
            else:
                print(f"[sched] WARNING worker for {w['ck']['name']} failed (exit {ret}) -- skipped", flush=True)
        running = still
        # compact per-benchmark progress summary every ~30s (not one line per worker).
        now = time.time()
        if running and now - last_report >= 30:
            print(_progress_summary(running, len(results), total), flush=True)
            last_report = now
        time.sleep(2)

    results.sort(key=lambda r: r["iter"])
    return results


# ----------------------------- aggregate / plot / select / prune -----------------------------
def write_tables(results, sweep_dir):
    with open(os.path.join(sweep_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    csv_path = os.path.join(sweep_dir, "results.csv")
    with open(csv_path, "w") as f:
        f.write("iter,ckpt,dataset," + ",".join(CSV_KEYS) + "\n")
        for r in results:
            for ds, m in r["datasets"].items():
                row = [str(r["iter"]), os.path.basename(r["ckpt"]), ds]
                row += [f"{m.get(k, 0.0):.6f}" for k in CSV_KEYS]
                f.write(",".join(row) + "\n")
    print(f">>> wrote {csv_path}")


def mean_overlap(r, datasets):
    vals = [r["datasets"][d][OVERLAP_F] for d in datasets if d in r["datasets"]]
    return sum(vals) / len(vals) if vals else 0.0


def select_winners(results, datasets):
    """Best checkpoint (by Overlap-F) for each dataset + best 3-way mean. Returns
    {category: result_dict}; categories with no data are omitted."""
    winners = {}
    for ds in datasets:
        scored = [r for r in results if ds in r["datasets"]]
        if scored:
            winners[ds] = max(scored, key=lambda r: r["datasets"][ds][OVERLAP_F])
    if results:
        winners["mean"] = max(results, key=lambda r: mean_overlap(r, datasets))
    return winners


def plot_curves(results, datasets, sweep_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iters = [r["iter"] for r in results]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ds in datasets:
        y_of = [r["datasets"].get(ds, {}).get(OVERLAP_F, float("nan")) for r in results]
        y_bf = [r["datasets"].get(ds, {}).get(BOUNDARY_F, float("nan")) for r in results]
        axes[0].plot(iters, y_of, marker="o", label=ds.upper())
        axes[1].plot(iters, y_bf, marker="o", label=ds.upper())
    axes[0].plot(iters, [mean_overlap(r, datasets) for r in results],
                 marker="s", linestyle="--", color="black", label="mean")
    axes[0].set_title("Overlap F-measure")
    axes[1].set_title("Boundary F-measure")
    for ax in axes:
        ax.set_xlabel("training iteration")
        ax.set_ylabel("F-measure")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    png = os.path.join(sweep_dir, "curve.png")
    fig.savefig(png, dpi=120)
    print(f">>> wrote {png}")


def prune(results, winners, keep_final, dry_run):
    """Delete every evaluated checkpoint that is not a winner. Returns (kept, deleted) path
    lists. `model_final.pth` is retained regardless when keep_final is True."""
    keep_paths = {w["ckpt"] for w in winners.values()}
    win_by_path = {}
    for cat, w in winners.items():
        win_by_path.setdefault(w["ckpt"], []).append(cat)

    kept, deleted = [], []
    for r in results:
        path = r["ckpt"]
        is_final = os.path.basename(path) == "model_final.pth"
        if path in keep_paths or (keep_final and is_final):
            cats = win_by_path.get(path, [])
            reason = "+".join(cats) if cats else ("keep-final" if is_final else "?")
            kept.append((path, reason, r))
        else:
            deleted.append(path)

    print("\n=== retention ===")
    for path, reason, r in kept:
        of = {ds: round(r["datasets"][ds][OVERLAP_F] * 100, 1) for ds in r["datasets"]}
        print(f"  KEEP  {os.path.basename(path):<22} [{reason}]  OverlapF%={of}")
    freed = 0
    for path in deleted:
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        freed += size
        if dry_run:
            print(f"  del?  {os.path.basename(path):<22} ({size/1e6:.0f} MB)  [dry-run]")
        else:
            os.remove(path)
            print(f"  DEL   {os.path.basename(path):<22} ({size/1e6:.0f} MB)")
    print(f"{'would free' if dry_run else 'freed'} {freed/1e9:.2f} GB "
          f"({len(kept)} kept, {len(deleted)} {'to delete' if dry_run else 'deleted'})")
    return kept, deleted


def log_to_wandb(results, datasets, output_dir, curve_png, winners):
    """Log the sweep's benchmark curves to W&B. If training stashed a run id in
    <output_dir>/wandb_run.json, resume THAT run so eval and loss share one timeline;
    otherwise start a standalone '<run>-sweep' run. Uses a custom 'eval/step' x-axis
    (= training iteration) so resuming a finished run doesn't fight W&B's global step."""
    if os.environ.get("WANDB_MODE", "").lower() == "disabled":
        print("[sweep] WANDB_MODE=disabled -- skipping W&B logging")
        return
    try:
        import wandb
    except ImportError:
        print("[sweep] wandb not installed -- skipping W&B logging")
        return

    project = os.environ.get("WANDB_PROJECT", "da-fusion")
    info_path = os.path.join(output_dir, "wandb_run.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        wandb.init(id=info["id"], project=info.get("project", project),
                   entity=info.get("entity"), resume="allow", dir=output_dir)
    else:
        wandb.init(project=project, dir=output_dir,
                   name=os.path.basename(os.path.normpath(output_dir)) + "-sweep")

    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    for r in results:  # sorted by iter
        log = {"eval/step": r["iter"], "eval/mean/OverlapF": mean_overlap(r, datasets)}
        for ds in datasets:
            m = r["datasets"].get(ds)
            if m:
                log[f"eval/{ds}/OverlapF"] = m[OVERLAP_F]
                log[f"eval/{ds}/BoundaryF"] = m[BOUNDARY_F]
                log[f"eval/{ds}/pct75"] = m[PCT75]
        wandb.log(log)
    if os.path.exists(curve_png):
        wandb.log({"eval/curve": wandb.Image(curve_png)})
    for cat, w in winners.items():
        wandb.summary[f"best/{cat}_iter"] = w["iter"]
        wandb.summary[f"best/{cat}_ckpt"] = os.path.basename(w["ckpt"])
    wandb.finish()
    print("[sweep] logged benchmark curves to W&B")


def run_orchestrator(args):
    output_dir, max_iter = args.output_dir, None
    if args.config:
        cfg_out, max_iter = resolve_config_scalars(args.config)
        output_dir = output_dir or cfg_out
    if not output_dir:
        sys.exit("error: could not resolve OUTPUT_DIR (pass --output-dir or --config)")
    output_dir = os.path.abspath(output_dir)

    ckpts = discover_checkpoints(output_dir, max_iter)
    if not ckpts:
        sys.exit(f"error: no model_*.pth checkpoints under {output_dir}")
    gpus = detect_gpus(args.gpus)
    datasets = [d for d in args.datasets.split(",") if d]
    print(f"Found {len(ckpts)} checkpoints in {output_dir}; GPUs={gpus}; datasets={datasets}")
    for c in ckpts:
        print(f"  {c['name']:<22} iter={c['iter']}")

    # Eval outputs go OUTSIDE the checkpoints tree: data/eval_results/<run>/ by default.
    sweep_dir = args.sweep_dir or os.path.join(eval_result_root(),
                                               os.path.basename(os.path.normpath(output_dir)))
    os.makedirs(sweep_dir, exist_ok=True)
    print(f"Eval outputs -> {sweep_dir}")

    if args.dry_run:
        print("\n[dry-run] would schedule the above across GPUs, then plot/prune. "
              "No workers launched, nothing deleted.")
        return

    if not args.config:
        sys.exit("error: --config is required to evaluate (workers build the model from it)")

    print(f"Scheduling {len(ckpts)} checkpoints over {len(gpus)} GPU(s) "
          f"x {args.workers_per_gpu} worker(s) = {len(gpus) * args.workers_per_gpu} concurrent")
    results = schedule(
        ckpts, gpus, sweep_dir,
        common_args={
            "config": args.config or "", "datasets": args.datasets,
            "input_type": args.input_type, "use_cgnet": args.use_cgnet,
            "fg_filter": args.fg_filter, "cgnet_weight": args.cgnet_weight,
            "save_viz": args.save_viz,
        },
        workers_per_gpu=args.workers_per_gpu,
    )
    if not results:
        sys.exit("error: all workers failed; nothing to aggregate")

    write_tables(results, sweep_dir)
    plot_curves(results, datasets, sweep_dir)
    winners = select_winners(results, datasets)
    with open(os.path.join(sweep_dir, "kept.json"), "w") as f:
        json.dump({cat: os.path.basename(w["ckpt"]) for cat, w in winners.items()}, f, indent=2)
    if not args.no_wandb:
        try:
            log_to_wandb(results, datasets, output_dir,
                         os.path.join(sweep_dir, "curve.png"), winners)
        except Exception as e:  # never let W&B block the actual pruning
            print(f"[sweep] WARNING W&B logging failed: {e}")
    if not args.prune:
        print("[sweep] no --prune: keeping ALL checkpoints (winners recorded in kept.json). "
              "Re-run with --prune to delete non-winners.")
        return
    # --prune: show the deletion plan, then confirm before removing anything.
    prune(results, winners, keep_final=args.keep_final, dry_run=True)
    if args.yes:
        confirmed = True
    elif not sys.stdin.isatty():
        print("[sweep] --prune given but no TTY to confirm; keeping all. Use --yes to force.")
        confirmed = False
    else:
        confirmed = input("Delete the non-winner checkpoints listed above? [y/N] ").strip().lower() in ("y", "yes")
    if confirmed:
        prune(results, winners, keep_final=args.keep_final, dry_run=False)
    else:
        print("[sweep] pruning cancelled -- all checkpoints kept.")


# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser("DA-Fusion multi-checkpoint sweep")
    ap.add_argument("--config", default=None, help="run config (resolves OUTPUT_DIR + MAX_ITER)")
    ap.add_argument("--output-dir", default=None, help="run dir with model_*.pth (overrides config)")
    ap.add_argument("--sweep-dir", default=None,
                    help="where to write eval outputs (default: data/eval_results/<run>)")
    ap.add_argument("--datasets", default=",".join(ALL_DATASETS), help="comma list: ocid,osd,ocbd")
    ap.add_argument("--input_type", default="rgbd", choices=["rgb", "depth", "rgbd"])
    ap.add_argument("--gpus", default=None, help="comma list e.g. 0,1,2,3 (default: all)")
    ap.add_argument("--workers-per-gpu", type=int, default=1,
                    help="concurrent workers per GPU; eval is CPU-bound so >1 overlaps CPU "
                         "metric with GPU inference (~3.4GB VRAM each). Try 3-4 on 48GB.")
    ap.add_argument("--fg_filter", default="depth", choices=["none", "cgnet", "depth"],
                    help="foreground filter: none | cgnet (UOAIS) | depth (UCN/MSMFormer depth-validity)")
    ap.add_argument("--use_cgnet", action="store_true", help="legacy alias for --fg_filter cgnet")
    ap.add_argument("--cgnet_weight", default=None,
                    help="CG-Net weights (default: dafusion.paths.CGNET_WEIGHTS, resolved in worker)")
    ap.add_argument("--no-wandb", action="store_true",
                    help="skip W&B logging (also skipped if WANDB_MODE=disabled)")
    ap.add_argument("--prune", action="store_true",
                    help="after scoring, delete non-winner checkpoints (asks y/N first). "
                         "Default: keep everything.")
    ap.add_argument("--yes", action="store_true",
                    help="with --prune, skip the confirmation prompt (non-interactive)")
    ap.add_argument("--save_viz", type=int, default=0, metavar="K",
                    help="per checkpoint, save K good/medium/bad prediction panels per benchmark "
                         "-> <sweep_dir>/viz/<iter>/")
    ap.add_argument("--keep-final", dest="keep_final", action="store_true", default=True,
                    help="always keep model_final.pth (default)")
    ap.add_argument("--no-keep-final", dest="keep_final", action="store_false")
    ap.add_argument("--dry-run", action="store_true", help="discover + show plan; no eval, no delete")
    # worker-only
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--gpu", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--ckpt", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--iter", type=int, default=0, help=argparse.SUPPRESS)
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        run_worker(args)
    else:
        run_orchestrator(args)


if __name__ == "__main__":
    main()
