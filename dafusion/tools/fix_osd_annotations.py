"""Interactive SAM tool to fix OSD annotations that are misaligned with the RGB image.

OSD ground-truth masks are registered to the depth-camera frame, so on the RGB image they can
be spatially offset (worse for closer objects). This tool lets you re-segment objects directly
on the RGB with Segment Anything (SAM) and write RGB-aligned instance label maps.

Only 111 OSD images, so eyeballing + fixing the wrong ones is quick.

Run (needs a display / X11 forwarding; DISPLAY must be set):
    dafusion/.venv/bin/python dafusion/tools/fix_osd_annotations.py
    # jump straight to specific frames you know are bad:
    dafusion/.venv/bin/python dafusion/tools/fix_osd_annotations.py --only test25,test26,test49

Workflow per image: click objects, commit each as an instance, then Save. Skipped images copy
the ORIGINAL annotation through, so <out-dir> ends up a complete, drop-in replacement set.

Controls (also printed on start):
    left click     add positive point (object)      right click   add negative point (background)
    n / Enter      commit current mask as instance   m            cycle SAM's 3 mask candidates
    u              undo last point                    c            clear current points
    z              undo last committed instance       g            toggle OLD-GT reference overlay
    s              SAVE new annotation, next image    k            keep ORIGINAL, next image
    [ / ]          prev / next image (no save)        q            quit
"""
import argparse
import glob
import os
import shutil

import cv2
import imageio.v2 as imageio
import numpy as np

try:
    from dafusion.paths import DATASET_PATHS
    DEFAULT_OSD = DATASET_PATHS["osd"]
except Exception:
    DEFAULT_OSD = None

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
           (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212),
           (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200), (128, 0, 0)]


def color(i):
    return np.array(PALETTE[(i - 1) % len(PALETTE)], dtype=np.float32)


class Annotator:
    def __init__(self, predictor, items, out_dir):
        self.predictor = predictor
        self.items = items          # list of (name, rgb_path, ann_path)
        self.out_dir = out_dir
        self.idx = 0
        self.show_gt = True
        self._load()

    # ---------- per-image state ----------
    def _load(self):
        name, rgb_path, ann_path = self.items[self.idx]
        self.name = name
        self.rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        self.h, self.w = self.rgb.shape[:2]
        self.old_gt = imageio.imread(ann_path) if ann_path and os.path.exists(ann_path) else None
        self.label = np.zeros((self.h, self.w), np.uint8)   # committed instances
        self.n_inst = 0
        self.pts, self.lbls = [], []                        # current object's prompt points
        self.cand, self.cand_i = None, 0                    # SAM candidate masks for current pts
        self.predictor.set_image(self.rgb)
        print(f"[{self.idx + 1}/{len(self.items)}] {self.name}  (out exists: "
              f"{os.path.exists(self._out_path())})")

    def _out_path(self):
        return os.path.join(self.out_dir, self.name + ".png")

    # ---------- SAM ----------
    def _predict(self):
        if not self.pts:
            self.cand = None
            return
        m, scores, _ = self.predictor.predict(
            point_coords=np.array(self.pts), point_labels=np.array(self.lbls),
            multimask_output=True)
        order = np.argsort(scores)[::-1]
        self.cand = m[order]
        self.cand_i = 0

    def _cur_mask(self):
        return None if self.cand is None else self.cand[self.cand_i]

    # ---------- rendering ----------
    def render(self):
        vis = self.rgb.astype(np.float32).copy()
        for i in range(1, self.n_inst + 1):
            m = self.label == i
            vis[m] = 0.5 * vis[m] + 0.5 * color(i)
        cm = self._cur_mask()
        if cm is not None:
            vis[cm] = 0.5 * vis[cm] + 0.5 * np.array([255, 255, 255], np.float32)
        vis = vis.astype(np.uint8)
        if self.show_gt and self.old_gt is not None:
            for lbl in np.unique(self.old_gt):
                if lbl == 0:
                    continue
                cnts, _ = cv2.findContours((self.old_gt == lbl).astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = tuple(int(v) for v in color(int(lbl)))   # old GT: per-instance color, thick
                cv2.drawContours(vis, cnts, -1, c, 3)
        return vis

    # ---------- actions ----------
    def add_point(self, x, y, positive):
        self.pts.append([int(x), int(y)])
        self.lbls.append(1 if positive else 0)
        self._predict()

    def undo_point(self):
        if self.pts:
            self.pts.pop(); self.lbls.pop(); self._predict()

    def clear_points(self):
        self.pts, self.lbls, self.cand = [], [], None

    def cycle_mask(self):
        if self.cand is not None:
            self.cand_i = (self.cand_i + 1) % len(self.cand)

    def commit(self):
        cm = self._cur_mask()
        if cm is None:
            print("  (no mask to commit)"); return
        self.n_inst += 1
        self.label[cm] = self.n_inst    # later instances win overlaps
        self.clear_points()
        print(f"  committed instance {self.n_inst}")

    def undo_instance(self):
        if self.n_inst:
            self.label[self.label == self.n_inst] = 0
            self.n_inst -= 1
            print(f"  removed instance {self.n_inst + 1}")

    def save(self):
        os.makedirs(self.out_dir, exist_ok=True)
        imageio.imwrite(self._out_path(), self.label)
        print(f"  SAVED {self._out_path()}  ({self.n_inst} instances)")

    def keep_original(self):
        os.makedirs(self.out_dir, exist_ok=True)
        _, _, ann_path = self.items[self.idx]
        if ann_path and os.path.exists(ann_path):
            shutil.copy(ann_path, self._out_path())
            print(f"  kept ORIGINAL -> {self._out_path()}")

    def go(self, delta):
        self.idx = max(0, min(len(self.items) - 1, self.idx + delta))
        self._load()


def main():
    ap = argparse.ArgumentParser("Fix OSD annotations interactively with SAM")
    ap.add_argument("--osd-root", default=DEFAULT_OSD, required=DEFAULT_OSD is None)
    ap.add_argument("--out-dir", default=None, help="default: <osd-root>/annotation_fixed")
    ap.add_argument("--sam-checkpoint", default="/data1/jokim/weights/SAM/sam_vit_h_4b8939.pth")
    ap.add_argument("--model-type", default="vit_h")
    ap.add_argument("--only", default=None, help="comma list of image names (no ext) to load, e.g. test25,test26")
    ap.add_argument("--start", type=int, default=0, help="start index")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.osd_root, "annotation_fixed")
    rgbs = sorted(glob.glob(os.path.join(args.osd_root, "image_color", "*.png")))
    items = []
    for r in rgbs:
        name = os.path.splitext(os.path.basename(r))[0]
        if args.only and name not in args.only.split(","):
            continue
        items.append((name, r, os.path.join(args.osd_root, "annotation", name + ".png")))
    if not items:
        raise SystemExit("no OSD images found (check --osd-root / --only)")

    import torch
    from segment_anything import sam_model_registry, SamPredictor
    print(f">>> loading SAM {args.model_type} ...")
    sam = sam_model_registry[args.model_type](checkpoint=args.sam_checkpoint)
    sam.to("cuda" if torch.cuda.is_available() else "cpu")
    predictor = SamPredictor(sam)

    import matplotlib
    import matplotlib.pyplot as plt
    # Disable matplotlib's built-in single-key shortcuts so they don't fight our handlers
    # (s=save-dialog, g=grid, k=xscale, c/left=back, l=yscale, q=quit, etc.).
    for _km in ("keymap.save", "keymap.grid", "keymap.grid_minor", "keymap.xscale",
                "keymap.yscale", "keymap.back", "keymap.forward", "keymap.home",
                "keymap.pan", "keymap.zoom", "keymap.fullscreen", "keymap.quit",
                "keymap.quit_all", "keymap.all_axes"):
        if _km in matplotlib.rcParams:      # some keys vary by matplotlib version
            matplotlib.rcParams[_km] = []
    ann = Annotator(predictor, items, out_dir)
    ann.idx = min(args.start, len(items) - 1); ann._load()

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(ann.render())
    ax.set_axis_off()

    def refresh():
        im.set_data(ann.render())
        cm = ann._cur_mask()
        cand = "" if cm is None else f" | mask {ann.cand_i + 1}/{len(ann.cand)}"
        ax.set_title(f"[{ann.idx + 1}/{len(items)}] {ann.name} | instances={ann.n_inst} | "
                     f"pts={len(ann.pts)}{cand} | GT_ref={'on' if ann.show_gt else 'off'}")
        fig.canvas.draw_idle()

    def on_click(e):
        if e.inaxes != ax or e.xdata is None:
            return
        ann.add_point(e.xdata, e.ydata, positive=(e.button == 1)); refresh()

    def on_key(e):
        k = e.key
        if k in ("enter", "n"):
            ann.commit()
        elif k == "m":
            ann.cycle_mask()
        elif k == "u":
            ann.undo_point()
        elif k == "c":
            ann.clear_points()
        elif k == "z":
            ann.undo_instance()
        elif k == "g":
            ann.show_gt = not ann.show_gt
        elif k == "s":
            ann.save(); ann.go(1) if ann.idx < len(items) - 1 else None
        elif k == "k":
            ann.keep_original(); ann.go(1) if ann.idx < len(items) - 1 else None
        elif k == "]":
            ann.go(1)
        elif k == "[":
            ann.go(-1)
        elif k == "q":
            plt.close(fig); return
        refresh()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    print(__doc__[__doc__.index("Controls"):])
    refresh()
    plt.show()
    print(f">>> done. Fixed/kept annotations in: {out_dir}")


if __name__ == "__main__":
    main()
