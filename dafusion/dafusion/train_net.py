"""DA-Fusion training entrypoint (detectron2). Uses the DA-Fusion dataset mapper and
the Mask2Former-style optimizer (per-param-group LR/weight-decay, backbone multiplier,
optional full-model gradient clipping, AMP)."""
import copy
import itertools
import json
import logging
import os
from typing import Any, Dict, List, Set

import torch

import weakref

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer, default_argument_parser, default_setup, launch,
    create_ddp_model, AMPTrainer, SimpleTrainer, HookBase,
)
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils import comm
from detectron2.utils.events import EventWriter, get_event_storage
from detectron2.utils.logger import setup_logger
from detectron2.data import build_detection_train_loader

import dafusion.modeling  # noqa: F401  registers DAFusion arch + dual backbone + heads
import dafusion.data       # noqa: F401  registers datasets + mapper
from dafusion.config import add_dafusion_config
from dafusion.data.dataset_mappers.dafusion_mapper import DAFusionDatasetMapper


def _wandb_enabled():
    # Opt out with WANDB_MODE=disabled; "offline"/"online" both log (offline stays local).
    return os.environ.get("WANDB_MODE", "").lower() != "disabled"


class WandbWriter(EventWriter):
    """Mirror EventStorage scalars (losses, lr, timings) to Weights & Biases. Rank-0 only;
    appended by Trainer.build_writers alongside the default console/JSON/TensorBoard writers."""

    def __init__(self, window_size: int = 20):
        self._window_size = window_size
        self._last_write = -1
        import wandb
        self._wandb = wandb

    def write(self):
        storage = get_event_storage()
        logs, new_last = {}, self._last_write
        for k, (v, itr) in storage.latest_with_smoothing_hint(self._window_size).items():
            if itr > self._last_write:
                logs[k] = v
                new_last = max(new_last, itr)
        self._last_write = new_last
        if logs:
            self._wandb.log(logs, step=storage.iter)

    def close(self):
        if self._wandb.run is not None:
            self._wandb.finish()



class EMAHook(HookBase):
    """Exponential moving average of the model weights, written out as `model_ema.pth`.

    Motivated by a measurement, not by fashion: three runs of the IDENTICAL config with different seeds
    scored OCID 91.40 / 89.93 / 91.00 -- a 1.47 spread (sd 0.62), larger than nine of the ten variant
    effects this project has called "refuted". EMA is the standard fix for exactly that: it averages the
    late-training trajectory, which both raises the expected score and shrinks run-to-run spread.

    Distinct from the SWA soup already refuted (Track 16d): that averaged three saved checkpoints AFTER
    training, sampling the trajectory 3 times at 1500-iteration spacing. This averages EVERY step with
    decay 0.9998 (~5000-iteration effective window), which is a far denser and better-conditioned average.

    Kept deliberately simple: float32 shadow of the float parameters and buffers on the training device,
    updated after each step on rank 0 only, saved alongside the normal checkpoints. Off unless
    SOLVER.EMA_ENABLED, so every existing run is byte-identical.
    """

    def __init__(self, model, decay, period, out_dir):
        self.model = model.module if hasattr(model, "module") else model
        self.decay = decay
        self.period = period
        self.out_dir = out_dir
        self.shadow = {k: v.detach().clone().float()
                       for k, v in self.model.state_dict().items()
                       if v.dtype.is_floating_point}

    def after_step(self):
        it = self.trainer.iter
        # warm up the average so the first steps do not dominate a 0.9998 decay
        d = min(self.decay, (1.0 + it) / (10.0 + it))
        sd = self.model.state_dict()
        with torch.no_grad():
            for k, v in self.shadow.items():
                v.mul_(d).add_(sd[k].detach().float(), alpha=1.0 - d)
        if self.period and (it + 1) % self.period == 0:
            self.save()

    def save(self):
        if not comm.is_main_process():
            return
        sd = self.model.state_dict()
        out = {k: (self.shadow[k].to(sd[k].dtype) if k in self.shadow else v)
               for k, v in sd.items()}
        torch.save({"model": out}, os.path.join(self.out_dir, "model_ema.pth"))

    def after_train(self):
        self.save()


class Trainer(DefaultTrainer):
    def __init__(self, cfg):
        # Same as DefaultTrainer.__init__, but wrap DDP with find_unused_parameters=True:
        # at high-resolution stages the relative-position-bias MLP is skipped (too many
        # tokens to materialise the bias), so those params get no grad on some iterations.
        super(DefaultTrainer, self).__init__()
        if not logging.getLogger("detectron2").isEnabledFor(logging.INFO):
            setup_logger(name="detectron2")
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)
        model = create_ddp_model(model, broadcast_buffers=False, find_unused_parameters=True)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(model, cfg.OUTPUT_DIR, trainer=weakref.proxy(self))
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        hooks_ = self.build_hooks()
        if getattr(cfg.SOLVER, "EMA_ENABLED", False):
            hooks_.append(EMAHook(model, cfg.SOLVER.EMA_DECAY, cfg.SOLVER.CHECKPOINT_PERIOD,
                                  cfg.OUTPUT_DIR))
        self.register_hooks(hooks_)

    @classmethod
    def build_train_loader(cls, cfg):
        mapper = DAFusionDatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, mapper=mapper)

    def build_writers(self):
        # Default console/JSON/TensorBoard writers + W&B when a run is active (see main()).
        writers = super().build_writers()
        import wandb
        if comm.is_main_process() and wandb.run is not None:
            writers.append(WandbWriter())
        return writers

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED
        defaults = {"lr": cfg.SOLVER.BASE_LR, "weight_decay": cfg.SOLVER.WEIGHT_DECAY}
        norm_types = (
            torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm, torch.nn.GroupNorm, torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d, torch.nn.InstanceNorm3d, torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )
        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for pname, value in module.named_parameters(recurse=False):
                if not value.requires_grad or value in memo:
                    continue
                memo.add(value)
                hyp = copy.copy(defaults)
                if "backbone" in module_name:
                    hyp["lr"] = hyp["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if "relative_position_bias_table" in pname or "absolute_pos_embed" in pname:
                    hyp["weight_decay"] = 0.0
                if isinstance(module, norm_types):
                    hyp["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyp["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyp})

        def maybe_full_clip(optim):
            clip_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                      and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                      and clip_val > 0.0)

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        opt_type = cfg.SOLVER.OPTIMIZER
        if opt_type == "SGD":
            optimizer = maybe_full_clip(torch.optim.SGD)(params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM)
        elif opt_type == "ADAMW":
            optimizer = maybe_full_clip(torch.optim.AdamW)(params, cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"no optimizer type {opt_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    cfg = get_cfg()
    add_dafusion_config(cfg)
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=0, name="dafusion")
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
        return
    if comm.is_main_process() and _wandb_enabled():
        import wandb
        import yaml
        wandb.init(
            project=os.environ.get("WANDB_PROJECT", "da-fusion"),
            name=os.path.basename(os.path.normpath(cfg.OUTPUT_DIR)),
            config=yaml.safe_load(cfg.dump()),
            dir=cfg.OUTPUT_DIR,
            resume="allow",
        )
        # Stash run id so the post-training eval sweep can resume THIS run and log its
        # benchmark curves onto the same timeline (see dafusion/eval/sweep.py).
        with open(os.path.join(cfg.OUTPUT_DIR, "wandb_run.json"), "w") as f:
            json.dump({"id": wandb.run.id, "project": wandb.run.project,
                       "entity": wandb.run.entity}, f)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(main, args.num_gpus, num_machines=args.num_machines, machine_rank=args.machine_rank,
           dist_url=args.dist_url, args=(args,))
