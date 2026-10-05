#!/usr/bin/env python
"""Train one model for one seed.

Example
-------
    python scripts/train.py --model configs/models/stgcn.yaml --seed 0
    python scripts/train.py --model configs/models/i3d.yaml --seed 1 \
        --set train.epochs=60 data.batch_size=8
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import torch

from slr.config import build_config
from slr.data.build import build_loaders
from slr.engine import set_seed, build_optimizer, build_scheduler, Trainer
from slr.models import build_model, MODALITY


def parse_args():
    ap = argparse.ArgumentParser(description="Train one model/seed.")
    ap.add_argument("--model", required=True, help="path to configs/models/<name>.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--default", default="configs/default.yaml")
    ap.add_argument("--dataset", default=None, help="override dataset config path")
    ap.add_argument("--out-root", default=None, help="override output_root")
    ap.add_argument("--set", nargs="*", dest="overrides", default=[], help="a.b=value overrides")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = build_config(args.model, args.default, args.dataset, args.overrides)
    cfg["seed"] = args.seed

    model_name = cfg["model"]["name"]
    modality = cfg["model"].get("modality", MODALITY.get(model_name, "skeleton"))
    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] CUDA unavailable -> using CPU")
        device = "cpu"

    out_root = args.out_root or cfg.get("output_root", "runs")
    out_dir = Path(out_root) / model_name / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    set_seed(args.seed, deterministic=cfg.get("deterministic", False))

    loaders, num_classes = build_loaders(cfg, modality)
    model = build_model(model_name, num_classes, cfg["model"])

    tcfg = cfg["train"]
    optimizer = build_optimizer(model.parameters(), tcfg["optimizer"])
    scheduler = build_scheduler(optimizer, tcfg["scheduler"], tcfg["epochs"])

    trainer = Trainer(
        model=model, loaders=loaders, optimizer=optimizer, scheduler=scheduler,
        num_classes=num_classes, out_dir=str(out_dir), device=device,
        epochs=tcfg["epochs"], amp=cfg.get("amp", True),
        label_smoothing=tcfg.get("label_smoothing", 0.1),
        grad_accum=tcfg.get("grad_accum", 1), clip_grad=tcfg.get("clip_grad"),
        select_metric=tcfg.get("eval_metric", "top1"),
    )
    print(f"[train] model={model_name} seed={args.seed} modality={modality} "
          f"classes={num_classes} device={device}")
    trainer.fit()


if __name__ == "__main__":
    main()
