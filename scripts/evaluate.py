#!/usr/bin/env python
"""Evaluate a trained checkpoint on the test split -> test_metrics.json."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import torch

from slr.data.build import build_loaders
from slr.engine import Trainer
from slr.models import build_model, MODALITY
from slr.engine.optim import build_optimizer, build_scheduler


def parse_args():
    ap = argparse.ArgumentParser(description="Evaluate a checkpoint on test.")
    ap.add_argument("--run-dir", required=True, help="runs/<model>/seed<k> (holds config.json + best.pt)")
    ap.add_argument("--ckpt", default="best.pt")
    return ap.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    model_name = cfg["model"]["name"]
    modality = cfg["model"].get("modality", MODALITY.get(model_name, "skeleton"))
    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    loaders, num_classes = build_loaders(cfg, modality)
    model = build_model(model_name, num_classes, cfg["model"])
    opt = build_optimizer(model.parameters(), cfg["train"]["optimizer"])
    sch = build_scheduler(opt, cfg["train"]["scheduler"], cfg["train"]["epochs"])
    trainer = Trainer(model, loaders, opt, sch, num_classes, str(run_dir),
                      device=device, epochs=cfg["train"]["epochs"], amp=cfg.get("amp", True))
    trainer.load_checkpoint(args.ckpt)
    metrics = trainer.evaluate("test")
    print(json.dumps(metrics, indent=2))
    (run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
