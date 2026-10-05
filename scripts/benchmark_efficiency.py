#!/usr/bin/env python
"""Measure efficiency (params, FLOPs, FPS, VRAM) for each model -> runs/efficiency.json.

Builds every model with a dummy input of the correct shape. Skeleton models always
build (native/stub); external GCNs and video models require their deps/third_party.
Failures are recorded, not fatal.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import torch

from slr.config import build_config
from slr.models import build_model, MODALITY, ALL_MODELS
from slr.results.efficiency import measure_efficiency


def dummy_input(modality: str, cfg: dict) -> torch.Tensor:
    if modality == "skeleton":
        C = cfg.get("model", {}).get("in_channels", 3)
        T = cfg["data"].get("num_frames", 64)
        from slr.data.graph import NUM_NODES
        return torch.randn(1, C, T, NUM_NODES, 1)
    T = cfg["data"].get("clip_frames", 32)
    S = cfg["data"].get("crop_size", 224)
    return torch.randn(1, 3, T, S, S)


def parse_args():
    ap = argparse.ArgumentParser(description="Benchmark model efficiency.")
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    ap.add_argument("--config-dir", default="configs/models")
    ap.add_argument("--out", default="runs/efficiency.json")
    ap.add_argument("--device", default="cuda")
    return ap.parse_args()


def main():
    args = parse_args()
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    out = {}
    for name in args.models:
        cfg_path = Path(args.config_dir) / f"{name}.yaml"
        if not cfg_path.exists():
            out[name] = {"error": "no config"}
            continue
        cfg = build_config(str(cfg_path))
        modality = cfg["model"].get("modality", MODALITY.get(name, "skeleton"))
        try:
            model = build_model(name, cfg["data"]["num_classes"], cfg["model"])
            dummy = dummy_input(modality, cfg)
            out[name] = measure_efficiency(model, dummy, device=device)
            print(f"[eff] {name}: {out[name]}")
        except Exception as exc:  # noqa: BLE001
            out[name] = {"error": str(exc)}
            print(f"[eff] {name}: FAILED ({exc})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[eff] wrote {args.out}")


if __name__ == "__main__":
    main()
