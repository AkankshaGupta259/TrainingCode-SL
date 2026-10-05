#!/usr/bin/env python
"""Orchestrate the full benchmark: every model x every seed, then aggregate.

    python scripts/run_all.py --seeds 0 1 2
    python scripts/run_all.py --models stgcn ctrgcn --seeds 0 1 2 --dry-run

Runs each training as a subprocess (isolates GPU memory between models). Skips a
run whose test_metrics.json already exists, so it is resumable.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import subprocess
import sys
from pathlib import Path

from slr.models import ALL_MODELS


def parse_args():
    ap = argparse.ArgumentParser(description="Run the whole benchmark.")
    ap.add_argument("--models", nargs="*", default=ALL_MODELS)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--config-dir", default="configs/models")
    ap.add_argument("--out-root", default="runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-efficiency", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    here = Path(__file__).parent
    py = sys.executable

    for model in args.models:
        cfg = Path(args.config_dir) / f"{model}.yaml"
        if not cfg.exists():
            print(f"[run_all] skip {model}: no config at {cfg}")
            continue
        for seed in args.seeds:
            done = Path(args.out_root) / model / f"seed{seed}" / "test_metrics.json"
            if done.exists():
                print(f"[run_all] already done: {model} seed{seed}")
                continue
            cmd = [py, str(here / "train.py"), "--model", str(cfg),
                   "--seed", str(seed), "--out-root", args.out_root]
            print("[run_all] $", " ".join(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=False)

    if not args.skip_efficiency and not args.dry_run:
        subprocess.run([py, str(here / "benchmark_efficiency.py"),
                        "--models", *args.models,
                        "--out", str(Path(args.out_root) / "efficiency.json")], check=False)

    if not args.dry_run:
        subprocess.run([py, str(here / "aggregate_results.py"),
                        "--runs-dir", args.out_root, "--out-dir", args.out_root], check=False)


if __name__ == "__main__":
    main()
