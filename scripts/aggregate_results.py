#!/usr/bin/env python
"""Aggregate all runs into runs/comparison.{csv,md} (mean ± std across seeds)."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

from slr.results import aggregate_runs, write_comparison


def parse_args():
    ap = argparse.ArgumentParser(description="Aggregate results into a comparison table.")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="runs")
    return ap.parse_args()


def main():
    args = parse_args()
    summary = aggregate_runs(args.runs_dir)
    write_comparison(summary, args.out_dir)


if __name__ == "__main__":
    main()
