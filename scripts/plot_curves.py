#!/usr/bin/env python
"""Plot train/val loss and validation metric curves from a run's history.csv."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse

from slr.utils.curves import plot_history


def parse_args():
    ap = argparse.ArgumentParser(description="Plot training curves.")
    ap.add_argument("--history", required=True, help="path to history.csv")
    ap.add_argument("--out-dir", default=None, help="where to save PNGs (default: next to the CSV)")
    return ap.parse_args()


def main():
    args = parse_args()
    plot_history(args.history, args.out_dir)


if __name__ == "__main__":
    main()
