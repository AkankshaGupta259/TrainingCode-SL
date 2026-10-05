#!/usr/bin/env python
"""Report what the WLASL download actually produced (per-split + per-class health)."""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
from pathlib import Path

from slr.data.verify import read_manifest, summarize, print_report


def parse_args():
    ap = argparse.ArgumentParser(description="Verify a WLASL download.")
    ap.add_argument("--out-dir", default="data/wlasl", help="dataset root (holds download_manifest.csv)")
    ap.add_argument("--manifest", default=None, help="explicit manifest path (overrides --out-dir)")
    ap.add_argument("--min-samples-warn", type=int, default=5)
    return ap.parse_args()


def main():
    args = parse_args()
    manifest = Path(args.manifest) if args.manifest else Path(args.out_dir) / "download_manifest.csv"
    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}. Run download_wlasl.py first.")
    rows = read_manifest(manifest)
    print_report(summarize(rows, min_samples_warn=args.min_samples_warn))


if __name__ == "__main__":
    main()
