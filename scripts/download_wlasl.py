#!/usr/bin/env python
"""Download WLASL (top-N glosses), trim clips, and write a manifest.

Usage
-----
    python scripts/download_wlasl.py --out-dir data/wlasl --num-classes 1000 --num-workers 8

If you already have raw videos from a mirror, point --video-dir at the folder of
``{video_id}.mp4`` files; existing files are skipped (status=exists) and only the
manifest is (re)built.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from slr.data.wlasl_metadata import (
    download_metadata,
    load_metadata,
    build_subset,
    iter_instances,
    save_label_map,
)
from slr.data.download import download_instance
from slr.data.verify import summarize, print_report


def parse_args():
    ap = argparse.ArgumentParser(description="Download the WLASL subset.")
    ap.add_argument("--out-dir", default="data/wlasl", help="dataset root")
    ap.add_argument("--num-classes", type=int, default=1000, help="top-N glosses")
    ap.add_argument("--json", default=None, help="path to WLASL_v0.3.json (auto-downloads if absent)")
    ap.add_argument("--video-dir", default=None, help="where trimmed clips go (default: <out-dir>/videos)")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--splits", nargs="*", default=None, help="limit to splits, e.g. train val test")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap total instances")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = Path(args.json) if args.json else out_dir / "WLASL_v0.3.json"
    download_metadata(json_path)
    metadata = load_metadata(json_path)

    entries, gloss_to_idx = build_subset(metadata, args.num_classes)
    save_label_map(gloss_to_idx, out_dir / "label_map.json")

    video_dir = Path(args.video_dir) if args.video_dir else out_dir / "videos"
    raw_dir = out_dir / "_raw"

    jobs = list(iter_instances(entries, splits=args.splits))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[download] {len(jobs)} instances across {len(entries)} classes -> {video_dir}")

    manifest_rows: list[dict] = []

    def work(pair):
        entry, inst = pair
        status = download_instance(
            inst, raw_dir, video_dir,
            timeout=args.timeout, retries=args.retries, overwrite=args.overwrite,
        )
        return {
            "video_id": inst.video_id,
            "gloss": entry.gloss,
            "class_idx": entry.class_idx,
            "split": inst.split,
            "status": status,
            "path": str((video_dir / f"{inst.video_id}.mp4")),
            "url": inst.url,
        }

    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = [ex.submit(work, pair) for pair in jobs]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="downloading"):
            manifest_rows.append(fut.result())

    manifest_path = out_dir / "download_manifest.csv"
    fieldnames = ["video_id", "gloss", "class_idx", "split", "status", "path", "url"]
    with open(manifest_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"[download] manifest -> {manifest_path}")

    # Clean up raw dir if empty.
    try:
        raw_dir.rmdir()
    except OSError:
        pass

    print_report(summarize(manifest_rows))


if __name__ == "__main__":
    main()
