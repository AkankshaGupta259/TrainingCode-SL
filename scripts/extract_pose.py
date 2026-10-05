#!/usr/bin/env python
"""Extract 27-node skeletons from downloaded clips and cache them as .npy.

    python scripts/extract_pose.py --out-dir data/wlasl --backend mediapipe --workers 4

Reads the download manifest, processes every available clip, writes
``data/wlasl/poses/{video_id}.npy`` of shape (T, 27, 3), and records an
extraction manifest (frames per clip, failures).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import csv
from pathlib import Path

from tqdm import tqdm

from slr.data.pose_extraction import get_backend, extract_and_cache


def parse_args():
    ap = argparse.ArgumentParser(description="Extract skeletons from clips.")
    ap.add_argument("--out-dir", default="data/wlasl")
    ap.add_argument("--backend", default="mediapipe")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="debug: cap clips")
    return ap.parse_args()


def main():
    args = parse_args()
    root = Path(args.out_dir)
    manifest = Path(args.manifest) if args.manifest else root / "download_manifest.csv"
    pose_dir = root / "poses"
    pose_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with open(manifest, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r["status"] in ("ok", "exists") and Path(r["path"]).exists():
                rows.append(r)
    if args.limit:
        rows = rows[: args.limit]
    print(f"[pose] {len(rows)} clips -> {pose_dir} (backend={args.backend})")

    backend = get_backend(args.backend)
    results = []
    ok = fail = 0
    try:
        for r in tqdm(rows, desc="extracting"):
            vid = r["video_id"]
            out_path = pose_dir / f"{vid}.npy"
            if out_path.exists() and not args.overwrite:
                results.append({"video_id": vid, "frames": -1, "status": "exists"})
                continue
            n = extract_and_cache(r["path"], str(out_path), backend)
            status = "ok" if n > 0 else "failed"
            ok += n > 0
            fail += n == 0
            results.append({"video_id": vid, "frames": n, "status": status})
    finally:
        if hasattr(backend, "close"):
            backend.close()

    man_out = root / "pose_manifest.csv"
    with open(man_out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["video_id", "frames", "status"])
        w.writeheader()
        w.writerows(results)
    print(f"[pose] done: {ok} ok, {fail} failed -> {man_out}")


if __name__ == "__main__":
    main()
