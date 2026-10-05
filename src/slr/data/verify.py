"""Summarize a WLASL download: per-split totals and per-class health.

Reads the manifest produced by ``scripts/download_wlasl.py`` and reports what you
actually have on disk, so the numbers can be cited in results.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


def read_manifest(manifest_path: str | Path) -> list[dict]:
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict], min_samples_warn: int = 5) -> dict:
    """Return a summary dict of download outcomes and per-class availability."""
    status_counts = Counter(r["status"] for r in rows)
    ok_statuses = {"ok", "exists"}

    # Per (class, split) counts of successfully available clips.
    per_class_split: dict[str, Counter] = defaultdict(Counter)
    per_split = Counter()
    for r in rows:
        if r["status"] in ok_statuses:
            per_class_split[r["gloss"]][r["split"]] += 1
            per_split[r["split"]] += 1

    expected_per_class = Counter(r["gloss"] for r in rows)
    available_per_class = {g: sum(c.values()) for g, c in per_class_split.items()}

    weak_classes = sorted(
        (
            (g, available_per_class.get(g, 0), expected_per_class[g])
            for g in expected_per_class
            if available_per_class.get(g, 0) < min_samples_warn
        ),
        key=lambda x: x[1],
    )

    return {
        "total_instances": len(rows),
        "status_counts": dict(status_counts),
        "available_total": sum(per_split.values()),
        "per_split": dict(per_split),
        "num_classes": len(expected_per_class),
        "classes_with_zero": sum(1 for g in expected_per_class if available_per_class.get(g, 0) == 0),
        "weak_classes": weak_classes,
        "min_samples_warn": min_samples_warn,
    }


def print_report(summary: dict) -> None:
    print("=" * 60)
    print("WLASL download report")
    print("=" * 60)
    print(f"Total instances in subset : {summary['total_instances']}")
    print(f"Available on disk         : {summary['available_total']}")
    print(f"Classes                   : {summary['num_classes']}")
    print(f"Classes with ZERO clips   : {summary['classes_with_zero']}")
    print("\nBy status:")
    for status, n in sorted(summary["status_counts"].items(), key=lambda x: -x[1]):
        print(f"  {status:20s} {n}")
    print("\nBy split (available):")
    for split, n in sorted(summary["per_split"].items()):
        print(f"  {split:10s} {n}")

    weak = summary["weak_classes"]
    if weak:
        print(f"\nClasses with < {summary['min_samples_warn']} clips ({len(weak)}):")
        for gloss, have, expected in weak[:40]:
            print(f"  {gloss:25s} {have}/{expected}")
        if len(weak) > 40:
            print(f"  ... and {len(weak) - 40} more")
    print("=" * 60)
