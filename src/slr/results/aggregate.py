"""Aggregate per-run test metrics into a model-vs-model comparison (mean ± std).

Scans ``runs/<model>/seed<k>/test_metrics.json`` across seeds, merges optional
efficiency numbers, and writes both a CSV and a Markdown table sorted by Top-1.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPORT_METRICS = ["top1", "top5", "mean_class_acc", "precision_macro", "recall_macro", "f1_macro"]
EFF_METRICS = ["params_M", "gflops", "fps"]


def aggregate_runs(runs_dir: str) -> dict:
    runs_dir = Path(runs_dir)
    per_model: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))

    for metrics_file in runs_dir.glob("*/seed*/test_metrics.json"):
        model = metrics_file.parent.parent.name
        data = json.loads(metrics_file.read_text(encoding="utf-8"))
        for m in REPORT_METRICS:
            if m in data:
                per_model[model][m].append(float(data[m]))

    eff_path = runs_dir / "efficiency.json"
    eff = json.loads(eff_path.read_text(encoding="utf-8")) if eff_path.exists() else {}

    summary = {}
    for model, metrics in per_model.items():
        row = {"n_seeds": len(next(iter(metrics.values()), []))}
        for m in REPORT_METRICS:
            vals = metrics.get(m, [])
            if vals:
                row[f"{m}_mean"] = float(np.mean(vals))
                row[f"{m}_std"] = float(np.std(vals))
        for em in EFF_METRICS:
            if model in eff and eff[model].get(em) is not None:
                row[em] = eff[model][em]
        summary[model] = row
    return summary


def write_comparison(summary: dict, out_dir: str) -> None:
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not summary:
        print("[aggregate] no runs found.")
        return

    df = pd.DataFrame(summary).T
    sort_key = "top1_mean" if "top1_mean" in df else df.columns[0]
    df = df.sort_values(sort_key, ascending=False)
    df.to_csv(out_dir / "comparison.csv")

    # Markdown table
    def fmt(model):
        r = summary[model]
        def ms(m):
            if f"{m}_mean" in r:
                return f"{100*r[f'{m}_mean']:.2f} ± {100*r[f'{m}_std']:.2f}"
            return "-"
        params = f"{r['params_M']:.2f}" if "params_M" in r else "-"
        gflops = f"{r['gflops']:.1f}" if r.get("gflops") is not None else "-"
        fps = f"{r['fps']:.0f}" if "fps" in r else "-"
        return (f"| {model} | {ms('top1')} | {ms('top5')} | {ms('mean_class_acc')} "
                f"| {ms('f1_macro')} | {params} | {gflops} | {fps} |")

    lines = [
        "# Model comparison — WLASL1000 (isolated)",
        "",
        "Accuracy/F1 are mean ± std (%) across seeds.",
        "",
        "| Model | Top-1 | Top-5 | Mean-class acc | F1 (macro) | Params (M) | GFLOPs | FPS |",
        "|---|---|---|---|---|---|---|---|",
    ]
    lines += [fmt(m) for m in df.index]
    (out_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[aggregate] wrote {out_dir/'comparison.csv'} and {out_dir/'comparison.md'}")
    print("\n".join(lines))
