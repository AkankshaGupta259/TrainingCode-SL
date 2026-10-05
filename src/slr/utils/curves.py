"""Plot training/validation curves from a HistoryLogger CSV."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless-safe (servers without a display)
import matplotlib.pyplot as plt  # noqa: E402


def _read(csv_path: str | Path):
    import pandas as pd

    return pd.read_csv(csv_path)


def plot_history(csv_path: str | Path, out_dir: Optional[str | Path] = None) -> list[Path]:
    """Create loss and accuracy/metric curves. Returns the saved figure paths."""
    csv_path = Path(csv_path)
    out_dir = Path(out_dir) if out_dir else csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _read(csv_path)
    epochs = df["epoch"]
    saved: list[Path] = []

    # --- Loss curve (train vs val) ---
    if any(c in df for c in ("train_loss", "val_loss")):
        fig, ax = plt.subplots(figsize=(7, 5))
        if "train_loss" in df:
            ax.plot(epochs, df["train_loss"], label="train loss")
        if "val_loss" in df:
            ax.plot(epochs, df["val_loss"], label="val loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title("Training / validation loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        p = out_dir / "loss_curve.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    # --- Metric curves (anything that looks like accuracy / f1) ---
    metric_cols = [
        c for c in df.columns
        if any(key in c for key in ("top1", "top5", "mean_class_acc", "f1", "acc"))
        and c not in ("train_loss", "val_loss")
    ]
    if metric_cols:
        fig, ax = plt.subplots(figsize=(7, 5))
        for c in metric_cols:
            ax.plot(epochs, df[c], label=c)
        ax.set_xlabel("epoch")
        ax.set_ylabel("metric")
        ax.set_title("Validation metrics")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        p = out_dir / "metrics_curve.png"
        fig.tight_layout()
        fig.savefig(p, dpi=150)
        plt.close(fig)
        saved.append(p)

    for p in saved:
        print(f"[curves] saved -> {p}")
    return saved
