"""Per-epoch history logging for training runs.

A training loop calls ``logger.log(epoch=..., train_loss=..., val_loss=..., ...)``
once per epoch. Rows are appended to ``history.csv`` (and mirrored to
``history.json``) so train/val loss curves can be plotted later or live.

Example
-------
>>> logger = HistoryLogger("runs/ctrgcn_seed0")
>>> for epoch in range(epochs):
...     logger.log(epoch=epoch, lr=lr,
...                train_loss=tr_loss, train_top1=tr_acc,
...                val_loss=va_loss, **val_metrics)   # val_metrics from compute_metrics
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional

try:  # optional TensorBoard mirroring
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # noqa: BLE001
    SummaryWriter = None  # type: ignore


class HistoryLogger:
    def __init__(
        self,
        log_dir: str | Path,
        csv_name: str = "history.csv",
        use_tensorboard: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.log_dir / csv_name
        self.json_path = self.log_dir / "history.json"
        self.rows: list[dict] = []
        self._fieldnames: Optional[list[str]] = None

        self.tb = None
        if use_tensorboard and SummaryWriter is not None:
            self.tb = SummaryWriter(str(self.log_dir / "tb"))

    def log(self, epoch: int, **metrics: Any) -> None:
        """Record one epoch. ``epoch`` is always the first column."""
        row = {"epoch": int(epoch)}
        row.update({k: _to_float(v) for k, v in metrics.items()})
        self.rows.append(row)

        # (Re)write CSV with a stable, growing set of columns.
        if self._fieldnames is None:
            self._fieldnames = list(row.keys())
        else:
            for k in row:
                if k not in self._fieldnames:
                    self._fieldnames.append(k)
        self._flush_csv()

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.rows, f, indent=2)

        if self.tb is not None:
            for k, v in row.items():
                if k != "epoch" and isinstance(v, (int, float)):
                    self.tb.add_scalar(k, v, epoch)

        pretty = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in row.items()
        )
        print(f"[epoch {epoch:03d}] {pretty}")

    def _flush_csv(self) -> None:
        with open(self.csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writeheader()
            for r in self.rows:
                writer.writerow(r)

    def close(self) -> None:
        if self.tb is not None:
            self.tb.close()


def _to_float(v: Any) -> Any:
    """Best-effort convert tensors/np scalars to plain python floats."""
    if hasattr(v, "item"):
        try:
            return float(v.item())
        except Exception:  # noqa: BLE001
            return v
    return v
