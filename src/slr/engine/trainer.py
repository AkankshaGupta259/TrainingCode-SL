"""Model-agnostic training / validation / test engine.

Works identically for skeleton and video models (the model's forward consumes the
right tensor shape). Records train+val loss curves and the full metric bundle each
epoch via HistoryLogger, checkpoints the best model, and runs a final test pass.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..metrics import compute_metrics
from ..utils import HistoryLogger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        loaders: dict[str, DataLoader],
        optimizer,
        scheduler,
        num_classes: int,
        out_dir: str,
        device: str = "cuda",
        epochs: int = 80,
        amp: bool = True,
        label_smoothing: float = 0.1,
        grad_accum: int = 1,
        clip_grad: float | None = None,
        select_metric: str = "top1",
        log_every: int = 50,
    ):
        self.model = model.to(device)
        self.loaders = loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_classes = num_classes
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.epochs = epochs
        self.amp = amp and device.startswith("cuda")
        self.grad_accum = max(1, grad_accum)
        self.clip_grad = clip_grad
        self.select_metric = select_metric
        self.log_every = log_every

        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.logger = HistoryLogger(out_dir)
        self.best_score = -1.0

    def _train_epoch(self, epoch: int) -> dict:
        self.model.train()
        loader = self.loaders["train"]
        running, correct, total = 0.0, 0, 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(loader):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y) / self.grad_accum
            self.scaler.scale(loss).backward()

            if (step + 1) % self.grad_accum == 0:
                if self.clip_grad:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            running += loss.item() * self.grad_accum * y.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
            if self.log_every and step % self.log_every == 0:
                print(f"  epoch {epoch} step {step}/{len(loader)} loss={loss.item()*self.grad_accum:.3f}")

        return {"train_loss": running / max(total, 1), "train_top1": correct / max(total, 1)}

    @torch.no_grad()
    def evaluate(self, split: str) -> dict:
        self.model.eval()
        loader = self.loaders[split]
        all_logits, all_targets, loss_sum, n = [], [], 0.0, 0
        for x, y in loader:
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=self.amp):
                logits = self.model(x)
                loss = self.criterion(logits, y)
            loss_sum += loss.item() * y.size(0)
            n += y.size(0)
            all_logits.append(logits.float().cpu().numpy())
            all_targets.append(y.cpu().numpy())
        logits = np.concatenate(all_logits)
        targets = np.concatenate(all_targets)
        metrics = compute_metrics(logits, targets, num_classes=self.num_classes)
        metrics[f"{split}_loss"] = loss_sum / max(n, 1)
        return metrics

    def fit(self) -> dict:
        for epoch in range(self.epochs):
            t0 = time.time()
            tr = self._train_epoch(epoch)
            val = self.evaluate("val")
            self.scheduler.step()

            row = {**tr, "val_loss": val.pop("val_loss"),
                   "lr": self.optimizer.param_groups[0]["lr"],
                   "epoch_time_s": time.time() - t0}
            row.update(val)  # top1, top5, mean_class_acc, precision/recall/f1
            self.logger.log(epoch=epoch, **row)

            score = val.get(self.select_metric, val["top1"])
            if score > self.best_score:
                self.best_score = score
                self.save_checkpoint("best.pt", epoch, score)
        self.save_checkpoint("last.pt", self.epochs - 1, self.best_score)
        self.logger.close()

        # Final test with the best checkpoint.
        self.load_checkpoint("best.pt")
        test_metrics = self.evaluate("test")
        test_metrics["val_best_" + self.select_metric] = self.best_score
        import json
        with open(self.out_dir / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(test_metrics, f, indent=2)
        print(f"[test] {test_metrics}")
        return test_metrics

    def save_checkpoint(self, name: str, epoch: int, score: float):
        torch.save({"model": self.model.state_dict(), "epoch": epoch, "score": score},
                   self.out_dir / name)

    def load_checkpoint(self, name: str):
        ckpt = torch.load(self.out_dir / name, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
