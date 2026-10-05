"""Optimizer and LR-scheduler builders driven by config dicts."""
from __future__ import annotations

import math

import torch
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(params, cfg: dict):
    name = cfg.get("name", "sgd").lower()
    lr = cfg.get("lr", 0.1)
    wd = cfg.get("weight_decay", 5e-4)
    if name == "sgd":
        return SGD(params, lr=lr, momentum=cfg.get("momentum", 0.9),
                   weight_decay=wd, nesterov=cfg.get("nesterov", True))
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=wd, betas=cfg.get("betas", (0.9, 0.999)))
    raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, cfg: dict, epochs: int, steps_per_epoch: int = 1):
    """Per-epoch scheduler with optional linear warmup. Call ``.step()`` once per epoch."""
    name = cfg.get("name", "cosine").lower()
    warmup = cfg.get("warmup_epochs", 0)
    min_ratio = cfg.get("min_lr", 0.0) / max(optimizer.param_groups[0]["lr"], 1e-12)

    def lr_lambda(epoch: int) -> float:
        if warmup and epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        if name == "cosine":
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if name == "multistep":
            milestones = cfg.get("milestones", [int(epochs * 0.5), int(epochs * 0.75)])
            gamma = cfg.get("gamma", 0.1)
            factor = 1.0
            for m in milestones:
                if epoch >= m:
                    factor *= gamma
            return factor
        if name in ("none", "constant"):
            return 1.0
        raise ValueError(f"Unknown scheduler: {name}")

    return LambdaLR(optimizer, lr_lambda)
