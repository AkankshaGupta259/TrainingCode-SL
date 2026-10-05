"""Measure model efficiency: #params, FLOPs, inference latency/FPS, peak VRAM.

FLOPs use fvcore or thop if installed; otherwise reported as null (not fatal).
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_flops(model: nn.Module, dummy: torch.Tensor):
    try:
        from fvcore.nn import FlopCountAnalysis

        return float(FlopCountAnalysis(model, dummy).total())
    except Exception:  # noqa: BLE001
        pass
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(macs) * 2  # MACs -> FLOPs
    except Exception:  # noqa: BLE001
        return None


@torch.no_grad()
def measure_latency(model: nn.Module, dummy: torch.Tensor, device: str,
                    warmup: int = 5, iters: int = 30) -> dict:
    model.eval().to(device)
    dummy = dummy.to(device)
    for _ in range(warmup):
        model(dummy)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model(dummy)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / iters
    bs = dummy.shape[0]
    return {"latency_ms": elapsed * 1000, "fps": bs / elapsed}


def measure_efficiency(model: nn.Module, dummy: torch.Tensor, device: str = "cpu") -> dict:
    out = {"params_M": count_params(model) / 1e6}
    flops = measure_flops(model, dummy.to(device))
    out["gflops"] = flops / 1e9 if flops else None
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    lat = measure_latency(model, dummy, device)
    out.update(lat)
    if device.startswith("cuda"):
        out["peak_vram_MB"] = torch.cuda.max_memory_allocated() / 1e6
    return out
