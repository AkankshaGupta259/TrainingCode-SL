"""RGB video models: I3D and SlowFast, loaded from PyTorchVideo with Kinetics-400 weights.

Both are wrapped so the training engine can treat them like any other model:
forward takes a clip tensor ``(N, C, T, H, W)`` and returns class logits. SlowFast's
two-pathway packing is done inside the wrapper, so datasets stay model-agnostic.

Requires ``pip install pytorchvideo`` on the training box (imported lazily).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _load_hub(model_name: str, pretrained: bool):
    try:
        return torch.hub.load(
            "facebookresearch/pytorchvideo", model=model_name, pretrained=pretrained
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load '{model_name}' from pytorchvideo hub. "
            f"Install it (`pip install pytorchvideo`) and ensure internet for weights. "
            f"Original error: {exc}"
        ) from exc


def _replace_head(module: nn.Module, num_classes: int) -> None:
    """Swap the final Linear (Kinetics-400 -> num_classes) in a pytorchvideo net head."""
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear) and child.out_features == 400:
            parent = module
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, nn.Linear(child.in_features, num_classes))
            return
    raise RuntimeError("Could not find a 400-way classifier head to replace.")


class I3D(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.net = _load_hub("i3d_r50", pretrained)
        _replace_head(self.net, num_classes)

    def forward(self, x):  # (N, C, T, H, W)
        return self.net(x)


class SlowFast(nn.Module):
    def __init__(self, num_classes: int, pretrained: bool = True, alpha: int = 4):
        super().__init__()
        self.net = _load_hub("slowfast_r50", pretrained)
        _replace_head(self.net, num_classes)
        self.alpha = alpha

    def _pack_pathways(self, x):
        # Fast pathway = all frames; Slow pathway = temporally subsampled by alpha.
        fast = x
        idx = torch.linspace(0, x.shape[2] - 1, x.shape[2] // self.alpha).long().to(x.device)
        slow = torch.index_select(x, 2, idx)
        return [slow, fast]

    def forward(self, x):  # (N, C, T, H, W)
        return self.net(self._pack_pathways(x))


def build_video_model(name: str, num_classes: int, model_cfg: dict) -> nn.Module:
    pretrained = model_cfg.get("pretrained", True)
    if name == "i3d":
        return I3D(num_classes, pretrained=pretrained)
    if name == "slowfast":
        return SlowFast(num_classes, pretrained=pretrained, alpha=model_cfg.get("alpha", 4))
    raise ValueError(f"Unknown video model: {name}")
