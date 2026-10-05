"""Model registry.

``build_model(name, num_classes, model_cfg)`` returns an ``nn.Module`` whose forward
takes the modality-appropriate tensor:
  * skeleton models -> (N, C, T, V, M)
  * video models    -> (N, C, T, H, W)

``MODALITY[name]`` tells the training script which dataset/loader to use.
"""
from __future__ import annotations

import torch.nn as nn

SKELETON_MODELS = ["stgcn", "ctrgcn", "hdgcn", "infogcn", "hagcn", "msg3d", "motionbert"]
VIDEO_MODELS = ["i3d", "slowfast"]
ALL_MODELS = SKELETON_MODELS + VIDEO_MODELS

MODALITY = {m: "skeleton" for m in SKELETON_MODELS}
MODALITY.update({m: "video" for m in VIDEO_MODELS})

# GCNs loaded from third_party/ via the uniform adapter.
_EXTERNAL_GCN = {"ctrgcn", "hdgcn", "infogcn", "hagcn", "msg3d"}


def build_model(name: str, num_classes: int, model_cfg: dict | None = None) -> nn.Module:
    name = name.lower()
    model_cfg = model_cfg or {}

    if name == "stgcn":
        from .stgcn import STGCN

        return STGCN(num_classes=num_classes, **model_cfg.get("args", {}))

    if name in _EXTERNAL_GCN:
        from .external_gcn import build_external_gcn

        return build_external_gcn(num_classes, {"name": name, **model_cfg})

    if name == "motionbert":
        from .motionbert_adapter import build_motionbert

        return build_motionbert(num_classes, model_cfg)

    if name in VIDEO_MODELS:
        from .video_models import build_video_model

        return build_video_model(name, num_classes, model_cfg)

    raise ValueError(f"Unknown model '{name}'. Known: {ALL_MODELS}")
