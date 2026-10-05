"""Adapter for MotionBERT (Zhu et al., ICCV 2023) used as a pretrained encoder.

MotionBERT's value is its pretrained DSTformer motion encoder, so we load the official
code from ``third_party/MotionBERT`` and (optionally) its pretrained checkpoint, then
attach an action-recognition head.

Important caveats for this benchmark (documented so results are read correctly):
  * MotionBERT is trained on BODY joints (Human3.6M, 17 kpts) and does NOT model hands.
    Our SLR skeleton is hand-centric, so either feed the 17 body-compatible joints or
    expect it to trail the hand-aware GCNs. This is an architectural finding, not a bug.
  * Input is 2D keypoints shaped (N, T, J, C). Our skeleton tensors are (N, C, T, V, M);
    the adapter permutes and drops the person axis.

Config (configs/models/motionbert.yaml) specifies the repo path, the model-builder
dotted path, its args, and an optional pretrained checkpoint to load.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _import_from(dotted: str, search_path: str | None):
    if search_path:
        p = str(Path(search_path).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
    module_path, _, attr = dotted.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import '{module_path}'. Clone MotionBERT into '{search_path}' "
            f"(see third_party/README.md). Original error: {exc}"
        ) from exc
    return getattr(module, attr)


class MotionBERTWrapper(nn.Module):
    """Wraps the official action-recognition model and adapts our input layout."""

    def __init__(self, core: nn.Module, num_joints_expected: int | None = None):
        super().__init__()
        self.core = core
        self.num_joints_expected = num_joints_expected

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, T, V, M) -> take first person -> (N, T, V, C)
        if x.dim() == 5:
            x = x[..., 0]                 # (N, C, T, V)
            x = x.permute(0, 2, 3, 1)     # (N, T, V, C)
        return self.core(x)


def build_motionbert(num_classes: int, model_cfg: dict) -> nn.Module:
    builder = _import_from(model_cfg["import"], model_cfg.get("third_party"))
    args = dict(model_cfg.get("args", {}))
    args.setdefault("num_classes", num_classes)
    core = builder(**args)

    ckpt = model_cfg.get("pretrained")
    if ckpt:
        state = torch.load(ckpt, map_location="cpu")
        state = state.get("model_pos", state.get("state_dict", state))
        missing, unexpected = core.load_state_dict(state, strict=False)
        print(f"[motionbert] loaded {ckpt} (missing={len(missing)}, unexpected={len(unexpected)})")

    return MotionBERTWrapper(core, num_joints_expected=model_cfg.get("num_joints"))
