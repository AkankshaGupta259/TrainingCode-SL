"""Adapters for GCNs whose official implementations are dropped into ``third_party/``.

Reimplementing CTR-GCN / HD-GCN / InfoGCN / HA-GCN / MS-G3D from memory risks subtle
deviations from the papers that would invalidate a benchmark. Instead we load each
model's *official* code and instantiate it through a small, uniform adapter so the
training engine treats every model identically.

How it works
------------
Each model config (``configs/models/<name>.yaml``) specifies:

    model:
      name: ctrgcn
      import: "model.ctrgcn.Model"     # dotted path inside the official repo
      third_party: "third_party/CTR-GCN"
      args: { num_point: 27, num_person: 1, graph: "...", graph_args: {...} }

The adapter adds ``third_party`` to ``sys.path``, imports the class, and builds it.
Most of these repos already accept ``(N, C, T, V, M)`` input and return class logits,
matching our native ST-GCN, so no further glue is needed. See ``third_party/README.md``.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import torch.nn as nn


def _import_from(dotted: str, search_path: str | None):
    if search_path:
        p = str(Path(search_path).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
    module_path, _, cls_name = dotted.rpartition(".")
    if not module_path:
        raise ValueError(f"'import' must be a dotted path to a class, got {dotted!r}")
    try:
        module = importlib.import_module(module_path)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import '{module_path}'. Clone the official repo into "
            f"'{search_path}' (see third_party/README.md). Original error: {exc}"
        ) from exc
    return getattr(module, cls_name)


def build_external_gcn(num_classes: int, model_cfg: dict) -> nn.Module:
    dotted = model_cfg.get("import")
    if not dotted:
        raise ValueError(
            f"Model '{model_cfg.get('name')}' needs an 'import' dotted path in its config."
        )
    cls = _import_from(dotted, model_cfg.get("third_party"))
    args: dict[str, Any] = dict(model_cfg.get("args", {}))

    # Most GCN repos use 'num_class' (occasionally 'num_classes'); supply the right one.
    try:
        import inspect

        sig = inspect.signature(cls.__init__)
        if "num_class" in sig.parameters:
            args.setdefault("num_class", num_classes)
        elif "num_classes" in sig.parameters:
            args.setdefault("num_classes", num_classes)
        else:
            args.setdefault("num_class", num_classes)
    except (TypeError, ValueError):
        args.setdefault("num_class", num_classes)

    return cls(**args)
