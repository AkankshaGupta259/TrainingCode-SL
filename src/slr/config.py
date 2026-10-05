"""Tiny config system: YAML load + deep-merge + dotted overrides.

A run config is built by merging, in order:
    configs/default.yaml  <-  dataset yaml  <-  model yaml  <-  CLI overrides
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _coerce(val: str) -> Any:
    for cast in (int, float):
        try:
            return cast(val)
        except ValueError:
            pass
    if val.lower() in ("true", "false"):
        return val.lower() == "true"
    if val.lower() in ("null", "none"):
        return None
    return val


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply ``a.b.c=value`` style overrides in place-ish (returns new dict)."""
    cfg = copy.deepcopy(cfg)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, val = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _coerce(val)
    return cfg


def build_config(
    model_cfg_path: str,
    default_path: str = "configs/default.yaml",
    dataset_cfg_path: str | None = None,
    overrides: list[str] | None = None,
) -> dict:
    cfg = load_yaml(default_path)
    model_cfg = load_yaml(model_cfg_path)
    ds_path = dataset_cfg_path or cfg.get("data", {}).get("dataset_config")
    if ds_path and Path(ds_path).exists():
        cfg = deep_merge(cfg, load_yaml(ds_path))
    cfg = deep_merge(cfg, model_cfg)
    cfg = apply_overrides(cfg, overrides or [])
    return cfg
