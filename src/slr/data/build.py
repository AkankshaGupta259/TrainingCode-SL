"""Build train/val/test DataLoaders for a given modality from a config dict."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .skeleton_dataset import SkeletonDataset
from .video_dataset import VideoDataset


def build_loaders(cfg: dict, modality: str) -> dict:
    data = cfg["data"]
    root = data["root"]
    num_classes = data["num_classes"]
    bs = data.get("batch_size", 32)
    workers = data.get("num_workers", 8)

    def make(split: str):
        augment = split == "train"
        if modality == "skeleton":
            ds = SkeletonDataset(
                root=root, split=split,
                num_frames=data.get("num_frames", 64),
                in_channels=cfg.get("model", {}).get("in_channels", 3),
                augment=augment,
            )
        elif modality == "video":
            ds = VideoDataset(
                root=root, split=split,
                num_frames=data.get("clip_frames", 32),
                crop_size=data.get("crop_size", 224),
                augment=augment,
            )
        else:
            raise ValueError(modality)
        return DataLoader(
            ds, batch_size=bs, shuffle=augment, num_workers=workers,
            pin_memory=torch.cuda.is_available(), drop_last=augment,
            persistent_workers=workers > 0,
        )

    return {"train": make("train"), "val": make("val"), "test": make("test")}, num_classes
