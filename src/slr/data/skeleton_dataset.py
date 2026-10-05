"""Dataset of cached skeleton sequences for the GCN / MotionBERT models.

Reads the download manifest + label map, loads per-clip pose caches
(``poses/{video_id}.npy`` of shape ``(T, V, C)``), samples a fixed number of frames,
normalizes, augments (train only), and returns a tensor shaped ``(C, T, V, M=1)``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import transforms as TF


class SkeletonDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        num_frames: int = 64,
        in_channels: int = 3,
        augment: bool = True,
        manifest: str | None = None,
        label_map: str | None = None,
        pose_dirname: str = "poses",
    ):
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.in_channels = in_channels
        self.augment = augment and split == "train"
        self.pose_dir = self.root / pose_dirname

        manifest = Path(manifest) if manifest else self.root / "download_manifest.csv"
        label_map = Path(label_map) if label_map else self.root / "label_map.json"
        with open(label_map, "r", encoding="utf-8") as f:
            self.gloss_to_idx = json.load(f)

        self.samples = []  # (pose_path, class_idx)
        with open(manifest, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] != split or row["status"] not in ("ok", "exists"):
                    continue
                vid = row["video_id"]
                pose_path = self.pose_dir / f"{vid}.npy"
                if not pose_path.exists():
                    continue
                self.samples.append((pose_path, int(row["class_idx"])))
        if not self.samples:
            raise RuntimeError(
                f"No {split} samples found under {self.root}. "
                f"Did you run download + extract_pose?"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        pose_path, label = self.samples[i]
        rng = np.random.default_rng(None if self.augment else i)
        x = np.load(pose_path).astype(np.float32)        # (T, V, C)
        if x.ndim != 3:
            raise ValueError(f"Bad pose shape {x.shape} in {pose_path}")

        idx = TF.temporal_sample(x.shape[0], self.num_frames, self.augment, rng)
        x = x[idx]                                        # (num_frames, V, C)
        x = TF.normalize_skeleton(x)
        if self.augment:
            x = TF.augment_skeleton(x, rng)

        x = x[..., : self.in_channels]
        t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(-1).contiguous()  # (C, T, V, 1)
        return t, label
