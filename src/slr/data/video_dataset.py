"""Dataset of RGB clips for the I3D / SlowFast models.

Decodes trimmed ``videos/{video_id}.mp4`` with OpenCV (no decord dependency),
uniformly samples ``num_frames``, crops/normalizes, and returns ``(C, T, H, W)``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import transforms as TF


def _decode_video(path: str, want_indices: np.ndarray) -> np.ndarray:
    """Return frames (len(want_indices), H, W, 3) RGB uint8 using OpenCV."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frames = []
    want = set(int(i) for i in want_indices)
    grabbed = {}
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx in want:
            grabbed[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        idx += 1
        if len(grabbed) == len(want):
            break
    cap.release()
    if not grabbed:  # unreadable -> black clip (kept so batch shape is stable)
        return np.zeros((len(want_indices), 224, 224, 3), dtype=np.uint8)
    last = max(grabbed)
    for i in want_indices:
        frames.append(grabbed.get(int(i), grabbed[last]))
    return np.stack(frames)


class VideoDataset(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        num_frames: int = 32,
        crop_size: int = 224,
        augment: bool = True,
        manifest: str | None = None,
        label_map: str | None = None,
        video_dirname: str = "videos",
    ):
        self.root = Path(root)
        self.split = split
        self.num_frames = num_frames
        self.crop_size = crop_size
        self.augment = augment and split == "train"
        self.video_dir = self.root / video_dirname

        manifest = Path(manifest) if manifest else self.root / "download_manifest.csv"
        label_map = Path(label_map) if label_map else self.root / "label_map.json"
        with open(label_map, "r", encoding="utf-8") as f:
            self.gloss_to_idx = json.load(f)

        self.samples = []
        with open(manifest, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["split"] != split or row["status"] not in ("ok", "exists"):
                    continue
                p = self.video_dir / f"{row['video_id']}.mp4"
                if p.exists():
                    self.samples.append((p, int(row["class_idx"])))
        if not self.samples:
            raise RuntimeError(f"No {split} videos found under {self.video_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        rng = np.random.default_rng(None if self.augment else i)
        import cv2

        cap = cv2.VideoCapture(str(path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        cap.release()
        idx = TF.temporal_sample(max(total, 1), self.num_frames, self.augment, rng)
        frames = _decode_video(path, idx)
        clip = TF.process_clip(frames, self.crop_size, self.augment, rng)
        return torch.from_numpy(clip).float(), label
