"""Augmentation / preprocessing for skeleton and RGB clips.

Skeleton arrays are handled as ``(T, V, C)`` numpy (C = x, y, score). RGB clips are
``(T, H, W, 3)`` uint8 numpy. Note: horizontal flip is OFF by default for both, since
it swaps handedness and changes the sign.
"""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Temporal sampling (shared idea, used by both modalities)
# --------------------------------------------------------------------------- #

def temporal_sample(n_frames: int, num_out: int, train: bool, rng: np.random.Generator) -> np.ndarray:
    """Return ``num_out`` frame indices. Train: random within equal segments; eval: centers."""
    if n_frames <= 0:
        return np.zeros(num_out, dtype=np.int64)
    if n_frames >= num_out:
        bounds = np.linspace(0, n_frames, num_out + 1).astype(int)
        if train:
            idx = [rng.integers(bounds[i], max(bounds[i] + 1, bounds[i + 1])) for i in range(num_out)]
        else:
            idx = [(bounds[i] + bounds[i + 1]) // 2 for i in range(num_out)]
        return np.clip(np.array(idx), 0, n_frames - 1)
    # fewer frames than requested -> pad by repeating the last
    idx = np.arange(n_frames)
    pad = np.full(num_out - n_frames, n_frames - 1)
    return np.concatenate([idx, pad])


# --------------------------------------------------------------------------- #
# Skeleton
# --------------------------------------------------------------------------- #

def normalize_skeleton(x: np.ndarray, ref_nodes=(1, 2)) -> np.ndarray:
    """Center on the midpoint of ``ref_nodes`` (shoulders) and scale by their distance."""
    x = x.copy()
    coords = x[..., :2]
    center = coords[:, list(ref_nodes), :].mean(axis=1, keepdims=True)  # (T,1,2)
    coords = coords - center
    # scale by mean shoulder width across valid frames
    d = np.linalg.norm(coords[:, ref_nodes[0], :] - coords[:, ref_nodes[1], :], axis=-1)
    scale = np.median(d[d > 1e-6]) if np.any(d > 1e-6) else 1.0
    coords = coords / (scale + 1e-6)
    x[..., :2] = coords
    return x


def augment_skeleton(x: np.ndarray, rng: np.random.Generator,
                     rot=0.17, scale=0.1, shear=0.1, jitter=0.01) -> np.ndarray:
    """Random rotation, scaling, shear, and joint jitter on the (x, y) coords."""
    x = x.copy()
    coords = x[..., :2]
    theta = rng.uniform(-rot, rot)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    sx, sy = 1 + rng.uniform(-scale, scale), 1 + rng.uniform(-scale, scale)
    S = np.array([[sx, 0], [0, sy]])
    hx, hy = rng.uniform(-shear, shear), rng.uniform(-shear, shear)
    H = np.array([[1, hx], [hy, 1]])
    M = R @ S @ H
    coords = coords @ M.T
    coords = coords + rng.normal(0, jitter, size=coords.shape)
    x[..., :2] = coords
    return x


# --------------------------------------------------------------------------- #
# RGB clips
# --------------------------------------------------------------------------- #

IMAGENET_MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
IMAGENET_STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)


def process_clip(frames: np.ndarray, size: int, train: bool, rng: np.random.Generator) -> np.ndarray:
    """(T,H,W,3) uint8 -> (3,T,size,size) float32, normalized. Random/center crop."""
    import cv2

    out = np.empty((frames.shape[0], size, size, 3), dtype=np.float32)
    resize_to = int(size * 1.15)
    for i, fr in enumerate(frames):
        fr = cv2.resize(fr, (resize_to, resize_to), interpolation=cv2.INTER_LINEAR)
        if train:
            y = rng.integers(0, resize_to - size + 1)
            xo = rng.integers(0, resize_to - size + 1)
        else:
            y = xo = (resize_to - size) // 2
        out[i] = fr[y:y + size, xo:xo + size, :]
    out = out / 255.0
    out = (out - IMAGENET_MEAN) / IMAGENET_STD
    return out.transpose(3, 0, 1, 2)  # (C, T, H, W)
