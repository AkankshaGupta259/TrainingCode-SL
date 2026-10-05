# Word-Level Sign Language Recognition — Model Comparison

Benchmark of **9 models** on **isolated (word-level) Sign Language Recognition** using **WLASL1000**.

| Family | Models | Input | Init |
|---|---|---|---|
| Skeleton GCN | ST-GCN, CTR-GCN, HA-GCN, HD-GCN, InfoGCN, MS-G3D | keypoints | from scratch |
| Skeleton Transformer | MotionBERT | keypoints | pretrained encoder |
| RGB video CNN | I3D, SlowFast | RGB clips | Kinetics-400 |

**Protocol:** WLASL1000 · 3 seeds per model (mean ± std) · published configs tuned lightly on val ·
single RTX 4090. Metrics: Top-1/Top-5, mean-per-class accuracy, macro precision/recall/F1,
plus an efficiency table (#params, GFLOPs, FPS, VRAM). See `configs/` and the docstrings.

## Repo layout

```
configs/            experiment + dataset configs (YAML)
src/slr/
  data/             WLASL metadata parsing, download, verification
  metrics/          classification metrics (top-k, per-class, P/R/F1)
  utils/            history/loss-curve logging + plotting
scripts/            CLI entry points (download, verify, plot)
data/               (gitignored) downloaded videos + extracted poses land here
```

## Setup

```bash
# 1. Python env (conda)
conda env create -f environment.yml
conda activate slr

# 2. System deps (required for video download/trim)
#    - ffmpeg must be on PATH
#    - yt-dlp is installed via pip (in environment.yml)
```

## 1) Download WLASL1000

The script fetches the official `WLASL_v0.3.json` metadata, selects the **top-1000 glosses by
sample count** (this is how the WLASL paper defines the subset), downloads each clip
(YouTube via yt-dlp, direct links via HTTP), trims it to the annotated frame range, and writes a
**manifest** recording success/failure per clip.

```bash
python scripts/download_wlasl.py \
    --out-dir data/wlasl \
    --num-classes 1000 \
    --num-workers 8
```

> ⚠️ Many WLASL source links are dead. Expect to lose some clips; rarer glosses (toward rank
> 1000) suffer most. The manifest + verify step below tell you exactly what you got. If too many
> are missing, start from a community mirror of the pre-downloaded videos and point
> `--video-dir` at it to skip re-downloading.

## 2) Verify what you actually downloaded

```bash
python scripts/verify_wlasl.py --out-dir data/wlasl
```

Prints per-split totals, per-class counts, and flags classes with too few samples. **Always report
these numbers** — they are part of a reproducible WLASL result.

## 3) Training (next phase)

Pose extraction + the 9 model configs/training loop come next. The metrics
(`src/slr/metrics`) and loss-curve logging (`src/slr/utils`) are already built so the training
loop just calls them. Plot any run's curves with:

```bash
python scripts/plot_curves.py --history runs/<exp>/history.csv --out-dir runs/<exp>
```
