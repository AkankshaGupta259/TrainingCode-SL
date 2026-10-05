"""Parse WLASL metadata and build the top-N-gloss subset (e.g. WLASL1000).

The WLASL paper defines WLASL{100,300,1000,2000} as the N glosses with the most
samples. We reproduce that here directly from ``WLASL_v0.3.json`` so the subset is
self-contained, and we use each instance's own ``split`` field for train/val/test.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional

# Official metadata location (adjust if the upstream repo layout changes).
WLASL_JSON_URL = (
    "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"
)


@dataclass
class Instance:
    video_id: str
    url: str
    split: str
    fps: float
    frame_start: int
    frame_end: int
    bbox: Optional[list] = None
    signer_id: Optional[int] = None
    source: Optional[str] = None

    @classmethod
    def from_raw(cls, raw: dict) -> "Instance":
        return cls(
            video_id=str(raw["video_id"]),
            url=raw.get("url", ""),
            split=raw.get("split", "train"),
            fps=float(raw.get("fps", 25) or 25),
            frame_start=int(raw.get("frame_start", 1) or 1),
            frame_end=int(raw.get("frame_end", -1) if raw.get("frame_end") is not None else -1),
            bbox=raw.get("bbox"),
            signer_id=raw.get("signer_id"),
            source=raw.get("source"),
        )


@dataclass
class GlossEntry:
    gloss: str
    class_idx: int
    instances: list = field(default_factory=list)


def download_metadata(dest: str | os.PathLike, url: str = WLASL_JSON_URL) -> Path:
    """Download WLASL_v0.3.json if not already present. Returns the local path."""
    import requests

    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[metadata] downloading {url}")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    print(f"[metadata] saved -> {dest}")
    return dest


def load_metadata(json_path: str | os.PathLike) -> list:
    """Load the raw WLASL metadata list."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_subset(metadata: list, num_classes: int) -> tuple[list[GlossEntry], dict]:
    """Select the top-``num_classes`` glosses by instance count.

    Returns (entries, gloss_to_idx). Glosses are ranked by descending instance
    count, ties broken alphabetically, so the subset is deterministic.
    """
    ranked = sorted(
        metadata,
        key=lambda e: (-len(e.get("instances", [])), e.get("gloss", "")),
    )
    selected = ranked[:num_classes]

    entries: list[GlossEntry] = []
    gloss_to_idx: dict[str, int] = {}
    for idx, entry in enumerate(selected):
        gloss = entry["gloss"]
        gloss_to_idx[gloss] = idx
        instances = [Instance.from_raw(r) for r in entry.get("instances", [])]
        entries.append(GlossEntry(gloss=gloss, class_idx=idx, instances=instances))
    return entries, gloss_to_idx


def iter_instances(
    entries: Iterable[GlossEntry],
    splits: Optional[Iterable[str]] = None,
) -> Iterator[tuple[GlossEntry, Instance]]:
    """Yield (GlossEntry, Instance) pairs, optionally filtered by split."""
    split_set = set(splits) if splits is not None else None
    for entry in entries:
        for inst in entry.instances:
            if split_set is None or inst.split in split_set:
                yield entry, inst


def save_label_map(gloss_to_idx: dict, path: str | os.PathLike) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(gloss_to_idx, f, indent=2, ensure_ascii=False)
    print(f"[metadata] label map ({len(gloss_to_idx)} classes) -> {path}")
