"""Download and trim a single WLASL clip.

Handles two source types:
  * YouTube URLs      -> yt-dlp
  * Direct video URLs -> HTTP streaming download

After fetching the raw file, the clip is trimmed to the annotated frame range
([frame_start, frame_end], 1-indexed in the metadata) using ffmpeg's frame-accurate
``select`` filter. ffmpeg and yt-dlp must be available on PATH.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .wlasl_metadata import Instance

# Statuses recorded in the manifest.
OK = "ok"
SKIP_EXISTS = "exists"
FAIL_DOWNLOAD = "download_failed"
FAIL_TRIM = "trim_failed"
FAIL_UNSUPPORTED = "unsupported_source"


def is_youtube(url: str) -> bool:
    u = url.lower()
    return "youtube.com" in u or "youtu.be" in u


def _run(cmd: list[str], timeout: int) -> tuple[bool, str]:
    """Run a subprocess quietly; return (success, stderr-tail)."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        if proc.returncode == 0:
            return True, ""
        return False, proc.stderr.decode("utf-8", "ignore")[-500:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except FileNotFoundError as exc:
        return False, f"missing executable: {exc}"


def download_youtube(url: str, out_path: Path, timeout: int) -> bool:
    cmd = [
        "yt-dlp", "--quiet", "--no-warnings", "--no-playlist",
        "-f", "mp4/best",
        "-o", str(out_path),
        url,
    ]
    ok, err = _run(cmd, timeout)
    if not ok and err:
        print(f"    yt-dlp: {err.strip().splitlines()[-1] if err.strip() else err}")
    return ok and out_path.exists()


def download_direct(url: str, out_path: Path, timeout: int) -> bool:
    import requests

    headers = {"User-Agent": "Mozilla/5.0 (research; WLASL downloader)"}
    try:
        with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        f.write(chunk)
        return out_path.exists() and out_path.stat().st_size > 0
    except Exception as exc:  # noqa: BLE001 - we log and continue to the next clip
        print(f"    http: {exc}")
        return False


def trim_video(
    src: Path, dst: Path, frame_start: int, frame_end: int, fps: float
) -> bool:
    """Trim ``src`` to [frame_start, frame_end] (1-indexed) using ffmpeg select.

    If no trimming is needed (whole clip), just copy the file.
    """
    s0 = max(frame_start - 1, 0)          # ffmpeg select n is 0-indexed
    end = frame_end - 1 if frame_end and frame_end > 0 else -1

    if s0 <= 0 and end < 0:
        shutil.copyfile(src, dst)
        return dst.exists()

    if end >= 0:
        select = f"select='between(n\\,{s0}\\,{end})'"
    else:
        select = f"select='gte(n\\,{s0})'"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", f"{select},setpts=N/FRAME_RATE/TB",
        "-an",
        str(dst),
    ]
    ok, err = _run(cmd, timeout=120)
    if not ok and err:
        print(f"    ffmpeg: {err.strip().splitlines()[-1] if err.strip() else err}")
    return ok and dst.exists()


def download_instance(
    inst: Instance,
    raw_dir: Path,
    out_dir: Path,
    timeout: int = 30,
    retries: int = 2,
    overwrite: bool = False,
) -> str:
    """Download + trim one instance. Returns a status string (see constants above)."""
    out_path = out_dir / f"{inst.video_id}.mp4"
    if out_path.exists() and not overwrite:
        return SKIP_EXISTS

    if not inst.url:
        return FAIL_UNSUPPORTED
    # .swf (e.g. aslpro) and other non-video sources are not handled here.
    if inst.url.lower().endswith(".swf"):
        return FAIL_UNSUPPORTED

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{inst.video_id}_raw.mp4"

    downloaded = False
    for _ in range(max(retries, 1)):
        if is_youtube(inst.url):
            downloaded = download_youtube(inst.url, raw_path, timeout)
        else:
            downloaded = download_direct(inst.url, raw_path, timeout)
        if downloaded:
            break
    if not downloaded:
        return FAIL_DOWNLOAD

    trimmed = trim_video(raw_path, out_path, inst.frame_start, inst.frame_end, inst.fps)
    try:
        raw_path.unlink(missing_ok=True)  # keep only the trimmed clip
    except OSError:
        pass
    return OK if trimmed else FAIL_TRIM
