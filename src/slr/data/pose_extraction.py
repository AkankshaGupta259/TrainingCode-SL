"""Extract the 27-node sign skeleton from RGB videos and cache it as ``(T, V, 3)`` npy.

Default backend is MediaPipe Holistic (pip-installable, no CUDA toolchain needed).
Each output frame holds the 27 nodes defined in ``graph.NODE_SPEC`` with channels
(x, y, score), x/y normalized to [0, 1] image coordinates. Missing landmarks are
written as zeros with score 0.

For higher accuracy on the 4090 you can swap in RTMPose/MMPose by implementing a
backend with the same ``extract(video_path) -> (T, 27, 3)`` contract.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .graph import NODE_SPEC, NUM_NODES


class MediaPipeHolisticBackend:
    def __init__(self, min_det=0.5, min_track=0.5, complexity=1):
        try:
            import mediapipe as mp
        except ImportError as exc:  # noqa: BLE001
            raise ImportError(
                "mediapipe is required for pose extraction "
                "(`pip install mediapipe`)."
            ) from exc
        self.mp = mp
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=complexity,
            min_detection_confidence=min_det,
            min_tracking_confidence=min_track,
        )

    def _collect(self, results) -> np.ndarray:
        groups = {
            "pose": results.pose_landmarks,
            "left_hand": results.left_hand_landmarks,
            "right_hand": results.right_hand_landmarks,
        }
        frame = np.zeros((NUM_NODES, 3), dtype=np.float32)
        for out_idx, (src, lm_idx) in NODE_SPEC.items():
            lms = groups.get(src)
            if lms is None:
                continue
            lm = lms.landmark[lm_idx]
            vis = getattr(lm, "visibility", 1.0) or 1.0
            frame[out_idx] = (lm.x, lm.y, vis)
        return frame

    def extract(self, video_path: str) -> np.ndarray:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            img.flags.writeable = False
            res = self.holistic.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            frames.append(self._collect(res))
        cap.release()
        if not frames:
            return np.zeros((0, NUM_NODES, 3), dtype=np.float32)
        return np.stack(frames)

    def close(self):
        self.holistic.close()


def get_backend(name: str = "mediapipe", **kwargs):
    if name == "mediapipe":
        return MediaPipeHolisticBackend(**kwargs)
    raise ValueError(f"Unknown pose backend: {name} (only 'mediapipe' is built in)")


def extract_and_cache(video_path: str, out_path: str, backend) -> int:
    """Extract one video and save the cache. Returns number of frames (0 if failed)."""
    arr = backend.extract(video_path)
    if arr.shape[0] == 0:
        return 0
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    return arr.shape[0]
