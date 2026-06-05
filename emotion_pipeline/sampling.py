import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def target_timestamps(duration_s: float, fps_target: int) -> List[float]:
    """Fixed time grid at fps_target, inclusive of t=0 and the final whole step.

    Independent of the source video fps, so every film is sampled on the same
    0.5s (for fps_target=2) grid -> counts are comparable across films.
    """
    if duration_s <= 0 or fps_target <= 0:
        return []
    step = 1.0 / fps_target
    n = int(math.floor(duration_s / step)) + 1
    return [round(i * step, 4) for i in range(n)]


def _nearest_index(t: float, src_fps: float) -> int:
    """Round-half-up nearest source-frame index (Python's round() is banker's)."""
    return int(math.floor(t * src_fps + 0.5))


def frame_indices_for_timestamps(timestamps: List[float], src_fps: float) -> List[int]:
    """Nearest source-frame index for each target timestamp (for CFR seeking)."""
    return [_nearest_index(t, src_fps) for t in timestamps]


def read_frames_at_timestamps(
    video_path: Path, fps_target: int
) -> Tuple[List[Tuple[int, float, np.ndarray]], float, float, int, int]:
    """Seek to each target timestamp via POS_MSEC (handles variable frame rate).

    Returns (frames, src_fps, duration_s, width, height) where each frame is
    (frame_idx, t_sec, bgr_ndarray). Frames that fail to decode are skipped.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = (total_f / src_fps) if src_fps else 0.0

    frames: List[Tuple[int, float, np.ndarray]] = []
    for t in target_timestamps(duration_s, fps_target):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        idx = _nearest_index(t, src_fps)
        frames.append((idx, round(t, 4), frame))
    cap.release()
    return frames, float(src_fps), float(duration_s), width, height
