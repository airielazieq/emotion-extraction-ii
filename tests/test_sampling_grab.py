import importlib.util
from pathlib import Path

import numpy as np
import pytest

HAVE_CV2 = importlib.util.find_spec("cv2") is not None


@pytest.mark.skipif(not HAVE_CV2, reason="cv2 not installed")
def test_grab_sampler_matches_timestamp_sampler_on_cfr(tmp_path: Path):
    import cv2
    from emotion_pipeline.sampling import (
        read_frames_at_timestamps, read_frames_grab,
    )

    # 3s, 30fps constant-frame-rate synthetic clip.
    path = tmp_path / "clip [abcdefghijk].mp4"
    w, h, fps, secs = 320, 240, 30, 3
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(fps * secs):
        writer.write(np.full((h, w, 3), i % 255, dtype=np.uint8))
    writer.release()

    ts_frames, ts_fps, ts_dur, ts_w, ts_h = read_frames_at_timestamps(path, 2)
    gb_frames, gb_fps, gb_dur, gb_w, gb_h = read_frames_grab(path, 2)

    # Same metadata
    assert (gb_w, gb_h) == (ts_w, ts_h) == (w, h)
    assert abs(gb_fps - fps) < 2
    # Frame indices line up with the timestamp grid (CFR => identical sampling)
    gb_idx = [fi for fi, _, _ in gb_frames]
    assert gb_idx == sorted(set(gb_idx))           # strictly increasing, unique
    assert gb_idx[:4] == [0, 15, 30, 45]           # 2 fps on a 30 fps source
    assert 5 <= len(gb_frames) <= 8
    # Each decoded frame is a real image of the right shape
    assert all(f.shape == (h, w, 3) for _, _, f in gb_frames)
