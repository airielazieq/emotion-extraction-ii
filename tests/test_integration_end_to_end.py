import importlib.util
from pathlib import Path

import numpy as np
import pytest

HAVE_CV2 = importlib.util.find_spec("cv2") is not None


@pytest.mark.skipif(not HAVE_CV2, reason="cv2 not installed")
def test_sampling_reads_expected_frame_count(tmp_path: Path):
    import cv2
    from emotion_pipeline.sampling import read_frames_at_timestamps

    # Write a 3-second, 30fps synthetic video (solid colour frames).
    path = tmp_path / "clip [abcdefghijk].mp4"
    w, h, fps, secs = 320, 240, 30, 3
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(fps * secs):
        writer.write(np.full((h, w, 3), i % 255, dtype=np.uint8))
    writer.release()

    frames, src_fps, dur, vw, vh = read_frames_at_timestamps(path, fps_target=2)
    # ~3s at 2 fps -> ~6-7 frames; allow tolerance for container rounding
    assert 5 <= len(frames) <= 8
    assert vw == w and vh == h
    assert abs(src_fps - fps) < 2


@pytest.mark.skipif(not HAVE_CV2, reason="cv2 not installed")
def test_full_pipeline_with_fake_models(tmp_path: Path):
    """faces -> frames -> film_summary with injected fake detector/classifier."""
    from emotion_pipeline import EMOTIONS
    from emotion_pipeline.config import PipelineConfig
    from emotion_pipeline.extract import extract_video_faces
    from emotion_pipeline.aggregate import faces_to_frames, film_summary
    import pandas as pd

    cfg = PipelineConfig()
    frames_in = [(0, 0.0, np.zeros((500, 500, 3), np.uint8)),
                 (15, 0.5, np.zeros((500, 500, 3), np.uint8)),
                 (30, 1.0, np.zeros((500, 500, 3), np.uint8))]

    def det(bgr):
        return [((100, 100, 300, 300), 0.9)]

    def clf(crop):
        v = [0.0] * 8
        v[EMOTIONS.index("sadness")] = 0.85
        v[EMOTIONS.index("neutral")] = 0.15
        return v

    recs = extract_video_faces("vid", frames_in, det, clf, cfg)
    faces = pd.DataFrame(recs)
    sampled = [(0, 0.0), (15, 0.5), (30, 1.0)]
    frames = faces_to_frames(faces, sampled, "vid", method=cfg.frame_emotion_method)
    summary = film_summary(frames, "vid", low_coverage_min_faced=1)
    assert summary["n_frames_face"] == 3
    assert summary["dominant_emotion"] == "sadness"
    assert abs(summary["prop_sadness"] - 1.0) < 1e-9
