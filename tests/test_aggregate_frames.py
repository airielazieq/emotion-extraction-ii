import pandas as pd
from emotion_pipeline import EMOTIONS
from emotion_pipeline.aggregate import frame_emotion_from_faces, faces_to_frames


def _face(video_id, frame_idx, t, face_id, area, **probs):
    rec = {"video_id": video_id, "frame_idx": frame_idx, "t_sec": t,
           "face_id": face_id, "face_area_frac": area}
    for e in EMOTIONS:
        rec[f"p_{e}"] = probs.get(e, 0.0)
    rec["label"] = max(EMOTIONS, key=lambda e: rec[f"p_{e}"])
    return rec


def test_mean_softmax_collapses_multiple_faces_to_one_frame_emotion():
    faces = pd.DataFrame([
        _face("v", 0, 0.0, 0, 0.2, sadness=0.9),
        _face("v", 0, 0.0, 1, 0.5, happiness=0.6, neutral=0.4),
    ])
    emo, mp = frame_emotion_from_faces(faces, method="mean_softmax")
    # mean: sadness 0.45 vs happiness 0.30 vs neutral 0.20 -> sadness
    assert emo == "sadness"
    assert 0.0 < mp <= 1.0


def test_dominant_largest_uses_biggest_face():
    faces = pd.DataFrame([
        _face("v", 0, 0.0, 0, 0.2, sadness=0.9),
        _face("v", 0, 0.0, 1, 0.8, happiness=0.7),
    ])
    emo, _ = frame_emotion_from_faces(faces, method="dominant_largest")
    assert emo == "happiness"  # largest-area face wins


def test_faces_to_frames_one_row_per_frame_and_marks_no_face():
    faces = pd.DataFrame([
        _face("v", 0, 0.0, 0, 0.5, happiness=0.8),
        _face("v", 0, 0.0, 1, 0.3, happiness=0.7),
        _face("v", 15, 0.5, 0, 0.5, sadness=0.9),
    ])
    # frame 30 had no face -> supplied via sampled_frames
    sampled = [(0, 0.0), (15, 0.5), (30, 1.0)]
    frames = faces_to_frames(faces, sampled, "v", method="mean_softmax")
    assert len(frames) == 3
    no_face_row = frames[frames["frame_idx"] == 30].iloc[0]
    assert bool(no_face_row["no_face"]) is True
    assert no_face_row["n_faces"] == 0
