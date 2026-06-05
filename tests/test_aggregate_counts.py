import pandas as pd
from emotion_pipeline import EMOTIONS
from emotion_pipeline.aggregate import emotion_counts_face, emotion_counts_frame


def _faces(labels):
    rows = []
    for i, lab in enumerate(labels):
        rec = {"video_id": "v", "frame_idx": i, "face_id": 0, "face_area_frac": 0.1,
               "label": lab}
        for e in EMOTIONS:
            rec[f"p_{e}"] = 1.0 if e == lab else 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def _frames(frame_emotions):
    rows = []
    for i, e in enumerate(frame_emotions):
        rows.append({"video_id": "v", "frame_idx": i, "t_sec": i * 0.5,
                     "n_faces": 0 if e == "no_face" else 1,
                     "no_face": e == "no_face", "frame_emotion": e,
                     "frame_max_prob": 0.0 if e == "no_face" else 0.8})
    return pd.DataFrame(rows)


def test_emotion_counts_face_includes_uncertain_and_totals():
    faces = _faces(["happiness", "happiness", "sadness", "uncertain", "uncertain"])
    row = emotion_counts_face(faces, "v")
    assert row["video_id"] == "v"
    assert row["n_happiness"] == 2
    assert row["n_sadness"] == 1
    assert row["n_uncertain"] == 2
    assert row["n_anger"] == 0
    assert row["n_faces_total"] == 5
    # every emotion plus uncertain has a column
    for e in EMOTIONS:
        assert f"n_{e}" in row


def test_emotion_counts_frame_breaks_out_no_face():
    frames = _frames(["happiness", "sadness", "sadness", "no_face"])
    row = emotion_counts_frame(frames, "v")
    assert row["n_happiness"] == 1
    assert row["n_sadness"] == 2
    assert row["n_no_face"] == 1
    assert row["n_frames_total"] == 4
    # no_face is NOT counted as an emotion bucket
    assert "n_no_face" in row and row["n_uncertain"] == 0
