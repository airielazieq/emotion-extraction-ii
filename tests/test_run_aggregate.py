from pathlib import Path
import pandas as pd
from emotion_pipeline import EMOTIONS
from scripts.run_aggregate import aggregate_one, join_labels


def _write_faces(out: Path, vid: str):
    rows = []
    for fi, t, emo in [(0, 0.0, "happiness"), (0, 0.0, "happiness"), (15, 0.5, "sadness")]:
        rec = {"video_id": vid, "frame_idx": fi, "t_sec": t, "face_id": 0,
               "face_area_frac": 0.5, "label": emo}
        for e in EMOTIONS:
            rec[f"p_{e}"] = 0.9 if e == emo else 0.0
        rows.append(rec)
    pd.DataFrame(rows).to_parquet(out / f"{vid}_faces.parquet", index=False)
    pd.DataFrame([{"video_id": vid, "frame_idx": 0, "t_sec": 0.0},
                  {"video_id": vid, "frame_idx": 15, "t_sec": 0.5},
                  {"video_id": vid, "frame_idx": 30, "t_sec": 1.0}]
                 ).to_parquet(out / f"{vid}_grid.parquet", index=False)


def test_aggregate_one_produces_frames_and_summary(tmp_path: Path):
    _write_faces(tmp_path, "vid1")
    frames, summary = aggregate_one(tmp_path, "vid1", method="mean_softmax")
    assert len(frames) == 3                  # incl. the no-face frame 30
    assert summary["n_frames_face"] == 2
    assert summary["dominant_emotion"] in ("happiness", "sadness")


def test_join_labels_attaches_likes():
    films = pd.DataFrame({"video_id": ["a", "b"], "dominant_emotion": ["fear", "neutral"]})
    labels = pd.DataFrame({"video_id": ["a", "b"], "likes": [100, 200]})
    merged = join_labels(films, labels)
    assert list(merged.sort_values("video_id")["likes"]) == [100, 200]
