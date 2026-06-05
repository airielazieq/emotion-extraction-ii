from pathlib import Path
import pandas as pd
from scripts.run_validate import sample_for_labeling, score_labels


def test_sample_for_labeling_is_deterministic_and_sized(tmp_path: Path):
    faces = pd.DataFrame({
        "video_id": ["v"] * 10,
        "frame_idx": list(range(10)),
        "face_id": [0] * 10,
        "argmax_emotion": ["happiness"] * 10,
        "max_prob": [0.9] * 10,
    })
    s1 = sample_for_labeling(faces, n=5, seed=42)
    s2 = sample_for_labeling(faces, n=5, seed=42)
    assert len(s1) == 5
    assert list(s1["frame_idx"]) == list(s2["frame_idx"])  # deterministic
    assert "human_label" in s1.columns                      # blank column to fill


def test_score_labels_reports_agreement_and_confusion():
    labeled = pd.DataFrame({
        "argmax_emotion": ["fear", "fear", "surprise", "happiness"],
        "human_label":    ["fear", "surprise", "surprise", "happiness"],
    })
    report = score_labels(labeled)
    assert abs(report["agreement"] - 0.75) < 1e-9
    assert report["confusion"].loc["fear", "surprise"] == 1
    assert report["n_labeled"] == 4
