from emotion_pipeline.validate import agreement, confusion, coverage_report
import pandas as pd


def test_agreement_simple():
    y_true = ["happiness", "sadness", "neutral", "fear"]
    y_pred = ["happiness", "sadness", "neutral", "anger"]
    assert abs(agreement(y_true, y_pred) - 0.75) < 1e-9


def test_confusion_counts_offdiagonal():
    y_true = ["fear", "fear", "surprise"]
    y_pred = ["fear", "surprise", "surprise"]
    cm = confusion(y_true, y_pred, labels=["fear", "surprise"])
    # rows=true, cols=pred
    assert cm.loc["fear", "fear"] == 1
    assert cm.loc["fear", "surprise"] == 1
    assert cm.loc["surprise", "surprise"] == 1


def test_coverage_report_flags_low_face_films():
    films = pd.DataFrame({
        "video_id": ["a", "b", "c"],
        "n_frames_face": [500, 80, 20],
        "face_rate": [0.7, 0.5, 0.1],
        "uncertain_frame_rate": [0.05, 0.2, 0.6],
    })
    rep = coverage_report(films, min_faced=100, max_uncertain=0.3)
    assert set(rep[rep["flagged"]]["video_id"]) == {"b", "c"}
