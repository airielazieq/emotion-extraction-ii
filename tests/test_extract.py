import numpy as np
from emotion_pipeline import EMOTIONS
from emotion_pipeline.config import PipelineConfig
from emotion_pipeline.extract import extract_video_faces


def _frame(h=1000, w=1000):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_extract_records_have_full_schema_and_filter_small_faces():
    cfg = PipelineConfig(min_face_px=40, min_face_area_frac=0.003, abstain_tau=0.4)
    frames = [(0, 0.0, _frame()), (15, 0.5, _frame())]

    # One big face (passes) + one tiny face (filtered) per frame.
    def fake_detector(bgr):
        return [((100, 100, 300, 300), 0.9), ((0, 0, 20, 20), 0.8)]

    # Big face -> happiness 0.8; the tiny face is filtered before classify.
    def fake_classifier(crop):
        v = [0.0] * 8
        v[EMOTIONS.index("happiness")] = 0.8
        v[EMOTIONS.index("neutral")] = 0.2
        return v

    recs = extract_video_faces("vid123", frames, fake_detector, fake_classifier, cfg)
    assert len(recs) == 2  # one kept face per frame
    r = recs[0]
    for col in ["video_id", "frame_idx", "t_sec", "face_id", "x1", "y1", "x2", "y2",
                "det_conf", "face_area_frac", "argmax_emotion", "max_prob", "label",
                *[f"p_{e}" for e in EMOTIONS]]:
        assert col in r
    assert r["video_id"] == "vid123"
    assert r["argmax_emotion"] == "happiness"
    assert r["label"] == "happiness"


def test_extract_abstains_on_low_confidence():
    cfg = PipelineConfig(abstain_tau=0.5)
    frames = [(0, 0.0, _frame())]

    def fake_detector(bgr):
        return [((100, 100, 400, 400), 0.9)]

    def fake_classifier(crop):
        return [0.2, 0.18, 0.15, 0.12, 0.1, 0.1, 0.08, 0.07]  # max 0.2 < 0.5

    recs = extract_video_faces("v", frames, fake_detector, fake_classifier, cfg)
    assert recs[0]["label"] == "uncertain"
    assert recs[0]["argmax_emotion"] == "anger"
