import math
import pandas as pd
from emotion_pipeline import EMOTIONS
from emotion_pipeline.aggregate import film_summary


def _frames(emotions, video_id="v"):
    rows = []
    for i, e in enumerate(emotions):
        rows.append({"video_id": video_id, "frame_idx": i * 15, "t_sec": i * 0.5,
                     "n_faces": 0 if e == "no_face" else 1,
                     "no_face": e == "no_face",
                     "frame_emotion": e, "frame_max_prob": 0.0 if e == "no_face" else 0.8})
    return pd.DataFrame(rows)


def test_film_summary_proportions_and_coverage():
    frames = _frames(["happiness", "happiness", "sadness", "no_face"])
    s = film_summary(frames, video_id="v", low_coverage_min_faced=2)
    assert s["video_id"] == "v"
    assert s["n_frames_sampled"] == 4
    assert s["n_frames_face"] == 3
    assert abs(s["face_rate"] - 0.75) < 1e-9
    # proportions computed over faced frames only
    assert abs(s["prop_happiness"] - (2 / 3)) < 1e-9
    assert abs(s["prop_sadness"] - (1 / 3)) < 1e-9
    assert s["dominant_emotion"] == "happiness"
    assert s["low_coverage_flag"] is False  # 3 faced >= 2


def test_film_summary_entropy_and_uncertain_rate():
    frames = _frames(["happiness", "sadness"])  # 50/50
    s = film_summary(frames, video_id="v", low_coverage_min_faced=100)
    assert abs(s["entropy_bits"] - 1.0) < 1e-6        # 2 equal classes -> 1 bit
    assert abs(s["norm_entropy"] - (1.0 / math.log2(8))) < 1e-6
    assert s["low_coverage_flag"] is True             # 2 faced < 100
    frames2 = _frames(["uncertain", "happiness"])
    s2 = film_summary(frames2, video_id="v", low_coverage_min_faced=1)
    assert abs(s2["uncertain_frame_rate"] - 0.5) < 1e-9
