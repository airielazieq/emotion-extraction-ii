from typing import List, Tuple

import numpy as np
import pandas as pd

from emotion_pipeline import EMOTIONS

_PCOLS = [f"p_{e}" for e in EMOTIONS]


def frame_emotion_from_faces(faces_in_frame: pd.DataFrame,
                             method: str = "mean_softmax") -> Tuple[str, float]:
    """Collapse all faces in ONE frame to a single (emotion, confidence).

    'mean_softmax'     : average the 8 probabilities across faces, take argmax.
    'dominant_largest' : the single largest-area face's argmax.
    'majority'         : most common per-face label (ties -> EMOTIONS order).
    """
    if faces_in_frame.empty:
        return "uncertain", 0.0
    if method == "dominant_largest":
        row = faces_in_frame.loc[faces_in_frame["face_area_frac"].idxmax()]
        probs = [float(row[c]) for c in _PCOLS]
    elif method == "majority":
        counts = faces_in_frame["label"].value_counts()
        counts = counts[counts.index != "uncertain"]
        if counts.empty:
            return "uncertain", 0.0
        top = counts.max()
        winners = [lab for lab in EMOTIONS if counts.get(lab, -1) == top]
        return winners[0], float(top) / float(len(faces_in_frame))
    else:  # mean_softmax
        probs = faces_in_frame[_PCOLS].mean(axis=0).tolist()
    best = int(np.argmax(probs))
    return EMOTIONS[best], float(probs[best])


def faces_to_frames(faces: pd.DataFrame, sampled_frames: List[Tuple[int, float]],
                    video_id: str, method: str = "mean_softmax") -> pd.DataFrame:
    """Build one row per SAMPLED frame, including no-face frames."""
    by_frame = {fidx: grp for fidx, grp in faces.groupby("frame_idx")}
    rows = []
    for frame_idx, t_sec in sampled_frames:
        grp = by_frame.get(frame_idx)
        if grp is None or grp.empty:
            rows.append({"video_id": video_id, "frame_idx": frame_idx, "t_sec": t_sec,
                         "n_faces": 0, "no_face": True,
                         "frame_emotion": "no_face", "frame_max_prob": 0.0})
        else:
            emo, mp = frame_emotion_from_faces(grp, method=method)
            rows.append({"video_id": video_id, "frame_idx": frame_idx, "t_sec": t_sec,
                         "n_faces": int(len(grp)), "no_face": False,
                         "frame_emotion": emo, "frame_max_prob": round(mp, 4)})
    return pd.DataFrame(rows)
