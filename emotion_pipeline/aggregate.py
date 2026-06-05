import math
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


def label_change_rate(labels_in_order: List[str]) -> float:
    """Fraction of consecutive (faced) frames whose emotion label changes."""
    seq = [l for l in labels_in_order if l not in ("no_face",)]
    if len(seq) < 2:
        return 0.0
    changes = sum(1 for a, b in zip(seq[:-1], seq[1:]) if a != b)
    return changes / (len(seq) - 1)


def film_summary(frames: pd.DataFrame, video_id: str,
                 low_coverage_min_faced: int = 100) -> dict:
    """One-row film-level summary from a frames table (per-frame emotions)."""
    n_sampled = int(len(frames))
    faced = frames[~frames["no_face"]]
    n_faced = int(len(faced))
    # proportions over faced frames, excluding 'uncertain' from the emotion mix
    emo_frames = faced[faced["frame_emotion"].isin(EMOTIONS)]
    n_emo = int(len(emo_frames))
    counts = emo_frames["frame_emotion"].value_counts()
    props = {f"prop_{e}": (float(counts.get(e, 0)) / n_emo if n_emo else 0.0)
             for e in EMOTIONS}
    probs = [props[f"prop_{e}"] for e in EMOTIONS]
    nz = [p for p in probs if p > 0]
    entropy_bits = float(-sum(p * math.log2(p) for p in nz)) if nz else 0.0
    dominant = max(EMOTIONS, key=lambda e: props[f"prop_{e}"]) if n_emo else "uncertain"
    non_neutral = sum(props[f"prop_{e}"] for e in EMOTIONS if e != "neutral")
    uncertain_rate = (float((faced["frame_emotion"] == "uncertain").sum()) / n_faced
                      if n_faced else 0.0)
    out = {
        "video_id": video_id,
        "n_frames_sampled": n_sampled,
        "n_frames_face": n_faced,
        "face_rate": (n_faced / n_sampled if n_sampled else 0.0),
        "n_faces": int(faced["n_faces"].sum()),
        "faces_per_faced_frame": (float(faced["n_faces"].sum()) / n_faced
                                  if n_faced else 0.0),
        **props,
        "dominant_emotion": dominant,
        "entropy_bits": round(entropy_bits, 4),
        "norm_entropy": round(entropy_bits / math.log2(len(EMOTIONS)), 6),
        "non_neutral_share": round(non_neutral, 4),
        "uncertain_frame_rate": round(uncertain_rate, 4),
        "label_change_rate": round(
            label_change_rate(faced["frame_emotion"].tolist()), 4),
        "low_coverage_flag": n_faced < low_coverage_min_faced,
    }
    return out
