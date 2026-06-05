from typing import List

import pandas as pd


def agreement(y_true: List[str], y_pred: List[str]) -> float:
    """Plain accuracy / agreement between hand labels and classifier labels."""
    if not y_true:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def confusion(y_true: List[str], y_pred: List[str], labels: List[str]) -> pd.DataFrame:
    """Confusion matrix as a DataFrame (rows=true, cols=pred)."""
    cm = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in labels and p in labels:
            cm.loc[t, p] += 1
    return cm


def coverage_report(films: pd.DataFrame, min_faced: int = 100,
                    max_uncertain: float = 0.3) -> pd.DataFrame:
    """Flag films with too few faced frames or too many uncertain frames."""
    rep = films.copy()
    rep["flagged"] = (rep["n_frames_face"] < min_faced) | \
                     (rep["uncertain_frame_rate"] > max_uncertain)
    return rep
