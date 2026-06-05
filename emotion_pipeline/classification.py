from typing import Sequence, Tuple

from emotion_pipeline import EMOTIONS


def classify_with_abstain(prob_vec: Sequence[float],
                          tau: float) -> Tuple[str, str, float]:
    """Map an 8-class probability vector (aligned to EMOTIONS) to a label.

    Returns (label, argmax_emotion, max_prob). When max_prob < tau the label is
    'uncertain' (the argmax is still recorded) -- errors are never silently
    counted as neutral.
    """
    if len(prob_vec) != len(EMOTIONS):
        raise ValueError(f"prob_vec must have {len(EMOTIONS)} entries")
    best_i = max(range(len(prob_vec)), key=lambda i: prob_vec[i])
    max_prob = float(prob_vec[best_i])
    argmax = EMOTIONS[best_i]
    label = argmax if max_prob >= tau else "uncertain"
    return label, argmax, max_prob
