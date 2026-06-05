from emotion_pipeline import EMOTIONS
from emotion_pipeline.classification import classify_with_abstain


def test_high_confidence_uses_argmax():
    probs = [0.02, 0.01, 0.01, 0.03, 0.80, 0.10, 0.02, 0.01]  # happiness
    label, argmax, mp = classify_with_abstain(probs, tau=0.40)
    assert argmax == "happiness"
    assert label == "happiness"
    assert abs(mp - 0.80) < 1e-9


def test_low_confidence_abstains_but_records_argmax():
    probs = [0.20, 0.18, 0.15, 0.12, 0.10, 0.10, 0.08, 0.07]  # max 0.20 < 0.40
    label, argmax, mp = classify_with_abstain(probs, tau=0.40)
    assert argmax == "anger"
    assert label == "uncertain"   # NOT silently 'neutral'
    assert abs(mp - 0.20) < 1e-9


def test_argmax_aligns_to_emotions_order():
    for i, emo in enumerate(EMOTIONS):
        probs = [0.0] * 8
        probs[i] = 1.0
        label, argmax, mp = classify_with_abstain(probs, tau=0.40)
        assert argmax == emo and label == emo
