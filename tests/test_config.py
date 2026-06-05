from emotion_pipeline import EMOTIONS
from emotion_pipeline.config import PipelineConfig


def test_emotions_canonical_order():
    assert EMOTIONS == ["anger", "contempt", "disgust", "fear",
                        "happiness", "neutral", "sadness", "surprise"]


def test_config_defaults():
    cfg = PipelineConfig()
    assert cfg.fps_target == 2
    assert 0.0 < cfg.abstain_tau < 1.0
    assert cfg.min_face_px >= 1
    assert cfg.frame_emotion_method == "mean_softmax"
    assert cfg.crop_margin > 0
