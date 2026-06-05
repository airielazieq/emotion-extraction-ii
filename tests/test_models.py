import importlib.util
import pytest

HAVE_DEPS = all(
    importlib.util.find_spec(m) is not None
    for m in ("cv2", "ultralytics", "hsemotion_onnx")
)


def test_models_module_imports():
    import emotion_pipeline.models as m
    assert hasattr(m, "load_face_model")
    assert hasattr(m, "load_emotion_model")
    assert hasattr(m, "detect_faces")
    assert hasattr(m, "classify_crop")


@pytest.mark.skipif(not HAVE_DEPS, reason="model deps not installed")
def test_emotion_model_label_order_matches_emotions():
    # Guards against silent re-ordering of the 8 classes between model versions.
    from emotion_pipeline import EMOTIONS
    from emotion_pipeline.models import load_emotion_model, emotion_class_order
    rec = load_emotion_model()
    assert sorted(emotion_class_order(rec)) == sorted(EMOTIONS)
