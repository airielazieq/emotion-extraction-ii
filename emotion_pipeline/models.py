import os
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from emotion_pipeline import EMOTIONS
from emotion_pipeline.detection import clamp_box

Box = Tuple[int, int, int, int]

_HSEMOTION_BASE = ("https://github.com/HSE-asavchenko/face-emotion-recognition/"
                   "raw/main/models/affectnet_emotions/onnx/")


def ensure_emotion_model(model_name: str = "enet_b2_8") -> Path:
    """Pre-download the hsemotion ONNX weights into ~/.hsemotion if missing.

    Works around a bug in hsemotion-onnx whose own downloader calls
    ``urllib.request`` without importing it (fails on Python 3.13). We fetch the
    file ourselves so a fresh checkout runs with no manual setup.
    """
    cache_dir = Path(os.path.expanduser("~")) / ".hsemotion"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fpath = cache_dir / f"{model_name}.onnx"
    if not fpath.is_file():
        url = f"{_HSEMOTION_BASE}{model_name}.onnx"
        print(f"[models] downloading emotion model {model_name} -> {fpath}")
        urllib.request.urlretrieve(url, fpath)
    return fpath


def load_face_model(weights_path: str, device: Optional[str] = None):
    """Load the YOLO face detector. ``device`` is passed through to Ultralytics
    (e.g. 'cpu', 'cuda', 0); None lets Ultralytics auto-select."""
    from ultralytics import YOLO
    model = YOLO(weights_path)
    if device is not None:
        model.to(device)
    return model


def preferred_ort_providers() -> List[str]:
    """Pick the best available ONNX Runtime execution provider.

    Prefers a GPU provider when the matching onnxruntime build is installed
    (DirectML on Windows / ROCm on AMD Linux / CUDA on NVIDIA), else CPU. This
    lets the AMD Radeon 8060S accelerate inference with no code change once
    `onnxruntime-directml` (Windows) or `onnxruntime-rocm` (Linux) is installed.
    """
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:
        return ["CPUExecutionProvider"]
    for p in ("DmlExecutionProvider", "ROCMExecutionProvider", "CUDAExecutionProvider"):
        if p in avail:
            return [p, "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_emotion_model(model_name: str = "enet_b0_8_best_vgaf",
                       providers: Optional[List[str]] = None):
    """Load the hsemotion ONNX recognizer, using a GPU execution provider when
    available. `providers` overrides the auto-selection."""
    path = ensure_emotion_model(model_name)
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    rec = HSEmotionRecognizer(model_name=model_name)
    providers = providers or preferred_ort_providers()
    # hsemotion hardcodes CPU; rebuild the session if a GPU provider is on offer.
    if providers != ["CPUExecutionProvider"]:
        import onnxruntime as ort
        rec.ort_session = ort.InferenceSession(str(path), providers=providers)
    return rec


def emotion_class_order(recognizer) -> List[str]:
    """Return the recognizer's class labels lowercased, in its native index order."""
    idx_to_class: Dict[int, str] = recognizer.idx_to_class
    return [idx_to_class[i].lower() for i in range(len(idx_to_class))]


def detect_faces(model, frame_bgr: np.ndarray, conf_thresh: float,
                 iou_thresh: float, imgsz: int) -> List[Tuple[Box, float]]:
    """Return [(clamped_box, det_conf), ...] for one BGR frame."""
    h, w = frame_bgr.shape[:2]
    results = model(frame_bgr, conf=conf_thresh, iou=iou_thresh,
                    imgsz=imgsz, verbose=False)
    out: List[Tuple[Box, float]] = []
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        for box, conf in zip(xyxy, confs):
            cb = clamp_box((box[0], box[1], box[2], box[3]), w, h)
            if cb[2] > cb[0] and cb[3] > cb[1]:
                out.append((cb, float(conf)))
    return out


def classify_crop(recognizer, crop_bgr: np.ndarray) -> List[float]:
    """Return an 8-prob vector aligned to EMOTIONS (reordered from model order)."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    _, scores = recognizer.predict_emotions(rgb, logits=False)
    order = emotion_class_order(recognizer)
    prob = [0.0] * len(EMOTIONS)
    for i, s in enumerate(scores):
        prob[EMOTIONS.index(order[i])] = float(s)
    return prob
