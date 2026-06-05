from typing import Dict, List, Tuple

import cv2
import numpy as np

from emotion_pipeline import EMOTIONS
from emotion_pipeline.detection import clamp_box

Box = Tuple[int, int, int, int]


def load_face_model(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)


def load_emotion_model(model_name: str = "enet_b2_8"):
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
    return HSEmotionRecognizer(model_name=model_name)


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
