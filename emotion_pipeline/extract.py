from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from emotion_pipeline import EMOTIONS
from emotion_pipeline.classification import classify_with_abstain
from emotion_pipeline.config import PipelineConfig
from emotion_pipeline.detection import (
    box_area_frac, expand_to_square_with_margin, passes_min_size,
)

Box = Tuple[int, int, int, int]
Frame = Tuple[int, float, np.ndarray]
Detector = Callable[[np.ndarray], List[Tuple[Box, float]]]
Classifier = Callable[[np.ndarray], Sequence[float]]


def extract_video_faces(video_id: str, frames: List[Frame],
                        detector: Detector, classifier: Classifier,
                        cfg: PipelineConfig) -> List[Dict]:
    """Produce face-level records for one video. Pure orchestration: detector and
    classifier are injected so this is unit-testable without the heavy models.
    """
    records: List[Dict] = []
    for frame_idx, t_sec, bgr in frames:
        h, w = bgr.shape[:2]
        for face_no, (box, det_conf) in enumerate(detector(bgr)):
            if not passes_min_size(box, w, h, cfg.min_face_px, cfg.min_face_area_frac):
                continue
            cx1, cy1, cx2, cy2 = expand_to_square_with_margin(box, w, h, cfg.crop_margin)
            crop = bgr[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            probs = classifier(crop)
            label, argmax, max_prob = classify_with_abstain(probs, cfg.abstain_tau)
            rec = {
                "video_id": video_id, "frame_idx": frame_idx, "t_sec": t_sec,
                "face_id": face_no, "x1": box[0], "y1": box[1], "x2": box[2],
                "y2": box[3], "det_conf": round(det_conf, 4),
                "face_area_frac": round(box_area_frac(box, w, h), 6),
                "argmax_emotion": argmax, "max_prob": round(max_prob, 4),
                "label": label,
            }
            for e, p in zip(EMOTIONS, probs):
                rec[f"p_{e}"] = round(float(p), 4)
            records.append(rec)
    return records


def build_manifest_row(video_id: str, source_path: Path, sha256: str,
                       duration_s: float, src_fps: float, width: int, height: int,
                       n_target_samples: int, cfg: PipelineConfig) -> Dict:
    import cv2
    try:
        import ultralytics
        ultra_v = ultralytics.__version__
    except Exception:
        ultra_v = "unknown"
    return {
        "video_id": video_id, "source_path": str(source_path), "sha256": sha256,
        "duration_s": round(duration_s, 3), "src_fps": round(src_fps, 4),
        "width": width, "height": height, "n_target_samples": n_target_samples,
        "fps_target": cfg.fps_target, "yolo_weights": cfg.yolo_weights,
        "emotion_model": cfg.emotion_model, "conf_thresh": cfg.conf_thresh,
        "iou_thresh": cfg.iou_thresh, "min_face_px": cfg.min_face_px,
        "min_face_area_frac": cfg.min_face_area_frac, "crop_margin": cfg.crop_margin,
        "abstain_tau": cfg.abstain_tau, "frame_emotion_method": cfg.frame_emotion_method,
        "cv2_version": cv2.__version__, "ultralytics_version": ultra_v,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }
