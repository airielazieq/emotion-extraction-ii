from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineConfig:
    fps_target: int = 2
    conf_thresh: float = 0.40          # YOLO face detection confidence
    iou_thresh: float = 0.45           # YOLO NMS IoU
    detector_imgsz: int = 640
    min_face_px: int = 40              # drop faces whose shorter side < this
    min_face_area_frac: float = 0.003  # drop faces smaller than 0.3% of frame
    crop_margin: float = 0.25          # expand box 25% before cropping
    abstain_tau: float = 0.40          # below this max-prob -> "uncertain"
    frame_emotion_method: str = "mean_softmax"  # or "dominant_largest" / "majority"
    yolo_weights: str = "yolov12m-face.pt"
    # enet_b0_8_best_vgaf is trained on in-the-wild video and is far better
    # calibrated on candid film footage than the static-image enet_b2_8
    # (committed-label rate 54% vs 12% at tau=0.40 on a 400-face A/B).
    emotion_model: str = "enet_b0_8_best_vgaf"
