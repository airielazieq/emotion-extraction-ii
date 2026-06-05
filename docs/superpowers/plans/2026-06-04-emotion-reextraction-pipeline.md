# Emotion Re-Extraction Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-extract per-frame facial emotions from the 677 original short films at a true 2 fps, producing a reproducible, provenance-tracked, face-level raw dataset plus per-frame and per-film aggregations — and a validation report quantifying the classifier's measurement error.

**Architecture:** A small Python package (`emotion_pipeline/`) split into pure, unit-testable logic modules (ids, sampling, detection geometry, classification post-processing, aggregation, validation) and thin model-I/O wrappers. A face-level "raw" table is captured first; per-frame and per-film views are derived in a separate aggregation step so the unit-of-analysis decision (per-frame vs per-face) is reversible and explicit. Everything is keyed by the stable YouTube video ID, never by sanitized folder names. CLI scripts orchestrate batch extraction (resumable), aggregation, and validation.

**Tech Stack:** Python 3.13, OpenCV (`cv2`), Ultralytics YOLO (face weights), `hsemotion-onnx` (EfficientNet-B2 AffectNet), pandas + pyarrow (parquet), scikit-learn (confusion matrix), pytest.

---

## Canonical Schema (referenced by every task — do not deviate)

`EMOTIONS` order is fixed everywhere:
```python
EMOTIONS = ["anger", "contempt", "disgust", "fear",
            "happiness", "neutral", "sadness", "surprise"]
```

**`faces.parquet`** — one row per detected face per sampled frame (the raw measurement):
`video_id, frame_idx, t_sec, face_id, x1, y1, x2, y2, det_conf, face_area_frac,`
`p_anger, p_contempt, p_disgust, p_fear, p_happiness, p_neutral, p_sadness, p_surprise,`
`argmax_emotion, max_prob, label` (label = argmax_emotion, or `"uncertain"` if max_prob < tau)

**`frames.parquet`** — one row per sampled frame:
`video_id, frame_idx, t_sec, n_faces, no_face (bool), frame_emotion, frame_max_prob`

**`films_summary.csv`** — one row per film:
`video_id, n_frames_sampled, n_frames_face, face_rate, n_faces, faces_per_faced_frame,`
`prop_<emotion> (×8), dominant_emotion, entropy_bits, norm_entropy, non_neutral_share,`
`uncertain_frame_rate, label_change_rate, low_coverage_flag`

**`manifest.csv`** — one row per video (provenance):
`video_id, source_path, sha256, duration_s, src_fps, width, height, n_target_samples,`
`fps_target, yolo_weights, emotion_model, conf_thresh, iou_thresh, min_face_px,`
`min_face_area_frac, crop_margin, abstain_tau, frame_emotion_method,`
`cv2_version, ultralytics_version, run_timestamp`

**`labels.csv`** (supplied/maintained separately, joined downstream):
`video_id, likes, views, subs, publish_date, scrape_timestamp`

---

## File Structure

- `emotion_pipeline/__init__.py` — package marker, exports `EMOTIONS`
- `emotion_pipeline/config.py` — `PipelineConfig` dataclass (all thresholds/versions)
- `emotion_pipeline/ids.py` — `parse_video_id`, `file_sha256`
- `emotion_pipeline/sampling.py` — `target_timestamps`, `frame_indices_for_timestamps`, `read_frames_at_timestamps`
- `emotion_pipeline/detection.py` — `clamp_box`, `box_area_frac`, `passes_min_size`, `expand_to_square_with_margin`
- `emotion_pipeline/classification.py` — `classify_with_abstain`, `faces_to_records`
- `emotion_pipeline/models.py` — `load_face_model`, `load_emotion_model`, `detect_faces`, `classify_crop` (thin I/O wrappers)
- `emotion_pipeline/extract.py` — `extract_video_faces`, `build_manifest_row`
- `emotion_pipeline/aggregate.py` — `frame_emotion_from_faces`, `faces_to_frames`, `film_summary`
- `emotion_pipeline/validate.py` — `confusion`, `agreement`, `label_change_rate`, `coverage_report`
- `scripts/run_extract.py` — batch CLI (resumable, keyed by video_id)
- `scripts/run_aggregate.py` — faces→frames→films_summary, join labels
- `scripts/run_validate.py` — sample faces for hand-labeling + score a completed label file
- `tests/test_*.py` — one per logic module
- `requirements.txt`, `README.md`

---

### Task 0: Project scaffold, config, and constants

**Files:**
- Create: `requirements.txt`
- Create: `emotion_pipeline/__init__.py`
- Create: `emotion_pipeline/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Initialize git and create requirements.txt**

Run:
```bash
git init
```

Create `requirements.txt`:
```
opencv-python>=4.9
ultralytics>=8.2
hsemotion-onnx>=0.3
pandas>=2.2
pyarrow>=15
scikit-learn>=1.4
pytest>=8.0
numpy>=1.26
```

- [ ] **Step 2: Write the failing test for the package + config**

Create `tests/test_config.py`:
```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'emotion_pipeline'`

- [ ] **Step 4: Create the package and config**

Create `emotion_pipeline/__init__.py`:
```python
EMOTIONS = ["anger", "contempt", "disgust", "fear",
            "happiness", "neutral", "sadness", "surprise"]
```

Create `emotion_pipeline/config.py`:
```python
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
    emotion_model: str = "enet_b2_8"
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 6: Commit**

```bash
git add requirements.txt emotion_pipeline/__init__.py emotion_pipeline/config.py tests/test_config.py
git commit -m "feat: scaffold emotion_pipeline package with config and constants"
```

---

### Task 1: Stable IDs and file hashing

**Files:**
- Create: `emotion_pipeline/ids.py`
- Test: `tests/test_ids.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ids.py`:
```python
from pathlib import Path
from emotion_pipeline.ids import parse_video_id, file_sha256


def test_parse_video_id_extracts_bracketed_youtube_id():
    assert parse_video_id("CHAI Horror Short [hurjP3jlGbI].mp4") == "hurjP3jlGbI"
    assert parse_video_id("2 RUNNERS SHORT FILM [YbM1nNleeZ0]") == "YbM1nNleeZ0"
    assert parse_video_id("id with dashes [-fTjFT53d-k].webm") == "-fTjFT53d-k"


def test_parse_video_id_returns_none_when_absent():
    assert parse_video_id("Some Movie Without An Id.mp4") is None
    assert parse_video_id("truncated_name_7a65b589") is None  # 8-char hash, not an id


def test_file_sha256_is_stable(tmp_path: Path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"hello world")
    h1 = file_sha256(f)
    h2 = file_sha256(f)
    assert h1 == h2
    assert len(h1) == 64
    assert h1 == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_ids.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'emotion_pipeline.ids'`

- [ ] **Step 3: Write the implementation**

Create `emotion_pipeline/ids.py`:
```python
import hashlib
import re
from pathlib import Path
from typing import Optional

# YouTube IDs are exactly 11 chars from [A-Za-z0-9_-], wrapped in square brackets.
_YT_ID_RE = re.compile(r"\[([A-Za-z0-9_-]{11})\]")


def parse_video_id(filename: str) -> Optional[str]:
    """Extract the 11-char YouTube id from a filename, or None if not present."""
    m = _YT_ID_RE.search(filename)
    return m.group(1) if m else None


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    """Stream a file through SHA-256 and return the hex digest."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_ids.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add emotion_pipeline/ids.py tests/test_ids.py
git commit -m "feat: video-id parsing and file hashing for provenance"
```

---

### Task 2: Timestamp-grid frame sampling (VFR-safe logic)

**Files:**
- Create: `emotion_pipeline/sampling.py`
- Test: `tests/test_sampling.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sampling.py`:
```python
from emotion_pipeline.sampling import target_timestamps, frame_indices_for_timestamps


def test_target_timestamps_fixed_grid():
    assert target_timestamps(2.0, 2) == [0.0, 0.5, 1.0, 1.5, 2.0]
    assert target_timestamps(1.0, 2) == [0.0, 0.5, 1.0]


def test_target_timestamps_edge_cases():
    assert target_timestamps(0.0, 2) == []
    assert target_timestamps(-5.0, 2) == []
    assert target_timestamps(3.0, 0) == []


def test_target_timestamps_independent_of_source_fps():
    # The grid is identical regardless of source fps -> cross-film comparability
    a = target_timestamps(10.0, 2)
    b = target_timestamps(10.0, 2)
    assert a == b
    assert all(abs((a[i + 1] - a[i]) - 0.5) < 1e-9 for i in range(len(a) - 1))


def test_frame_indices_round_to_nearest_source_frame():
    ts = [0.0, 0.5, 1.0]
    assert frame_indices_for_timestamps(ts, 30.0) == [0, 15, 30]
    assert frame_indices_for_timestamps(ts, 25.0) == [0, 13, 25]  # 0.5*25=12.5 -> 13
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sampling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'emotion_pipeline.sampling'`

- [ ] **Step 3: Write the implementation**

Create `emotion_pipeline/sampling.py`:
```python
import math
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def target_timestamps(duration_s: float, fps_target: int) -> List[float]:
    """Fixed time grid at fps_target, inclusive of t=0 and the final whole step.

    Independent of the source video fps, so every film is sampled on the same
    0.5s (for fps_target=2) grid -> counts are comparable across films.
    """
    if duration_s <= 0 or fps_target <= 0:
        return []
    step = 1.0 / fps_target
    n = int(math.floor(duration_s / step)) + 1
    return [round(i * step, 4) for i in range(n)]


def frame_indices_for_timestamps(timestamps: List[float], src_fps: float) -> List[int]:
    """Nearest source-frame index for each target timestamp (for CFR seeking)."""
    return [int(round(t * src_fps)) for t in timestamps]


def read_frames_at_timestamps(
    video_path: Path, fps_target: int
) -> Tuple[List[Tuple[int, float, np.ndarray]], float, float, int, int]:
    """Seek to each target timestamp via POS_MSEC (handles variable frame rate).

    Returns (frames, src_fps, duration_s, width, height) where each frame is
    (frame_idx, t_sec, bgr_ndarray). Frames that fail to decode are skipped.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = (total_f / src_fps) if src_fps else 0.0

    frames: List[Tuple[int, float, np.ndarray]] = []
    for t in target_timestamps(duration_s, fps_target):
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        idx = int(round(t * src_fps))
        frames.append((idx, round(t, 4), frame))
    cap.release()
    return frames, float(src_fps), float(duration_s), width, height
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sampling.py -v`
Expected: PASS (4 passed). `read_frames_at_timestamps` is exercised in the Task 12 integration test.

- [ ] **Step 5: Commit**

```bash
git add emotion_pipeline/sampling.py tests/test_sampling.py
git commit -m "feat: VFR-safe timestamp-grid frame sampling"
```

---

### Task 3: Detection geometry (size filter + margined square crop)

**Files:**
- Create: `emotion_pipeline/detection.py`
- Test: `tests/test_detection.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_detection.py`:
```python
from emotion_pipeline.detection import (
    clamp_box, box_area_frac, passes_min_size, expand_to_square_with_margin,
)


def test_clamp_box_clips_to_frame():
    assert clamp_box((-5, -5, 50, 60), 100, 100) == (0, 0, 50, 60)
    assert clamp_box((10, 10, 200, 300), 100, 100) == (10, 10, 100, 100)


def test_box_area_frac():
    assert abs(box_area_frac((0, 0, 10, 10), 100, 100) - 0.01) < 1e-9


def test_passes_min_size_filters_small_and_thin_faces():
    # 50x50 face in 1000x1000 frame: side 50 >= 40, area 0.0025 < 0.003 -> fail
    assert passes_min_size((0, 0, 50, 50), 1000, 1000, 40, 0.003) is False
    # 100x100 face: side 100 >= 40, area 0.01 >= 0.003 -> pass
    assert passes_min_size((0, 0, 100, 100), 1000, 1000, 40, 0.003) is True
    # 30px side fails the pixel floor regardless of area frac
    assert passes_min_size((0, 0, 30, 200), 1000, 1000, 40, 0.0) is False


def test_expand_to_square_with_margin_is_square_and_clamped():
    box = (40, 40, 60, 80)  # 20 wide, 40 tall, center (50,60)
    x1, y1, x2, y2 = expand_to_square_with_margin(box, 200, 200, margin=0.25)
    w, h = x2 - x1, y2 - y1
    assert abs(w - h) <= 1            # square (within rounding)
    assert w >= 40                    # max side 40 * 1.25 = 50
    # near a border it stays inside the frame
    bx = expand_to_square_with_margin((0, 0, 30, 30), 100, 100, margin=0.5)
    assert bx[0] >= 0 and bx[1] >= 0 and bx[2] <= 100 and bx[3] <= 100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'emotion_pipeline.detection'`

- [ ] **Step 3: Write the implementation**

Create `emotion_pipeline/detection.py`:
```python
from typing import Tuple

Box = Tuple[int, int, int, int]


def clamp_box(box: Box, w: int, h: int) -> Box:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return (x1, y1, x2, y2)


def box_area_frac(box: Box, w: int, h: int) -> float:
    x1, y1, x2, y2 = box
    return ((x2 - x1) * (y2 - y1)) / float(max(w * h, 1))


def passes_min_size(box: Box, w: int, h: int,
                    min_px: int, min_area_frac: float) -> bool:
    """Reject tiny/background/thin faces that produce garbage emotion labels."""
    x1, y1, x2, y2 = box
    short_side = min(x2 - x1, y2 - y1)
    return short_side >= min_px and box_area_frac(box, w, h) >= min_area_frac


def expand_to_square_with_margin(box: Box, w: int, h: int,
                                 margin: float = 0.25) -> Box:
    """Expand to a square box with margin (AffectNet models expect margined crops)."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    half = max(x2 - x1, y2 - y1) * (1.0 + margin) / 2.0
    new = (round(cx - half), round(cy - half), round(cx + half), round(cy + half))
    return clamp_box(new, w, h)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_detection.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add emotion_pipeline/detection.py tests/test_detection.py
git commit -m "feat: face-box size filter and margined square crop geometry"
```

---

### Task 4: Classification post-processing (abstain, never silent-neutral)

**Files:**
- Create: `emotion_pipeline/classification.py`
- Test: `tests/test_classification.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_classification.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_classification.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'emotion_pipeline.classification'`

- [ ] **Step 3: Write the implementation**

Create `emotion_pipeline/classification.py`:
```python
from typing import List, Sequence, Tuple

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_classification.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add emotion_pipeline/classification.py tests/test_classification.py
git commit -m "feat: confidence-based abstain instead of silent neutral fallback"
```

---

### Task 5: Model I/O wrappers (probability-preserving, order-checked)

**Files:**
- Create: `emotion_pipeline/models.py`
- Test: `tests/test_models.py`

- [ ] **Step 1: Write the failing test (import + skip-if-unavailable smoke)**

Create `tests/test_models.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'emotion_pipeline.models'`

- [ ] **Step 3: Write the implementation**

Create `emotion_pipeline/models.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_models.py -v`
Expected: PASS (1 passed, 1 may skip if deps absent)

- [ ] **Step 5: Commit**

```bash
git add emotion_pipeline/models.py tests/test_models.py
git commit -m "feat: model wrappers preserving full softmax with class-order guard"
```

---

### Task 6: Extraction orchestrator + manifest (dependency-injected, testable)

**Files:**
- Create: `emotion_pipeline/extract.py`
- Test: `tests/test_extract.py`

- [ ] **Step 1: Write the failing test (fakes for detector/classifier)**

Create `tests/test_extract.py`:
```python
import numpy as np
from emotion_pipeline import EMOTIONS
from emotion_pipeline.config import PipelineConfig
from emotion_pipeline.extract import extract_video_faces


def _frame(h=1000, w=1000):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_extract_records_have_full_schema_and_filter_small_faces():
    cfg = PipelineConfig(min_face_px=40, min_face_area_frac=0.003, abstain_tau=0.4)
    frames = [(0, 0.0, _frame()), (15, 0.5, _frame())]

    # One big face (passes) + one tiny face (filtered) per frame.
    def fake_detector(bgr):
        return [((100, 100, 300, 300), 0.9), ((0, 0, 20, 20), 0.8)]

    # Big face -> happiness 0.8; the tiny face is filtered before classify.
    def fake_classifier(crop):
        v = [0.0] * 8
        v[EMOTIONS.index("happiness")] = 0.8
        v[EMOTIONS.index("neutral")] = 0.2
        return v

    recs = extract_video_faces("vid123", frames, fake_detector, fake_classifier, cfg)
    assert len(recs) == 2  # one kept face per frame
    r = recs[0]
    for col in ["video_id", "frame_idx", "t_sec", "face_id", "x1", "y1", "x2", "y2",
                "det_conf", "face_area_frac", "argmax_emotion", "max_prob", "label",
                *[f"p_{e}" for e in EMOTIONS]]:
        assert col in r
    assert r["video_id"] == "vid123"
    assert r["argmax_emotion"] == "happiness"
    assert r["label"] == "happiness"


def test_extract_abstains_on_low_confidence():
    cfg = PipelineConfig(abstain_tau=0.5)
    frames = [(0, 0.0, _frame())]

    def fake_detector(bgr):
        return [((100, 100, 400, 400), 0.9)]

    def fake_classifier(crop):
        return [0.2, 0.18, 0.15, 0.12, 0.1, 0.1, 0.08, 0.07]  # max 0.2 < 0.5

    recs = extract_video_faces("v", frames, fake_detector, fake_classifier, cfg)
    assert recs[0]["label"] == "uncertain"
    assert recs[0]["argmax_emotion"] == "anger"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'emotion_pipeline.extract'`

- [ ] **Step 3: Write the implementation**

Create `emotion_pipeline/extract.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_extract.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add emotion_pipeline/extract.py tests/test_extract.py
git commit -m "feat: dependency-injected face extraction orchestrator + manifest"
```

---

### Task 7: Aggregate faces → frames (explicit unit-of-analysis choice)

**Files:**
- Create: `emotion_pipeline/aggregate.py`
- Test: `tests/test_aggregate_frames.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_aggregate_frames.py`:
```python
import pandas as pd
from emotion_pipeline import EMOTIONS
from emotion_pipeline.aggregate import frame_emotion_from_faces, faces_to_frames


def _face(video_id, frame_idx, t, face_id, area, **probs):
    rec = {"video_id": video_id, "frame_idx": frame_idx, "t_sec": t,
           "face_id": face_id, "face_area_frac": area}
    for e in EMOTIONS:
        rec[f"p_{e}"] = probs.get(e, 0.0)
    rec["label"] = max(EMOTIONS, key=lambda e: rec[f"p_{e}"])
    return rec


def test_mean_softmax_collapses_multiple_faces_to_one_frame_emotion():
    faces = pd.DataFrame([
        _face("v", 0, 0.0, 0, 0.2, sadness=0.9),
        _face("v", 0, 0.0, 1, 0.5, happiness=0.6, neutral=0.4),
    ])
    emo, mp = frame_emotion_from_faces(faces, method="mean_softmax")
    # mean: sadness 0.45 vs happiness 0.30 vs neutral 0.20 -> sadness
    assert emo == "sadness"
    assert 0.0 < mp <= 1.0


def test_dominant_largest_uses_biggest_face():
    faces = pd.DataFrame([
        _face("v", 0, 0.0, 0, 0.2, sadness=0.9),
        _face("v", 0, 0.0, 1, 0.8, happiness=0.7),
    ])
    emo, _ = frame_emotion_from_faces(faces, method="dominant_largest")
    assert emo == "happiness"  # largest-area face wins


def test_faces_to_frames_one_row_per_frame_and_marks_no_face():
    faces = pd.DataFrame([
        _face("v", 0, 0.0, 0, 0.5, happiness=0.8),
        _face("v", 0, 0.0, 1, 0.3, happiness=0.7),
        _face("v", 15, 0.5, 0, 0.5, sadness=0.9),
    ])
    # frame 30 had no face -> supplied via sampled_frames
    sampled = [(0, 0.0), (15, 0.5), (30, 1.0)]
    frames = faces_to_frames(faces, sampled, "v", method="mean_softmax")
    assert len(frames) == 3
    no_face_row = frames[frames["frame_idx"] == 30].iloc[0]
    assert bool(no_face_row["no_face"]) is True
    assert no_face_row["n_faces"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aggregate_frames.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'emotion_pipeline.aggregate'`

- [ ] **Step 3: Write the implementation**

Create `emotion_pipeline/aggregate.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_aggregate_frames.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add emotion_pipeline/aggregate.py tests/test_aggregate_frames.py
git commit -m "feat: faces->frames aggregation with explicit per-frame methods"
```

---

### Task 8: Aggregate frames → film summary

**Files:**
- Modify: `emotion_pipeline/aggregate.py` (add `film_summary`)
- Test: `tests/test_aggregate_film.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_aggregate_film.py`:
```python
import math
import pandas as pd
from emotion_pipeline import EMOTIONS
from emotion_pipeline.aggregate import film_summary


def _frames(emotions, video_id="v"):
    rows = []
    for i, e in enumerate(emotions):
        rows.append({"video_id": video_id, "frame_idx": i * 15, "t_sec": i * 0.5,
                     "n_faces": 0 if e == "no_face" else 1,
                     "no_face": e == "no_face",
                     "frame_emotion": e, "frame_max_prob": 0.0 if e == "no_face" else 0.8})
    return pd.DataFrame(rows)


def test_film_summary_proportions_and_coverage():
    frames = _frames(["happiness", "happiness", "sadness", "no_face"])
    s = film_summary(frames, video_id="v", low_coverage_min_faced=2)
    assert s["video_id"] == "v"
    assert s["n_frames_sampled"] == 4
    assert s["n_frames_face"] == 3
    assert abs(s["face_rate"] - 0.75) < 1e-9
    # proportions computed over faced frames only
    assert abs(s["prop_happiness"] - (2 / 3)) < 1e-9
    assert abs(s["prop_sadness"] - (1 / 3)) < 1e-9
    assert s["dominant_emotion"] == "happiness"
    assert s["low_coverage_flag"] is False  # 3 faced >= 2


def test_film_summary_entropy_and_uncertain_rate():
    frames = _frames(["happiness", "sadness"])  # 50/50
    s = film_summary(frames, video_id="v", low_coverage_min_faced=100)
    assert abs(s["entropy_bits"] - 1.0) < 1e-6        # 2 equal classes -> 1 bit
    assert abs(s["norm_entropy"] - (1.0 / math.log2(8))) < 1e-6
    assert s["low_coverage_flag"] is True             # 2 faced < 100
    frames2 = _frames(["uncertain", "happiness"])
    s2 = film_summary(frames2, video_id="v", low_coverage_min_faced=1)
    assert abs(s2["uncertain_frame_rate"] - 0.5) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_aggregate_film.py -v`
Expected: FAIL — `ImportError: cannot import name 'film_summary'`

- [ ] **Step 3: Add the implementation to `emotion_pipeline/aggregate.py`**

Append to `emotion_pipeline/aggregate.py`:
```python
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
        "norm_entropy": round(entropy_bits / math.log2(len(EMOTIONS)), 4),
        "non_neutral_share": round(non_neutral, 4),
        "uncertain_frame_rate": round(uncertain_rate, 4),
        "label_change_rate": round(
            label_change_rate(faced["frame_emotion"].tolist()), 4),
        "low_coverage_flag": n_faced < low_coverage_min_faced,
    }
    return out
```

Add `import math` to the top of `emotion_pipeline/aggregate.py` if not present.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_aggregate_film.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add emotion_pipeline/aggregate.py tests/test_aggregate_film.py
git commit -m "feat: frames->film summary with entropy, coverage and jitter flags"
```

---

### Task 9: Validation utilities (measurement-error quantification)

**Files:**
- Create: `emotion_pipeline/validate.py`
- Test: `tests/test_validate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_validate.py`:
```python
from emotion_pipeline.validate import agreement, confusion, coverage_report
import pandas as pd


def test_agreement_simple():
    y_true = ["happiness", "sadness", "neutral", "fear"]
    y_pred = ["happiness", "sadness", "neutral", "anger"]
    assert abs(agreement(y_true, y_pred) - 0.75) < 1e-9


def test_confusion_counts_offdiagonal():
    y_true = ["fear", "fear", "surprise"]
    y_pred = ["fear", "surprise", "surprise"]
    cm = confusion(y_true, y_pred, labels=["fear", "surprise"])
    # rows=true, cols=pred
    assert cm.loc["fear", "fear"] == 1
    assert cm.loc["fear", "surprise"] == 1
    assert cm.loc["surprise", "surprise"] == 1


def test_coverage_report_flags_low_face_films():
    films = pd.DataFrame({
        "video_id": ["a", "b", "c"],
        "n_frames_face": [500, 80, 20],
        "face_rate": [0.7, 0.5, 0.1],
        "uncertain_frame_rate": [0.05, 0.2, 0.6],
    })
    rep = coverage_report(films, min_faced=100, max_uncertain=0.3)
    assert set(rep[rep["flagged"]]["video_id"]) == {"b", "c"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'emotion_pipeline.validate'`

- [ ] **Step 3: Write the implementation**

Create `emotion_pipeline/validate.py`:
```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_validate.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add emotion_pipeline/validate.py tests/test_validate.py
git commit -m "feat: validation utilities for agreement, confusion, coverage"
```

---

### Task 10: Batch extraction CLI (resumable, keyed by video_id)

**Files:**
- Create: `scripts/run_extract.py`
- Test: `tests/test_run_extract.py`

- [ ] **Step 1: Write the failing test for the resume helper**

Create `tests/test_run_extract.py`:
```python
from pathlib import Path
from scripts.run_extract import already_done, faces_path_for


def test_faces_path_for_uses_video_id(tmp_path: Path):
    p = faces_path_for(tmp_path, "abc12345678")
    assert p.name == "abc12345678_faces.parquet"
    assert p.parent == tmp_path


def test_already_done_detects_existing_output(tmp_path: Path):
    vid = "abc12345678"
    assert already_done(tmp_path, vid) is False
    faces_path_for(tmp_path, vid).write_bytes(b"x")
    assert already_done(tmp_path, vid) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_extract'`

- [ ] **Step 3: Write the implementation**

Create `scripts/__init__.py` (empty file).

Create `scripts/run_extract.py`:
```python
"""Batch face-level emotion extraction, keyed by stable YouTube video_id.

Usage:
  python -m scripts.run_extract --videos DIR --out DIR/faces [--weights models/yolov12m-face.pt]
Resumable: skips any video whose <video_id>_faces.parquet already exists.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from emotion_pipeline.config import PipelineConfig
from emotion_pipeline.extract import build_manifest_row, extract_video_faces
from emotion_pipeline.ids import file_sha256, parse_video_id
from emotion_pipeline.sampling import read_frames_at_timestamps

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def faces_path_for(out_dir: Path, video_id: str) -> Path:
    return out_dir / f"{video_id}_faces.parquet"


def already_done(out_dir: Path, video_id: str) -> bool:
    return faces_path_for(out_dir, video_id).exists()


def discover_videos(videos_dir: Path):
    return sorted([p for p in videos_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
                  key=lambda p: p.name.lower())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--weights", default="models/yolov12m-face.pt")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = PipelineConfig(yolo_weights=args.weights)

    from emotion_pipeline.models import (
        classify_crop, detect_faces, load_emotion_model, load_face_model,
    )
    face_model = load_face_model(args.weights)
    emo = load_emotion_model(cfg.emotion_model)

    manifest_rows = []
    videos = discover_videos(args.videos)
    for i, vpath in enumerate(videos, 1):
        vid = parse_video_id(vpath.name) or vpath.stem
        if args.skip_existing and already_done(args.out, vid):
            print(f"[{i}/{len(videos)}] skip {vid} (exists)")
            continue
        print(f"[{i}/{len(videos)}] {vid}  {vpath.name}")
        frames, src_fps, dur, w, h = read_frames_at_timestamps(vpath, cfg.fps_target)

        def detector(bgr):
            return detect_faces(face_model, bgr, cfg.conf_thresh,
                                cfg.iou_thresh, cfg.detector_imgsz)

        def classifier(crop):
            return classify_crop(emo, crop)

        recs = extract_video_faces(vid, frames, detector, classifier, cfg)
        pd.DataFrame(recs).to_parquet(faces_path_for(args.out, vid), index=False)

        # frame grid (incl. no-face frames) needed for aggregation later
        pd.DataFrame([{"video_id": vid, "frame_idx": fi, "t_sec": t}
                      for fi, t, _ in frames]).to_parquet(
            args.out / f"{vid}_grid.parquet", index=False)

        manifest_rows.append(build_manifest_row(
            vid, vpath, file_sha256(vpath), dur, src_fps, w, h, len(frames), cfg))

    if manifest_rows:
        man_path = args.out / "manifest.csv"
        df = pd.DataFrame(manifest_rows)
        if man_path.exists():
            df = pd.concat([pd.read_csv(man_path), df]).drop_duplicates(
                "video_id", keep="last")
        df.to_csv(man_path, index=False)
        print(f"[manifest] wrote {man_path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_extract.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/__init__.py scripts/run_extract.py tests/test_run_extract.py
git commit -m "feat: resumable batch extraction CLI keyed by video id"
```

---

### Task 11: Aggregation CLI (faces → frames → films_summary + label join)

**Files:**
- Create: `scripts/run_aggregate.py`
- Test: `tests/test_run_aggregate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_aggregate.py`:
```python
from pathlib import Path
import pandas as pd
from emotion_pipeline import EMOTIONS
from scripts.run_aggregate import aggregate_one, join_labels


def _write_faces(out: Path, vid: str):
    rows = []
    for fi, t, emo in [(0, 0.0, "happiness"), (0, 0.0, "happiness"), (15, 0.5, "sadness")]:
        rec = {"video_id": vid, "frame_idx": fi, "t_sec": t, "face_id": 0,
               "face_area_frac": 0.5, "label": emo}
        for e in EMOTIONS:
            rec[f"p_{e}"] = 0.9 if e == emo else 0.0
        rows.append(rec)
    pd.DataFrame(rows).to_parquet(out / f"{vid}_faces.parquet", index=False)
    pd.DataFrame([{"video_id": vid, "frame_idx": 0, "t_sec": 0.0},
                  {"video_id": vid, "frame_idx": 15, "t_sec": 0.5},
                  {"video_id": vid, "frame_idx": 30, "t_sec": 1.0}]
                 ).to_parquet(out / f"{vid}_grid.parquet", index=False)


def test_aggregate_one_produces_frames_and_summary(tmp_path: Path):
    _write_faces(tmp_path, "vid1")
    frames, summary = aggregate_one(tmp_path, "vid1", method="mean_softmax")
    assert len(frames) == 3                  # incl. the no-face frame 30
    assert summary["n_frames_face"] == 2
    assert summary["dominant_emotion"] in ("happiness", "sadness")


def test_join_labels_attaches_likes():
    films = pd.DataFrame({"video_id": ["a", "b"], "dominant_emotion": ["fear", "neutral"]})
    labels = pd.DataFrame({"video_id": ["a", "b"], "likes": [100, 200]})
    merged = join_labels(films, labels)
    assert list(merged.sort_values("video_id")["likes"]) == [100, 200]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_aggregate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_aggregate'`

- [ ] **Step 3: Write the implementation**

Create `scripts/run_aggregate.py`:
```python
"""Aggregate face-level parquet outputs into frames + films_summary, then join labels.

Usage:
  python -m scripts.run_aggregate --faces DIR --out DIR [--labels labels.csv]
                                  [--method mean_softmax]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from emotion_pipeline.aggregate import faces_to_frames, film_summary


def aggregate_one(faces_dir: Path, video_id: str, method: str):
    faces = pd.read_parquet(faces_dir / f"{video_id}_faces.parquet")
    grid = pd.read_parquet(faces_dir / f"{video_id}_grid.parquet")
    sampled = list(zip(grid["frame_idx"].tolist(), grid["t_sec"].tolist()))
    frames = faces_to_frames(faces, sampled, video_id, method=method)
    summary = film_summary(frames, video_id)
    return frames, summary


def join_labels(films: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return films.merge(labels, on="video_id", how="left")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--method", default="mean_softmax")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    video_ids = sorted(p.name[:-len("_faces.parquet")]
                       for p in args.faces.glob("*_faces.parquet"))
    all_frames, summaries = [], []
    for vid in video_ids:
        frames, summary = aggregate_one(args.faces, vid, args.method)
        all_frames.append(frames)
        summaries.append(summary)
        print(f"[agg] {vid}: {summary['n_frames_face']} faced frames, "
              f"dominant={summary['dominant_emotion']}")

    if all_frames:
        pd.concat(all_frames, ignore_index=True).to_parquet(
            args.out / "frames.parquet", index=False)
    films = pd.DataFrame(summaries)
    if args.labels and args.labels.exists():
        films = join_labels(films, pd.read_csv(args.labels))
    films.to_csv(args.out / "films_summary.csv", index=False)
    print(f"[agg] wrote films_summary.csv ({len(films)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_aggregate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_aggregate.py tests/test_run_aggregate.py
git commit -m "feat: aggregation CLI building frames + films_summary with label join"
```

---

### Task 12: Validation CLI (sample faces for hand-labeling + score)

**Files:**
- Create: `scripts/run_validate.py`
- Test: `tests/test_run_validate.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_run_validate.py`:
```python
from pathlib import Path
import pandas as pd
from scripts.run_validate import sample_for_labeling, score_labels


def test_sample_for_labeling_is_deterministic_and_sized(tmp_path: Path):
    faces = pd.DataFrame({
        "video_id": ["v"] * 10,
        "frame_idx": list(range(10)),
        "face_id": [0] * 10,
        "argmax_emotion": ["happiness"] * 10,
        "max_prob": [0.9] * 10,
    })
    s1 = sample_for_labeling(faces, n=5, seed=42)
    s2 = sample_for_labeling(faces, n=5, seed=42)
    assert len(s1) == 5
    assert list(s1["frame_idx"]) == list(s2["frame_idx"])  # deterministic
    assert "human_label" in s1.columns                      # blank column to fill


def test_score_labels_reports_agreement_and_confusion():
    labeled = pd.DataFrame({
        "argmax_emotion": ["fear", "fear", "surprise", "happiness"],
        "human_label":    ["fear", "surprise", "surprise", "happiness"],
    })
    report = score_labels(labeled)
    assert abs(report["agreement"] - 0.75) < 1e-9
    assert report["confusion"].loc["fear", "surprise"] == 1
    assert report["n_labeled"] == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_run_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.run_validate'`

- [ ] **Step 3: Write the implementation**

Create `scripts/run_validate.py`:
```python
"""Validation: (1) sample a random face set for hand-labeling, (2) score a
completed label file to report classifier agreement + confusion matrix.

Usage:
  python -m scripts.run_validate sample --faces DIR --out sample_to_label.csv --n 300
  python -m scripts.run_validate score  --labeled sample_to_label.csv --out report.txt
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from emotion_pipeline import EMOTIONS
from emotion_pipeline.validate import agreement, confusion


def sample_for_labeling(faces: pd.DataFrame, n: int, seed: int = 42) -> pd.DataFrame:
    take = min(n, len(faces))
    s = faces.sample(n=take, random_state=seed).sort_values(
        ["video_id", "frame_idx", "face_id"]).reset_index(drop=True)
    s = s[["video_id", "frame_idx", "face_id", "argmax_emotion", "max_prob"]].copy()
    s["human_label"] = ""  # to be filled by a human annotator
    return s


def score_labels(labeled: pd.DataFrame) -> dict:
    df = labeled[labeled["human_label"].astype(str).str.len() > 0]
    y_true = df["human_label"].tolist()
    y_pred = df["argmax_emotion"].tolist()
    return {
        "n_labeled": int(len(df)),
        "agreement": agreement(y_true, y_pred),
        "confusion": confusion(y_true, y_pred, labels=EMOTIONS),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sample")
    sp.add_argument("--faces", required=True, type=Path)
    sp.add_argument("--out", required=True, type=Path)
    sp.add_argument("--n", type=int, default=300)
    sp.add_argument("--seed", type=int, default=42)
    sc = sub.add_parser("score")
    sc.add_argument("--labeled", required=True, type=Path)
    sc.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "sample":
        faces = pd.concat([pd.read_parquet(p) for p in args.faces.glob("*_faces.parquet")],
                          ignore_index=True)
        sample_for_labeling(faces, args.n, args.seed).to_csv(args.out, index=False)
        print(f"[validate] wrote {args.out} — fill the 'human_label' column, then 'score'")
    else:
        rep = score_labels(pd.read_csv(args.labeled))
        text = (f"n_labeled = {rep['n_labeled']}\n"
                f"agreement = {rep['agreement']:.4f}\n\n"
                f"confusion (rows=human, cols=model):\n{rep['confusion'].to_string()}\n")
        print(text)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_run_validate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add scripts/run_validate.py tests/test_run_validate.py
git commit -m "feat: validation CLI for hand-label sampling and scoring"
```

---

### Task 13: End-to-end integration test on a synthetic video

**Files:**
- Test: `tests/test_integration_end_to_end.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_integration_end_to_end.py`:
```python
import importlib.util
from pathlib import Path

import numpy as np
import pytest

HAVE_CV2 = importlib.util.find_spec("cv2") is not None


@pytest.mark.skipif(not HAVE_CV2, reason="cv2 not installed")
def test_sampling_reads_expected_frame_count(tmp_path: Path):
    import cv2
    from emotion_pipeline.sampling import read_frames_at_timestamps

    # Write a 3-second, 30fps synthetic video (solid colour frames).
    path = tmp_path / "clip [abcdefghijk].mp4"
    w, h, fps, secs = 320, 240, 30, 3
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(fps * secs):
        writer.write(np.full((h, w, 3), i % 255, dtype=np.uint8))
    writer.release()

    frames, src_fps, dur, vw, vh = read_frames_at_timestamps(path, fps_target=2)
    # ~3s at 2 fps -> ~6-7 frames; allow tolerance for container rounding
    assert 5 <= len(frames) <= 8
    assert vw == w and vh == h
    assert abs(src_fps - fps) < 2


@pytest.mark.skipif(not HAVE_CV2, reason="cv2 not installed")
def test_full_pipeline_with_fake_models(tmp_path: Path):
    """faces -> frames -> film_summary with injected fake detector/classifier."""
    from emotion_pipeline import EMOTIONS
    from emotion_pipeline.config import PipelineConfig
    from emotion_pipeline.extract import extract_video_faces
    from emotion_pipeline.aggregate import faces_to_frames, film_summary
    import pandas as pd

    cfg = PipelineConfig()
    frames_in = [(0, 0.0, np.zeros((500, 500, 3), np.uint8)),
                 (15, 0.5, np.zeros((500, 500, 3), np.uint8)),
                 (30, 1.0, np.zeros((500, 500, 3), np.uint8))]

    def det(bgr):
        return [((100, 100, 300, 300), 0.9)]

    def clf(crop):
        v = [0.0] * 8
        v[EMOTIONS.index("sadness")] = 0.85
        v[EMOTIONS.index("neutral")] = 0.15
        return v

    recs = extract_video_faces("vid", frames_in, det, clf, cfg)
    faces = pd.DataFrame(recs)
    sampled = [(0, 0.0), (15, 0.5), (30, 1.0)]
    frames = faces_to_frames(faces, sampled, "vid", method=cfg.frame_emotion_method)
    summary = film_summary(frames, "vid", low_coverage_min_faced=1)
    assert summary["n_frames_face"] == 3
    assert summary["dominant_emotion"] == "sadness"
    assert abs(summary["prop_sadness"] - 1.0) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails (or skips if cv2 absent)**

Run: `python -m pytest tests/test_integration_end_to_end.py -v`
Expected: FAIL initially if any wiring is wrong; PASS once Tasks 2/6/7/8 are correct. (Skips the cv2 test if OpenCV is not installed.)

- [ ] **Step 3: Fix any wiring issues surfaced, then re-run the full suite**

Run: `python -m pytest -v`
Expected: PASS (all tasks' tests green; model/cv2 tests may skip).

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration_end_to_end.py
git commit -m "test: end-to-end synthetic-video and fake-model integration"
```

---

### Task 14: README documenting the run order and design decisions

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write the README**

Create `README.md`:
```markdown
# Emotion Re-Extraction Pipeline

Re-extracts per-frame facial emotions from short films at a true 2 fps, with
provenance tracking and measurement-error validation.

## Why this exists / what was wrong before
- Old pipeline counted **one emotion per face**, so multi-face frames inflated
  counts. Here the raw unit is the face, but the **default film view is per-frame**
  (mean-softmax over faces) — the unit choice is explicit and reversible.
- Old pipeline silently mapped classifier errors to `neutral`. Here low-confidence
  faces become `uncertain` (see `abstain_tau`).
- Old pipeline sampled by `round(src_fps/2)` (drifts, breaks on VFR). Here we seek
  by **timestamp** on a fixed grid identical across all films.
- Old pipeline keyed outputs by truncated folder names. Here everything is keyed by
  the **stable YouTube video id**, with a `manifest.csv` recording file hash, fps,
  model versions, and thresholds.

## Run order
```bash
# 1. Extract face-level raw data + manifest (resumable)
python -m scripts.run_extract --videos path/to/videos --out out/faces \
       --weights models/yolov12m-face.pt

# 2. Aggregate to frames + films_summary, joining the likes label table
python -m scripts.run_aggregate --faces out/faces --out out --labels labels.csv \
       --method mean_softmax

# 3a. Sample faces to hand-label, then fill the 'human_label' column
python -m scripts.run_validate sample --faces out/faces --out out/to_label.csv --n 300
# 3b. Score the classifier against your hand labels
python -m scripts.run_validate score --labeled out/to_label.csv --out out/report.txt
```

## Outputs
- `out/faces/<video_id>_faces.parquet` — raw per-face rows (full 8-class softmax)
- `out/frames.parquet` — one row per sampled frame
- `out/films_summary.csv` — modeling table (proportions, entropy, coverage flags)
- `out/faces/manifest.csv` — provenance per video

## Tests
```bash
python -m pytest -v
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: pipeline README with run order and design rationale"
```

---

## Self-Review Notes (verified before handoff)

- **Spec coverage:** timestamp sampling (T2), min-face filter + margin crop (T3), full-softmax + abstain (T4/T5), face-level raw storage (T6), per-frame aggregation with explicit method (T7), film summary incl. entropy/coverage/jitter (T8), provenance manifest keyed by video id (T6/T10), label join (T11), hand-label validation + confusion matrix + jitter (T8/T9/T12), end-to-end test (T13). All methodology points covered.
- **Type/name consistency:** `EMOTIONS` order fixed in T0 and reused everywhere; `faces_to_frames`/`film_summary`/`frame_emotion_from_faces` signatures match across T7/T8/T11/T13; parquet column names match the canonical schema in every task.
- **No placeholders:** every code/test step contains complete runnable content.
- **Known external prerequisites (not code tasks):** `models/yolov12m-face.pt` weights file, the original 677 video files, and a maintained `labels.csv` (video_id→likes/views/…). These are inputs the engineer supplies at run time.
