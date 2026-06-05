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

## Models
- **Face detection:** `models/yolov12m-face.pt` (Ultralytics YOLO). ~90% frame face
  coverage on test footage — no change needed.
- **Emotion classification:** `enet_b0_8_best_vgaf` (HSEmotion, 8-class AffectNet
  labels, trained on in-the-wild video). Chosen over the static-image `enet_b2_8`
  after an A/B on 400 real faces: committed-label rate **54% vs 12%** at the same
  `abstain_tau=0.40`, i.e. far fewer `uncertain` faces with no threshold fudging.
  The ONNX weights download automatically on first run into `~/.hsemotion/`.

## Setup (fresh machine)
```bash
git clone <your-repo-url> emotion_extraction
cd emotion_extraction
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Put the YOLO face weights here (not stored in git):
#   models/yolov12m-face.pt
# Put the source videos in:
#   videos/   (filenames must contain the YouTube id in [brackets])

python -m scripts.setup_check        # verifies deps, GPU providers, models, weights
```

### GPU acceleration (AMD Ryzen AI Max+ 395 / Radeon 8060S)
The emotion model runs on ONNX Runtime and will **auto-use a GPU provider** when the
matching build is installed (no code change). Swap the CPU runtime for one of:
```bash
pip uninstall -y onnxruntime
pip install onnxruntime-directml      # Windows (drives the Radeon 8060S)
# or, on Linux:  pip install onnxruntime-rocm
```
`scripts.setup_check` prints which provider will be used. The YOLO detector device is
set with `--device` on `run_extract` (e.g. `--device cpu`); on this 16-core CPU the
detector is already fast, and CPU is the reliable default.

## Run order
```bash
# 1. Extract face-level raw data + manifest (resumable; skips finished videos)
python -m scripts.run_extract --videos videos --out out/faces \
       --weights models/yolov12m-face.pt

# 2. Aggregate to frames + films_summary + per-film counts + a combined corpus
python -m scripts.run_aggregate --faces out/faces --out out --method mean_softmax \
       [--labels labels.csv]

# 3a. Sample faces to hand-label, then fill the 'human_label' column
python -m scripts.run_validate sample --faces out/faces --out out/to_label.csv --n 300
# 3b. Score the classifier against your hand labels (agreement + confusion matrix)
python -m scripts.run_validate score --labeled out/to_label.csv --out out/report.txt
```

## Outputs
Per-film raw (in `out/faces/`):
- `<video_id>_faces.parquet` — raw per-face rows (full 8-class softmax, boxes, conf)
- `<video_id>_grid.parquet` — the sampled-frame grid (incl. no-face frames)
- `manifest.csv` — provenance per video

Aggregated (in `out/`):
- `frames.parquet` — one row per sampled frame, all films
- `films_summary.csv` — modeling table (proportions, entropy, coverage flags)
- `films_emotion_counts_face.csv` — one row per film: **count of each per-face label**
  (incl. `uncertain`)
- `films_emotion_counts_frame.csv` — one row per film: **count of each per-frame
  emotion** (with `no_face` broken out)

Combined corpus (in `out/combined/`) — every output unified into a single file each:
- `all_faces.parquet`, `all_frames.parquet`, `films_summary.csv`,
  `films_emotion_counts_face.csv`, `films_emotion_counts_frame.csv`

## Per-face vs per-frame counts
Each sampled **frame** can contain several **faces**. The per-face table counts every
detected face (honest raw data, but crowd scenes dominate). The per-frame table first
collapses all faces in a frame to one emotion (`mean_softmax`), so every half-second of
film counts once — comparable across films regardless of how crowded each is. The
per-frame table is the recommended modeling unit.

## Performance note (for the full 677-film run)
Sampling seeks by timestamp (`cv2 POS_MSEC`) for VFR-safety and holds a film's sampled
frames in memory. On the 128 GB target machine memory is a non-issue; throughput is
bound by detector+classifier inference per frame. The run is **resumable** — rerun the
same command and it skips any video whose `_faces.parquet` already exists.

## Tests
```bash
python -m pytest -v
```
