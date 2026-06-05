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
