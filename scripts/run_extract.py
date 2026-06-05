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
