"""Batch face-level emotion extraction, keyed by stable YouTube video_id.

Usage:
  python -m scripts.run_extract --videos DIR --out DIR/faces
         [--weights models/yolov12m-face.pt] [--device cpu]
         [--sampler timestamp|grab] [--shards N --shard-id K] [--threads T]

Resumable: skips any video whose <video_id>_faces.parquet already exists.

Parallelism: launch N processes, each with the SAME --shards N and a distinct
--shard-id 0..N-1; they process disjoint videos (round-robin) so there is no
double work and no write races. Each shard writes its own manifest_shardKK.csv.
"""
import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

from emotion_pipeline.config import PipelineConfig
from emotion_pipeline.extract import build_manifest_row, extract_video_faces
from emotion_pipeline.ids import file_sha256, parse_video_id
from emotion_pipeline.sampling import read_frames_at_timestamps, read_frames_grab

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def faces_path_for(out_dir: Path, video_id: str) -> Path:
    return out_dir / f"{video_id}_faces.parquet"


def already_done(out_dir: Path, video_id: str) -> bool:
    return faces_path_for(out_dir, video_id).exists()


def discover_videos(videos_dir: Path):
    return sorted([p for p in videos_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTS],
                  key=lambda p: p.name.lower())


def select_shard(items: List, shards: int, shard_id: int) -> List:
    """Round-robin partition: shard K gets items[K], items[K+shards], ...

    Disjoint across shards and together covering every item, so N parallel
    workers never process the same video twice.
    """
    if shards <= 1:
        return list(items)
    if not (0 <= shard_id < shards):
        raise ValueError(f"shard_id must be in [0,{shards})")
    return [x for i, x in enumerate(items) if i % shards == shard_id]


def _apply_thread_limits(threads):
    """Best-effort cap of intra-op threads so N parallel workers don't thrash."""
    if not threads:
        return
    try:
        import cv2
        cv2.setNumThreads(int(threads))
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(int(threads))
    except Exception:
        pass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--weights", default="models/yolov12m-face.pt")
    ap.add_argument("--device", default=None,
                    help="detector device for Ultralytics: cpu, cuda, 0, ... "
                         "(default: auto-select)")
    ap.add_argument("--sampler", choices=["timestamp", "grab"], default="timestamp",
                    help="timestamp=VFR-safe seek (default, validated); "
                         "grab=fast single-pass, assumes constant frame rate")
    ap.add_argument("--shards", type=int, default=1,
                    help="total number of parallel workers")
    ap.add_argument("--shard-id", type=int, default=0,
                    help="this worker's index in [0, shards)")
    ap.add_argument("--threads", type=int, default=None,
                    help="cap intra-op threads per worker (e.g. cores // shards)")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    _apply_thread_limits(args.threads)
    cfg = PipelineConfig(yolo_weights=args.weights)
    read_frames = read_frames_grab if args.sampler == "grab" else read_frames_at_timestamps

    from emotion_pipeline.models import (
        classify_crop, detect_faces, load_emotion_model, load_face_model,
    )
    face_model = load_face_model(args.weights, device=args.device)
    emo = load_emotion_model(cfg.emotion_model, intra_op_threads=args.threads)
    tag = f"shard {args.shard_id}/{args.shards}" if args.shards > 1 else "single"
    print(f"[setup] {tag}, sampler={args.sampler}, device={args.device or 'auto'}, "
          f"threads={args.threads or 'auto'}, emotion_model={cfg.emotion_model}")

    manifest_rows = []
    videos = select_shard(discover_videos(args.videos), args.shards, args.shard_id)
    for i, vpath in enumerate(videos, 1):
        vid = parse_video_id(vpath.name) or vpath.stem
        if args.skip_existing and already_done(args.out, vid):
            print(f"[{i}/{len(videos)}] skip {vid} (exists)")
            continue
        print(f"[{i}/{len(videos)}] {vid}  {vpath.name}")
        frames, src_fps, dur, w, h = read_frames(vpath, cfg.fps_target)

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
        # Per-shard manifest avoids concurrent writers clobbering one file.
        name = ("manifest.csv" if args.shards == 1
                else f"manifest_shard{args.shard_id:02d}.csv")
        man_path = args.out / name
        df = pd.DataFrame(manifest_rows)
        if man_path.exists():
            df = pd.concat([pd.read_csv(man_path), df]).drop_duplicates(
                "video_id", keep="last")
        df.to_csv(man_path, index=False)
        print(f"[manifest] wrote {man_path} ({len(df)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
