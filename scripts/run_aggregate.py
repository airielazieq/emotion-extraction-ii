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
