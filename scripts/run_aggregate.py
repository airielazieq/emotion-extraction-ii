"""Aggregate face-level parquet outputs into frames + films_summary + counts.

Usage:
  python -m scripts.run_aggregate --faces DIR --out DIR [--labels labels.csv]
                                  [--method mean_softmax] [--combined-dir DIR]

Per run it writes, into --out:
  frames.parquet                     one row per sampled frame (all films)
  films_summary.csv                  one row per film (proportions, entropy, ...)
  films_emotion_counts_face.csv      one row per film: count of each per-face label
  films_emotion_counts_frame.csv     one row per film: count of each per-frame emotion
And into --combined-dir (default: <out>/combined) every output unified into one
file each, so the whole corpus lives in a single folder:
  all_faces.parquet, all_frames.parquet, films_summary.csv,
  films_emotion_counts_face.csv, films_emotion_counts_frame.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from emotion_pipeline.aggregate import (
    emotion_counts_face, emotion_counts_frame, faces_to_frames, film_summary,
)


def aggregate_one(faces_dir: Path, video_id: str, method: str):
    faces = pd.read_parquet(faces_dir / f"{video_id}_faces.parquet")
    grid = pd.read_parquet(faces_dir / f"{video_id}_grid.parquet")
    sampled = list(zip(grid["frame_idx"].tolist(), grid["t_sec"].tolist()))
    frames = faces_to_frames(faces, sampled, video_id, method=method)
    summary = film_summary(frames, video_id)
    return faces, frames, summary


def join_labels(films: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    return films.merge(labels, on="video_id", how="left")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--method", default="mean_softmax")
    ap.add_argument("--combined-dir", type=Path, default=None,
                    help="folder for unified corpus files (default: <out>/combined)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    combined_dir = args.combined_dir or (args.out / "combined")
    combined_dir.mkdir(parents=True, exist_ok=True)

    video_ids = sorted(p.name[:-len("_faces.parquet")]
                       for p in args.faces.glob("*_faces.parquet"))
    all_faces, all_frames = [], []
    summaries, counts_face, counts_frame = [], [], []
    for vid in video_ids:
        faces, frames, summary = aggregate_one(args.faces, vid, args.method)
        all_faces.append(faces)
        all_frames.append(frames)
        summaries.append(summary)
        counts_face.append(emotion_counts_face(faces, vid))
        counts_frame.append(emotion_counts_frame(frames, vid))
        print(f"[agg] {vid}: {summary['n_frames_face']} faced frames, "
              f"dominant={summary['dominant_emotion']}")

    frames_df = (pd.concat(all_frames, ignore_index=True)
                 if all_frames else pd.DataFrame())
    faces_df = (pd.concat(all_faces, ignore_index=True)
                if all_faces else pd.DataFrame())
    films = pd.DataFrame(summaries)
    if args.labels and args.labels.exists():
        films = join_labels(films, pd.read_csv(args.labels))
    cf = pd.DataFrame(counts_face)
    cfr = pd.DataFrame(counts_frame)

    # Per-run outputs in --out
    frames_df.to_parquet(args.out / "frames.parquet", index=False)
    films.to_csv(args.out / "films_summary.csv", index=False)
    cf.to_csv(args.out / "films_emotion_counts_face.csv", index=False)
    cfr.to_csv(args.out / "films_emotion_counts_frame.csv", index=False)

    # Unified corpus, one file per output type, in the combined folder
    faces_df.to_parquet(combined_dir / "all_faces.parquet", index=False)
    frames_df.to_parquet(combined_dir / "all_frames.parquet", index=False)
    films.to_csv(combined_dir / "films_summary.csv", index=False)
    cf.to_csv(combined_dir / "films_emotion_counts_face.csv", index=False)
    cfr.to_csv(combined_dir / "films_emotion_counts_frame.csv", index=False)

    print(f"[agg] wrote films_summary.csv ({len(films)} films); "
          f"combined corpus in {combined_dir} "
          f"({len(faces_df)} faces, {len(frames_df)} frames)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
