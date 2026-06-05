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
    human = df["human_label"].tolist()
    model = df["argmax_emotion"].tolist()
    # Confusion oriented rows=model (argmax), cols=human label.
    return {
        "n_labeled": int(len(df)),
        "agreement": agreement(human, model),
        "confusion": confusion(model, human, labels=EMOTIONS),
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
                f"confusion (rows=model argmax, cols=human):\n{rep['confusion'].to_string()}\n")
        print(text)
        if args.out:
            args.out.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
