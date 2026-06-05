"""One-shot environment check for a fresh checkout.

Verifies dependencies, reports the active ONNX Runtime execution providers
(so you can confirm GPU acceleration), pre-downloads the emotion model, and
confirms the YOLO face weights are present and loadable.

    python -m scripts.setup_check [--weights models/yolov12m-face.pt]
"""
import argparse
import importlib
import sys
from pathlib import Path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="models/yolov12m-face.pt", type=Path)
    args = ap.parse_args(argv)

    ok = True

    print("== dependencies ==")
    for mod in ["cv2", "numpy", "pandas", "pyarrow", "sklearn",
                "onnxruntime", "ultralytics", "hsemotion_onnx"]:
        try:
            m = importlib.import_module(mod)
            v = getattr(m, "__version__", "?")
            print(f"  OK  {mod:16s} {v}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  MISSING  {mod:16s} ({e})")

    print("\n== ONNX Runtime execution providers ==")
    try:
        from emotion_pipeline.models import preferred_ort_providers
        import onnxruntime as ort
        print(f"  available: {ort.get_available_providers()}")
        chosen = preferred_ort_providers()
        print(f"  will use : {chosen}")
        if chosen == ["CPUExecutionProvider"]:
            print("  note: CPU only. For GPU install onnxruntime-directml "
                  "(Windows) or onnxruntime-rocm (Linux).")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  ERROR: {e}")

    print("\n== emotion model ==")
    try:
        from emotion_pipeline.config import PipelineConfig
        from emotion_pipeline.models import ensure_emotion_model, load_emotion_model
        name = PipelineConfig().emotion_model
        path = ensure_emotion_model(name)
        load_emotion_model(name)
        print(f"  OK  {name} -> {path}")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  ERROR: {e}")

    print("\n== YOLO face weights ==")
    if args.weights.is_file():
        try:
            from emotion_pipeline.models import load_face_model
            load_face_model(str(args.weights))
            print(f"  OK  loaded {args.weights}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  ERROR loading {args.weights}: {e}")
    else:
        ok = False
        print(f"  MISSING  {args.weights} — place the YOLO face weights here.")

    print("\n== result ==")
    print("  ALL GOOD — ready to run scripts.run_extract" if ok
          else "  PROBLEMS above — fix them before running the pipeline")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
