"""Unified CLI entrypoint for the Guitar Chord AI pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def run_dataset() -> None:
    from dataset.dataset_builder import download_balanced_dataset

    print("[1/2] Downloading balanced dataset...")
    download_balanced_dataset()


def run_preprocess() -> None:
    from dataset.dataset_filtering import clean_dataset, split_dataset

    print("[2/2] Cleaning and splitting dataset...")
    clean_dataset()
    split_dataset()


def run_train() -> None:
    from training.trainer import train

    print("Training model (2-stage fine-tuning)...")
    train()


def run_evaluate() -> None:
    from evaluation.evaluate import evaluate

    print("Running evaluation...")
    evaluate()


def run_predict(image: str) -> None:
    from inference.predict import predict

    chord, confidence, probs = predict(image)
    print(f"Predicted chord: {chord}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Class probabilities: {probs}")


def run_gradcam(image: str, method: str, model_path: str, output: str | None) -> None:
    from interpretability.gradcam import save_overlay

    save_path = save_overlay(model_path=model_path, image_path=image, method=method, out_path=output)
    print(f"Saved {method} overlay: {save_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Guitar Chord AI main entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("dataset", help="Download balanced raw dataset")
    subparsers.add_parser("preprocess", help="Clean/filter data and perform train/val/test split")
    subparsers.add_parser("train", help="Train the CNN model")
    subparsers.add_parser("evaluate", help="Evaluate model and save metrics")
    subparsers.add_parser("full", help="Run dataset + preprocess + train + evaluate")

    predict_parser = subparsers.add_parser("predict", help="Predict chord for one image")
    predict_parser.add_argument("--image", required=True, help="Path to image")

    gradcam_parser = subparsers.add_parser("gradcam", help="Generate Grad-CAM or Grad-CAM++ overlay")
    gradcam_parser.add_argument("--image", required=True, help="Path to input image")
    gradcam_parser.add_argument("--method", choices=["gradcam", "gradcam++"], default="gradcam")
    gradcam_parser.add_argument("--model-path", default="models/guitar_chord_model.keras")
    gradcam_parser.add_argument("--output", default=None, help="Optional output file path")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "dataset":
        run_dataset()
    elif args.command == "preprocess":
        run_preprocess()
    elif args.command == "train":
        run_train()
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "predict":
        run_predict(args.image)
    elif args.command == "gradcam":
        run_gradcam(args.image, args.method, args.model_path, args.output)
    elif args.command == "full":
        run_dataset()
        run_preprocess()
        run_train()
        run_evaluate()
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
