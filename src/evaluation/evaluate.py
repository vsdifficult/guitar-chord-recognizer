import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report

from evaluation.metrics import compute_calibration, compute_classification_metrics, compute_roc_per_class
from utils.config import ProjectConfig


def evaluate():
    cfg = ProjectConfig()
    model = tf.keras.models.load_model(cfg.train.model_dir / "guitar_chord_model.keras", compile=False)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        cfg.data.test_dir,
        image_size=(cfg.data.image_size, cfg.data.image_size),
        batch_size=cfg.data.batch_size,
        label_mode="categorical",
        shuffle=False,
    ).prefetch(tf.data.AUTOTUNE)

    class_names = test_ds.class_names

    y_true_onehot, y_proba = [], []
    for x, y in test_ds:
        p = model.predict(x, verbose=0)
        y_true_onehot.append(y.numpy())
        y_proba.append(p)

    y_true_onehot = np.concatenate(y_true_onehot, axis=0)
    y_proba = np.concatenate(y_proba, axis=0)
    y_true = np.argmax(y_true_onehot, axis=1)
    y_pred = np.argmax(y_proba, axis=1)

    metrics = compute_classification_metrics(y_true, y_pred)
    roc = compute_roc_per_class(y_true_onehot, y_proba, class_names)
    calibration = compute_calibration(y_true_onehot, y_proba, class_names)

    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Metrics:", {k: v for k, v in metrics.items() if k != "confusion_matrix"})
    print("Confusion matrix:\n", metrics["confusion_matrix"])

    out = Path("models/eval_artifacts")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v) if isinstance(v, (float, np.floating)) else str(v) for k, v in metrics.items()}, f, indent=2)
    with open(out / "roc_auc.json", "w", encoding="utf-8") as f:
        json.dump({k: float(v["auc"]) for k, v in roc.items()}, f, indent=2)
    with open(out / "calibration.json", "w", encoding="utf-8") as f:
        json.dump({k: {kk: np.asarray(vv).tolist() for kk, vv in v.items()} for k, v in calibration.items()}, f, indent=2)


if __name__ == "__main__":
    evaluate()
