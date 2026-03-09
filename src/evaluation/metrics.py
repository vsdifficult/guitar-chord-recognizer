import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_curve, auc
from sklearn.calibration import calibration_curve


def compute_classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "confusion_matrix": cm}


def compute_roc_per_class(y_true_onehot, y_proba, class_names):
    curves = {}
    for idx, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_onehot[:, idx], y_proba[:, idx])
        curves[name] = {"fpr": fpr, "tpr": tpr, "auc": auc(fpr, tpr)}
    return curves


def compute_calibration(y_true_onehot, y_proba, class_names, n_bins=10):
    output = {}
    for idx, name in enumerate(class_names):
        frac_pos, mean_pred = calibration_curve(y_true_onehot[:, idx], y_proba[:, idx], n_bins=n_bins)
        output[name] = {"fraction_positive": frac_pos, "mean_predicted": mean_pred}
    return output
