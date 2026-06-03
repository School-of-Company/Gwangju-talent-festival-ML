from typing import List

from sklearn.metrics import confusion_matrix


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    total_rows: int,
    train_rows: int,
) -> dict:
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true ({len(y_true)}) and y_pred ({len(y_pred)}) must have the same length."
        )
    tn, fp, fn, tp = map(int, confusion_matrix(y_true, y_pred, labels=["normal", "anomaly"]).ravel())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "totalRows":         total_rows,
        "trainRows":         train_rows,
        "normalRows":        tn + fp,
        "anomalyRows":       tp + fn,
        "precision":         round(precision, 4),
        "recall":            round(recall, 4),
        "f1":                round(f1, 4),
        "falsePositiveRate": round(fpr, 4),
        "truePositive":      tp,
        "falsePositive":     fp,
        "trueNegative":      tn,
        "falseNegative":     fn,
    }
