from typing import List


def compute_metrics(
    y_true: List[str],
    y_pred: List[str],
    total_rows: int,
    train_rows: int,
) -> dict:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == "anomaly" and p == "anomaly")
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == "normal"  and p == "anomaly")
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == "normal"  and p == "normal")
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == "anomaly" and p == "normal")

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
