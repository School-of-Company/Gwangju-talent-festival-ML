import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

sys.path.insert(0, str(Path(__file__).parent))
from preprocessing import load_and_preprocess
from metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Train IsolationForest anomaly detector")
    p.add_argument("--dataset-path", required=True, help="Path to input CSV")
    p.add_argument("--output-dir",   required=True, help="Directory to save artifacts")
    p.add_argument(
        "--contamination", default="auto",
        help="IsolationForest contamination: float (0, 0.5] or 'auto'. Default: auto",
    )
    p.add_argument("--n-estimators", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()

    contamination = args.contamination
    if contamination != "auto":
        contamination = float(contamination)
        if not (0 < contamination <= 0.5):
            raise ValueError(
                f"--contamination must be in (0, 0.5] or 'auto', got {contamination}"
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading dataset: {args.dataset_path}")
    X, y, feature_cols, categories = load_and_preprocess(args.dataset_path)

    normal_mask = y == "normal"
    X_train = X[normal_mask].reset_index(drop=True)
    if len(X_train) == 0:
        raise ValueError(
            "No training data found with label 'normal'. "
            "Isolation Forest requires at least one normal sample to train."
        )
    print(f"[2/5] Training on {len(X_train)} normal rows (total {len(X)} rows)")

    model = IsolationForest(
        n_estimators=args.n_estimators,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X_train)

    print("[3/5] Computing predictions and anomaly scores")
    raw_preds   = model.predict(X)
    anom_scores = model.score_samples(X)
    pred_labels = ["anomaly" if p == -1 else "normal" for p in raw_preds]

    print("[4/5] Computing metrics")
    metrics = compute_metrics(y.tolist(), pred_labels, len(X), len(X_train))

    print("[5/5] Saving artifacts")

    model_artifact = {
        "model": model,
        "feature_columns": feature_cols,
        "label_mapping": {
            "iforest_predict_1":  "normal",
            "iforest_predict_-1": "anomaly",
        },
        "metadata": {
            "contamination":        str(contamination),
            "n_estimators":         args.n_estimators,
            "random_state":         42,
            "train_rows":           len(X_train),
            "total_rows":           len(X),
            "featureColumns":       feature_cols,
            "domainCategories":     categories["domainCategories"],
            "metricNameCategories": categories["metricNameCategories"],
            "dayOfWeekCategories":  categories["dayOfWeekCategories"],
        },
    }
    joblib.dump(model_artifact, output_dir / "model.joblib")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame({
        "trueLabel":      y.tolist(),
        "predictedLabel": pred_labels,
        "anomalyScore":   anom_scores,
    }).to_csv(output_dir / "predictions.csv", index=False)

    print(f"\nArtifacts saved to: {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
