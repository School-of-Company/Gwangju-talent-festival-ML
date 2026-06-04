import pandas as pd

from app.model_loader import ModelArtifact


def build_feature_row(
    artifact: ModelArtifact,
    domain: str,
    metric_name: str,
    value: float,
    hour_of_day: int,
    day_of_week: int,
) -> pd.DataFrame:
    row = {col: 0 for col in artifact.feature_columns}
    row["value"] = value
    row["hourOfDay"] = hour_of_day

    for col_key, col_val in [
        (f"domain_{domain}", 1),
        (f"metricName_{metric_name}", 1),
        (f"dayOfWeek_{day_of_week}", 1),
    ]:
        if col_key in row:
            row[col_key] = col_val

    return pd.DataFrame([row], columns=artifact.feature_columns)


def run_inference(
    artifact: ModelArtifact,
    domain: str,
    metric_name: str,
    value: float,
    hour_of_day: int,
    day_of_week: int,
) -> tuple[float, str]:
    X = build_feature_row(artifact, domain, metric_name, value, hour_of_day, day_of_week)
    score = float(artifact.model.decision_function(X)[0])
    predicted_label = "anomaly" if score < 0 else "normal"
    anomaly_score = -score
    return anomaly_score, predicted_label
