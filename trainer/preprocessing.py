import pandas as pd

REQUIRED_COLUMNS = ["domain", "metricName", "timestamp", "value", "hourOfDay", "dayOfWeek", "label"]
VALID_LABELS = {"normal", "anomaly"}
VALID_DOMAINS = {"SEAT", "JUDGE"}
VALID_METRIC_NAMES = {"failure_rate", "p95_duration"}
CATEGORICAL_COLS = ["domain", "metricName", "dayOfWeek"]
NUMERIC_COLS = ["value", "hourOfDay"]


def load_and_preprocess(csv_path: str) -> tuple:
    df = pd.read_csv(csv_path)
    _validate_columns(df)
    _validate_no_missing(df)
    _validate_labels(df)
    _validate_categorical_values(df)
    _validate_numeric_ranges(df)
    df = df.sort_values("timestamp").reset_index(drop=True)
    X, feature_cols = _build_features(df)
    y = df["label"].reset_index(drop=True)
    categories = {
        "domainCategories":     sorted(VALID_DOMAINS),
        "metricNameCategories": sorted(VALID_METRIC_NAMES),
        "dayOfWeekCategories":  [str(i) for i in range(1, 8)],
    }
    return X, y, feature_cols, categories


def _validate_columns(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def _validate_no_missing(df: pd.DataFrame) -> None:
    counts = df[REQUIRED_COLUMNS].isnull().sum()
    bad = counts[counts > 0]
    if not bad.empty:
        raise ValueError(f"Null values found:\n{bad.to_string()}")


def _validate_labels(df: pd.DataFrame) -> None:
    invalid = set(df["label"].unique()) - VALID_LABELS
    if invalid:
        raise ValueError(
            f"Invalid label values: {sorted(invalid)}. "
            f"Only {sorted(VALID_LABELS)} are allowed. "
            "Check Spring export -- values like 'ignored' or 'open' must be filtered before training."
        )


def _validate_categorical_values(df: pd.DataFrame) -> None:
    bad_domains = set(df["domain"].unique()) - VALID_DOMAINS
    if bad_domains:
        raise ValueError(
            f"Invalid domain values: {sorted(bad_domains)}. Allowed: {sorted(VALID_DOMAINS)}"
        )
    bad_metrics = set(df["metricName"].unique()) - VALID_METRIC_NAMES
    if bad_metrics:
        raise ValueError(
            f"Invalid metricName values: {sorted(bad_metrics)}. Allowed: {sorted(VALID_METRIC_NAMES)}"
        )


def _validate_numeric_ranges(df: pd.DataFrame) -> None:
    bad_dow = df[(df["dayOfWeek"] < 1) | (df["dayOfWeek"] > 7)]
    if not bad_dow.empty:
        raise ValueError(
            f"dayOfWeek must be 1-7 (Java DayOfWeek). "
            f"Invalid values: {sorted(bad_dow['dayOfWeek'].unique())}"
        )
    bad_hod = df[(df["hourOfDay"] < 0) | (df["hourOfDay"] > 23)]
    if not bad_hod.empty:
        raise ValueError(
            f"hourOfDay must be 0-23. "
            f"Invalid values: {sorted(bad_hod['hourOfDay'].unique())}"
        )


def _build_features(df: pd.DataFrame) -> tuple:
    cat_df = pd.DataFrame()
    cat_df["domain"]     = pd.Categorical(df["domain"],                    categories=sorted(VALID_DOMAINS))
    cat_df["metricName"] = pd.Categorical(df["metricName"],                categories=sorted(VALID_METRIC_NAMES))
    cat_df["dayOfWeek"]  = pd.Categorical(df["dayOfWeek"].astype(str),     categories=[str(i) for i in range(1, 8)])
    dummies = pd.get_dummies(cat_df, prefix=CATEGORICAL_COLS, dtype=int)
    numeric = df[NUMERIC_COLS].reset_index(drop=True)
    X = pd.concat([numeric, dummies], axis=1)
    return X, list(X.columns)
