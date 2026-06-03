import logging
import os
from pathlib import Path
from typing import Optional

import joblib

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "./models/iforest-v1/model.joblib"

_artifact: Optional["ModelArtifact"] = None


class ModelArtifact:
    def __init__(self, raw: dict, model_path: str):
        self.model = raw["model"]
        self.feature_columns: list = raw.get("feature_columns") or raw["metadata"]["featureColumns"]
        meta = raw["metadata"]
        self.domain_categories: list = meta["domainCategories"]
        self.metric_name_categories: list = meta["metricNameCategories"]
        self.day_of_week_categories: list = meta["dayOfWeekCategories"]
        self.contamination = meta["contamination"]
        self.n_estimators: int = meta["n_estimators"]
        self.random_state: int = meta["random_state"]
        self.model_path: str = model_path


def load_model() -> Optional[ModelArtifact]:
    global _artifact
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    path = Path(model_path)
    if not path.exists():
        logger.warning("Model file not found: %s", model_path)
        _artifact = None
        return None
    try:
        raw = joblib.load(path)
        _artifact = ModelArtifact(raw, str(path))
        logger.info("Model loaded from %s (features=%d)", model_path, len(_artifact.feature_columns))
        return _artifact
    except Exception as exc:
        logger.error("Failed to load model from %s: %s", model_path, exc)
        _artifact = None
        return None


def get_artifact() -> Optional[ModelArtifact]:
    return _artifact
