import math
from typing import List, Literal, Union

from pydantic import BaseModel, field_validator


class AnomalyScoreRequest(BaseModel):
    domain: str
    metricName: str
    value: float
    hourOfDay: int
    dayOfWeek: int

    @field_validator("value")
    @classmethod
    def validate_value(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError("value must be a finite number (NaN and Infinity are not allowed)")
        return v

    @field_validator("hourOfDay")
    @classmethod
    def validate_hour(cls, v: int) -> int:
        if not 0 <= v <= 23:
            raise ValueError(f"hourOfDay must be 0-23, got {v}")
        return v

    @field_validator("dayOfWeek")
    @classmethod
    def validate_dow(cls, v: int) -> int:
        if not 1 <= v <= 7:
            raise ValueError(f"dayOfWeek must be 1-7 (Java DayOfWeek), got {v}")
        return v


class AnomalyScoreResponse(BaseModel):
    anomalyScore: float
    predictedLabel: Literal["normal", "anomaly"]
    modelVersion: str
    modelLoaded: bool


class HealthResponse(BaseModel):
    status: str
    modelLoaded: bool


class ModelInfoResponse(BaseModel):
    modelLoaded: bool
    modelPath: str
    featureColumns: List[str]
    domainCategories: List[str]
    metricNameCategories: List[str]
    dayOfWeekCategories: List[str]
    contamination: Union[float, str, None]
    n_estimators: int
    random_state: int
