from typing import List, Literal, Union

from pydantic import BaseModel, Field


class AnomalyScoreRequest(BaseModel):
    domain: str
    metricName: str
    value: float = Field(allow_inf_nan=False)
    hourOfDay: int = Field(ge=0, le=23)
    dayOfWeek: int = Field(ge=1, le=7)


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
