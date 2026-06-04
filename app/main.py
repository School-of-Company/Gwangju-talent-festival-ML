import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from app.model_loader import DEFAULT_MODEL_PATH, get_artifact, load_model
from app.schemas import (
    AnomalyScoreRequest,
    AnomalyScoreResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.inference import run_inference

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Gwangju Talent Festival ML", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok", modelLoaded=get_artifact() is not None)


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    artifact = get_artifact()
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    if artifact is None:
        return ModelInfoResponse(
            modelLoaded=False,
            modelPath=model_path,
            featureColumns=[],
            domainCategories=[],
            metricNameCategories=[],
            dayOfWeekCategories=[],
            contamination=None,
            n_estimators=0,
            random_state=0,
        )
    return ModelInfoResponse(
        modelLoaded=True,
        modelPath=artifact.model_path,
        featureColumns=artifact.feature_columns,
        domainCategories=artifact.domain_categories,
        metricNameCategories=artifact.metric_name_categories,
        dayOfWeekCategories=artifact.day_of_week_categories,
        contamination=artifact.contamination,
        n_estimators=artifact.n_estimators,
        random_state=artifact.random_state,
    )


@app.post("/anomaly-score", response_model=AnomalyScoreResponse)
def anomaly_score(req: AnomalyScoreRequest):
    artifact = get_artifact()
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check MODEL_PATH environment variable.")

    if req.domain not in artifact.domain_categories:
        raise HTTPException(status_code=400, detail=f"Invalid domain: {req.domain}. Allowed: {artifact.domain_categories}")
    if req.metricName not in artifact.metric_name_categories:
        raise HTTPException(status_code=400, detail=f"Invalid metricName: {req.metricName}. Allowed: {artifact.metric_name_categories}")

    score, label = run_inference(
        artifact,
        req.domain,
        req.metricName,
        req.value,
        req.hourOfDay,
        req.dayOfWeek,
    )
    return AnomalyScoreResponse(
        anomalyScore=score,
        predictedLabel=label,
        modelVersion=Path(artifact.model_path).parent.name or "unknown",
        modelLoaded=True,
    )
