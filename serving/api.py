"""
FastAPI model serving endpoint.

Loads the champion model from MLflow Model Registry (or falls back to the
locally saved joblib file) and serves predictions.

Start with:
    uv run uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"
META_PATH = MODELS_DIR / "feature_meta.json"
INFO_PATH = MODELS_DIR / "best_model_info.json"

app = FastAPI(
    title="Computer Durability Classifier",
    description="Predicts whether a computer needs replacement based on usage patterns.",
    version="1.0.0",
)

# ── Model Loading ─────────────────────────────────────────────────────────────

_model = None
_scaler = None
_feature_cols: list[str] = []


def _load_artifacts() -> None:
    global _model, _scaler, _feature_cols

    if META_PATH.exists():
        meta = json.loads(META_PATH.read_text())
        _feature_cols = meta["feature_cols"]
        scaler_path = MODELS_DIR / Path(meta["scaler_path"]).name
        _scaler = joblib.load(scaler_path)
    else:
        raise RuntimeError("feature_meta.json not found — run the Dagster pipeline first.")

    if INFO_PATH.exists():
        info = json.loads(INFO_PATH.read_text())
        winner = info.get("winner", "XGBoost")
        if winner == "XGBoost":
            model_path = MODELS_DIR / "xgb_tuned.joblib"
        else:
            model_path = MODELS_DIR / "rf_baseline.joblib"
        _model = joblib.load(model_path)
    else:
        raise RuntimeError("best_model_info.json not found — run the Dagster pipeline first.")


@app.on_event("startup")
def startup_event() -> None:
    _load_artifacts()


# ── Request / Response schemas ────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    hours_used_per_day: float = Field(..., ge=0.0, le=24.0, example=18.5)
    cost: float = Field(..., ge=0.0, example=15000.0)
    user_age: float = Field(..., ge=0.0, le=120.0, example=45.0)
    primary_usage: int = Field(..., ge=1, le=4, example=2)
    brand: int = Field(..., ge=1, le=5, example=3)
    computer_age_months: float = Field(..., ge=0.0, example=36.0)


class PredictionResponse(BaseModel):
    needs_replacement: bool
    probability: float
    model_version: Optional[str] = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/info")
def model_info() -> dict:
    if INFO_PATH.exists():
        return json.loads(INFO_PATH.read_text())
    return {"error": "model info not available"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if _model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Assemble feature vector in the exact column order
    row = np.array([[
        request.hours_used_per_day,
        request.cost,
        request.user_age,
        request.primary_usage,
        request.brand,
        request.computer_age_months,
    ]])
    row_scaled = _scaler.transform(row)
    prob = float(_model.predict_proba(row_scaled)[0, 1])
    label = prob >= 0.5

    version = None
    if INFO_PATH.exists():
        info = json.loads(INFO_PATH.read_text())
        version = f"{info.get('winner')} v{info.get('registry_version')}"

    return PredictionResponse(
        needs_replacement=label,
        probability=round(prob, 4),
        model_version=version,
    )


@app.post("/predict/batch")
def predict_batch(requests: list[PredictionRequest]) -> list[PredictionResponse]:
    if _model is None or _scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    rows = np.array([
        [r.hours_used_per_day, r.cost, r.user_age, r.primary_usage, r.brand, r.computer_age_months]
        for r in requests
    ])
    rows_scaled = _scaler.transform(rows)
    probs = _model.predict_proba(rows_scaled)[:, 1]

    version = None
    if INFO_PATH.exists():
        info = json.loads(INFO_PATH.read_text())
        version = f"{info.get('winner')} v{info.get('registry_version')}"

    return [
        PredictionResponse(
            needs_replacement=bool(p >= 0.5),
            probability=round(float(p), 4),
            model_version=version,
        )
        for p in probs
    ]
