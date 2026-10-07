"""
FastAPI Backend Application for Indian Railways Predictive Maintenance
Production ASGI Server (Azure App Service Ready)
"""

import os
import io
import pandas as pd
from contextlib import asynccontextmanager
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.schemas import (
    AssetTelemetryInput,
    PredictionResult,
    BatchPredictionResponse,
    PresetProfile,
    ModelInfoResponse
)
from backend.inference import model_manager, FEATURE_ORDER
from backend.data_store import PRESET_PROFILES, MODEL_EVALUATION_METRICS

# Lifespan context manager to load models once at server startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Initializing FastAPI Predictive Maintenance Server...")
    model_manager.load_models()
    yield
    print("🛑 Shutting down FastAPI server.")

app = FastAPI(
    title="Indian Railways Predictive Maintenance API",
    description="Production ASGI REST API for Track, Rolling Stock, and Mechanical Predictive Health Diagnostics",
    version="2.0.0",
    lifespan=lifespan
)

# Dynamic CORS Configuration for Azure / Vercel deployment
allowed_origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",") if o.strip()]
if not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static plot directories if present
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
eda_plots_dir = os.path.join(BASE_DIR, "eda_plots")
model_eval_dir = os.path.join(BASE_DIR, "model_eval_plots")

if os.path.exists(eda_plots_dir):
    app.mount("/plots/eda", StaticFiles(directory=eda_plots_dir), name="eda_plots")
if os.path.exists(model_eval_dir):
    app.mount("/plots/eval", StaticFiles(directory=model_eval_dir), name="eval_plots")


# ==============================================================================
# API ENDPOINTS
# ==============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check probe endpoint for Azure App Service"""
    return {
        "status": "healthy",
        "service": "ir-predictive-maintenance-api",
        "models_loaded": model_manager.is_loaded
    }


@app.get("/api/presets", response_model=List[PresetProfile], tags=["Presets"])
async def get_presets():
    """Returns 6 realistic railway operational presets with full telemetry profiles"""
    return PRESET_PROFILES


@app.post("/api/predict", response_model=PredictionResult, tags=["Inference"])
async def predict_single_asset(telemetry: AssetTelemetryInput):
    """
    Accepts one railway asset's telemetry input, runs inference, and returns:
    - Verdict (Maintenance Required vs Healthy)
    - Risk Probability %
    - Urgency level (CRITICAL, HIGH, MEDIUM, LOW)
    - Recommended Track Maintenance Action
    - Subsystem Health indices (Track, Mechanical, Pneumatic, Electrical)
    - Top 3 root cause contributing factors
    """
    try:
        return model_manager.predict_single(telemetry)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Inference execution failed: {str(e)}"
        )


@app.post("/api/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch_assets(file: UploadFile = File(...)):
    """
    Accepts a CSV file containing multiple railway asset records.
    Returns a ranked list of predictions sorted by risk probability (highest risk first),
    along with summary fleet statistics.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a valid CSV (.csv) file."
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded CSV file is empty."
            )

        return model_manager.predict_batch_df(df)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse and process CSV file: {str(e)}"
        )


@app.get("/api/sample-csv", tags=["Inference"])
async def download_sample_csv():
    """Generates and downloads a sample 10-asset CSV file formatted for batch upload"""
    sample_rows = []
    for p in PRESET_PROFILES:
        row = dict(p["data"])
        row["asset_id"] = p["data"].get("asset_id", f"RAKE-{p['id']}")
        sample_rows.append(row)

    # Add 4 more diverse synthetic variations
    extra_variations = [
        {"asset_id": "NR-EMU-102", "region": "Northern Railway", "train_type": "Metro", "rail_wear_mm": 14.2, "wheel_wear_percent": 82.0, "bearing_temperature_c": 98.0, "brake_pressure_psi": 62.0, "track_vibration_level": 7.8, "ballast_condition": "Poor"},
        {"asset_id": "CR-EXP-552", "region": "Central Railway", "train_type": "Express", "rail_wear_mm": 4.1, "wheel_wear_percent": 28.0, "bearing_temperature_c": 55.0, "brake_pressure_psi": 98.0, "track_vibration_level": 2.2, "ballast_condition": "Good"},
        {"asset_id": "WR-FRT-809", "region": "Western Railway", "train_type": "Freight", "rail_wear_mm": 16.8, "wheel_wear_percent": 89.0, "bearing_temperature_c": 112.0, "brake_pressure_psi": 52.0, "track_vibration_level": 8.4, "ballast_condition": "Poor"},
        {"asset_id": "SR-PSG-330", "region": "Southern Railway", "train_type": "Passenger", "rail_wear_mm": 6.5, "wheel_wear_percent": 42.0, "bearing_temperature_c": 64.0, "brake_pressure_psi": 90.0, "track_vibration_level": 3.4, "ballast_condition": "Fair"}
    ]

    for extra in extra_variations:
        base = dict(PRESET_PROFILES[0]["data"])
        base.update(extra)
        sample_rows.append(base)

    df_sample = pd.DataFrame(sample_rows)
    stream = io.StringIO()
    df_sample.to_csv(stream, index=False)

    return StreamingResponse(
        io.BytesIO(stream.getvalue().encode('utf-8')),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sample_railway_fleet.csv"}
    )


@app.get("/api/model-info", response_model=ModelInfoResponse, tags=["Analytics"])
async def get_model_info():
    """
    Returns precomputed evaluation metrics (Accuracy, Precision, Recall, F1, ROC-AUC),
    confusion matrix counts, ROC curve points, and top feature importance rankings.
    """
    return MODEL_EVALUATION_METRICS


# Custom Exception Handler for clear 4xx error formatting
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "status": "error",
            "error_code": exc.status_code,
            "message": exc.detail
        }
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port, reload=False)
