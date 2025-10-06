from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from contextlib import asynccontextmanager
import os
from ExoMACModel import ExoMACModel
from models.requests import PredictRequest
from models.responses import PredictResponse
from typing import Optional
from fastapi import HTTPException
import pandas as pd

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = ExoMACModel(
        repo_id=os.getenv("EXOMAC_REPO", "ZapatoProgramming/ExoMAC-KKT"),
        local_dir=os.getenv("EXOMAC_LOCAL_DIR", "ExoMACModel/ExoMAC-KKT"),
        prefer_snapshot=True,         
        always_download=False,         
        verbose=True,
    )
    app.state.model = model
    yield

app = FastAPI(
    title="NASA SpaceApp API",
    description="API para el proyecto NASA SpaceApp 2025",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://nasa-space-app-2025-nine.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Endpoint raíz de la API"""
    return {
        "message": "Bienvenido a NASA SpaceApp API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Endpoint de health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "NASA SpaceApp API"
    }


@app.get("/test/predict")
async def test_predict():
    """Endpoint de prueba para la predicción"""
    return {
        "status": 200,
        "message": "SOY EL CHESTNUT",
    }

@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
):
    m: Optional[ExoMACModel] = getattr(app.state, "model", None)
    if m is None:
        raise HTTPException(503, "Model not loaded")

    data = dict(req.features)

    try:
        label, probabilities = m.predict(
            data,
            return_proba=True,
            compute_engineered_if_missing=True,
        )
    except Exception as e:
        raise HTTPException(500, f"Prediction error")
    
    cols = m.feature_columns
    recognized = [c for c in cols if c in data]
    unknown = [k for k in data.keys() if k not in cols]

    used = m._ensure_engineered_features(dict(data))
    X = pd.DataFrame([used], dtype=float).reindex(columns=cols)
    missing = X.columns[X.iloc[0].isna()].tolist()

    # Engineered features: those added beyond the original input keys
    engineered_only = {k: used.get(k) for k in used.keys() if k not in data}
    # JSON-safe (convert NaN to None and numpy floats to float)
    engineered_json = {
        k: (None if pd.isna(v) else float(v)) if isinstance(v, (int, float)) or hasattr(v, "__float__") else None
        for k, v in engineered_only.items()
    }

    return PredictResponse(
        label=label,
        probabilities=probabilities,
        recognized=recognized,
        unknown=unknown,
        missing=missing,
        feature_order=cols,
        engineered=engineered_json,
    )
