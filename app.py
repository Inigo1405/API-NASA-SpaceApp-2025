from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from contextlib import asynccontextmanager
import os
from ExoMACModel import ExoMACModel
from models.requests import PredictRequest
from models.responses import (
    PredictResponse, 
    ExoplanetCumulative, 
    ExoplanetK2, 
    ExoplanetTOI,
    DatasetListResponse
)
from typing import Optional
from fastapi import HTTPException, Query
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


# ============================================================================
# HELPER FUNCTIONS PARA CARGA DE DATASETS
# ============================================================================

def load_csv_dataset(filename: str) -> pd.DataFrame:
    """Carga un CSV de NASA con manejo de comentarios y errores"""
    try:
        filepath = os.path.join("NASA_datasets", filename)
        df = pd.read_csv(filepath, comment='#')
        return df
    except Exception as e:
        raise HTTPException(500, f"Error loading dataset {filename}: {str(e)}")

def filter_dataframe(df: pd.DataFrame, limit: int = 100, offset: int = 0, **filters) -> pd.DataFrame:
    """Filtra un dataframe y aplica paginación"""
    filtered_df = df.copy()
    
    # Aplicar filtros si existen
    for key, value in filters.items():
        if value is not None and key in filtered_df.columns:
            filtered_df = filtered_df[filtered_df[key] == value]
    
    # Aplicar paginación
    return filtered_df.iloc[offset:offset + limit]


def df_to_dict_list(df: pd.DataFrame) -> list:
    """Convierte DataFrame a lista de diccionarios, manejando NaN"""
    return df.where(pd.notna(df), None).to_dict('records')


# ============================================================================
# ENDPOINTS PARA DATASET CUMULATIVE (KEPLER)
# ============================================================================

@app.get("/kepler", response_model=DatasetListResponse)
async def get_kepler_exoplanets(
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Número máximo de resultados (si no se especifica, devuelve todos)"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    koi_disposition: Optional[str] = Query(None, description="Filtrar por disposición (CONFIRMED, FALSE POSITIVE, CANDIDATE)")
):
    """
    Obtiene lista de exoplanetas del dataset Cumulative (Kepler).
    
    Incluye las 11 características esenciales:
    - koi_period: Periodo orbital
    - koi_duration: Duración del tránsito
    - koi_depth: Profundidad del tránsito
    - koi_impact: Parámetro de impacto
    - koi_prad: Radio del planeta
    - koi_slogg: Gravedad superficial estelar
    - koi_sma: Semi-eje mayor
    - koi_smet: Metalicidad estelar
    - koi_srad: Radio estelar
    - koi_steff: Temperatura efectiva estelar
    - koi_snr: Relación señal-ruido (nota: no disponible en el dataset)
    """
    df = load_csv_dataset("cumulative_2025.10.05_10.28.27.csv")
    total = len(df)
    
    # Filtrar si se especifica disposición
    if koi_disposition:
        df = df[df['koi_disposition'] == koi_disposition]
    
    # Aplicar paginación solo si limit está definido
    if limit is not None:
        df_page = df.iloc[offset:offset + limit]
    else:
        df_page = df.iloc[offset:]
    
    # Seleccionar columnas relevantes (solo con ≤50% nulos)
    columns_to_include = [
        # Identificadores
        'kepid', 'kepoi_name', 'kepler_name',
        # Disposición
        'koi_disposition', 'koi_pdisposition', 'koi_score',
        # Características orbitales
        'koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
        'koi_prad', 'koi_sma', 'koi_teq', 'koi_insol', 'koi_dor', 'koi_ror',
        'koi_eccen', 'koi_incl',
        # Características estelares
        'koi_steff', 'koi_slogg', 'koi_srad', 'koi_smass', 'koi_smet', 'koi_kepmag',
        # Coordenadas
        'ra', 'dec'
    ]
    
    # Filtrar solo columnas que existen
    available_columns = [col for col in columns_to_include if col in df_page.columns]
    df_result = df_page[available_columns]
    
    return DatasetListResponse(
        total=total,
        count=len(df_result),
        data=df_to_dict_list(df_result)
    )


@app.get("/kepler/{id}", response_model=ExoplanetCumulative)
async def get_cumulative_exoplanet_by_id(id: str):
    """
    Obtiene un exoplaneta específico del dataset Cumulative por su nombre KOI o nombre Kepler.
    Ejemplo: K00001.01, K00002.01, Kepler-227 b, etc.
    """
    df = load_csv_dataset("cumulative_2025.10.05_10.28.27.csv")
    # Buscar por kepoi_name o kepler_name
    exoplanet = df[(df['kepoi_name'] == id) | (df['kepler_name'] == id)]
    if len(exoplanet) == 0:
        raise HTTPException(404, f"Exoplanet with kepoi_name or kepler_name '{id}' not found")
    data = exoplanet.iloc[0].where(pd.notna(exoplanet.iloc[0]), None).to_dict()
    return ExoplanetCumulative(**data)


@app.get("/keplerSummary")
async def get_kepler_summary():
    """
    Devuelve el conteo de exoplanetas confirmados, candidatos, falsos positivos y el total en el dataset Kepler.
    """
    df = load_csv_dataset("cumulative_2025.10.05_10.28.27.csv")
    disposition_counts = df['koi_disposition'].value_counts().to_dict()
    # Normalizar claves
    summary = {
        "CONFIRMED": disposition_counts.get("CONFIRMED", 0),
        "CANDIDATE": disposition_counts.get("CANDIDATE", 0),
        "FALSE POSITIVE": disposition_counts.get("FALSE POSITIVE", 0),
        "TOTAL": int(df.shape[0])
    }
    return summary

# ============================================================================
# ENDPOINTS PARA DATASET K2
# ============================================================================

@app.get("/k2", response_model=DatasetListResponse)
async def get_k2_exoplanets(
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Número máximo de resultados (si no se especifica, devuelve todos)"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    disposition: Optional[str] = Query(None, description="Filtrar por disposición")
):
    """
    Obtiene lista de exoplanetas del dataset K2.
    
    Incluye datos de planetas y estrellas para visualización.
    """
    df = load_csv_dataset("k2pandc_2025.10.05_10.29.57.csv")
    total = len(df)
    
    # Filtrar si se especifica disposición
    if disposition:
        df = df[df['disposition'] == disposition]
    
    # Aplicar paginación solo si limit está definido
    if limit is not None:
        df_page = df.iloc[offset:offset + limit]
    else:
        df_page = df.iloc[offset:]
    
    # Seleccionar columnas relevantes (solo con ≤50% nulos)
    columns_to_include = [
        # Identificadores
        'pl_name', 'hostname', 'epic_hostname', 'tic_id', 'gaia_id',
        'disposition', 'discoverymethod', 'disc_year',
        # Datos del planeta
        'pl_orbper', 'pl_rade', 'pl_radj', 'pl_trandep', 'pl_trandur',
        'pl_tranmid', 'pl_imppar',
        # Datos estelares
        'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg',
        # Datos del sistema
        'sy_dist', 'sy_vmag', 'sy_kmag', 'sy_jmag', 'sy_hmag', 'sy_gaiamag',
        # Coordenadas
        'ra', 'dec'
    ]
    
    # Filtrar solo columnas que existen
    available_columns = [col for col in columns_to_include if col in df_page.columns]
    df_result = df_page[available_columns]
    
    return DatasetListResponse(
        total=total,
        count=len(df_result),
        data=df_to_dict_list(df_result)
    )


@app.get("/k2/{pl_name}", response_model=ExoplanetK2)
async def get_k2_exoplanet_by_name(pl_name: str):
    """
    Obtiene un exoplaneta específico del dataset K2 por su nombre.
    
    Ejemplo: K2-1 b, K2-2 b, etc.
    """
    df = load_csv_dataset("k2pandc_2025.10.05_10.29.57.csv")
    
    # Buscar por pl_name
    exoplanet = df[df['pl_name'] == pl_name]
    
    if len(exoplanet) == 0:
        raise HTTPException(404, f"Exoplanet with name '{pl_name}' not found")
    
    # Convertir a diccionario
    data = exoplanet.iloc[0].where(pd.notna(exoplanet.iloc[0]), None).to_dict()
    
    return ExoplanetK2(**data)


# ============================================================================
# ENDPOINTS PARA DATASET TOI (TESS)
# ============================================================================

@app.get("/tess", response_model=DatasetListResponse)
async def get_tess_exoplanets(
    limit: Optional[int] = Query(None, ge=1, le=1000, description="Número máximo de resultados (si no se especifica, devuelve todos)"),
    offset: int = Query(0, ge=0, description="Offset para paginación"),
    tfopwg_disp: Optional[str] = Query(None, description="Filtrar por disposición del grupo de trabajo")
):
    """
    Obtiene lista de TESS Objects of Interest (TOI).
    
    Incluye datos de candidatos a exoplanetas del telescopio TESS.
    """
    df = load_csv_dataset("TOI_2025.10.05_10.30.20.csv")
    total = len(df)
    
    # Filtrar si se especifica disposición
    if tfopwg_disp:
        df = df[df['tfopwg_disp'] == tfopwg_disp]
    
    # Aplicar paginación solo si limit está definido
    if limit is not None:
        df_page = df.iloc[offset:offset + limit]
    else:
        df_page = df.iloc[offset:]
    
    # Seleccionar columnas relevantes (solo con ≤50% nulos)
    columns_to_include = [
        # Identificadores
        'toi', 'tid', 'ctoi_alias', 'tfopwg_disp', 'pl_pnum',
        # Datos del planeta
        'pl_orbper', 'pl_rade', 'pl_eqt', 'pl_insol', 'pl_trandep',
        'pl_trandurh', 'pl_tranmid', 'pl_imppar', 'pl_orbsmax',
        # Datos estelares
        'st_teff', 'st_rad', 'st_mass', 'st_met', 'st_logg', 'st_dist', 'st_tmag',
        # Movimiento propio
        'st_pmra', 'st_pmdec',
        # Coordenadas
        'ra', 'dec', 'rastr', 'decstr'
    ]
    
    # Filtrar solo columnas que existen
    available_columns = [col for col in columns_to_include if col in df_page.columns]
    df_result = df_page[available_columns]
    
    # Convertir campos que deben ser strings
    string_fields = ['ctoi_alias', 'rastr', 'decstr']
    for field in string_fields:
        if field in df_result.columns:
            df_result[field] = df_result[field].apply(lambda x: str(x) if pd.notna(x) else None)
    
    return DatasetListResponse(
        total=total,
        count=len(df_result),
        data=df_to_dict_list(df_result)
    )


@app.get("/tess/{toi_id}", response_model=ExoplanetTOI)
async def get_tess_by_id(toi_id: float):
    """
    Obtiene un TOI específico por su ID.
    
    Ejemplo: 100.01, 101.01, etc.
    """
    df = load_csv_dataset("TOI_2025.10.05_10.30.20.csv")
    
    # Buscar por toi
    toi_obj = df[df['toi'] == toi_id]
    
    if len(toi_obj) == 0:
        raise HTTPException(404, f"TOI with id '{toi_id}' not found")
    
    # Convertir a diccionario
    data = toi_obj.iloc[0].where(pd.notna(toi_obj.iloc[0]), None).to_dict()
    
    # Convertir campos que deben ser strings
    string_fields = ['ctoi_alias', 'rastr', 'decstr']
    for field in string_fields:
        if field in data and data[field] is not None:
            data[field] = str(data[field])
    
    return ExoplanetTOI(**data)


