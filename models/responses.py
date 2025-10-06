from typing import Dict, List, Optional
from pydantic import BaseModel

class PredictResponse(BaseModel):
    label: str
    probabilities: Optional[Dict[str, float]] = None
    recognized: List[str]
    unknown: List[str]
    missing: List[str]
    feature_order: List[str]
    engineered: Dict[str, Optional[float]]


# Modelos para los datasets de NASA (Solo columnas con ≤50% nulos)
class ExoplanetCumulative(BaseModel):
    """Modelo para exoplanetas del dataset Cumulative (Kepler) - Solo datos reales"""
    # Identificadores (0% nulos)
    kepid: Optional[int] = None
    kepoi_name: Optional[str] = None
    kepler_name: Optional[str] = None
    
    # Disposición y clasificación (0% nulos)
    koi_disposition: Optional[str] = None
    koi_pdisposition: Optional[str] = None
    koi_score: Optional[float] = None
    
    # Características orbitales esenciales (0% nulos)
    koi_period: Optional[float] = None  # Periodo orbital (días)
    koi_time0bk: Optional[float] = None  # Tiempo de referencia (BJD)
    koi_impact: Optional[float] = None  # Parámetro de impacto
    koi_duration: Optional[float] = None  # Duración del tránsito (horas)
    koi_depth: Optional[float] = None  # Profundidad del tránsito (ppm)
    koi_prad: Optional[float] = None  # Radio del planeta (radios terrestres)
    koi_sma: Optional[float] = None  # Semi-eje mayor (AU)
    
    # Parámetros derivados (0-1% nulos)
    koi_teq: Optional[float] = None  # Temperatura de equilibrio (K)
    koi_insol: Optional[float] = None  # Flujo de insolación (flujo terrestre)
    koi_dor: Optional[float] = None  # Distancia planeta-estrella / radio estelar
    koi_ror: Optional[float] = None  # Radio planeta / radio estrella
    koi_eccen: Optional[float] = None  # Excentricidad orbital
    koi_incl: Optional[float] = None  # Inclinación orbital (grados)
    
    # Características estelares (0-2% nulos)
    koi_steff: Optional[float] = None  # Temperatura efectiva estelar (K)
    koi_slogg: Optional[float] = None  # Gravedad superficial estelar
    koi_srad: Optional[float] = None  # Radio estelar (radios solares)
    koi_smass: Optional[float] = None  # Masa estelar (masas solares)
    koi_smet: Optional[float] = None  # Metalicidad estelar [Fe/H]
    koi_kepmag: Optional[float] = None  # Magnitud Kepler
    
    # Coordenadas (0% nulos)
    ra: Optional[float] = None
    dec: Optional[float] = None


class ExoplanetK2(BaseModel):
    """Modelo para exoplanetas del dataset K2 - Solo datos reales (≤50% nulos)"""
    # Identificadores (0% nulos)
    pl_name: Optional[str] = None
    hostname: Optional[str] = None
    epic_hostname: Optional[str] = None
    tic_id: Optional[str] = None
    gaia_id: Optional[str] = None
    
    # Clasificación y descubrimiento (0% nulos)
    disposition: Optional[str] = None
    discoverymethod: Optional[str] = None
    disc_year: Optional[int] = None
    
    # Datos del planeta (0-50% nulos)
    pl_orbper: Optional[float] = None  # Periodo orbital (días)
    pl_rade: Optional[float] = None  # Radio del planeta (radios terrestres)
    pl_radj: Optional[float] = None  # Radio del planeta (radios jovianos)
    pl_trandep: Optional[float] = None  # Profundidad del tránsito (%)
    pl_trandur: Optional[float] = None  # Duración del tránsito (horas)
    pl_tranmid: Optional[float] = None  # Tiempo medio del tránsito (BJD)
    pl_imppar: Optional[float] = None  # Parámetro de impacto (≤50% nulos)
    
    # Datos estelares (0-50% nulos)
    st_teff: Optional[float] = None  # Temperatura efectiva estelar (K)
    st_rad: Optional[float] = None  # Radio estelar (radios solares)
    st_mass: Optional[float] = None  # Masa estelar (masas solares)
    st_met: Optional[float] = None  # Metalicidad estelar [Fe/H]
    st_logg: Optional[float] = None  # Gravedad superficial estelar
    
    # Datos del sistema (0-50% nulos)
    sy_dist: Optional[float] = None  # Distancia al sistema (parsecs)
    sy_vmag: Optional[float] = None  # Magnitud V
    sy_kmag: Optional[float] = None  # Magnitud K
    sy_jmag: Optional[float] = None  # Magnitud J
    sy_hmag: Optional[float] = None  # Magnitud H
    sy_gaiamag: Optional[float] = None  # Magnitud Gaia
    
    # Coordenadas (0% nulos)
    ra: Optional[float] = None
    dec: Optional[float] = None


class ExoplanetTOI(BaseModel):
    """Modelo para exoplanetas del dataset TESS Objects of Interest (TOI) - Solo datos reales"""
    # Identificadores (0% nulos)
    toi: Optional[float] = None
    tid: Optional[int] = None  # TESS Input Catalog ID
    ctoi_alias: Optional[str] = None
    
    # Clasificación (0% nulos)
    tfopwg_disp: Optional[str] = None  # Disposición del grupo de trabajo
    pl_pnum: Optional[int] = None  # Número de planetas en el sistema
    
    # Datos del planeta (0% nulos)
    pl_orbper: Optional[float] = None  # Periodo orbital (días)
    pl_rade: Optional[float] = None  # Radio del planeta (radios terrestres)
    pl_eqt: Optional[float] = None  # Temperatura de equilibrio (K)
    pl_insol: Optional[float] = None  # Flujo de insolación (flujo terrestre)
    pl_trandep: Optional[float] = None  # Profundidad del tránsito (ppm)
    pl_trandurh: Optional[float] = None  # Duración del tránsito (horas)
    pl_tranmid: Optional[float] = None  # Tiempo medio del tránsito (BJD)
    pl_imppar: Optional[float] = None  # Parámetro de impacto
    pl_orbsmax: Optional[float] = None  # Semi-eje mayor (AU)
    
    # Datos estelares (0-3% nulos)
    st_teff: Optional[float] = None  # Temperatura efectiva estelar (K)
    st_rad: Optional[float] = None  # Radio estelar (radios solares)
    st_mass: Optional[float] = None  # Masa estelar (masas solares)
    st_met: Optional[float] = None  # Metalicidad estelar [Fe/H]
    st_logg: Optional[float] = None  # Gravedad superficial estelar
    st_dist: Optional[float] = None  # Distancia (parsecs)
    st_tmag: Optional[float] = None  # Magnitud TESS
    
    # Movimiento propio (0% nulos)
    st_pmra: Optional[float] = None  # Movimiento propio en ascensión recta (mas/yr)
    st_pmdec: Optional[float] = None  # Movimiento propio en declinación (mas/yr)
    
    # Coordenadas (0% nulos)
    ra: Optional[float] = None
    dec: Optional[float] = None
    rastr: Optional[str] = None
    decstr: Optional[str] = None


class DatasetListResponse(BaseModel):
    """Respuesta para listado de exoplanetas"""
    total: int
    count: int
    data: List[Dict]
