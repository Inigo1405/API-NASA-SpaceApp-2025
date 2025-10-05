from __future__ import annotations

import json
import os
import threading
from typing import Dict, Tuple, Optional, List
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download


class _Singleton(type):
    """Thread-safe Singleton metaclass (una instancia por proceso)."""
    _instances: Dict[type, object] = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # Double-checked locking
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class ExoMACModel(metaclass=_Singleton):
    """
    Misión-agnóstico: cargador de modelo (Pipeline sklearn) entrenado con Kepler/K2/TESS.
    - Descarga artefactos desde Hugging Face SOLO si no existen localmente.
    - Guarda/lee desde una carpeta local del proyecto (por defecto: ./models/ExoMAC-KKT).
    - Exposición de helpers de predicción y de features ingenierizadas.
    """

    DEFAULT_REPO = "ZapatoProgramming/ExoMAC-KKT"
    _FILENAMES = {
        "model":  "exoplanet_best_model.joblib",
        "feats":  "exoplanet_feature_columns.json",
        "labels": "exoplanet_class_labels.json",
        "meta":   "exoplanet_metadata.json",
    }

    def __init__(
        self,
        repo_id: Optional[str] = None,
        token: Optional[str] = None,
        prefer_snapshot: bool = True,
        allow_patterns: Optional[List[str]] = None,
        local_dir: Optional[str | os.PathLike] = None,
        always_download: bool = False,
        verbose: bool = True,
    ):
        """
        Args:
            repo_id: Hugging Face repo id. Por defecto 'ZapatoProgramming/ExoMAC-KKT'.
            token:   Token HF si el repo es privado.
            prefer_snapshot: Si True, usa snapshot_download (descarga por patrón).
            allow_patterns:  Patrones a descargar cuando prefer_snapshot=True.
            local_dir:       Carpeta donde se guardan/leen artefactos en tu proyecto.
            always_download: Si True, fuerza descarga (útil para actualizar).
            verbose:         Imprime mensajes útiles.
        """
        self.repo_id = repo_id or self.DEFAULT_REPO
        self.token = token
        self.prefer_snapshot = prefer_snapshot
        self.allow_patterns = allow_patterns or ["artifacts/*", "*.joblib", "*.json"]
        self.local_dir = Path(local_dir or (Path("models") / self.repo_id.split("/")[-1]))
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.always_download = always_download
        self.verbose = verbose

        self._model = None
        self._feature_columns: List[str] = []
        self._class_labels: List[str] = []
        self._metadata: Dict = {}

        self._load_artifacts()

    # ------------------------- PUBLIC API -------------------------

    @property
    def model(self):
        return self._model

    @property
    def feature_columns(self) -> List[str]:
        return list(self._feature_columns)

    @property
    def class_labels(self) -> List[str]:
        return list(self._class_labels)

    @property
    def metadata(self) -> Dict:
        return dict(self._metadata)

    def predict(
        self,
        params: Dict[str, float],
        return_proba: bool = True,
        compute_engineered_if_missing: bool = True,
    ) -> Tuple[str, Optional[Dict[str, float]]]:
        """
        Predice una etiqueta y (opcionalmente) probabilidades para un dict de features.
        - Rellena features ingenierizadas si el modelo las espera y no están.
        """
        if compute_engineered_if_missing:
            params = self._ensure_engineered_features(dict(params))

        X = pd.DataFrame([params], dtype=float).reindex(columns=self._feature_columns)
        y_idx = int(self._model.predict(X)[0])
        label = self._class_labels[y_idx]

        if not return_proba:
            return label, None

        proba = None
        try:
            p = self._model.predict_proba(X)[0]
            proba = {lbl: float(prob) for lbl, prob in zip(self._class_labels, p)}
        except Exception:
            pass
        return label, proba

    def predict_with_debug(self, params: Dict[str, float]) -> Tuple[str, Optional[Dict[str, float]]]:
        """
        Igual que predict(), pero imprime features reconocidas/desconocidas y faltantes.
        """
        params2 = self._ensure_engineered_features(dict(params))
        X = pd.DataFrame([params2], dtype=float).reindex(columns=self._feature_columns)

        recognized = [c for c in self._feature_columns if c in params2]
        unknown = [k for k in params2.keys() if k not in self._feature_columns]
        missing = X.columns[X.iloc[0].isna()].tolist()

        print(f"Recognized: {len(recognized)}/{len(self._feature_columns)}")
        if recognized:
            print("  •", ", ".join(recognized[:16]) + (" ..." if len(recognized) > 16 else ""))
        if unknown:
            print(f"Unknown keys: {len(unknown)}")
            if unknown:
                print("  •", ", ".join(unknown[:16]) + (" ..." if len(unknown) > 16 else ""))
        if missing:
            print(f"Missing (imputed): {len(missing)}")
            if missing:
                print("  •", ", ".join(missing[:16]) + (" ..." if len(missing) > 16 else ""))

        return self.predict(params2, return_proba=True, compute_engineered_if_missing=False)

    # ------------------------- INTERNALS -------------------------

    def _load_artifacts(self) -> None:
        """
        1) Si ya existen archivos locales y always_download=False -> NO descarga.
        2) Si faltan archivos o always_download=True -> descarga (snapshot o per-file).
        3) Carga el modelo + metadata desde disco.
        """
        paths: Optional[Dict[str, str]] = None

        # (0) Intentar leer desde local sin tocar red
        if not self.always_download:
            local_paths = self._try_local_paths()
            if local_paths is not None:
                paths = local_paths
                if self.verbose:
                    print(f"[ExoMAC] Using cached artifacts in {self.local_dir}")
            else:
                if self.verbose:
                    print(f"[ExoMAC] Local artifacts not found. Will download to {self.local_dir}.")

        # (1) Descargar si hace falta
        if paths is None:
            if self.prefer_snapshot:
                # Descarga patrones a la carpeta local (la API ya no usa symlinks)
                snapshot_download(
                    repo_id=self.repo_id,
                    token=self.token,
                    allow_patterns=self.allow_patterns,
                    local_dir=str(self.local_dir),
                )
                paths = self._resolve_from_dir(self.local_dir)
            else:
                paths = {}
                for key, fname in self._FILENAMES.items():
                    paths[key] = self._get_artifact_to_local_dir(fname)

        # (2) Cargar desde disco
        self._model = joblib.load(paths["model"])
        self._feature_columns = json.load(open(paths["feats"], "r", encoding="utf-8"))
        self._class_labels    = json.load(open(paths["labels"], "r", encoding="utf-8"))
        self._metadata        = json.load(open(paths["meta"], "r", encoding="utf-8"))

        if self.verbose:
            print(f"[ExoMAC] Loaded model from {paths['model']}")

    # --- Local path helpers ---

    def _have_all_files(self, base: Path) -> bool:
        """¿Están TODOS los artefactos (en artifacts/ o raíz) en 'base'?"""
        base = Path(base)
        for _, name in self._FILENAMES.items():
            p1 = base / "artifacts" / name
            p2 = base / name
            if not (p1.exists() or p2.exists()):
                return False
        return True

    def _try_local_paths(self) -> Optional[Dict[str, str]]:
        """Devuelve rutas locales si todo existe; si falta algo, None."""
        if self._have_all_files(self.local_dir):
            return self._resolve_from_dir(self.local_dir)
        return None

    def _resolve_from_dir(self, base_dir: Path | str) -> Dict[str, str]:
        """
        Selecciona artifacts/<name> si existe; si no, <base>/<name>.
        """
        base_dir = Path(base_dir)
        out: Dict[str, str] = {}
        for key, name in self._FILENAMES.items():
            p1 = base_dir / "artifacts" / name
            p2 = base_dir / name
            if p1.exists():
                out[key] = str(p1)
            elif p2.exists():
                out[key] = str(p2)
            else:
                raise FileNotFoundError(f"Could not find {name} under {base_dir}")
        return out

    def _get_artifact_to_local_dir(self, fname: str) -> str:
        """
        Descarga a self.local_dir con hf_hub_download (si tu versión soporta local_dir).
        Si no, descarga a la caché global y copia a self.local_dir.
        """
        self.local_dir.mkdir(parents=True, exist_ok=True)

        for candidate in (f"artifacts/{fname}", fname):
            try:
                # huggingface_hub >= 0.23 soporta local_dir
                path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=candidate,
                    token=self.token,
                    local_dir=str(self.local_dir),
                )
                return path
            except TypeError:
                # Fallback: versión antigua sin local_dir
                cache_path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=candidate,
                    token=self.token,
                )
                dst = self.local_dir / Path(candidate).name
                os.makedirs(self.local_dir, exist_ok=True)
                if not os.path.exists(dst):
                    from shutil import copy2
                    copy2(cache_path, dst)
                return str(dst)
            except Exception:
                # prueba siguiente candidato (raíz en lugar de artifacts/)
                continue

        raise FileNotFoundError(f"Could not download {fname} from {self.repo_id}")

    # --- Engineered features helpers ---

    def _ensure_engineered_features(self, d: Dict[str, float]) -> Dict[str, float]:
        """
        Rellena features ingenierizadas si el modelo las espera y no están:
        - duty_cycle, log_koi_period, log_koi_depth, teq_proxy
        - koi_snr/log_koi_snr  o  snr_proxy/log_snr_proxy (proxy)
        """
        need = set(self._feature_columns)

        # Duty cycle
        if "duty_cycle" in need and "duty_cycle" not in d:
            if all(k in d for k in ("koi_duration", "koi_period")) and d.get("koi_period"):
                d["duty_cycle"] = d["koi_duration"] / (d["koi_period"] * 24.0)

        # Logs
        if "log_koi_period" in need and "log_koi_period" not in d and d.get("koi_period", 0) > 0:
            d["log_koi_period"] = np.log10(d["koi_period"])
        if "log_koi_depth" in need and "log_koi_depth" not in d and d.get("koi_depth", 0) > 0:
            d["log_koi_depth"] = np.log10(d["koi_depth"])

        # teq_proxy (simple)
        if "teq_proxy" in need and "teq_proxy" not in d and "koi_steff" in d:
            d["teq_proxy"] = d["koi_steff"]

        # SNR real o proxy
        if "koi_snr" in need and "koi_snr" not in d:
            d["koi_snr"] = np.nan
        if "log_koi_snr" in need and "log_koi_snr" not in d and d.get("koi_snr", 0) > 0:
            d["log_koi_snr"] = np.log10(d["koi_snr"])

        if "snr_proxy" in need and "snr_proxy" not in d:
            if all(k in d for k in ("koi_depth", "koi_duration", "koi_period")) and d.get("koi_period", 0) > 0:
                d["snr_proxy"] = d["koi_depth"] * np.sqrt(max(d["koi_duration"] / (d["koi_period"] * 24.0), 1e-12))
        if "log_snr_proxy" in need and "log_snr_proxy" not in d and d.get("snr_proxy", 0) > 0:
            d["log_snr_proxy"] = np.log10(d["snr_proxy"])

        return d


# ------------------------- DEMO -------------------------
if __name__ == "__main__":
    # Primera ejecución: descargará a ./models/ExoMAC-KKT si no existe.
    model = ExoMACModel(
        local_dir="./ExoMACModel/ExoMAC-KKT",
        prefer_snapshot=True,
        always_download=False,   # <- ejecuciones siguientes NO vuelven a descargar
        verbose=True,
    )

    # Subsecuentes: misma instancia (singleton) y SIN descarga.
    same_model = ExoMACModel(local_dir="./ExoMACModel/ExoMAC-KKT")
    assert model is same_model

    # Ejemplo mínimo de predicción
    params = {
        "koi_period": 12.0, "koi_duration": 3.5, "koi_depth": 600.0, "koi_impact": 0.20,
        "koi_prad": 2.1, "koi_slogg": 4.4, "koi_sma": 0.10, "koi_smet": 0.0,
        "koi_srad": 1.0, "koi_steff": 5700.0, "koi_snr": 12.0,
    }
    label, proba = model.predict_with_debug(params)
    print("Predicted:", label)
    print("Local dir:", model.local_dir.resolve())
