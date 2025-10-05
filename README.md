# Exoplanet Classification using NASA Datasets (Kepler, K2, TESS) — README

This README explains **how we trained** a **mission-agnostic** classifier to predict whether a detected object is **CONFIRMED**, **CANDIDATE**, or **FALSE POSITIVE** using public catalogs from **Kepler**, **K2**, and **TESS**. It documents the data ingestion, harmonization, feature engineering (including SNR logic), preprocessing, model selection, evaluation, class-imbalance handling, explainability, and inference artifacts.

---

## Project Structure (key files)

```
.
├─ Exoplanet_Classification_NASA_Kepler_K2_TESS_withSNR_ROBUST.ipynb   # main notebook
└─ artifacts/
   ├─ exoplanet_best_model.joblib            # full sklearn Pipeline (preprocessing + estimator)
   ├─ exoplanet_feature_columns.json         # exact feature order used during training
   ├─ exoplanet_class_labels.json            # label names in prediction index order
   └─ exoplanet_metadata.json                # summary (best model name, n_features, timestamp)
```

---

## Datasets

- **Kepler Cumulative:** `cumulative_YYYY.MM.DD_HH.MM.SS.csv`
- **K2 Planets & Candidates:** `k2pandc_YYYY.MM.DD_HH.MM.SS.csv`
- **TESS Objects of Interest (TOI):** `TOI_YYYY.MM.DD_HH.MM.SS.csv`

Each table contains per-candidate features and a mission-specific disposition field. These missions are **complementary** (different cadences/systems), so merging increases coverage and diversity.

---

## Environment

- Python ≥ 3.9  
- Libraries: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `imblearn`, optional `xgboost`, optional `shap`, `joblib`.

> If `xgboost` or `shap` aren’t installed, the notebook automatically skips those steps.

---

## Reproducibility

- Random seeds set to **42** in model splits and estimators.
- Stratified splits used for consistent class composition.
- The exact **feature order** and **class label order** are saved to JSON and re-used at inference.

---

## Data Loading (robust, no `low_memory`)

CSV parsing is **robust**:

- Try multiple separators and engines (prefers **Python engine** with `on_bad_lines="skip"`).
- Normalize column names to `snake_case`.
- Re-detect header if the first read looks suspicious (e.g., numeric column names).
- Drop columns that are entirely empty.

> We deliberately **do not** pass `low_memory` (it’s unsupported by the Python engine and can degrade type inference).

---

## Label Harmonization

Different catalogs use different labels. We map all to a **unified set**:

- **CONFIRMED**: `CONFIRMED`, `CP`, `KP`
- **CANDIDATE**: `CANDIDATE`, `PC`, `CAND`, `APC`
- **FALSE POSITIVE**: `FALSE POSITIVE`, `FP`, `FA`, `REFUTED`

Rows without a recognized disposition are dropped before modeling.

---

## Feature Harmonization (mission-agnostic)

We build a **canonical numeric feature set** by scanning a list of **alias groups** and picking the **first numeric-coercible** column present per mission, e.g.:

- Period: `['koi_period','orbital_period','period','pl_orbper','per']`
- Duration: `['koi_duration','duration','tran_dur']`
- Depth: `['koi_depth','depth','tran_depth']`
- Stellar context: `['koi_steff','st_teff','teff']`, `['koi_slogg','st_logg','logg']`, `['koi_smet','st_metfe','feh']`, `['koi_srad','st_rad','star_radius']`
- Transit geometry: `['koi_impact','impact','b']`, `['koi_sma','sma','a_rs','semi_major_axis']`
- **SNR/MES (broad):** `['koi_snr','koi_model_snr','koi_mes','mes','max_mes','snr','detection_snr','transit_snr','signal_to_noise','koi_max_snr']`

We **coerce to numeric** (`errors='coerce'`) and accept if any non-null values remain.

---

## Feature Engineering

Physics-motivated features improve separability:

- **Duty cycle**:  
  `duty_cycle = koi_duration / (koi_period * 24.0)`  
  (transit duration relative to orbital period)
- **Log transforms**:  
  `log_koi_period = log10(koi_period)`  
  `log_koi_depth  = log10(koi_depth)`
- **Equilibrium-temperature proxy**:  
  - Simple: `teq_proxy = koi_steff`  
  - Refined (if `a/R*` available or derivable): `teq_proxy = koi_steff / sqrt(2 * a_rs)`
- **SNR logic (guaranteed SNR-like feature)**:  
  - If any mission exposes a usable SNR/MES, we use **`koi_snr`** and also compute **`log_koi_snr`**.  
  - Otherwise we compute a **mission-agnostic proxy**:  
    ```
    snr_proxy = koi_depth * sqrt( koi_duration / (koi_period * 24.0) )
    log_snr_proxy = log10(snr_proxy)
    ```

> The exact features used in training are whatever appear in `artifacts/exoplanet_feature_columns.json` (this list is created dynamically from the actual files you load).

---

## Mission-Agnostic Policy

- We keep a temporary `mission` column only for auditing, then **drop it** before training.
- Features are derived from **physical quantities**, not the mission identity.

---

## Preprocessing & Split

- Keep **numeric** columns only.
- **Imputation**: median (`SimpleImputer(strategy="median")`).
- **Scaling**: `RobustScaler()` (robust to outliers).
- **Label encoding**: `LabelEncoder` → integer target (needed by XGBoost).
- **Split**: `train_test_split(test_size=0.2, stratify=y, random_state=42)`.

---

## Model Selection & Training

We wrap preprocessing and the classifier in a **single Pipeline**, and optimize **macro-F1** with `RandomizedSearchCV` using `StratifiedKFold`:

- **Random Forest** (class_weight="balanced")
  - `n_estimators`: 150–600 (fast mode: 150/300)
  - `max_depth`: None or small integers
  - `min_samples_split`, `min_samples_leaf`
- **SVM (RBF)** (class_weight="balanced")
  - `C`: logspace grid
  - `gamma`: `["scale","auto"]`
- **XGBoost** (optional; only if installed)
  - `objective="multi:softprob"`, `eval_metric="mlogloss"`, `tree_method="hist"`
  - Grid over `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`, `n_estimators`

> We use a `FAST` flag (defaults to **True**) to keep grids small for iteration speed. Disabling FAST expands the search.

---

## Evaluation

- **Primary metric**: **Macro-F1** (treats classes evenly).
- Also report **accuracy** and per-class **precision/recall/F1**.
- **Confusion matrices** (shared color scale across models).
- **ROC AUC** (one-vs-rest) when probability estimates are available.

---

## Handling Class Imbalance

We provide an **optional** SMOTE section using `imblearn.Pipeline`:

```
ImbPipeline([
  ("impute", SimpleImputer(median)),
  ("scale", RobustScaler()),
  ("smote", SMOTE()),
  ("clf", RandomForestClassifier or SVC(probability=True))
])
```

- We **avoid nested sklearn Pipelines** inside the imblearn pipeline to prevent:
  > `TypeError: All intermediate steps of the chain should not be Pipelines`.

The SMOTE models are tuned on macro-F1 and compared against the non-SMOTE runs.

---

## Explainability

- **Permutation importance** computed on the **full pipeline** (so it reflects preprocessing).
- **SHAP** (if installed) for tree-based models:
  - Global bar chart of mean |SHAP| across classes
  - Per-class bar chart (e.g., CONFIRMED)  
  - SHAP summary bar (optional)

If the best model isn’t tree-based, SHAP is skipped automatically.

---

## Saved Artifacts

After selecting the **best model** (highest macro-F1 on the held-out test set), we serialize:

- `artifacts/exoplanet_best_model.joblib` – the full sklearn **Pipeline**
- `artifacts/exoplanet_feature_columns.json` – **exact feature order** expected at inference
- `artifacts/exoplanet_class_labels.json` – class names in output index order
- `artifacts/exoplanet_metadata.json` – name, n_features, labels, timestamp

These provide **stable inference** even if the notebook or environment changes later.

---

## Inference Usage

We include a helper that builds an input row with the **exact training feature order**, warns about **unknown/missing** keys, and prints predictions and class probabilities:

```python
from pathlib import Path
import json, joblib, numpy as np, pandas as pd

ARTIFACTS_DIR = Path("artifacts")
model = joblib.load(ARTIFACTS_DIR / "exoplanet_best_model.joblib")
feature_columns = json.loads((ARTIFACTS_DIR / "exoplanet_feature_columns.json").read_text())
class_labels    = json.loads((ARTIFACTS_DIR / "exoplanet_class_labels.json").read_text())

def predict_with_debug(model, feature_columns, class_labels, params: dict):
    X = pd.DataFrame([params], dtype=float).reindex(columns=feature_columns)
    y_idx = int(model.predict(X)[0])
    label = class_labels[y_idx]
    print("Prediction:", label)
    try:
        proba = model.predict_proba(X)[0]
        for lbl, p in sorted(zip(class_labels, proba), key=lambda t: t[1], reverse=True):
            print(f"  {lbl:>15s}: {p:.3f}")
    except Exception:
        pass
    return label
```

### Engineered features to compute in your inputs

- `duty_cycle = koi_duration / (koi_period * 24.0)`
- `log_koi_period = log10(koi_period)`
- `log_koi_depth = log10(koi_depth)`
- `teq_proxy = koi_steff` (or `koi_steff / sqrt(2 * a_rs)` if using refined variant)
- If the trained feature list includes **`koi_snr`**: also send `log_koi_snr = log10(koi_snr)`  
- Else, if it includes **`snr_proxy`**: send `snr_proxy = koi_depth * sqrt(koi_duration / (koi_period*24.0))` and `log_snr_proxy`

> You can check which SNR flavor was used by inspecting `feature_columns.json`.

---

## Design Choices & Rationale

- **Mission-agnostic**: We drop mission identity; rely on physical features to generalize.
- **Macro-F1**: More balanced assessment when class counts differ (especially for FALSE POSITIVE).
- **RobustScaler**: Less sensitive to outliers than StandardScaler.
- **LabelEncoder**: Ensures XGBoost gets integers (fixes “Invalid classes inferred” errors).
- **SNR strategy**: If SNR/MES exists we use it; otherwise we inject a **physics-motivated proxy** to preserve detection strength information across missions.

---

## Troubleshooting

- **CSV ParserError / “python engine” warnings**  
  We avoid `low_memory` and use Python engine with `on_bad_lines="skip"`. The loader will try alternate separators and re-try with a detected header line.

- **SMOTE “intermediate steps should not be Pipelines”**  
  We use a single `imblearn.Pipeline` with inline steps (impute → scale → SMOTE → clf).

- **`PicklingError: QuantileCapper`**  
  We removed custom transformers and rely on standard sklearn components to ensure picklability.

- **`AttributeError: 'numpy.ndarray' has no attribute 'columns'`**  
  We compute permutation importance on the **pipeline** (not the bare estimator), and we take feature names from `X_train.columns`.

- **XGBoost “Invalid classes inferred”**  
  We label-encode `y` to integers (`LabelEncoder`) before fitting.

- **Different inputs return the “same” class**  
  Often caused by: (a) too few features provided (most imputed), (b) features outside training ranges (scaled to similar z-scores), or (c) providing keys not used by the model. Use `predict_with_debug` to see **recognized/unknown/missing** features and adjust your dict.

---

## How to Re-train

1. Open the notebook `Exoplanet_Classification_NASA_Kepler_K2_TESS_withSNR_ROBUST.ipynb`.
2. Ensure the three CSV files are available at the paths set in the **Data Loading** cell.
3. (Optional) Set `FAST = False` in the training cell to run a broader hyper-parameter search.
4. Run all cells. Artifacts are written to `./artifacts/`.

---

## Extending the Work

- Add more SNR/MES name variants if future catalogs introduce new headers.
- Experiment with **calibrated probabilities** (e.g., `CalibratedClassifierCV`) for better ROC behavior.
- Try additional models (LightGBM/CatBoost) if available.
- Incorporate vetting-pipeline shape features (e.g., V-shape metrics) when harmonizable across missions.
- Add **temporal cross-validation** per mission or per release to stress test generalization.
