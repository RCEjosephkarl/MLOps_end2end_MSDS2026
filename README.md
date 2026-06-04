<<<<<<< HEAD
---
title: MLOps End2end MSDS2026
emoji: 🏆
colorFrom: purple
colorTo: gray
sdk: gradio
sdk_version: 6.16.0
python_version: '3.13'
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
=======
# MLOps End-to-End Pipeline — Computer Durability Classifier

Predicts whether a computer **needs replacement** using a full MLOps stack:

| Component | Role |
|---|---|
| **Dagster** | Workflow orchestration (asset-based DAG) |
| **MLflow** | Experiment tracking + Model Registry |
| **Evidently** | Data drift detection + quality monitoring |
| **FastAPI** | REST prediction endpoint |
| **Gradio** | Interactive demo UI (HF Spaces–ready) |

---

## Dataset

| File | Rows | Description |
|---|---|---|
| `Computer_Durability.csv` | 999 | Original dataset |
| `Computer_Durability_Plus.csv` | 2,999 | Original + 2,000 synthetic rows with mild drift |

**Features:** Hours Used Per Day · Cost · User Age · Primary Usage · Brand · Computer Age (Months)  
**Target:** `Needs Replacement` (binary, ~5–6% positive)

**Drift injected:** synthetic cohort has +2h/day average usage and −$3k average cost — detectable by Evidently's drift report.

---

## Setup

```bash
# 1. Create venv with Python 3.12 and install all dependencies
uv venv --python 3.12
uv pip install -e ".[dev]"
```

---

## Running the Pipeline

### Option A — Programmatic (script)

```bash
uv run python scripts/run_pipeline.py
```

Runs all 9 Dagster assets in order. Produces:
- `models/` — trained RF + XGBoost models, scaler, Optuna trial CSV
- `reports/` — Evidently HTML reports (data quality, drift, classification)
- `mlruns/` — MLflow experiment + model registry

### Option B — Dagster Web UI

```bash
uv run dagster dev
# Opens http://localhost:3000
# Navigate to Assets → Materialize All
```

### Data synthesis only

```bash
uv run python scripts/synthesize_data.py
```

---

## Model Serving

### Option A — Docker Compose (recommended)

```bash
# Build images (first time, or after code changes)
docker compose build

# Start API + Gradio in the background
docker compose up -d api gradio

# Stop
docker compose down
```

- FastAPI: `http://localhost:8000`
- Gradio UI: `http://localhost:7860`
- `HF_API_URL` is pre-configured in `docker-compose.yml` — no extra env setup needed.

### Option B — Local (uv)

```bash
# Terminal 1
uv run uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2
uv run python app.py
# Opens http://localhost:7860
```

### API Endpoints

- `GET  /health` — liveness check
- `GET  /info` — model version + metrics
- `POST /predict` — single prediction
- `POST /predict/batch` — batch predictions

Example:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "hours_used_per_day": 20.0,
    "cost": 12000,
    "user_age": 50,
    "primary_usage": 1,
    "brand": 2,
    "computer_age_months": 48
  }'
```

### Running Tests

```bash
docker compose run --rm test
```

### MLflow UI

```bash
uv run mlflow ui --backend-store-uri mlruns/
# Opens http://localhost:5000
```

---

## Project Structure

```
├── app.py                           # Gradio frontend (HF Spaces–ready)
├── Dockerfile                       # Multi-service image (api, gradio, test)
├── docker-compose.yml               # Orchestrates api (8000), gradio (7860), test services
├── .dockerignore                    # Excludes venv, cache, mlruns from build context
├── pyproject.toml                   # uv/pip project + dependencies
├── .python-version                  # Python 3.12
│
├── src/mlops_pipeline/
│   ├── config.py                    # Paths, column names, MLflow settings
│   ├── resources.py                 # Dagster MLflow resource
│   ├── definitions.py               # Dagster Definitions (asset wiring)
│   └── assets/
│       ├── data_assets.py           # raw_data, augmented_data, train_test_split
│       ├── training_assets.py       # baseline_rf_model, tuned_xgb_model, best_model_info
│       └── evaluation_assets.py     # data_quality_report, data_drift_report, model_eval_report
│
├── serving/
│   └── api.py                       # FastAPI prediction server
│
├── scripts/
│   ├── synthesize_data.py           # Generates Computer_Durability_Plus.csv
│   └── run_pipeline.py              # Runs all Dagster assets programmatically
│
├── models/                          # Trained model artifacts (generated)
├── reports/                         # Evidently HTML reports (generated)
├── mlruns/                          # MLflow tracking store (generated)
└── data/raw/
    ├── Computer_Durability.csv      # Original 999-row dataset
    └── Computer_Durability_Plus.csv # Augmented 2,999-row dataset (generated)
```

---

## Pipeline Asset DAG

```
raw_data ──────────────────────────────────────────┐
                                                    │
augmented_data ──► train_test_split_asset ──► baseline_rf_model ──► best_model_info ──► model_eval_report
                         │                                                   │
                         └──────────────────► tuned_xgb_model ──────────────┘
                                                                             │
augmented_data ──────────────────────────────────────────────► data_quality_report
                                                              data_drift_report
```

---

## Results (actual run)

| Model | ROC-AUC | Avg Precision | F1 |
|---|---|---|---|
| RandomForest (baseline) | 0.827 | 0.246 | 0.314 |
| **XGBoost + Optuna (winner)** | **0.841** | **0.289** | **0.376** |

**Champion model:** `ComputerDurabilityClassifier v1 @champion` in MLflow Registry

High Avg Precision (~0.29) on a 6% positive-rate dataset is meaningful — random baseline would score 0.06.

---

## HF Spaces Deployment

The `app.py` at the project root is already structured for Hugging Face Spaces:
1. Push the repo to HF
2. Set `HF_API_URL` as a Space secret pointing to your deployed FastAPI instance
3. Spaces will auto-launch `app.py`
>>>>>>> c1e8f69 (Initial commit: end-to-end MLOps pipeline project)
