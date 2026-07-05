# Project Setup and Execution Guide

## Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/) package manager

---

## 1. Setup

```bash
# Create virtual environment with Python 3.12 and install all dependencies
uv venv --python 3.12
uv pip install -e ".[dev]"
```

---

## 2. Generate Augmented Dataset (one-time)

```bash
uv run python scripts/synthesize_data.py
```

Produces `Computer_Durability_Plus.csv` (2,999 rows: original 999 + 2,000 synthetic with drift).

---

## 3. Run the Full Pipeline

### Option A — Programmatic (recommended)

```bash
uv run python scripts/run_pipeline.py
```

Materializes all 9 Dagster assets in dependency order. Outputs:
- `models/` — trained RF + XGBoost models, scaler, Optuna trial CSV, best model JSON
- `reports/` — Evidently HTML reports (data quality, drift, classification)
- `mlruns/` — MLflow experiment tracking + model registry

### Option B — Dagster Web UI

```bash
uv run dagster dev
# Opens http://localhost:3000
# Navigate to: Assets → Materialize All
```

---

## 4. Model Serving

### Start FastAPI

```bash
uv run uvicorn serving.api:app --host 0.0.0.0 --port 8000 --reload
```

Available endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/info` | GET | Model version and metrics |
| `/predict` | POST | Single prediction |
| `/predict/batch` | POST | Batch predictions |

Example single prediction:

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

### Start Gradio Demo

```bash
# In a second terminal (API must be running on port 8000)
uv run python app.py
# Opens http://localhost:7860
```

---

## 5. MLflow UI

```bash
uv run mlflow ui --backend-store-uri mlruns/
# Opens http://localhost:5000
```

Browse experiment runs, compare metrics, and inspect the `ComputerDurabilityClassifier` model registry.

---

## 6. Docker Compose (all services)

```bash
docker-compose up
```

Starts three services simultaneously:
- `api` — FastAPI prediction server at `http://localhost:8000`
- `gradio` — Gradio frontend at `http://localhost:7860`
- `test` — runs pytest with coverage and exits

---

## 7. Hugging Face Spaces Deployment

1. Push this repository to a Hugging Face Space.
2. Deploy your FastAPI server separately (e.g., on a cloud VM or container service).
3. Set `HF_API_URL` as a Space secret pointing to your FastAPI instance URL.
4. Spaces will auto-launch `app.py`.

---

## Project Structure

```
├── Computer_Durability.csv          # Original 999-row dataset
├── Computer_Durability_Plus.csv     # Augmented 2,999-row dataset (generated)
├── app.py                           # Gradio frontend (HF Spaces–ready)
├── pyproject.toml                   # uv/pip project + dependencies
├── todo.md                          # This file
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
└── data/raw/                        # Raw CSV copies
```
