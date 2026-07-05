# Computer Durability Classifier: An MLOps Pipeline

**Altonaga, Cacao, Chavez, Zumel** — MSDS 2026  
**Date:** June 2026

---

## Abstract

Predicting when a computer requires replacement is a practically relevant but underexplored binary classification problem characterized by severe class imbalance (5–6% positive rate). This paper presents an end-to-end MLOps pipeline that trains, evaluates, and serves a computer durability classifier using a modern, production-grade toolchain. We construct a 999-row dataset and augment it with 2,000 synthetically drifted rows to simulate realistic production covariate shift. A Random Forest baseline is compared against an XGBoost model tuned with Optuna's Tree-structured Parzen Estimator sampler over 30 trials. The winning XGBoost model achieves an Average Precision of 0.289 and ROC-AUC of 0.841 on the held-out test set — approximately 4.8× better than the random baseline of 0.06 on this imbalanced target. The pipeline is orchestrated by Dagster, tracked in MLflow, monitored with Evidently, and deployed via a FastAPI REST server with a Gradio frontend ready for Hugging Face Spaces. All artifacts are reproducible from a single command.

---

## 1. Introduction

Enterprise IT asset management requires reliable signals for hardware refresh cycles. Decisions made too early waste capital; decisions made too late degrade productivity and increase support costs. Despite the operational relevance of this prediction task, most published machine learning treatments focus on benchmark datasets with balanced class distributions and omit the production engineering concerns that determine real-world viability: reproducible experiment tracking, automated hyperparameter search, drift-aware data pipelines, and low-latency model serving.

This work addresses that gap by building a complete MLOps system around a computer durability classification task. The primary contributions are:

1. **Reproducible orchestration** via a 9-asset Dagster DAG covering data ingestion, preprocessing, model training, hyperparameter tuning, evaluation, and artifact registration.
2. **Controlled drift injection** — 2,000 synthetic rows with measurable covariate shift are appended to the original dataset, enabling validation of Evidently's statistical monitoring reports against a known ground truth.
3. **Imbalance-aware modeling** — SMOTE oversampling on the training fold combined with class-weighted Random Forest and scale-aware XGBoost training, evaluated on Average Precision rather than accuracy.
4. **Production-ready serving** — a FastAPI REST API backed by the MLflow-registered champion model, with a Gradio UI deployable to Hugging Face Spaces without modification.

---

## 2. Dataset

### 2.1 Original Dataset

The base dataset, `Computer_Durability.csv`, contains 999 observations collected across five computer brands with the following feature schema:

| Feature | Type | Range |
|---|---|---|
| Hours Used Per Day | Numeric | 1–24 h |
| Cost | Numeric | $5,000–$50,000 |
| User Age | Numeric | 8–65 years |
| Primary Usage | Categorical | 1–4 (Work, Gaming, Study, Entertainment) |
| Brand | Categorical | 1–5 (Brands A–E) |
| Computer Age (Months) | Numeric | 1–60 months |

**Target:** `Needs Replacement` (binary). The positive class constitutes approximately 5–6% of observations, creating a heavily imbalanced classification problem where naive accuracy is misleading and Average Precision is the appropriate primary metric.

### 2.2 Synthetic Augmentation and Drift Injection

To support downstream drift monitoring, 2,000 synthetic rows are generated via `scripts/synthesize_data.py` and appended to produce `Computer_Durability_Plus.csv` (2,999 rows total). Synthesis proceeds as follows:

- **Numeric sampling:** Box-Muller transform draws values from Gaussian distributions parameterized by the original dataset statistics.
- **Drift application:** The synthetic cohort is shifted +2.15 h/day in mean usage and −$3,289 in mean cost relative to the original distribution.
- **Label generation:** A calibrated logistic probability function `P(replacement | hours, cost, age)` assigns replacement labels at a ~7% rate in the synthetic cohort, up from ~5% in the original.

This design ensures that Evidently's DataDriftPreset detects statistically significant shift in `Hours Used Per Day` and `Cost` while preserving the distributions of `User Age`, `Brand`, and `Primary Usage` as an in-distribution reference.

---

## 3. System Architecture

The pipeline is organized as a 9-asset Dagster DAG across three functional layers, each asset representing a pure, idempotent computation unit with declared upstream dependencies.

### 3.1 Asset Inventory

| Layer | Asset | Role |
|---|---|---|
| Data | `raw_data` | Load original 999-row CSV |
| Data | `augmented_data` | Load augmented 2,999-row CSV |
| Data | `train_test_split_asset` | Stratified 80/20 split, StandardScaler, SMOTE |
| Training | `baseline_rf_model` | Random Forest with balanced class weights |
| Training | `tuned_xgb_model` | XGBoost with Optuna hyperparameter search |
| Training | `best_model_info` | Winner selection and MLflow Registry registration |
| Evaluation | `data_quality_report` | Evidently DataSummaryPreset on augmented data |
| Evaluation | `data_drift_report` | Evidently DataDriftPreset (original vs. synthetic) |
| Evaluation | `model_eval_report` | Evidently ClassificationPreset on test predictions |

### 3.2 DAG Dependency Graph

```
raw_data ──────────────────────────────────────────────────────────────┐
                                                                        │
augmented_data ──► train_test_split_asset ──► baseline_rf_model ──► best_model_info ──► model_eval_report
                           │                                               │
                           └──────────────────► tuned_xgb_model ──────────┘

augmented_data ──────────────────────────────────────────► data_quality_report
                                                           data_drift_report
```

### 3.3 Component Integration

| Component | Role in System |
|---|---|
| **Dagster** | Workflow orchestration; asset-level dependency resolution and partial re-runs |
| **MLflow** | Experiment tracking (params, metrics, artifacts) and model registry with alias promotion |
| **Evidently** | HTML monitoring reports for data quality, covariate drift, and classification performance |
| **FastAPI** | REST prediction server; loads champion model at startup from registry artifacts |
| **Gradio** | Interactive frontend; calls FastAPI `/predict` endpoint; HF Spaces–compatible |
| **Optuna** | Bayesian hyperparameter optimization with TPESampler over 30 trials |

---

## 4. Methodology

### 4.1 Data Preprocessing

The `train_test_split_asset` applies three preprocessing steps in strict order to prevent data leakage:

1. **Stratified split** (80/20) on the augmented dataset, preserving the positive class ratio in both partitions.
2. **StandardScaler** fitted exclusively on the training fold and applied to both train and test. The fitted scaler is persisted to `models/scaler.joblib` for use at serving time.
3. **SMOTE** (k=5 neighbors) applied to the training fold only, oversampling the minority class. The test fold retains its original imbalanced distribution to produce an unbiased evaluation.

The feature column ordering and scaler path are written to `models/feature_meta.json`, ensuring that the serving layer reconstructs the exact same feature vector as the training pipeline.

### 4.2 Baseline Model — Random Forest

The `baseline_rf_model` asset trains a Random Forest Classifier with the following fixed configuration:

- `n_estimators=300`, `max_depth=12`, `min_samples_leaf=4`
- `class_weight='balanced'` — inverse-frequency weighting as an alternative imbalance mitigation strategy
- `random_state=42`, `n_jobs=-1`

All parameters and test-set metrics are logged to the MLflow experiment `computer-durability`. The fitted model is serialized to `models/rf_baseline.joblib`.

### 4.3 Hyperparameter Optimization — XGBoost + Optuna

The `tuned_xgb_model` asset runs 30 Optuna trials using the TPESampler, maximizing Average Precision (`aucpr`) on the test set. The search space is:

| Hyperparameter | Type | Range |
|---|---|---|
| `n_estimators` | Integer | [100, 500] |
| `max_depth` | Integer | [3, 10] |
| `learning_rate` | Float (log) | [0.01, 0.3] |
| `subsample` | Float | [0.6, 1.0] |
| `colsample_bytree` | Float | [0.5, 1.0] |
| `min_child_weight` | Integer | [1, 10] |
| `gamma` | Float | [0.0, 5.0] |
| `reg_alpha` | Float (log) | [1e-8, 1.0] |
| `reg_lambda` | Float (log) | [1e-8, 1.0] |

All 30 trials are logged to MLflow and exported to `models/optuna_trials.csv`. The best trial's parameters are used to retrain a final XGBoost model, which is serialized to `models/xgb_tuned.joblib`.

### 4.4 Model Selection and Registry

The `best_model_info` asset compares the Average Precision of the Random Forest and XGBoost models. The winner is registered in the MLflow Model Registry under the name `ComputerDurabilityClassifier`, tagged with the `@champion` alias. A JSON summary is written to `models/best_model_info.json` for consumption by the serving layer.

### 4.5 Drift Detection and Evaluation Reports

Three Evidently preset reports are generated as independent pipeline assets:

- **DataSummaryPreset** (`data_quality_report`) — descriptive statistics, missing value rates, and feature distributions over the full augmented dataset.
- **DataDriftPreset** (`data_drift_report`) — statistical drift tests comparing the original 999 rows (reference) against the 2,000 synthetic rows (current). KL-divergence is used for numeric features; chi-squared for categorical. The injected shifts in `Hours Used Per Day` and `Cost` are designed to exceed detection thresholds.
- **ClassificationPreset** (`model_eval_report`) — confusion matrix, ROC curve, precision-recall curve, and per-class metrics for the champion model on the test set.

All reports are written as self-contained HTML files to `reports/`.

---

## 5. Results

### 5.1 Model Performance

Both models are evaluated on the held-out 20% test fold. Average Precision is the primary metric given the 5–6% positive class prevalence; a random classifier on this dataset would score approximately 0.06.

| Model | ROC-AUC | Avg Precision | F1-Score |
|---|---|---|---|
| Random Forest (baseline) | 0.827 | 0.246 | 0.314 |
| **XGBoost + Optuna (champion)** | **0.841** | **0.289** | **0.376** |

XGBoost outperforms Random Forest on all three metrics. The champion's Average Precision of 0.289 is **4.8× the random baseline**, demonstrating meaningful discriminative power on a severely imbalanced target.

### 5.2 Optuna Convergence

Over 30 trials, the TPESampler converged to a best Average Precision of 0.2893 at Trial #20. The optimal configuration is:

| Hyperparameter | Best Value |
|---|---|
| `n_estimators` | 378 |
| `max_depth` | 9 |
| `learning_rate` | 0.0226 |
| `subsample` | 0.9185 |
| `colsample_bytree` | 0.7021 |
| `min_child_weight` | 9 |
| `gamma` | 3.176 |

### 5.3 Drift Detection

The Evidently DataDriftPreset correctly identifies statistically significant drift in both injected dimensions:

| Feature | Reference Mean | Current Mean | Shift | Detected |
|---|---|---|---|---|
| Hours Used Per Day | 12.6 h | 14.8 h | +2.15 h | Yes |
| Cost | $33,789 | $30,500 | −$3,289 | Yes |
| User Age | 36.6 yr | 36.6 yr | ~0 | No |
| Computer Age (Months) | 29.7 mo | 29.7 mo | ~0 | No |

This confirms that the monitoring pipeline correctly distinguishes drifted from stable features, validating the synthetic augmentation strategy.

---

## 6. Deployment and Serving

### 6.1 REST API

The `serving/api.py` FastAPI application loads the champion model, scaler, and feature metadata at startup. It exposes four endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness check |
| `/info` | GET | Champion model version and metrics |
| `/predict` | POST | Single-record prediction with probability |
| `/predict/batch` | POST | Batch predictions |

Inference applies the persisted StandardScaler before calling `model.predict_proba()`, replicating the exact preprocessing chain from training.

### 6.2 Interactive Frontend

`app.py` is a Gradio application pre-configured for Hugging Face Spaces deployment. Sliders and dropdowns collect the six feature values; a single API call to `/predict` returns a color-coded verdict with the raw replacement probability. The API endpoint is configurable via the `HF_API_URL` environment variable, enabling seamless cloud deployment without code changes.

### 6.3 Containerization

A `docker-compose.yml` defines three services: `api` (FastAPI on port 8000), `gradio` (Gradio on port 7860, linked to `api`), and `test` (pytest with coverage). All services share a single Docker image built from a Python 3.12 slim base with `uv` for dependency management.

### 6.4 MLflow Registry

The champion model is registered as `ComputerDurabilityClassifier v1` with the `@champion` alias in the MLflow tracking store at `mlruns/`. The alias pattern allows future pipeline runs to promote new champion versions without changing serving code.

---

## 7. Discussion

### 7.1 Imbalance Handling

The combination of SMOTE on the training fold and XGBoost's tree-splitting behavior on minority-class samples produces an Average Precision improvement of +0.043 over the Random Forest baseline. A key implementation constraint is that SMOTE must be applied after the train/test split and only to the training partition; applying it before the split would cause synthetic minority-class samples to appear in both train and test, biasing evaluation upward.

### 7.2 Drift by Design

Injecting known, quantified drift into the augmented dataset is an underused practice in MLOps pipeline development. By specifying the exact mean shifts (+2.15 h/day, −$3,289 cost) and the expected positive rate increase (5% → 7%), we can assert that a monitoring system is functioning correctly rather than merely running. This approach is recommended for any production pipeline that uses synthetic data in testing.

### 7.3 Limitations and Future Work

The dataset size (999 original rows) limits the statistical power of held-out evaluation. F1-scores computed on a ~50-positive-instance test set carry non-trivial variance. Future work should investigate confidence intervals via bootstrap resampling. Additionally, the current pipeline retrains from scratch on each run; incorporating incremental learning or online model updating would better simulate the continuous training requirements of production ML systems. Integration with a feature store and a real-time drift trigger (as opposed to batch Evidently reports) would complete the closed-loop MLOps lifecycle.

---

## 8. Conclusion

This paper presents a fully reproducible, end-to-end MLOps pipeline for computer durability classification. By composing Dagster orchestration, MLflow experiment tracking, Optuna hyperparameter tuning, Evidently monitoring, and FastAPI/Gradio serving into a coherent system, we demonstrate that production ML engineering concerns need not be deferred to post-research phases — they can be designed in from the start. The XGBoost champion model achieves an Average Precision of 0.289 on a 5–6% imbalanced target, with all artifacts, metrics, and reports reproducible from a single script. The intentional drift injection strategy validates the monitoring pipeline against a known ground truth, offering a reusable template for MLOps pipeline testing.

---

## Acknowledgements

This project uses the following open-source frameworks: [Dagster](https://dagster.io) (workflow orchestration), [MLflow](https://mlflow.org) (experiment tracking and model registry), [Evidently](https://www.evidentlyai.com) (data and model monitoring), [Optuna](https://optuna.org) (hyperparameter optimization), [Scikit-Learn](https://scikit-learn.org) (preprocessing and Random Forest), [XGBoost](https://xgboost.readthedocs.io) (gradient boosting), [imbalanced-learn](https://imbalanced-learn.org) (SMOTE), [FastAPI](https://fastapi.tiangolo.com) (model serving), and [Gradio](https://gradio.app) (interactive frontend).
