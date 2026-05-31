"""
Dagster training assets:
  baseline_rf_model  → RandomForest with class_weight='balanced'
  tuned_xgb_model    → XGBoost tuned via Optuna (30 trials)
  best_model_info    → compare RF vs XGBoost, pick winner
All runs logged to MLflow.
"""
import json
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import optuna
import xgboost as xgb
from dagster import AssetExecutionContext, asset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

from mlops_pipeline.config import (
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
)
from mlops_pipeline.resources import MLflowResource

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _eval_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "avg_precision": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


@asset(group_name="training", required_resource_keys={"mlflow"})
def baseline_rf_model(
    context: AssetExecutionContext,
    train_test_split_asset: dict[str, Any],
) -> dict[str, Any]:
    """Train a RandomForest baseline and log to MLflow."""
    mlflow_res: MLflowResource = context.resources.mlflow
    mlflow_res.set_experiment()

    X_train = train_test_split_asset["X_train"]
    y_train = train_test_split_asset["y_train"]
    X_test = train_test_split_asset["X_test"]
    y_test = train_test_split_asset["y_test"]

    with mlflow.start_run(run_name="RandomForest-baseline") as run:
        params = {
            "n_estimators": 300,
            "max_depth": 12,
            "min_samples_leaf": 4,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1,
        }
        mlflow.log_params(params)

        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)

        y_prob = rf.predict_proba(X_test)[:, 1]
        metrics = _eval_metrics(y_test, y_prob)
        mlflow.log_metrics(metrics)

        # Log model artifact
        mlflow.sklearn.log_model(rf, artifact_path="model")

        run_id = run.info.run_id
        context.log.info(f"RF  roc_auc={metrics['roc_auc']:.4f}  avg_pr={metrics['avg_precision']:.4f}")

    import joblib
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(rf, MODELS_DIR / "rf_baseline.joblib")

    context.add_output_metadata({"run_id": run_id, **metrics})
    return {"model_name": "RandomForest", "run_id": run_id, "metrics": metrics}


@asset(group_name="training", required_resource_keys={"mlflow"})
def tuned_xgb_model(
    context: AssetExecutionContext,
    train_test_split_asset: dict[str, Any],
) -> dict[str, Any]:
    """XGBoost hyperparameter search via Optuna (30 trials), log best run to MLflow."""
    mlflow_res: MLflowResource = context.resources.mlflow
    mlflow_res.set_experiment()

    X_train = train_test_split_asset["X_train"]
    y_train = train_test_split_asset["y_train"]
    X_test = train_test_split_asset["X_test"]
    y_test = train_test_split_asset["y_test"]

    scale_pos_weight = float((y_train == 0).sum()) / float((y_train == 1).sum())

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
            "use_label_encoder": False,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        y_prob = model.predict_proba(X_test)[:, 1]
        return average_precision_score(y_test, y_prob)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=30, show_progress_bar=False)
    context.log.info(f"Optuna best avg_pr={study.best_value:.4f} in {len(study.trials)} trials")

    # Retrain best params on full train set and log to MLflow
    best_params = {
        **study.best_params,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "use_label_encoder": False,
    }

    with mlflow.start_run(run_name="XGBoost-Optuna") as run:
        mlflow.log_params(best_params)
        mlflow.log_param("optuna_n_trials", len(study.trials))

        xgb_model = xgb.XGBClassifier(**best_params)
        xgb_model.fit(X_train, y_train)

        y_prob = xgb_model.predict_proba(X_test)[:, 1]
        metrics = _eval_metrics(y_test, y_prob)
        mlflow.log_metrics(metrics)

        mlflow.xgboost.log_model(xgb_model, artifact_path="model")

        # Log Optuna study summary
        trials_df = study.trials_dataframe()
        trials_path = MODELS_DIR / "optuna_trials.csv"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        trials_df.to_csv(trials_path, index=False)
        mlflow.log_artifact(str(trials_path), artifact_path="optuna")

        run_id = run.info.run_id
        context.log.info(f"XGB roc_auc={metrics['roc_auc']:.4f}  avg_pr={metrics['avg_precision']:.4f}")

    import joblib
    joblib.dump(xgb_model, MODELS_DIR / "xgb_tuned.joblib")

    context.add_output_metadata({"run_id": run_id, **metrics})
    return {"model_name": "XGBoost", "run_id": run_id, "metrics": metrics}


@asset(group_name="training", required_resource_keys={"mlflow"})
def best_model_info(
    context: AssetExecutionContext,
    baseline_rf_model: dict[str, Any],
    tuned_xgb_model: dict[str, Any],
) -> dict[str, Any]:
    """Pick the best model by avg_precision and register it in MLflow Model Registry."""
    from mlops_pipeline.config import MODEL_REGISTRY_NAME

    candidates = [baseline_rf_model, tuned_xgb_model]
    best = max(candidates, key=lambda m: m["metrics"]["avg_precision"])

    context.log.info(
        f"Winner: {best['model_name']} "
        f"avg_pr={best['metrics']['avg_precision']:.4f} "
        f"roc_auc={best['metrics']['roc_auc']:.4f}"
    )

    # Register in MLflow Model Registry
    mlflow_res: MLflowResource = context.resources.mlflow
    client = mlflow_res.get_client()

    model_uri = f"runs:/{best['run_id']}/model"
    try:
        client.create_registered_model(MODEL_REGISTRY_NAME)
    except Exception:
        pass  # already exists

    mv = client.create_model_version(
        name=MODEL_REGISTRY_NAME,
        source=model_uri,
        run_id=best["run_id"],
    )
    client.set_registered_model_alias(
        name=MODEL_REGISTRY_NAME,
        alias="champion",
        version=mv.version,
    )
    context.log.info(f"Registered {MODEL_REGISTRY_NAME} v{mv.version} as @champion")

    result = {
        "winner": best["model_name"],
        "run_id": best["run_id"],
        "registry_version": mv.version,
        "metrics": best["metrics"],
        "all_metrics": {m["model_name"]: m["metrics"] for m in candidates},
    }
    context.add_output_metadata(
        {
            "winner": best["model_name"],
            "avg_precision": best["metrics"]["avg_precision"],
            "roc_auc": best["metrics"]["roc_auc"],
            "registry_version": mv.version,
        }
    )

    # Write summary JSON for serving
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "best_model_info.json").write_text(json.dumps(result, indent=2))

    return result
