"""
Dagster evaluation assets using Evidently 0.7+ API:
  data_quality_report   → DataSummaryPreset on augmented_data
  data_drift_report     → DataDriftPreset: original vs synthetic cohort
  model_eval_report     → ClassificationPreset on test predictions
All HTML reports saved to reports/.
"""
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, asset
from evidently import DataDefinition, Dataset
from evidently.core.datasets import BinaryClassification
from evidently.core.report import Report
from evidently.presets import ClassificationPreset, DataDriftPreset, DataSummaryPreset

from mlops_pipeline.config import (
    FEATURE_COLS,
    MODELS_DIR,
    NUMERIC_COLS,
    CATEGORICAL_COLS,
    ORIG_CSV,
    PLUS_CSV,
    REPORTS_DIR,
    TARGET_COL,
)


def _save_report(snapshot, name: str) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / f"{name}.html"
    snapshot.save_html(str(out))
    return out


def _base_definition() -> DataDefinition:
    return DataDefinition(
        numerical_columns=NUMERIC_COLS,
        categorical_columns=CATEGORICAL_COLS + [TARGET_COL],
    )


@asset(group_name="evaluation")
def data_quality_report(
    context: AssetExecutionContext,
    augmented_data: pd.DataFrame,
) -> str:
    """Evidently data summary/quality report on the full augmented dataset."""
    dd = _base_definition()
    ds = Dataset.from_pandas(augmented_data, data_definition=dd)
    report = Report([DataSummaryPreset()])
    snapshot = report.run(ds, None)
    out = _save_report(snapshot, "data_quality")
    context.log.info(f"Data quality report → {out}")
    context.add_output_metadata({"report_path": str(out)})
    return str(out)


@asset(group_name="evaluation")
def data_drift_report(
    context: AssetExecutionContext,
) -> str:
    """Evidently drift report: original CSV (reference) vs 2000 synthetic rows (current)."""
    orig = pd.read_csv(ORIG_CSV)
    plus = pd.read_csv(PLUS_CSV)
    synth = plus.iloc[len(orig):]

    dd = _base_definition()
    ref_ds = Dataset.from_pandas(orig, data_definition=dd)
    cur_ds = Dataset.from_pandas(synth, data_definition=dd)

    report = Report([DataDriftPreset()])
    snapshot = report.run(cur_ds, ref_ds)
    out = _save_report(snapshot, "data_drift")
    context.log.info(f"Data drift report → {out}")
    context.add_output_metadata({"report_path": str(out)})
    return str(out)


@asset(group_name="evaluation")
def model_eval_report(
    context: AssetExecutionContext,
    train_test_split_asset: dict[str, Any],
    best_model_info: dict[str, Any],
) -> str:
    """Evidently classification report on the test set using the best model."""
    winner = best_model_info["winner"]
    model_path = MODELS_DIR / ("xgb_tuned.joblib" if winner == "XGBoost" else "rf_baseline.joblib")
    model = joblib.load(model_path)

    X_test = train_test_split_asset["X_test"]
    y_test = train_test_split_asset["y_test"]

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    test_df = pd.DataFrame(X_test, columns=FEATURE_COLS)
    test_df[TARGET_COL] = y_test
    test_df["prediction_proba"] = y_prob
    test_df["prediction_labels"] = y_pred

    dd = DataDefinition(
        numerical_columns=FEATURE_COLS,
        classification=[
            BinaryClassification(
                target=TARGET_COL,
                prediction_labels="prediction_labels",
                prediction_probas="prediction_proba",
            )
        ],
    )
    ds = Dataset.from_pandas(test_df, data_definition=dd)
    report = Report([ClassificationPreset()])
    snapshot = report.run(ds, None)
    out = _save_report(snapshot, "model_evaluation")

    context.log.info(f"Model eval report → {out}")
    context.add_output_metadata(
        {
            "report_path": str(out),
            "winner_model": winner,
            "test_rows": len(y_test),
        }
    )
    return str(out)
