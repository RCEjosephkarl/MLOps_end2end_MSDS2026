"""
Run the full Dagster asset pipeline programmatically (no web UI needed).
Executes all assets in dependency order:
  raw_data → augmented_data → train_test_split_asset
  → baseline_rf_model + tuned_xgb_model → best_model_info
  → data_quality_report + data_drift_report + model_eval_report
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dagster import materialize

from mlops_pipeline.assets.data_assets import (
    augmented_data,
    raw_data,
    train_test_split_asset,
)
from mlops_pipeline.assets.evaluation_assets import (
    data_drift_report,
    data_quality_report,
    model_eval_report,
)
from mlops_pipeline.assets.training_assets import (
    baseline_rf_model,
    best_model_info,
    tuned_xgb_model,
)
from mlops_pipeline.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI
from mlops_pipeline.resources import MLflowResource

ASSETS = [
    raw_data,
    augmented_data,
    train_test_split_asset,
    baseline_rf_model,
    tuned_xgb_model,
    best_model_info,
    data_quality_report,
    data_drift_report,
    model_eval_report,
]

if __name__ == "__main__":
    print("=" * 60)
    print("MLOps Pipeline: Computer Durability Classifier")
    print("=" * 60)

    result = materialize(
        ASSETS,
        resources={
            "mlflow": MLflowResource(
                tracking_uri=MLFLOW_TRACKING_URI,
                experiment_name=MLFLOW_EXPERIMENT,
            )
        },
    )

    if result.success:
        print("\n✓ Pipeline completed successfully!")
        print(f"  MLflow UI: mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")
        print(f"  Reports  : reports/")
        print(f"  Models   : models/")
        print("\nTo serve the model:")
        print("  uv run uvicorn serving.api:app --host 0.0.0.0 --port 8000")
        print("  uv run python app.py")
    else:
        print("\n✗ Pipeline failed. Check logs above.")
        sys.exit(1)
