"""Dagster Definitions: wires all assets and resources together."""
from dagster import Definitions, load_assets_from_modules

from mlops_pipeline.assets import data_assets, evaluation_assets, training_assets
from mlops_pipeline.config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI
from mlops_pipeline.resources import MLflowResource

all_assets = load_assets_from_modules(
    [data_assets, training_assets, evaluation_assets]
)

defs = Definitions(
    assets=all_assets,
    resources={
        "mlflow": MLflowResource(
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=MLFLOW_EXPERIMENT,
        ),
    },
)
