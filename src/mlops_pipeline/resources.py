"""Dagster resources: MLflow client wrapper."""
from __future__ import annotations

import mlflow
from dagster import ConfigurableResource


class MLflowResource(ConfigurableResource):
    tracking_uri: str
    experiment_name: str

    def get_client(self) -> mlflow.MlflowClient:
        mlflow.set_tracking_uri(self.tracking_uri)
        return mlflow.MlflowClient(tracking_uri=self.tracking_uri)

    def set_experiment(self) -> str:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        return mlflow.get_experiment_by_name(self.experiment_name).experiment_id
