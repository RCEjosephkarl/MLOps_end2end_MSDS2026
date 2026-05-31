"""
Dagster data assets:
  raw_data          → load original 999-row CSV
  augmented_data    → load 3000-row Plus CSV (original + 2000 synthetic)
  train_test_split  → stratified split with SMOTE oversampling on train set
"""
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dagster import AssetExecutionContext, asset
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from mlops_pipeline.config import (
    FEATURE_COLS,
    MODELS_DIR,
    ORIG_CSV,
    PLUS_CSV,
    TARGET_COL,
)


@asset(group_name="data")
def raw_data(context: AssetExecutionContext) -> pd.DataFrame:
    """Load the original 999-row Computer_Durability.csv."""
    df = pd.read_csv(ORIG_CSV)
    context.log.info(f"Loaded raw data: {df.shape}, positives={df[TARGET_COL].sum()}")
    context.add_output_metadata(
        {
            "num_rows": int(len(df)),
            "num_positives": int(df[TARGET_COL].sum()),
            "positive_rate": float(df[TARGET_COL].mean()),
            "columns": df.columns.tolist(),
        }
    )
    return df


@asset(group_name="data")
def augmented_data(context: AssetExecutionContext) -> pd.DataFrame:
    """Load the 3000-row Computer_Durability_Plus.csv (original + 2000 synthetic)."""
    df = pd.read_csv(PLUS_CSV)
    context.log.info(f"Loaded augmented data: {df.shape}, positives={df[TARGET_COL].sum()}")
    context.add_output_metadata(
        {
            "num_rows": int(len(df)),
            "num_positives": int(df[TARGET_COL].sum()),
            "positive_rate": float(df[TARGET_COL].mean()),
            "synthetic_rows": int(len(df) - 999),
        }
    )
    return df


@asset(group_name="data")
def train_test_split_asset(
    context: AssetExecutionContext,
    augmented_data: pd.DataFrame,
) -> dict[str, Any]:
    """
    Stratified 80/20 split, then SMOTE on the training set.
    Returns a dict with X_train, X_test, y_train, y_test as numpy arrays
    plus the fitted StandardScaler.
    """
    X = augmented_data[FEATURE_COLS].values
    y = augmented_data[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale numeric features (fit only on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE: oversample minority class on train set only
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    context.log.info(
        f"Train (after SMOTE): {X_train_res.shape}, pos={y_train_res.sum()}"
    )
    context.log.info(
        f"Test  (no SMOTE):   {X_test_scaled.shape}, pos={y_test.sum()}"
    )

    # Persist scaler for serving
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")

    # Persist feature metadata for serving
    meta = {
        "feature_cols": FEATURE_COLS,
        "target_col": TARGET_COL,
        "scaler_path": str(MODELS_DIR / "scaler.joblib"),
    }
    (MODELS_DIR / "feature_meta.json").write_text(json.dumps(meta, indent=2))

    context.add_output_metadata(
        {
            "train_rows": int(len(X_train_res)),
            "test_rows": int(len(X_test_scaled)),
            "train_positives_after_smote": int(y_train_res.sum()),
            "test_positives": int(y_test.sum()),
        }
    )

    return {
        "X_train": X_train_res,
        "X_test": X_test_scaled,
        "y_train": y_train_res,
        "y_test": y_test,
        "feature_cols": FEATURE_COLS,
    }
