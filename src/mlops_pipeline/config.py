"""Central configuration: paths, column names, model registry settings."""
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent  # project root: src/mlops_pipeline/ → src/ → root

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_RAW_DIR = ROOT / "data" / "raw"
ORIG_CSV = DATA_RAW_DIR / "Computer_Durability.csv"
PLUS_CSV = DATA_RAW_DIR / "Computer_Durability_Plus.csv"

FEATURE_COLS = [
    "Hours Used Per Day",
    "Cost",
    "User Age",
    "Primary Usage",
    "Brand",
    "Computer Age (Months)",
]
TARGET_COL = "Needs Replacement"
CATEGORICAL_COLS = ["Primary Usage", "Brand"]
NUMERIC_COLS = [c for c in FEATURE_COLS if c not in CATEGORICAL_COLS]

# ── MLflow ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = f"file://{ROOT / 'mlruns'}"
MLFLOW_EXPERIMENT = "computer-durability"
MODEL_REGISTRY_NAME = "ComputerDurabilityClassifier"

# ── Outputs ───────────────────────────────────────────────────────────────────
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
