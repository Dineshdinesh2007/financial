"""FinGuard AI Configuration Module.

Central configuration for paths, feature schemas, model hyperparameters,
API settings, and default decision thresholds.
"""

from pathlib import Path
import os

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
MODELS_DIR = PROJECT_ROOT / "models"

# Specific file paths
RAW_DATA_PATH = RAW_DATA_DIR / "creditcard.csv"
SYNTHETIC_DATA_PATH = SYNTHETIC_DATA_DIR / "demo_transactions.csv"
MODEL_FILE = MODELS_DIR / "best_model.joblib"
METADATA_FILE = MODELS_DIR / "metadata.json"
EVALUATION_FILE = MODELS_DIR / "evaluation_metrics.json"
DATABASE_PATH = PROJECT_ROOT / "finguard.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Feature definitions for Kaggle Credit Card Fraud dataset
TIME_COLUMN = "Time"
AMOUNT_COLUMN = "Amount"
V_COLUMNS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLUMNS = [TIME_COLUMN] + V_COLUMNS + [AMOUNT_COLUMN]
TARGET_COLUMN = "Class"
ALL_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]

# Dataset characteristics
EXPECTED_COLUMNS_COUNT = 31
EXPECTED_FEATURES_COUNT = 30
EXPECTED_ROWS_COUNT = 284807

# Defaults & Limits
DEFAULT_DECISION_THRESHOLD = 0.35
MAX_BATCH_ROWS = 25000
API_HOST = os.getenv("FINGUARD_API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("FINGUARD_API_PORT", "8000"))
API_URL = f"http://{API_HOST}:{API_PORT}"

# App Version
APP_VERSION = "1.0.0"
MODEL_VERSION = "v1.0.0"
