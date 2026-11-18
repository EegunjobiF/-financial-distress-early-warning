"""Central settings. Only the REPLACE section needs editing."""

import logging
import os
from pathlib import Path

# Load .env file locally if python-dotenv is installed (optional, safe to keep)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ----------------------------------------------------------------- REPLACE
# Secrets are loaded from environment variables, NOT hardcoded.
# Put the real values in your local .env file (never commit .env)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET")
SAGEMAKER_ROLE_ARN = os.getenv("SAGEMAKER_ROLE_ARN")

# Fail fast if secrets are missing, instead of failing later in deploy
if not S3_BUCKET:
    raise ValueError("Missing env var: S3_BUCKET - set it in your .env file")
if not SAGEMAKER_ROLE_ARN:
    raise ValueError("Missing env var: SAGEMAKER_ROLE_ARN - set it in your .env file")

# ---------------------------------------------------------------- ENDPOINT
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "bankruptcy-risk-endpoint")
DEPLOY_MODE = "serverless"          # "serverless" (pay per request) or "realtime" (pay per hour)
SERVERLESS_MEMORY_MB = 2048
SERVERLESS_MAX_CONCURRENCY = 2
REALTIME_INSTANCE_TYPE = "ml.t2.medium"
SKLEARN_FRAMEWORK_VERSION = "1.2-1"
S3_MODEL_PREFIX = "models/deploy"

# ------------------------------------------------------------------- PATHS
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE.parent / "model" / "bankruptcy_model_lightgbm.joblib"
SETTINGS_PATH = HERE.parent / "model" / "model_settings.json"
BUILD_DIR = HERE / "build"
ENDPOINT_CODE_DIR = HERE / "endpoint_code"

# ------------------------------------------------------------------ OUTPUT
S3_PREDICTIONS_KEY = "predictions/predictions.csv"
PREDICTION_COLUMNS = ["company_id", "risk_score", "prediction", "prediction_date", "scored_at"]
LABEL_HIGH_RISK = "High Risk"
LABEL_LOW_RISK = "Low Risk"

# ---------------------------------------------------------------- FEATURES
RAW_COLUMNS = [f"Attr{i}" for i in range(1, 65)]
ENGINEERED_COLUMNS = [
    "inventory_burden_stl",
    "receivables_burden_stl",
    "operating_profit_to_liabilities",
    "negative_profitability_share",
    "financial_red_flag_share",
]
EXPECTED_FEATURE_COUNT = 69
PROFIT_RATIOS = ["Attr1", "Attr7", "Attr22", "Attr23", "Attr35", "Attr42"]
MIN_DENOMINATOR = 1e-6          # same guard used in training to avoid dividing by ~0
WARN_MISSING_SHARE = 0.5        # warn (not reject) when more than half the ratios are missing

def get_logger(name):
    """Return a logger that prints timestamps, so each pipeline stage can be traced."""
    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    return logging.getLogger(name)