"""Runs inside the SageMaker container. Pure model serving: features in, scores out."""

import json
import os

import lightgbm as lgb
import pandas as pd

def model_fn(model_dir):
    """Load the LightGBM booster that deploy.py packaged as model.txt."""
    path = os.path.join(model_dir, "model.txt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"model.txt not found in {model_dir}")
    return lgb.Booster(model_file=path)

def input_fn(request_body, content_type="application/json"):
    """Parse the 'split' JSON the ETL sends: {"columns": [...], "data": [[...]]}."""
    if content_type != "application/json":
        raise ValueError(f"Unsupported content type: {content_type}")
    payload = json.loads(request_body)
    if "columns" not in payload or "data" not in payload:
        raise ValueError("Payload must contain 'columns' and 'data'")
    # dtype float64 turns JSON nulls into NaN, which LightGBM handles natively.
    return pd.DataFrame(payload["data"], columns=payload["columns"], dtype="float64")

def predict_fn(features, booster):
    """Reject a wrong feature order instead of silently scoring misaligned columns."""
    expected = booster.feature_name()
    if list(features.columns) != expected:
        raise ValueError("Feature names or order do not match the trained model")
    return booster.predict(features)

def output_fn(scores, accept="application/json"):
    return json.dumps({"risk_scores": [float(score) for score in scores]}), "application/json"