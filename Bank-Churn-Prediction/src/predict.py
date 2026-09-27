"""
predict.py
----------
Step 6 of the pipeline: score new/unseen customers with the trained model.

Loads the persisted model, scaler, and feature column list so the exact
same transformation used at training time is replayed on new data -- this
consistency is what prevents train/serve skew.

Usage:
    python predict.py                          # scores data/raw/bank_churn.csv
    python predict.py --input path/to/new.csv  # scores a custom file
"""

import argparse

import joblib
import pandas as pd

from src.feature_engineering import build_feature_set
from src.preprocess import clean_data
from src.utils import (
    FEATURE_NAMES_PATH,
    MODEL_PATH,
    MODELS_DIR,
    PREDICTIONS_PATH,
    RAW_DATA_PATH,
    SCALER_PATH,
    TARGET_COL,
    ensure_dirs,
    get_logger,
)

logger = get_logger(__name__)


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURE_NAMES_PATH)
    threshold_path = MODELS_DIR / "decision_threshold.pkl"
    threshold = joblib.load(threshold_path) if threshold_path.exists() else 0.5
    return model, scaler, feature_names, threshold


def prepare_input(df: pd.DataFrame, scaler, feature_names) -> pd.DataFrame:
    """Clean + feature-engineer new data using the FITTED scaler (fit=False),
    then align columns exactly to what the model was trained on."""
    has_target = TARGET_COL in df.columns
    df_clean = clean_data(df.copy())
    df_feat, _ = build_feature_set(df_clean, fit=False, scaler=scaler)

    # Align columns: add any missing dummy columns as 0, drop unexpected ones,
    # preserve training-time column order.
    for col in feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0
    extra_cols = [c for c in df_feat.columns if c not in feature_names and c != TARGET_COL]
    if extra_cols:
        df_feat = df_feat.drop(columns=extra_cols)

    X = df_feat[feature_names]
    return X, (df_feat[TARGET_COL] if has_target else None)


def predict(df: pd.DataFrame) -> pd.DataFrame:
    """Return the original rows with churn_probability and churn_prediction appended."""
    model, scaler, feature_names, threshold = load_artifacts()
    X, _ = prepare_input(df, scaler, feature_names)

    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)

    result = df.copy().reset_index(drop=True)
    result["churn_probability"] = proba.round(4)
    result["churn_prediction"] = preds
    logger.info("Scored %d customers. Predicted churners: %d (%.1f%%)",
                len(result), preds.sum(), preds.mean() * 100)
    return result


def main():
    parser = argparse.ArgumentParser(description="Score customers for churn risk.")
    parser.add_argument("--input", type=str, default=str(RAW_DATA_PATH), help="Path to input CSV.")
    parser.add_argument("--output", type=str, default=str(PREDICTIONS_PATH), help="Path to write predictions CSV.")
    args = parser.parse_args()

    ensure_dirs()
    df = pd.read_csv(args.input)
    result = predict(df)
    result.to_csv(args.output, index=False)
    logger.info("Predictions saved to %s", args.output)


if __name__ == "__main__":
    main()
