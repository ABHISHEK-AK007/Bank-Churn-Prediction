"""
feature_engineering.py
-----------------------
Step 4 of the pipeline: turn cleaned data into model-ready features.

- Encodes categoricals (one-hot for Geography, binary map for Gender)
- Engineers domain-informed features that materially help churn models:
    * BalanceSalaryRatio   -> financial exposure relative to income
    * IsZeroBalance        -> zero-balance accounts churn very differently
    * TenureByAge          -> relationship depth relative to customer age
    * ProductsPerTenureYr  -> product adoption pace
    * IsSeniorCustomer     -> age >= 60 flag (bank-specific risk cohort)
    * CreditScoreBucket    -> ordinal risk band, robust to score outliers
- Scales numeric features with StandardScaler (fit on train, applied to all)
- Persists the fitted scaler + final feature column list to models/ so
  predict.py can exactly reproduce this transformation on new data.
"""

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from preprocess import clean_data
from load_data import load_raw_data
from utils import (
    FEATURE_NAMES_PATH,
    FEATURED_DATA_PATH,
    SCALER_PATH,
    TARGET_COL,
    ensure_dirs,
    get_logger,
)

logger = get_logger(__name__)

NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
    "EstimatedSalary", "BalanceSalaryRatio", "TenureByAge", "ProductsPerTenureYr",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add domain-informed derived features."""
    df = df.copy()

    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["TenureByAge"] = df["Tenure"] / (df["Age"] + 1)
    df["ProductsPerTenureYr"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    df["IsSeniorCustomer"] = (df["Age"] >= 60).astype(int)

    df["CreditScoreBucket"] = pd.cut(
        df["CreditScore"],
        bins=[0, 580, 670, 740, 800, 850],
        labels=["Poor", "Fair", "Good", "VeryGood", "Excellent"],
    )

    logger.info("Engineered features: BalanceSalaryRatio, IsZeroBalance, TenureByAge, "
                "ProductsPerTenureYr, IsSeniorCustomer, CreditScoreBucket")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode nominal categoricals; binary-map Gender."""
    df = df.copy()

    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0}).fillna(0).astype(int)

    cat_cols = [c for c in ["Geography", "CreditScoreBucket"] if c in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    logger.info("Encoded categorical columns: %s", cat_cols + (["Gender"] if "Gender" in df.columns else []))
    return df


def scale_features(df: pd.DataFrame, fit: bool = True, scaler: StandardScaler = None):
    """Standard-scale numeric features. Fit a new scaler or reuse one for inference."""
    df = df.copy()
    cols_present = [c for c in NUMERIC_FEATURES if c in df.columns]

    if fit:
        scaler = StandardScaler()
        df[cols_present] = scaler.fit_transform(df[cols_present])
        logger.info("Fitted new StandardScaler on %d numeric features.", len(cols_present))
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit=False.")
        df[cols_present] = scaler.transform(df[cols_present])

    return df, scaler


def build_feature_set(df: pd.DataFrame, fit: bool = True, scaler: StandardScaler = None):
    """Full feature engineering pipeline: engineer -> encode -> scale."""
    df = engineer_features(df)
    df = encode_categoricals(df)
    df, scaler = scale_features(df, fit=fit, scaler=scaler)
    return df, scaler


def save_artifacts(df: pd.DataFrame, scaler: StandardScaler) -> None:
    ensure_dirs()
    df.to_csv(FEATURED_DATA_PATH, index=False)
    joblib.dump(scaler, SCALER_PATH)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    joblib.dump(feature_cols, FEATURE_NAMES_PATH)

    logger.info("Featured dataset saved to %s", FEATURED_DATA_PATH)
    logger.info("Scaler saved to %s", SCALER_PATH)
    logger.info("Feature column list (%d features) saved to %s", len(feature_cols), FEATURE_NAMES_PATH)


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    featured_df, fitted_scaler = build_feature_set(cleaned_df, fit=True)
    save_artifacts(featured_df, fitted_scaler)
