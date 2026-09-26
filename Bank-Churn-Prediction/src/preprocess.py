"""
preprocess.py
-------------
Step 2 of the pipeline: clean the raw data.

Handles:
- Dropping identifier columns that carry no predictive signal
- Missing value imputation
- Duplicate removal
- Outlier capping (winsorization) on skewed numeric columns
- Basic type enforcement

Writes data/processed/cleaned_data.csv
"""

import pandas as pd

from load_data import load_raw_data
from utils import CLEANED_DATA_PATH, ensure_dirs, get_logger

logger = get_logger(__name__)

ID_COLUMNS = ["RowNumber", "CustomerId", "Surname"]
NUMERIC_OUTLIER_COLS = ["CreditScore", "Age", "Balance", "EstimatedSalary"]


def drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are unique identifiers / free text with no signal."""
    cols_to_drop = [c for c in ID_COLUMNS if c in df.columns]
    if cols_to_drop:
        logger.info("Dropping identifier columns: %s", cols_to_drop)
        df = df.drop(columns=cols_to_drop)
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    missing_before = df.isnull().sum().sum()
    if missing_before == 0:
        return df

    numeric_cols = df.select_dtypes(include="number").columns
    categorical_cols = df.select_dtypes(include="object").columns

    for col in numeric_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Filled %d missing values in '%s' with median=%.2f",
                        df[col].isnull().sum(), col, median_val)

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode().iloc[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("Filled missing values in '%s' with mode='%s'", col, mode_val)

    logger.info("Total missing values resolved: %d", missing_before)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows.", removed)
    return df


def cap_outliers(df: pd.DataFrame, cols=NUMERIC_OUTLIER_COLS, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Winsorize outliers in numeric columns to the [1st, 99th] percentile range."""
    for col in cols:
        if col not in df.columns:
            continue
        lower, upper = df[col].quantile([lower_q, upper_q])
        n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        if n_capped:
            logger.info("Capped %d outliers in '%s' to [%.2f, %.2f]", n_capped, col, lower, upper)
    return df


def enforce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure categorical columns are strings and binary flags are ints."""
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    for col in ["HasCrCard", "IsActiveMember", "Exited"]:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline in the correct order."""
    logger.info("Starting data cleaning on %d rows...", len(df))
    df = drop_identifier_columns(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = cap_outliers(df)
    df = enforce_dtypes(df)
    logger.info("Cleaning complete: %d rows x %d columns remain.", df.shape[0], df.shape[1])
    return df


def save_cleaned_data(df: pd.DataFrame, path=CLEANED_DATA_PATH) -> None:
    ensure_dirs()
    df.to_csv(path, index=False)
    logger.info("Cleaned data saved to %s", path)


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    save_cleaned_data(cleaned_df)
