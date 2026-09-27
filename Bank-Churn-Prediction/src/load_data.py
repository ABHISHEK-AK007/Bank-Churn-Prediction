"""
load_data.py
------------
Step 1 of the pipeline: load the raw bank churn CSV.

If data/raw/bank_churn.csv does not exist, a synthetic dataset matching
the standard "Churn Modelling" schema (CustomerId, Surname, CreditScore,
Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard,
IsActiveMember, EstimatedSalary, Exited) is generated so the rest of the
pipeline is runnable immediately. Replace the CSV with your real export
from your core banking system / data warehouse whenever you have it --
no other code needs to change as long as the column names match.
"""

import pandas as pd

from src.utils import RAW_DATA_PATH, ensure_dirs, generate_synthetic_bank_data, get_logger

logger = get_logger(__name__)


def load_raw_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw dataset, generating a synthetic sample if not found."""
    ensure_dirs()

    if not path.exists():
        logger.warning(
            "%s not found. Generating a synthetic sample dataset so the "
            "pipeline can run end-to-end. Replace this file with your real "
            "export for production use.",
            path,
        )
        df = pd.DataFrame(generate_synthetic_bank_data())
        df.to_csv(path, index=False)
        logger.info("Synthetic dataset written to %s (%d rows).", path, len(df))

    df = pd.read_csv(path)
    logger.info("Loaded raw data: %d rows x %d columns from %s", df.shape[0], df.shape[1], path)
    return df


def basic_summary(df: pd.DataFrame) -> None:
    """Log a quick sanity-check summary: shape, dtypes, missing values, target balance."""
    logger.info("Columns: %s", list(df.columns))
    logger.info("Missing values per column:\n%s", df.isnull().sum().to_string())
    if "Exited" in df.columns:
        churn_rate = df["Exited"].mean() * 100
        logger.info("Churn rate: %.2f%%", churn_rate)


if __name__ == "__main__":
    data = load_raw_data()
    basic_summary(data)
