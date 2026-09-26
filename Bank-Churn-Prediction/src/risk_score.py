"""
risk_score.py
-------------
Converts raw churn probabilities into business-friendly risk tiers and
a prioritized retention worklist -- the bridge between the ML model and
the Power BI dashboard / relationship-manager action list.

Risk tiers (tunable):
    Low       : probability < 0.30
    Medium    : 0.30 <= probability < 0.60
    High      : 0.60 <= probability < 0.80
    Critical  : probability >= 0.80

Also computes an Estimated Value at Risk = churn_probability * Balance,
so relationship managers can prioritize high-balance, high-risk customers
first -- this is usually the single most useful column for the business.
"""

import pandas as pd

from predict import predict
from utils import PREDICTIONS_PATH, RAW_DATA_PATH, ensure_dirs, get_logger

logger = get_logger(__name__)

RISK_BINS = [0, 0.30, 0.60, 0.80, 1.01]
RISK_LABELS = ["Low", "Medium", "High", "Critical"]


def assign_risk_tier(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["risk_tier"] = pd.cut(df["churn_probability"], bins=RISK_BINS, labels=RISK_LABELS, right=False)
    return df


def compute_value_at_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Balance" in df.columns:
        df["estimated_value_at_risk"] = (df["churn_probability"] * df["Balance"]).round(2)
    return df


def build_retention_worklist(df: pd.DataFrame, top_n: int = 100) -> pd.DataFrame:
    """Rank customers for retention outreach: highest value-at-risk first."""
    sort_col = "estimated_value_at_risk" if "estimated_value_at_risk" in df.columns else "churn_probability"
    worklist = df.sort_values(sort_col, ascending=False).head(top_n)
    logger.info("Built retention worklist: top %d customers by %s.", len(worklist), sort_col)
    return worklist


def score_and_tier(df: pd.DataFrame) -> pd.DataFrame:
    scored = predict(df)
    scored = assign_risk_tier(scored)
    scored = compute_value_at_risk(scored)

    tier_counts = scored["risk_tier"].value_counts().reindex(RISK_LABELS)
    logger.info("Risk tier distribution:\n%s", tier_counts.to_string())
    return scored


if __name__ == "__main__":
    ensure_dirs()
    raw_df = pd.read_csv(RAW_DATA_PATH)
    scored_df = score_and_tier(raw_df)
    scored_df.to_csv(PREDICTIONS_PATH, index=False)
    logger.info("Risk-scored predictions saved to %s", PREDICTIONS_PATH)

    worklist = build_retention_worklist(scored_df)
    worklist_path = PREDICTIONS_PATH.parent / "retention_worklist_top100.csv"
    worklist.to_csv(worklist_path, index=False)
    logger.info("Retention worklist saved to %s", worklist_path)
