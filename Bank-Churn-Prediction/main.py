"""
main.py
-------
Runs the full Bank Churn Prediction pipeline end-to-end:

    load -> clean -> EDA -> feature engineering -> train -> predict -> risk score

Equivalent to running notebooks 01-06 in order, but as a single
reproducible script -- use this for scheduled retraining / CI, and use
the notebooks for interactive exploration.

Usage:
    python main.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from src.utils import ensure_dirs, get_logger, set_seed  # noqa: E402

logger = get_logger("main")


def run_pipeline():
    start = time.time()
    set_seed()
    ensure_dirs()

    logger.info("STEP 1/6: Loading raw data")
    from src.load_data import basic_summary, load_raw_data
    raw_df = load_raw_data()
    basic_summary(raw_df)

    logger.info("STEP 2/6: Cleaning data")
    from src.preprocess import clean_data, save_cleaned_data
    cleaned_df = clean_data(raw_df)
    save_cleaned_data(cleaned_df)

    logger.info("STEP 3/6: Running EDA")
    from src.eda import run_eda
    run_eda(cleaned_df)

    logger.info("STEP 4/6: Engineering features")
    from src.feature_engineering import build_feature_set, save_artifacts
    featured_df, scaler = build_feature_set(cleaned_df, fit=True)
    save_artifacts(featured_df, scaler)

    logger.info("STEP 5/6: Training model")
    from src.train_model import train_pipeline
    model, metrics, threshold = train_pipeline()

    logger.info("STEP 6/6: Scoring customers and computing risk tiers")
    from src.risk_score import build_retention_worklist, score_and_tier
    from src.utils import PREDICTIONS_PATH, RAW_DATA_PATH
    import pandas as pd
    raw_for_scoring = pd.read_csv(RAW_DATA_PATH)
    scored_df = score_and_tier(raw_for_scoring)
    scored_df.to_csv(PREDICTIONS_PATH, index=False)
    worklist = build_retention_worklist(scored_df)
    worklist.to_csv(PREDICTIONS_PATH.parent / "retention_worklist_top100.csv", index=False)

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1f seconds", elapsed)
    logger.info("Final ROC-AUC: %.4f | F1: %.4f | Threshold: %.3f",
                metrics["roc_auc"], metrics["f1"], threshold)
    logger.info("Outputs: data/processed/, models/, images/graphs/")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
