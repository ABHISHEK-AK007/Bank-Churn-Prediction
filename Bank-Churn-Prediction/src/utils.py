"""
utils.py
--------
Shared configuration, logging, and helper utilities used across the
Bank Churn Prediction pipeline. Centralizing paths and constants here
means every script/notebook stays in sync if the folder layout changes.
"""

import logging
import os
import random
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Project paths (resolved relative to this file, so it works no matter
# where the script is invoked from)
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
IMAGES_DIR = ROOT_DIR / "images" / "graphs"
REPORTS_DIR = ROOT_DIR / "reports"

RAW_DATA_PATH = DATA_RAW_DIR / "bank_churn.csv"
CLEANED_DATA_PATH = DATA_PROCESSED_DIR / "cleaned_data.csv"
FEATURED_DATA_PATH = DATA_PROCESSED_DIR / "featured_data.csv"
PREDICTIONS_PATH = DATA_PROCESSED_DIR / "predictions.csv"

MODEL_PATH = MODELS_DIR / "random_forest.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODER_PATH = MODELS_DIR / "encoders.pkl"
FEATURE_NAMES_PATH = MODELS_DIR / "feature_names.pkl"

TARGET_COL = "Exited"
RANDOM_STATE = 42


def ensure_dirs() -> None:
    """Create every project directory the pipeline writes to, if missing."""
    for d in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, IMAGES_DIR, REPORTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Fix all relevant random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger that prints timestamped, leveled messages."""
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on re-import
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def generate_synthetic_bank_data(n_rows: int = 10000, seed: int = RANDOM_STATE):
    """
    Generate a synthetic dataset that mirrors the schema of the classic
    Kaggle 'Churn Modelling' bank dataset. Used only as a stand-in so the
    pipeline is runnable end-to-end before you drop in the real
    data/raw/bank_churn.csv. Safe to delete once you have real data.
    """
    rng = np.random.default_rng(seed)

    geography = rng.choice(["France", "Germany", "Spain"], size=n_rows, p=[0.5, 0.25, 0.25])
    gender = rng.choice(["Male", "Female"], size=n_rows)
    age = rng.integers(18, 92, size=n_rows)
    credit_score = rng.integers(350, 851, size=n_rows)
    tenure = rng.integers(0, 11, size=n_rows)
    balance = np.round(rng.uniform(0, 250000, size=n_rows), 2)
    balance[rng.random(n_rows) < 0.35] = 0.0  # ~35% zero-balance accounts
    num_products = rng.integers(1, 5, size=n_rows)
    has_cr_card = rng.choice([0, 1], size=n_rows, p=[0.3, 0.7])
    is_active_member = rng.choice([0, 1], size=n_rows, p=[0.45, 0.55])
    estimated_salary = np.round(rng.uniform(10, 200000, size=n_rows), 2)

    # Build churn probability from a plausible signal mix so the model has
    # real relationships to learn (older, inactive, single-product,
    # high-balance-but-inactive customers churn more -- mirrors real bank data).
    churn_logit = (
        -3.0
        + 0.035 * (age - 40)
        + 0.9 * (is_active_member == 0)
        + 0.5 * (num_products == 1)
        + 0.8 * (num_products >= 4)
        + 0.4 * (geography == "Germany")
        + 0.000004 * balance
        - 0.002 * (credit_score - 650)
    )
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    exited = (rng.random(n_rows) < churn_prob).astype(int)

    df = {
        "CustomerId": np.arange(15600000, 15600000 + n_rows),
        "Surname": [f"Cust{i}" for i in range(n_rows)],
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Exited": exited,
    }
    return df
