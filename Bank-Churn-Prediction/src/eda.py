"""
eda.py
------
Step 3 of the pipeline: exploratory data analysis.

Generates and saves the core set of plots analysts / stakeholders need
to understand churn drivers, to images/graphs/. Also prints key summary
stats to the console/log for quick reporting.
"""

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for scripts/CI
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from preprocess import clean_data
from load_data import load_raw_data
from utils import IMAGES_DIR, TARGET_COL, ensure_dirs, get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


def plot_churn_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(6, 4))
    ax = sns.countplot(x=TARGET_COL, data=df, palette="Set2")
    ax.set_title("Customer Churn Distribution")
    ax.set_xlabel("Exited (0 = Retained, 1 = Churned)")
    total = len(df)
    for p in ax.patches:
        pct = f"{100 * p.get_height() / total:.1f}%"
        ax.annotate(pct, (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "01_churn_distribution.png", dpi=150)
    plt.close()


def plot_churn_by_categorical(df: pd.DataFrame) -> None:
    cat_cols = [c for c in ["Geography", "Gender", "NumOfProducts", "IsActiveMember", "HasCrCard"]
                if c in df.columns]
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(5 * len(cat_cols), 4))
    if len(cat_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cat_cols):
        rate = df.groupby(col)[TARGET_COL].mean().sort_values(ascending=False)
        sns.barplot(x=rate.index, y=rate.values, ax=ax, palette="coolwarm")
        ax.set_title(f"Churn Rate by {col}")
        ax.set_ylabel("Churn Rate")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "02_churn_by_categorical.png", dpi=150)
    plt.close()


def plot_numeric_distributions(df: pd.DataFrame) -> None:
    num_cols = [c for c in ["CreditScore", "Age", "Balance", "EstimatedSalary", "Tenure"]
                if c in df.columns]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, col in zip(axes, num_cols):
        sns.kdeplot(data=df, x=col, hue=TARGET_COL, common_norm=False, fill=True, alpha=0.4, ax=ax)
        ax.set_title(f"{col} distribution by churn")
    for ax in axes[len(num_cols):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "03_numeric_distributions.png", dpi=150)
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include="number")
    plt.figure(figsize=(9, 7))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "04_correlation_heatmap.png", dpi=150)
    plt.close()


def print_key_insights(df: pd.DataFrame) -> None:
    logger.info("=== KEY EDA INSIGHTS ===")
    logger.info("Overall churn rate: %.2f%%", df[TARGET_COL].mean() * 100)
    if "Geography" in df.columns:
        logger.info("Churn rate by Geography:\n%s",
                     df.groupby("Geography")[TARGET_COL].mean().round(3).to_string())
    if "NumOfProducts" in df.columns:
        logger.info("Churn rate by NumOfProducts:\n%s",
                     df.groupby("NumOfProducts")[TARGET_COL].mean().round(3).to_string())
    if "IsActiveMember" in df.columns:
        logger.info("Churn rate by IsActiveMember:\n%s",
                     df.groupby("IsActiveMember")[TARGET_COL].mean().round(3).to_string())


def run_eda(df: pd.DataFrame) -> None:
    ensure_dirs()
    logger.info("Running EDA and saving plots to %s ...", IMAGES_DIR)
    plot_churn_distribution(df)
    plot_churn_by_categorical(df)
    plot_numeric_distributions(df)
    plot_correlation_heatmap(df)
    print_key_insights(df)
    logger.info("EDA complete. 4 plots saved.")


if __name__ == "__main__":
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    run_eda(cleaned_df)
