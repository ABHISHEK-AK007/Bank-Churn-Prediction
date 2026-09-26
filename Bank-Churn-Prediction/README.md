# Bank Churn Prediction

An end-to-end customer churn prediction pipeline for a retail bank: data
cleaning, EDA, feature engineering, model training/tuning, prediction,
risk scoring, and a Power BI-ready output feed.

**Pipeline:** Python (VS Code/Jupyter) → Data Analysis → Machine Learning → CSV Output → Power BI Dashboard → Report → PPT

## Project Structure

```
Bank-Churn-Prediction/
├── data/
│   ├── raw/bank_churn.csv          # source data (auto-generated synthetic sample if missing)
│   └── processed/                  # cleaned_data.csv, featured_data.csv, predictions.csv, worklist
├── notebooks/                      # 01-06, interactive walkthroughs of each stage
├── src/                            # production pipeline code (importable modules)
├── models/                         # random_forest.pkl, scaler.pkl, feature_names.pkl
├── dashboard/                      # Power BI dashboard (see dashboard/README.md)
├── reports/                        # Final_Report.docx / .pdf
├── presentation/                   # Bank_Churn_Presentation.pptx
├── images/graphs/                  # auto-generated plots (EDA + model evaluation)
├── requirements.txt
├── main.py                         # runs the full pipeline end-to-end
└── README.md
```

## Quick Start

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Drop your real export at data/raw/bank_churn.csv (see schema below).
# If it's absent, main.py auto-generates a synthetic sample so you can
# try the pipeline immediately.

python main.py
```

This runs, in order: load → clean → EDA → feature engineering → train →
predict → risk score, and writes every output file described below.

To work through it interactively instead, open the notebooks in order
(`01_data_loading.ipynb` → `06_prediction.ipynb`) — each one imports the
same `src/` functions `main.py` uses, so results match exactly.

## Expected Input Schema (`data/raw/bank_churn.csv`)

Matches the standard "Churn Modelling" bank dataset schema:

| Column | Type | Notes |
|---|---|---|
| CustomerId | int | dropped before modeling |
| Surname | string | dropped before modeling |
| CreditScore | int | 350-850 |
| Geography | string | e.g. France, Germany, Spain |
| Gender | string | Male / Female |
| Age | int | |
| Tenure | int | years with the bank |
| Balance | float | account balance |
| NumOfProducts | int | 1-4 |
| HasCrCard | 0/1 | |
| IsActiveMember | 0/1 | |
| EstimatedSalary | float | |
| **Exited** | 0/1 | **target** — 1 = churned |

If your export uses different column names, update `TARGET_COL` /
`ID_COLUMNS` in `src/utils.py` and `src/preprocess.py` accordingly — the
rest of the pipeline adapts automatically.

## Modeling Approach ("best performance" choices)

- **Class imbalance**: churn is typically ~20-30% of customers. The
  training set (only) is rebalanced with **SMOTE** (falls back to random
  oversampling if `imbalanced-learn` isn't installed) so the model isn't
  biased toward always predicting "retained."
- **Model comparison**: Logistic Regression, Random Forest, Gradient
  Boosting, and XGBoost (if installed) are all trained under identical
  conditions and ranked by **ROC-AUC** — accuracy alone is misleading on
  imbalanced classes.
- **Hyperparameter tuning**: `RandomizedSearchCV` (25 iterations × 5-fold
  stratified CV, scored on ROC-AUC) tunes the production Random Forest.
- **Threshold tuning**: instead of the naive 0.5 cutoff, the decision
  threshold is chosen to maximize F1 on the precision-recall curve —
  churn use cases usually care more about catching at-risk customers
  (recall) than raw accuracy.
- **Engineered features** that matter for churn specifically:
  `BalanceSalaryRatio`, `IsZeroBalance`, `TenureByAge`,
  `ProductsPerTenureYr`, `IsSeniorCustomer`, `CreditScoreBucket`.
- **Train/serve consistency**: the fitted `StandardScaler` and exact
  feature column list are persisted to `models/`, so `predict.py` replays
  the identical transformation on new data — no train/serve skew.

## Outputs

| File | Produced by | Purpose |
|---|---|---|
| `data/processed/cleaned_data.csv` | `preprocess.py` | cleaned dataset |
| `data/processed/featured_data.csv` | `feature_engineering.py` | model-ready features |
| `data/processed/predictions.csv` | `risk_score.py` | every customer + churn probability + risk tier |
| `data/processed/retention_worklist_top100.csv` | `risk_score.py` | top 100 customers ranked by estimated value at risk |
| `models/random_forest.pkl` | `train_model.py` | final trained model |
| `models/scaler.pkl`, `feature_names.pkl` | `feature_engineering.py` | inference-time transform |
| `images/graphs/*.png` | `eda.py`, `train_model.py` | EDA + evaluation plots |

`predictions.csv` is the file to load into **Power BI** (`dashboard/`) —
see `dashboard/README.md` for the suggested visuals.

## Scoring New Customers

```bash
python src/predict.py --input path/to/new_customers.csv --output data/processed/new_predictions.csv
```

## Risk Tiers

| Tier | Churn probability |
|---|---|
| Low | < 0.30 |
| Medium | 0.30 - 0.60 |
| High | 0.60 - 0.80 |
| Critical | ≥ 0.80 |

`estimated_value_at_risk = churn_probability × Balance` prioritizes
outreach toward high-balance, high-risk customers first.
