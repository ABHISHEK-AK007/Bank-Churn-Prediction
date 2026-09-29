# Power BI Dashboard

## Deploy the Streamlit dashboard

For Streamlit Community Cloud, select `dashboard/app.py` as the app entrypoint.
Community Cloud will use `dashboard/requirements.txt`, keeping notebook and
model-training packages out of the dashboard build. The deployed app reads the
committed files in `data/processed/`.

The trained model file `models/random_forest.pkl` is excluded from Git and is
too large for a regular GitHub file, so the dashboard remains usable for
exploration but customer scoring is unavailable unless that model is hosted
separately and configured for the deployment.

`.pbix` files are binary Power BI Desktop files and can't be generated
from code — build `bank_dashboard.pbix` in Power BI Desktop using
`data/processed/predictions.csv` as the data source. Suggested layout:

## Data source
`Get Data → Text/CSV → data/processed/predictions.csv`
(Refresh this after every `python main.py` run so the dashboard reflects
the latest scored customers.)

## Suggested pages / visuals

**Page 1 — Executive Overview**
- KPI cards: total customers, overall churn rate, count of "Critical" risk tier, total estimated value at risk
- Donut chart: customers by `risk_tier`
- Bar chart: churn rate by `Geography`
- Bar chart: churn rate by `NumOfProducts`

**Page 2 — Customer Risk Explorer**
- Table/matrix: customer list with `churn_probability`, `risk_tier`, `estimated_value_at_risk`, filterable by Geography/Gender/Age band
- Slicers: `risk_tier`, `Geography`, `IsActiveMember`
- Scatter plot: `Age` vs `Balance`, colored by `risk_tier`

**Page 3 — Retention Worklist**
- Table sourced from `data/processed/retention_worklist_top100.csv`, sorted by `estimated_value_at_risk` descending — this is the list relationship managers action first.

## Suggested DAX measures

```
Churn Rate = DIVIDE(SUM(predictions[churn_prediction]), COUNTROWS(predictions))

Total Value at Risk = SUM(predictions[estimated_value_at_risk])

Critical Risk Customers = CALCULATE(COUNTROWS(predictions), predictions[risk_tier] = "Critical")
```
