"""Interactive Power BI-style bank churn dashboard.

Run from the project root with: streamlit run dashboard/app.py
"""

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))

from src.risk_score import score_and_tier  # noqa: E402


PREDICTIONS_PATH = ROOT_DIR / "data" / "processed" / "predictions.csv"
WORKLIST_PATH = ROOT_DIR / "data" / "processed" / "retention_worklist_top100.csv"

st.set_page_config(page_title="Bank Churn Intelligence", page_icon="🏦", layout="wide")

st.markdown(
	"""
	<style>
	[data-testid="stMetricValue"] { color: #0f766e; }
	.block-container { padding-top: 2rem; }
	</style>
	""",
	unsafe_allow_html=True,
)


@st.cache_data
def load_csv(path: Path) -> pd.DataFrame:
	return pd.read_csv(path)


def money(value: float) -> str:
	return f"${value:,.0f}"


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
	result = df.copy()
	result["Age band"] = pd.cut(
		result["Age"], bins=[0, 30, 45, 60, 120],
		labels=["18-30", "31-45", "46-60", "60+"], include_lowest=True,
	)
	return result


@st.cache_data
def score_customer(values: tuple) -> pd.DataFrame:
	columns = [
		"CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age",
		"Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
		"EstimatedSalary",
	]
	return score_and_tier(pd.DataFrame([values], columns=columns))


def validate_customer_input(values: tuple) -> str | None:
	"""Reject impossible values before they reach the persisted model."""
	_, _, credit_score, _, _, age, tenure, balance, products, _, _, salary = values
	if not 300 <= credit_score <= 900:
		return "Credit score must be between 300 and 900."
	if not 18 <= age <= 100:
		return "Age must be between 18 and 100."
	if not 0 <= tenure <= 10:
		return "Tenure must be between 0 and 10 years."
	if not 1 <= products <= 4:
		return "Number of products must be between 1 and 4."
	if balance < 0 or salary < 0:
		return "Balance and salary cannot be negative."
	return None


def render_customer_scorer() -> None:
	st.subheader("Score a new customer")
	st.caption("Enter customer details to run the fitted ML model and estimate retention priority.")
	with st.form("customer_scorer"):
		left, middle, right = st.columns(3)
		with left:
			credit_score = st.number_input("Credit score", 300, 900, 650)
			geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
			gender = st.selectbox("Gender", ["Female", "Male"])
			age = st.number_input("Age", 18, 100, 40)
		with middle:
			tenure = st.number_input("Tenure (years)", 0, 10, 5)
			balance = st.number_input("Balance", 0.0, 1_000_000.0, 50_000.0, step=1000.0)
			products = st.number_input("Number of products", 1, 4, 2)
			salary = st.number_input("Estimated salary", 0.0, 1_000_000.0, 75_000.0, step=1000.0)
		with right:
			has_card = st.checkbox("Has credit card", True)
			active = st.checkbox("Is active member", True)
			submitted = st.form_submit_button("Run churn model", type="primary")

	if submitted:
		values = (0, "New customer", credit_score, geography, gender, age, tenure, balance, products, int(has_card), int(active), salary)
		validation_error = validate_customer_input(values)
		if validation_error:
			st.error(validation_error)
			return
		try:
			scored = score_customer(values).iloc[0]
			probability = float(scored["churn_probability"])
			value_at_risk = float(scored["estimated_value_at_risk"])
			metrics = st.columns(3)
			metrics[0].metric("Churn probability", f"{probability:.1%}")
			metrics[1].metric("Risk tier", str(scored["risk_tier"]))
			metrics[2].metric("Estimated value at risk", money(value_at_risk))
			st.info("This is a model estimate, not a certainty. Use it to prioritize outreach and combine it with current customer context.")
		except (FileNotFoundError, KeyError, ValueError) as error:
			st.error(f"Model scoring failed: {error}. Run the training pipeline first.")


@st.cache_data
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame]:
	predictions = add_age_band(load_csv(PREDICTIONS_PATH))
	worklist = load_csv(WORKLIST_PATH) if WORKLIST_PATH.exists() else predictions.copy()
	return predictions, worklist.sort_values("estimated_value_at_risk", ascending=False)


st.title("Bank Churn Intelligence")
st.caption("Executive monitoring, customer-level risk exploration, and retention action queue")

if "refresh_message" not in st.session_state:
	st.session_state["refresh_message"] = False

if st.button("Refresh latest scored data", icon=":material/refresh:", key="refresh_dashboard"):
	st.cache_data.clear()
	st.session_state["refresh_message"] = True
	st.rerun()

try:
	predictions, worklist = load_dashboard_data()
except FileNotFoundError:
	st.error("predictions.csv not found. Run `python main.py` before opening the dashboard.")
	st.stop()

if st.session_state.pop("refresh_message", False):
	st.success("Latest predictions and retention worklist loaded.")

with st.sidebar:
	st.header("Dashboard filters")
	selected_tiers = st.multiselect("Risk tier", sorted(predictions["risk_tier"].dropna().unique()), default=sorted(predictions["risk_tier"].dropna().unique()))
	selected_geo = st.multiselect("Geography", sorted(predictions["Geography"].dropna().unique()), default=sorted(predictions["Geography"].dropna().unique()))
	selected_gender = st.multiselect("Gender", sorted(predictions["Gender"].dropna().unique()), default=sorted(predictions["Gender"].dropna().unique()))
	active_options = st.multiselect("Active member", [0, 1], default=[0, 1], format_func=lambda value: "Active" if value else "Inactive")

filtered = predictions[
	predictions["risk_tier"].isin(selected_tiers)
	& predictions["Geography"].isin(selected_geo)
	& predictions["Gender"].isin(selected_gender)
	& predictions["IsActiveMember"].isin(active_options)
]

overview, explorer, scorer, retention = st.tabs(["Executive Overview", "Customer Risk Explorer", "Score New Customer", "Retention Worklist"])

with overview:
	total_value = filtered["estimated_value_at_risk"].sum()
	kpis = st.columns(4)
	kpis[0].metric("Total customers", f"{len(filtered):,}")
	kpis[1].metric("Overall churn rate", f"{filtered['churn_prediction'].mean():.1%}" if len(filtered) else "0.0%")
	kpis[2].metric("Critical risk customers", f"{(filtered['risk_tier'] == 'Critical').sum():,}")
	kpis[3].metric("Total value at risk", money(total_value))

	left, right = st.columns(2)
	with left:
		tier_counts = filtered["risk_tier"].value_counts().rename_axis("risk_tier").reset_index(name="customers")
		st.plotly_chart(px.pie(tier_counts, names="risk_tier", values="customers", hole=0.55, title="Customers by risk tier", color_discrete_sequence=px.colors.qualitative.Set2))
	with right:
		geo_rate = filtered.groupby("Geography", as_index=False)["churn_prediction"].mean()
		geo_rate["churn_rate"] = geo_rate["churn_prediction"] * 100
		st.plotly_chart(px.bar(geo_rate, x="Geography", y="churn_rate", title="Churn rate by geography", labels={"churn_rate": "Churn rate (%)"}, color="Geography"))

	product_rate = filtered.groupby("NumOfProducts", as_index=False)["churn_prediction"].mean()
	product_rate["churn_rate"] = product_rate["churn_prediction"] * 100
	st.plotly_chart(px.bar(product_rate, x="NumOfProducts", y="churn_rate", title="Churn rate by number of products", labels={"churn_rate": "Churn rate (%)"}, color="NumOfProducts"))

with explorer:
	st.subheader("Customer risk explorer")
	age_bands = st.multiselect("Age band", sorted(filtered["Age band"].dropna().astype(str).unique()), default=sorted(filtered["Age band"].dropna().astype(str).unique()))
	explorer_data = filtered[filtered["Age band"].astype(str).isin(age_bands)]
	display_columns = ["CustomerId", "Geography", "Gender", "Age", "churn_probability", "risk_tier", "estimated_value_at_risk"]
	st.dataframe(
		explorer_data[display_columns].sort_values("churn_probability", ascending=False),
		column_config={
			"churn_probability": st.column_config.NumberColumn("Churn probability", format="percent"),
			"estimated_value_at_risk": st.column_config.NumberColumn("Estimated value at risk", format="$%,.2f"),
		},
		width="stretch",
		hide_index=True,
	)
	scatter = px.scatter(explorer_data, x="Age", y="Balance", color="risk_tier", hover_data=["CustomerId", "churn_probability", "estimated_value_at_risk"], title="Age vs balance", color_discrete_sequence=px.colors.qualitative.Set2)
	st.plotly_chart(scatter)

with scorer:
	render_customer_scorer()

with retention:
	st.subheader("Retention worklist")
	st.caption("Relationship managers should action the highest estimated value at risk first.")
	worklist_columns = [column for column in ["CustomerId", "Geography", "Gender", "Age", "churn_probability", "risk_tier", "estimated_value_at_risk"] if column in worklist.columns]
	st.dataframe(
		worklist[worklist_columns].sort_values("estimated_value_at_risk", ascending=False),
		column_config={
			"churn_probability": st.column_config.NumberColumn("Churn probability", format="percent"),
			"estimated_value_at_risk": st.column_config.NumberColumn("Estimated value at risk", format="$%,.2f"),
		},
		width="stretch",
		hide_index=True,
	)
