import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path
import sys
# Add project folder to Python path
PROJECT_DIR = Path(__file__).resolve().parent.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.predict import load_artifacts, prepare_input

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Bank Churn Prediction Dashboard",
    page_icon="🏦",
    layout="wide"
)
# ============================================================
# CUSTOM WHITE THEME
# ============================================================

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: white;
    }

    /* Main content */
    .main {
        background-color: white;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }

    /* Text */
    h1, h2, h3, h4, h5, h6, p, label {
        color: #000000 !important;
    }

    .main-title {
        color: #000000 !important;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #000000 !important;
        font-size: 18px;
        margin-bottom: 25px;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #dddddd;
        border-radius: 12px;
        padding: 15px;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"

MODEL_PATH = MODEL_DIR / "random_forest.pkl"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            color: #000000 !important;
            font-size: 38px;
            font-weight: 700;
            margin-bottom: 5px;
        }

        .subtitle {
            color: #000000 !important;
            font-size: 18px;
            margin-bottom: 25px;
        }

        .metric-card {
            padding: 20px;
            border-radius: 12px;
            background-color: #f7f7f7;
            text-align: center;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🏦 Bank Customer Churn Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Customer Churn & Risk Analytics Dashboard</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data() -> tuple[pd.DataFrame | None, Path | None]:

    possible_files = [
        DATA_DIR / "customer_risk_scores.csv",
        DATA_DIR / "processed" / "customer_risk_scores.csv",
        DATA_DIR / "processed" / "predictions.csv",
        DATA_DIR / "predictions.csv",
    ]

    for file in possible_files:

        if file.exists():

            df = pd.read_csv(file)

            return df, file

    return None, None


df, data_file = load_data()


# ============================================================
# DATA NOT FOUND
# ============================================================

if df is None or data_file is None:

    st.error(
        "Customer risk/prediction CSV file nahi mili."
    )

    st.info(
        "Pehle `python main.py` run karo aur generated CSV file check karo."
    )

    st.stop()

assert df is not None
assert data_file is not None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔎 Dashboard Filters")

if data_file is not None:
    st.sidebar.write(
        f"Loaded file: `{data_file.name}`"
    )


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(dataframe: pd.DataFrame, possible_names: list[str]) -> str | None:

    for name in possible_names:

        if name in dataframe.columns:
            return name

    return None


risk_col = find_column(
    df,
    [
        "risk_tier",
        "Risk_Tier",
        "Risk_Category",
        "risk_category"
    ]
)

prob_col = find_column(
    df,
    [
        "churn_probability",
        "Churn_Probability",
        "probability",
        "churn_prob"
    ]
)

prediction_col = find_column(
    df,
    [
        "prediction",
        "Prediction",
        "Churn",
        "Exited",
        "Predicted_Churn",
        "churn_prediction",
        "Churn_Prediction"
    ]
)


# ============================================================
# KPI CALCULATIONS
# ============================================================

total_customers = len(df)

if prediction_col:

    predicted_churners = int(
        pd.to_numeric(
            df[prediction_col],
            errors="coerce"
        ).fillna(0).sum()
    )

else:

    predicted_churners = 0


churn_rate = (
    predicted_churners / total_customers * 100
    if total_customers > 0
    else 0
)


if risk_col:

    critical_count = int(
        (df[risk_col].astype(str).str.lower() == "critical").sum()
    )

    high_count = int(
        (df[risk_col].astype(str).str.lower() == "high").sum()
    )

else:

    critical_count = 0
    high_count = 0


# ============================================================
# KPI CARDS
# ============================================================

st.subheader("📊 Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Customers",
        f"{total_customers:,}"
    )


with col2:

    st.metric(
        "Predicted Churners",
        f"{predicted_churners:,}"
    )


with col3:

    st.metric(
        "Churn Rate",
        f"{churn_rate:.1f}%"
    )


with col4:

    st.metric(
        "Critical Risk",
        f"{critical_count:,}"
    )


st.divider()


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader("⚠️ Customer Risk Distribution")


if risk_col:

    risk_counts = (
        df[risk_col]
        .astype(str)
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = [
        "Risk",
        "Customers"
    ]

    fig_risk = px.bar(
        risk_counts,
        x="Risk",
        y="Customers",
        title="Customers by Risk Tier",
        text="Customers"
    )

    fig_risk.update_layout(
        xaxis_title="Risk Tier",
        yaxis_title="Number of Customers"
    )

    st.plotly_chart(
        fig_risk,
        use_container_width=True
    )

else:

    st.warning(
        "Risk tier column nahi mili."
    )


# ============================================================
# CHURN DISTRIBUTION
# ============================================================

st.subheader("📈 Churn Prediction Distribution")


if prediction_col:

    churn_counts = (
        df[prediction_col]
        .value_counts()
        .reset_index()
    )

    churn_counts.columns = [
        "Prediction",
        "Customers"
    ]

    fig_churn = px.pie(
        churn_counts,
        names="Prediction",
        values="Customers",
        title="Predicted Churn Distribution"
    )

    st.plotly_chart(
        fig_churn,
        use_container_width=True
    )

else:

    st.info(
        "Prediction column nahi mili."
    )


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

if prob_col:

    st.subheader("🎯 Churn Probability")

    probability_data = pd.to_numeric(
        df[prob_col],
        errors="coerce"
    ).dropna()

    fig_probability = px.histogram(
        probability_data,
        x=probability_data,
        nbins=30,
        title="Customer Churn Probability Distribution"
    )

    fig_probability.update_layout(
        xaxis_title="Churn Probability",
        yaxis_title="Customers"
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True
    )
# ============================================================
# CUSTOMER CHURN PREDICTION INPUT
# ============================================================

st.markdown("---")
st.header("🔮 Predict Customer Churn")
st.write("Enter customer details to get a churn prediction.")

with st.container():
    col1, col2, col3 = st.columns(3)

    # Column 1
    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=650,
            step=1
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35,
            step=1
        )

    # Column 2
    with col2:
        tenure = st.number_input(
            "Tenure (Years)",
            min_value=0,
            max_value=10,
            value=5,
            step=1
        )

        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=4,
            value=2,
            step=1
        )

    # Column 3
    with col3:
        has_credit_card = st.selectbox(
            "Has Credit Card",
            ["Yes", "No"]
        )

        is_active_member = st.selectbox(
            "Is Active Member",
            ["Yes", "No"]
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=60000.0,
            step=1000.0
        )

# Predict button
predict_button = st.button(
    "🔮 Predict Churn",
    type="primary",
    width="stretch"
)

if predict_button:
    try:
        # ====================================================
        # STEP 3.1 - Load trained model
        # ====================================================

        model_path = BASE_DIR / "models" / "random_forest.pkl"
        threshold_path = BASE_DIR / "models" / "decision_threshold.pkl"

        if not model_path.exists():
            st.error("❌ Trained model not found: models/random_forest.pkl")
            st.stop()

        model, scaler, feature_names, decision_threshold = load_artifacts()

        # ====================================================
        # STEP 3.2 - Create user input DataFrame
        # ====================================================

        user_df = pd.DataFrame([{
            "CreditScore": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": num_products,
            "HasCrCard": 1 if has_credit_card == "Yes" else 0,
            "IsActiveMember": 1 if is_active_member == "Yes" else 0,
            "EstimatedSalary": estimated_salary
        }])

        # ====================================================
        # STEP 3.3 - Apply the training-time inference pipeline
        # ====================================================

        user_features, _ = prepare_input(
            user_df,
            scaler,
            feature_names
        )

        # ====================================================
        # STEP 3.8 - Prediction probability
        # ====================================================

        churn_probability = float(
            model.predict_proba(user_features)[0][1]
        )

        # Apply saved threshold
        prediction = int(
            churn_probability >= decision_threshold
        )

        # ====================================================
        # STEP 3.9 - Display result
        # ====================================================

        st.markdown("---")
        st.subheader("📊 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Churn Probability",
                f"{churn_probability * 100:.2f}%"
            )

        with result_col2:
            st.metric(
                "Decision Threshold",
                f"{decision_threshold * 100:.2f}%"
            )

        # Prediction message
        if prediction == 1:
            st.error(
                "⚠️ Customer is likely to CHURN"
            )

            st.warning(
                f"The predicted churn probability is "
                f"{churn_probability * 100:.2f}%."
            )

        else:
            st.success(
                "✅ Customer is likely to STAY"
            )

            st.info(
                f"The predicted churn probability is "
                f"{churn_probability * 100:.2f}%."
            )

        # ====================================================
        # Probability progress bar
        # ====================================================

        st.write("### Churn Probability")

        st.progress(
            min(churn_probability, 1.0)
        )

        # ====================================================
        # Input summary
        # ====================================================

        st.write("### 👤 Customer Input Summary")

        display_df = pd.DataFrame([{
            "Credit Score": credit_score,
            "Geography": geography,
            "Gender": gender,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "Products": num_products,
            "Credit Card": has_credit_card,
            "Active Member": is_active_member,
            "Estimated Salary": estimated_salary
        }])

        st.dataframe(
            display_df,
            width="stretch",
            hide_index=True
        )

    except Exception as e:

        st.error("❌ Prediction failed")

        st.exception(e)
   


# ============================================================
# NUMERIC FEATURE ANALYSIS
# ============================================================

st.subheader("📊 Customer Data Analysis")


numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


if len(numeric_columns) > 0:

    selected_column = st.selectbox(
        "Select a numeric feature",
        numeric_columns
    )

    fig_feature = px.histogram(
        df,
        x=selected_column,
        title=f"Distribution of {selected_column}",
        marginal="box"
    )

    st.plotly_chart(
        fig_feature,
        use_container_width=True
    )


# ============================================================
# CUSTOMER TABLE
# ============================================================

st.subheader("👥 Customer Risk Data")

search_text = st.text_input(
    "Search customer data"
)


display_df = df.copy()


if search_text:

    mask = display_df.astype(str).apply(
        lambda row: row.str.contains(
            search_text,
            case=False,
            na=False
        ).any(),
        axis=1
    )

    display_df = display_df[mask]


st.dataframe(
    display_df.head(100),
    use_container_width=True
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("🤖 Model Information")

if MODEL_PATH.exists():

    try:

        model = joblib.load(MODEL_PATH)

        st.success(
            "Random Forest model successfully loaded."
        )

        st.write(
            f"Model Type: `{type(model).__name__}`"
        )

    except Exception as e:

        st.warning(
            f"Model load nahi ho saka: {e}"
        )

else:

    st.warning(
        "random_forest.pkl file nahi mili."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Bank Customer Churn Prediction & Risk Scoring System"
)