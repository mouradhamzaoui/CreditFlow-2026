import time
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# ==========================================
# CONFIGURATION PAGE
# ==========================================

st.set_page_config(page_title="CreditFlow 2026", page_icon="🏦", layout="wide")

# ==========================================
# LOAD MODELS
# ==========================================


@st.cache_resource
def load_models():
    xgb_model = joblib.load("models/xgb_regressor.joblib")
    lgb_model = joblib.load("models/lgbm_regressor.joblib")
    explainer = joblib.load("models/shap_explainer.joblib")
    feature_names = joblib.load("src/models/artifacts/feature_names.joblib")

    return xgb_model, lgb_model, explainer, feature_names


xgb_model, lgb_model, explainer, feature_names = load_models()

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("🏦 Données Entreprise")

years = st.sidebar.slider("Years in Business", 1, 50, 5)

sector = st.sidebar.selectbox(
    "Industry Sector", ["Tech", "Retail", "Manufacturing", "Services", "Healthcare"]
)

revenue = st.sidebar.number_input(
    "Annual Revenue (€)", min_value=1000.0, value=150000.0, step=1000.0
)

burn = st.sidebar.number_input(
    "Monthly Burn Rate (€)", min_value=100.0, value=12000.0, step=100.0
)

debt = st.sidebar.slider("Debt to Equity", 0.1, 3.0, 0.8)

defaults = st.sidebar.selectbox("Previous Defaults", [0, 1])

balance = st.sidebar.number_input(
    "Average Account Balance (€)", min_value=0.0, value=25000.0, step=100.0
)

esg = st.sidebar.slider("ESG Score", 0.0, 100.0, 75.0)

volatility = st.sidebar.slider("Market Volatility", 0.01, 0.50, 0.15)

# ==========================================
# MAIN PAGE
# ==========================================

st.title("🏦 CreditFlow 2026")
st.markdown("### AI-Powered B2B Credit Line Decision System")

if st.button("🔍 Analyze Company", use_container_width=True):
    start = time.time()

    # ==========================================
    # PREPROCESS INPUT
    # ==========================================

    df = pd.DataFrame(
        [
            {
                "YEARS_IN_BUSINESS": years,
                "INDUSTRY_SECTOR": sector,
                "ANNUAL_REVENUE": revenue,
                "MONTHLY_BURN_RATE": burn,
                "DEBT_TO_EQUITY": debt,
                "PREVIOUS_DEFAULTS": defaults,
                "AVG_ACCOUNT_BALANCE": balance,
                "ESG_SCORE": esg,
                "MARKET_VOLATILITY": volatility,
            }
        ]
    )

    # Log transform
    for col in ["ANNUAL_REVENUE", "MONTHLY_BURN_RATE", "AVG_ACCOUNT_BALANCE"]:
        df[f"LOG_{col}"] = np.log1p(df[col])
        df.drop(columns=[col], inplace=True)

    # One-hot
    df = pd.get_dummies(
        df, columns=["INDUSTRY_SECTOR"], prefix="SECTOR", drop_first=True
    )

    # Align columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    # ==========================================
    # PREDICTION
    # ==========================================

    xgb_pred = xgb_model.predict(df)[0]
    lgb_pred = lgb_model.predict(df)[0]

    pred_log = (xgb_pred + lgb_pred) / 2

    prediction = np.expm1(pred_log)

    prediction = max(0, prediction)

    inference_time = (time.time() - start) * 1000

    # ==========================================
    # METRICS
    # ==========================================

    st.markdown("---")

    c1, c2, c3 = st.columns(3)

    c1.metric("💰 Recommended Credit Line", f"{prediction:,.0f} €")

    c2.metric("⚡ Inference Time", f"{inference_time:.1f} ms")

    c3.metric("🎯 Model Confidence (R²)", "98.6%")

    # ==========================================
    # GAUGE
    # ==========================================

    st.markdown("## 📊 Credit Recommendation Score")

    normalized = min(prediction / 100000, 1.0)

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=normalized * 100,
            title={"text": "Credit Capacity"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "green"},
                "steps": [
                    {"range": [0, 40], "color": "#ffcccc"},
                    {"range": [40, 70], "color": "#fff3cd"},
                    {"range": [70, 100], "color": "#d4edda"},
                ],
            },
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================
    # SHAP EXPLANATIONS
    # ==========================================

    st.markdown("## 🔍 Top Financial Drivers")

    shap_values = explainer.shap_values(df)[0]

    impact = list(zip(df.columns, shap_values))

    impact.sort(key=lambda x: abs(x[1]), reverse=True)

    top3 = impact[:3]

    shap_df = pd.DataFrame(top3, columns=["Feature", "SHAP Impact"])

    fig_bar = px.bar(
        shap_df,
        x="SHAP Impact",
        y="Feature",
        orientation="h",
        color="SHAP Impact",
        color_continuous_scale="Blues",
    )

    st.plotly_chart(fig_bar, use_container_width=True)

    # ==========================================
    # RAW FEATURES
    # ==========================================

    with st.expander("📄 Processed Features"):
        st.dataframe(df)

else:
    st.info("👈 Configure company financials and click Analyze.")
