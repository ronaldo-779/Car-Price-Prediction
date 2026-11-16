
"""
dashboard.py – Main Streamlit entry point for Used Car Price Prediction Dashboard.

Run with:
    streamlit run app/dashboard.py
"""

import streamlit as st

from data_loader import load_data
from overview import show_overview
from eda import show_eda
from records import show_records
from data_quality import show_data_quality
from outliers import show_outliers
from prediction import show_prediction
from model_comparison import show_model_comparison


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🚗 Used Car Price Prediction",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f77b4;
    }
    .stMetric {
        background-color: #f0f4f8;
        border-radius: 8px;
        padding: 10px;
    }
    .block-container {
        padding-top: 1rem;
    }
    section[data-testid="stSidebar"] {
        background-color: #1a1a2e;
        color: white;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #eee !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stCaption {
        color: #ccc !important;
    }
    section[data-testid="stSidebar"] table {
        color: #f0f0f0 !important;
        width: 100%;
    }
    section[data-testid="stSidebar"] th,
    section[data-testid="stSidebar"] td {
        color: #f0f0f0 !important;
        border-color: #444 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown a {
        color: #90caf9 !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown('<p class="main-header">🚗 Used Car Price Prediction Dashboard</p>', unsafe_allow_html=True)
st.caption("Machine Learning Project — Dataset Analysis, Model Comparison & Price Prediction")
st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# LOAD DATASET
# ──────────────────────────────────────────────────────────────────────────────

try:
    df = load_data()
except Exception as error:
    st.error(f"❌ Error loading dataset: {error}")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.markdown("## 🚗 Car Price Dashboard")
st.sidebar.caption("Navigate between sections below")
st.sidebar.divider()

PAGE_ICONS = {
    "📊 Overview":                 "📊 Overview",
    "🔍 EDA & Charts":             "🔍 EDA & Charts",
    "📋 All Records":              "📋 All Records",
    "🛡️ Data Quality":             "🛡️ Data Quality",
    "🔎 Outlier Review":           "🔎 Outlier Review",
    "🤖 Model Training":           "🤖 Model Training",
    "🔮 Price Prediction":         "🔮 Price Prediction",
}

page = st.sidebar.radio(
    "Select Page",
    list(PAGE_ICONS.keys()),
    label_visibility="collapsed",
)

st.sidebar.divider()

# Dataset summary in sidebar
st.sidebar.markdown("### 📂 Dataset Info")
st.sidebar.markdown(f"""
| Field | Value |
|-------|-------|
| **Records** | {len(df):,} |
| **Brands** | {df["Brand"].nunique()} |
| **Cities** | {df["City"].nunique()} |
| **Median Price** | ₹{df["Selling_Price"].median():,.0f} |
| **Year Range** | {int(df["Year"].min())}–{int(df["Year"].max())} |
""")

# Model status
from pathlib import Path
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.joblib"
if MODEL_PATH.exists():
    st.sidebar.success("✅ Trained model available")
else:
    st.sidebar.warning("⚠️ No trained model — go to Model Training")

st.sidebar.divider()
st.sidebar.caption("Indian Used Car Market Dataset | 2,000 Records")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE ROUTING
# ──────────────────────────────────────────────────────────────────────────────

if page == "📊 Overview":
    show_overview(df)

elif page == "🔍 EDA & Charts":
    show_eda(df)

elif page == "📋 All Records":
    show_records(df)

elif page == "🛡️ Data Quality":
    show_data_quality(df)

elif page == "🔎 Outlier Review":
    show_outliers(df)

elif page == "🤖 Model Training":
    show_model_comparison(df)

elif page == "🔮 Price Prediction":
    show_prediction(df)
