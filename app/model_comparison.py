
"""
model_comparison.py – Streamlit page: train multiple ML models & compare them.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import sys
import os

# Ensure the app directory is on the path
sys.path.insert(0, os.path.dirname(__file__))
from ml_trainer import train_and_evaluate, prepare_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
ALL_MODELS_PATH = MODEL_DIR / "all_models.joblib"
BEST_MODEL_PATH = MODEL_DIR / "best_model.joblib"


def show_model_comparison(df):
    st.header("🤖 Model Training & Comparison")
    st.caption(
        "Train 10 ML algorithms on the dataset and compare their performance. "
        "The best model is saved and used in Price Prediction."
    )

    # ── Training config ───────────────────────────────────────
    with st.expander("⚙️ Training Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider(
                "Test Split (%)", 10, 40, 20, 5, key="test_size"
            ) / 100
        with col2:
            cv_folds = st.slider(
                "Cross-Validation Folds", 3, 10, 5, key="cv_folds"
            )

    X, y = prepare_data(df)
    st.info(
        f"**Training data:** {len(X)} usable records "
        f"(after removing missing/invalid rows and 'Not available' transmissions)"
    )

    # ── Train button ──────────────────────────────────────────
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        progress_bar = st.progress(0.0)
        status_text  = st.empty()

        def _cb(pct, msg):
            progress_bar.progress(pct)
            status_text.text(msg)

        with st.spinner("Training models – this may take ~1-2 minutes…"):
            result = train_and_evaluate(
                df,
                test_size=test_size,
                cv_folds=cv_folds,
                progress_cb=_cb,
            )

        progress_bar.progress(1.0)
        status_text.text("✅ All models trained!")

        st.session_state["train_result"] = result
        st.success(
            f"🏆 **Best Model: {result['best_name']}** "
            f"(R² = {max(r['R²'] for r in result['results']):.4f})"
        )

    # ── Display results (from cache or fresh run) ─────────────
    result = st.session_state.get("train_result")

    if result is None and BEST_MODEL_PATH.exists():
        st.info(
            "Previously trained models detected. "
            "Click **Train All Models** to retrain and see fresh results."
        )
        return

    if result is None:
        st.warning(
            "No trained models found. Click **Train All Models** above to start."
        )
        return

    res_df = pd.DataFrame(result["results"]).sort_values("R²", ascending=False)

    # ── Metrics table ─────────────────────────────────────────
    st.subheader("📋 Model Performance Table")
    st.dataframe(
        res_df.style.format({
            "MAE": "₹{:,.0f}",
            "RMSE": "₹{:,.0f}",
            "R²": "{:.4f}",
            "CV R² Mean": "{:.4f}",
            "CV R² Std": "{:.4f}",
        }).background_gradient(subset=["R²"], cmap="RdYlGn")
        .background_gradient(subset=["RMSE"], cmap="RdYlGn_r"),
        use_container_width=True,
        hide_index=True,
    )

    # ── R² bar chart ──────────────────────────────────────────
    st.subheader("📊 R² Score Comparison")
    colors = [
        "#2ecc71" if r["Model"] == result["best_name"] else "#3498db"
        for r in result["results"]
    ]
    fig = go.Figure(go.Bar(
        x=res_df["Model"],
        y=res_df["R²"],
        marker_color=colors,
        text=res_df["R²"].round(4),
        textposition="outside",
    ))
    fig.update_layout(
        title="R² Score by Model (higher is better)",
        yaxis_title="R² Score",
        xaxis_tickangle=-30,
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── MAE & RMSE comparison ─────────────────────────────────
    st.subheader("📉 Error Metrics Comparison")
    col_a, col_b = st.columns(2)

    with col_a:
        fig2 = px.bar(
            res_df, x="Model", y="MAE",
            title="Mean Absolute Error (₹) – lower is better",
            color="MAE", color_continuous_scale="Reds_r",
        )
        fig2.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        fig3 = px.bar(
            res_df, x="Model", y="RMSE",
            title="RMSE (₹) – lower is better",
            color="RMSE", color_continuous_scale="Oranges_r",
        )
        fig3.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

    # ── CV R² with error bands ────────────────────────────────
    st.subheader("🔁 Cross-Validation R² (Mean ± Std)")
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(
        x=res_df["Model"],
        y=res_df["CV R² Mean"],
        error_y=dict(type="data", array=res_df["CV R² Std"].tolist()),
        marker_color="#9b59b6",
        name="CV R²",
    ))
    fig4.update_layout(
        title="Cross-Validation R² (5-fold) with Std Dev",
        yaxis_title="CV R²",
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig4, use_container_width=True)

    # ── Actual vs Predicted for best model ───────────────────
    st.subheader(f"🎯 Actual vs Predicted – {result['best_name']}")
    y_test  = result["y_test"]
    y_pred  = result["y_pred_best"]
    scatter_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    fig5 = px.scatter(
        scatter_df, x="Actual", y="Predicted",
        title=f"Actual vs Predicted Prices – {result['best_name']}",
        opacity=0.6,
        color_discrete_sequence=["#e74c3c"],
    )
    mn = min(y_test.min(), y_pred.min())
    mx = max(y_test.max(), y_pred.max())
    fig5.add_trace(go.Scatter(
        x=[mn, mx], y=[mn, mx],
        mode="lines", name="Perfect Fit",
        line=dict(color="green", dash="dash"),
    ))
    st.plotly_chart(fig5, use_container_width=True)

    # ── Residual plot ─────────────────────────────────────────
    st.subheader("📐 Residual Distribution")
    residuals = y_test - y_pred
    fig6 = px.histogram(
        x=residuals, nbins=50,
        title="Residuals (Actual − Predicted)",
        color_discrete_sequence=["#f39c12"],
        marginal="box",
    )
    fig6.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig6, use_container_width=True)

    st.success(
        f"✅ Best model saved: **{result['best_name']}**  "
        f"| Test R² = **{max(r['R²'] for r in result['results']):.4f}**  "
        f"| RMSE = **₹{min(r['RMSE'] for r in result['results']):,.0f}**"
    )
