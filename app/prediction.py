
"""
prediction.py – Price prediction page with multi-model support.
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR        = ROOT / "models"
BEST_MODEL_PATH  = MODEL_DIR / "best_model.joblib"
ALL_MODELS_PATH  = MODEL_DIR / "all_models.joblib"

FEATURE_COLS = ["Brand", "Year", "Kilometers_Driven", "Fuel_Type", "Transmission"]


def show_prediction(df):
    st.header("🔮 Used Car Price Prediction")
    st.write(
        "Fill in the vehicle details and get an estimated second-hand selling price. "
        "You can compare predictions from multiple trained models."
    )

    # ── Check models ──────────────────────────────────────────
    model_available    = BEST_MODEL_PATH.exists()
    all_models_available = ALL_MODELS_PATH.exists()

    if not model_available:
        st.warning(
            "⚠️ No trained model found. "
            "Go to **Model Training & Comparison** page and click **Train All Models** first."
        )
        return

    # Load models
    best_pipeline = joblib.load(BEST_MODEL_PATH)
    all_pipelines = joblib.load(ALL_MODELS_PATH) if all_models_available else None

    # Determine best model name
    best_name = st.session_state.get("train_result", {}).get("best_name", "Best Model")
    if "train_result" in st.session_state:
        best_name = st.session_state["train_result"]["best_name"]

    # ── Input form ────────────────────────────────────────────
    st.subheader("🚗 Vehicle Details")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)

        with col1:
            brand = st.selectbox(
                "🏷️ Car Brand",
                sorted(df["Brand"].dropna().unique()),
                help="Select the manufacturer brand",
            )

            filtered_models = sorted(
                df[df["Brand"] == brand]["Model"].dropna().unique()
            ) if brand else sorted(df["Model"].dropna().unique())

            year = st.number_input(
                "📅 Manufacturing Year",
                min_value=1990, max_value=2026, value=2018, step=1,
                help="Year the car was manufactured",
            )

            city = st.selectbox(
                "🏙️ City",
                sorted(df["City"].dropna().unique()),
                help="City where the car is being sold",
            )

        with col2:
            kilometers = st.number_input(
                "🛣️ Kilometers Driven",
                min_value=0, max_value=600000, value=45000, step=1000,
                help="Total kilometers driven by the car",
            )

            fuel = st.selectbox(
                "⛽ Fuel Type",
                sorted(df["Fuel_Type"].dropna().unique()),
                help="Type of fuel used by the car",
            )

            trans_opts = [
                v for v in sorted(df["Transmission"].dropna().unique())
                if v.lower() != "not available"
            ]
            transmission = st.selectbox(
                "⚙️ Transmission",
                trans_opts,
                help="Automatic or Manual transmission",
            )

        # Model selector
        st.divider()
        prediction_mode = st.radio(
            "🤖 Prediction Mode",
            ["Best Model Only", "Compare All Models"],
            horizontal=True,
            help="Choose to predict with the best model, or compare all trained models",
        )

        submitted = st.form_submit_button(
            "🔮 Predict Price", use_container_width=True, type="primary"
        )

    # ── Prediction ────────────────────────────────────────────
    if submitted:
        input_df = pd.DataFrame([{
            "Brand":            brand,
            "Year":             year,
            "City":             city,
            "Kilometers_Driven": kilometers,
            "Fuel_Type":        fuel,
            "Transmission":     transmission,
        }])

        try:
            st.divider()
            st.subheader("💰 Price Estimate")

            # Best model prediction
            best_pred = best_pipeline.predict(input_df[FEATURE_COLS])[0]

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric(
                f"🏆 {best_name}",
                f"₹ {best_pred:,.0f}",
                help="Prediction from the best-performing trained model",
            )
            col_m2.metric(
                "📊 Price Range (±15%)",
                f"₹ {best_pred * 0.85:,.0f} – ₹ {best_pred * 1.15:,.0f}",
            )
            col_m3.metric(
                "🏷️ Price in Lakhs",
                f"₹ {best_pred / 100000:.2f} L",
            )

            st.caption(
                "⚠️ This is an estimated value based on historical listing data. "
                "Actual market prices may vary based on condition, features, and negotiations."
            )

            # All-model comparison
            if prediction_mode == "Compare All Models" and all_pipelines:
                st.subheader("📊 Predictions from All Models")
                pred_rows = []
                for mname, pipe in all_pipelines.items():
                    try:
                        p = pipe.predict(input_df[FEATURE_COLS])[0]
                        pred_rows.append({
                            "Model": mname,
                            "Predicted Price (₹)": p,
                            "Price (Lakhs)": p / 100000,
                        })
                    except Exception:
                        pass

                pred_df = pd.DataFrame(pred_rows).sort_values(
                    "Predicted Price (₹)", ascending=False
                )

                st.dataframe(
                    pred_df.style.format({
                        "Predicted Price (₹)": "₹{:,.0f}",
                        "Price (Lakhs)": "{:.2f} L",
                    }).background_gradient(subset=["Predicted Price (₹)"], cmap="YlGn"),
                    use_container_width=True,
                    hide_index=True,
                )

                import plotly.express as px
                fig = px.bar(
                    pred_df, x="Model", y="Predicted Price (₹)",
                    title="Price Predictions by All Models",
                    color="Predicted Price (₹)",
                    color_continuous_scale="Viridis",
                    text=pred_df["Predicted Price (₹)"].apply(lambda x: f"₹{x:,.0f}"),
                )
                fig.update_layout(xaxis_tickangle=-30)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # ── Car age & KM guide ─────────────────────────────────────
    with st.expander("📖 Pricing Guidance"):
        st.markdown("""
        **Factors that affect used car prices:**
        | Factor | Impact |
        |--------|--------|
        | **Year** | Newer cars command higher prices |
        | **Kilometers Driven** | Lower KM = higher price |
        | **Brand** | Premium brands retain value better |
        | **Fuel Type** | Diesel cars tend to cost more |
        | **Transmission** | Automatics are priced higher |

        **Quick valuation guide (approximate):**
        - Well-maintained car (< 30,000 km): +15–20% premium
        - High mileage (> 150,000 km): −20–30% discount
        - First owner vs Third owner: −10–15% per ownership change
        """)
