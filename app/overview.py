
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def show_overview(df):
    st.header("📊 Dataset Overview")
    st.caption("High-level summary of the used car dataset.")

    # ── KPI row ──────────────────────────────────────────────
    total_records  = len(df)
    total_brands   = df["Brand"].nunique()
    total_cities   = df["City"].nunique()
    median_price   = df["Selling_Price"].median()
    avg_price      = df["Selling_Price"].mean()
    avg_km         = df["Kilometers_Driven"].mean()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("🚗 Total Cars",    f"{total_records:,}")
    c2.metric("🏷️ Brands",        f"{total_brands}")
    c3.metric("🏙️ Cities",        f"{total_cities}")
    c4.metric("💰 Median Price",  f"₹{median_price:,.0f}")
    c5.metric("📊 Avg Price",     f"₹{avg_price:,.0f}")
    c6.metric("🛣️ Avg KM",        f"{avg_km:,.0f}")

    st.divider()

    # ── Row 1 – Pie + Bar ─────────────────────────────────────
    left, right = st.columns(2)

    with left:
        source_counts = df["Source"].value_counts()
        fig = px.pie(
            names=source_counts.index,
            values=source_counts.values,
            title="Records by Data Source",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        brand_counts = (
            df["Brand"].value_counts().head(12).reset_index()
        )
        brand_counts.columns = ["Brand", "Count"]
        fig = px.bar(
            brand_counts, x="Brand", y="Count",
            title="Top 12 Car Brands",
            color="Count",
            color_continuous_scale="Blues",
        )
        fig.update_layout(showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2 – Fuel + Transmission ──────────────────────────
    left2, right2 = st.columns(2)

    with left2:
        fuel_counts = df["Fuel_Type"].value_counts().reset_index()
        fuel_counts.columns = ["Fuel_Type", "Count"]
        fig = px.pie(
            fuel_counts, names="Fuel_Type", values="Count",
            title="Distribution by Fuel Type",
            color_discrete_sequence=px.colors.qualitative.Pastel,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right2:
        trans = df["Transmission"].value_counts().reset_index()
        trans.columns = ["Transmission", "Count"]
        fig = px.bar(
            trans, x="Transmission", y="Count",
            title="Transmission Type Distribution",
            color="Transmission",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Year histogram ────────────────────────────────────────
    st.subheader("Manufacturing Year Distribution")
    fig = px.histogram(
        df, x="Year", nbins=30,
        title="Cars by Manufacturing Year",
        color_discrete_sequence=["#636EFA"],
        marginal="rug",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Dataset preview ───────────────────────────────────────
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)
