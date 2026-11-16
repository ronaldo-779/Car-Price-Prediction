
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def show_eda(df):
    st.header("🔍 Exploratory Data Analysis")
    st.caption("Explore patterns and relationships in the used-car dataset.")

    # ── Sidebar filters ───────────────────────────────────────
    with st.expander("🔧 Filter Data", expanded=True):
        col_f1, col_f2, col_f3 = st.columns(3)

        with col_f1:
            brands = ["All"] + sorted(df["Brand"].dropna().unique().tolist())
            sel_brand = st.selectbox("Brand", brands, key="eda_brand")

        with col_f2:
            fuels = ["All"] + sorted(df["Fuel_Type"].dropna().unique().tolist())
            sel_fuel = st.selectbox("Fuel Type", fuels, key="eda_fuel")

        with col_f3:
            yr_min = int(df["Year"].min())
            yr_max = int(df["Year"].max())
            yr_range = st.slider(
                "Year Range", yr_min, yr_max, (yr_min, yr_max), key="eda_year"
            )

    fdf = df.copy()
    if sel_brand != "All":
        fdf = fdf[fdf["Brand"] == sel_brand]
    if sel_fuel != "All":
        fdf = fdf[fdf["Fuel_Type"] == sel_fuel]
    fdf = fdf[(fdf["Year"] >= yr_range[0]) & (fdf["Year"] <= yr_range[1])]

    st.info(f"Showing **{len(fdf):,}** records after filtering.")

    # ── 1. Price distribution ─────────────────────────────────
    st.subheader("1. Selling Price Distribution")
    fig = px.histogram(
        fdf, x="Selling_Price", nbins=50,
        title="Selling Price Distribution",
        marginal="box",
        color_discrete_sequence=["#636EFA"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 2. Year vs Price ──────────────────────────────────────
    st.subheader("2. Manufacturing Year vs Selling Price")
    fig = px.scatter(
        fdf, x="Year", y="Selling_Price",
        color="Fuel_Type",
        hover_data=["Brand", "Model"],
        trendline="ols",
        title="Year vs Selling Price (with trend)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 3. KM vs Price ────────────────────────────────────────
    st.subheader("3. Kilometers Driven vs Selling Price")
    fig = px.scatter(
        fdf, x="Kilometers_Driven", y="Selling_Price",
        color="Transmission",
        hover_data=["Brand", "Model", "Year"],
        opacity=0.6,
        title="KM Driven vs Selling Price",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 4. Brand median price ─────────────────────────────────
    st.subheader("4. Median Price by Brand (Top 20)")
    brand_med = (
        fdf.groupby("Brand")["Selling_Price"]
        .median()
        .sort_values(ascending=False)
        .head(20)
        .reset_index()
    )
    brand_med.columns = ["Brand", "Median_Price"]
    fig = px.bar(
        brand_med, x="Median_Price", y="Brand",
        orientation="h",
        title="Median Selling Price by Brand",
        color="Median_Price",
        color_continuous_scale="Viridis",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

    # ── 5. Price by Fuel Type ─────────────────────────────────
    st.subheader("5. Selling Price by Fuel Type")
    fig = px.box(
        fdf, x="Fuel_Type", y="Selling_Price",
        color="Fuel_Type",
        title="Price Distribution by Fuel Type",
        points="outliers",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 6. Price by Transmission ──────────────────────────────
    st.subheader("6. Price by Transmission Type")
    fig = px.violin(
        fdf[fdf["Transmission"] != "Not available"],
        x="Transmission", y="Selling_Price",
        color="Transmission", box=True, points="outliers",
        title="Price Distribution by Transmission",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 7. City-wise average price ────────────────────────────
    st.subheader("7. City-wise Average Selling Price (Top 15)")
    city_avg = (
        fdf.groupby("City")["Selling_Price"]
        .mean()
        .sort_values(ascending=False)
        .head(15)
        .reset_index()
    )
    city_avg.columns = ["City", "Avg_Price"]
    fig = px.bar(
        city_avg, x="City", y="Avg_Price",
        title="Average Selling Price by City",
        color="Avg_Price",
        color_continuous_scale="Teal",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 8. Correlation heatmap ────────────────────────────────
    st.subheader("8. Numerical Feature Correlation")
    numeric = fdf[["Year", "Kilometers_Driven", "Selling_Price"]].dropna()
    fig = px.imshow(
        numeric.corr(),
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        title="Correlation Heatmap",
        zmin=-1, zmax=1,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 9. Source comparison ──────────────────────────────────
    st.subheader("9. Source-wise Price Comparison")
    fig = px.box(
        fdf, x="Source", y="Selling_Price",
        color="Source",
        title="Price by Data Source",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── 10. Descriptive stats ─────────────────────────────────
    st.subheader("10. Descriptive Statistics")
    st.dataframe(
        fdf[["Year", "Kilometers_Driven", "Selling_Price"]]
        .describe()
        .T.style.format("{:,.2f}"),
        use_container_width=True,
    )
