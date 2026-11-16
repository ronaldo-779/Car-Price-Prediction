
import streamlit as st
import plotly.express as px
import pandas as pd


def show_outliers(df):
    st.header("🔎 Outlier Review")
    st.caption(
        "Identify and inspect outlier records in the dataset "
        "using the IQR method on Selling Price."
    )

    # ── IQR method ────────────────────────────────────────────
    Q1  = df["Selling_Price"].quantile(0.25)
    Q3  = df["Selling_Price"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers     = df[(df["Selling_Price"] < lower) | (df["Selling_Price"] > upper)]
    non_outliers = df[(df["Selling_Price"] >= lower) & (df["Selling_Price"] <= upper)]

    # ── KPI row ───────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",   f"{len(df):,}")
    c2.metric("Outliers",        f"{len(outliers):,}")
    c3.metric("Non-Outliers",    f"{len(non_outliers):,}")
    c4.metric("Outlier %",       f"{len(outliers)/len(df)*100:.1f}%")

    st.info(
        f"**IQR Bounds:**  Lower = ₹{lower:,.0f}  |  Upper = ₹{upper:,.0f}"
    )

    # ── Box plot ──────────────────────────────────────────────
    st.subheader("1. Price Box Plot")
    fig = px.box(
        df, y="Selling_Price",
        title="Selling Price Distribution with Outliers",
        points="outliers",
        color_discrete_sequence=["#3498db"],
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Scatter: flag outliers ────────────────────────────────
    st.subheader("2. Price Scatter (Outliers Highlighted)")
    plot_df = df.copy()
    plot_df["Category"] = "Normal"
    plot_df.loc[
        (plot_df["Selling_Price"] < lower) | (plot_df["Selling_Price"] > upper),
        "Category",
    ] = "Outlier"

    fig2 = px.scatter(
        plot_df,
        x="Year", y="Selling_Price",
        color="Category",
        hover_data=["Brand", "Model", "City"],
        color_discrete_map={"Normal": "#3498db", "Outlier": "#e74c3c"},
        title="Outliers vs Normal Records (Year vs Price)",
    )
    fig2.add_hline(y=lower, line_dash="dash", line_color="orange", annotation_text="Lower Bound")
    fig2.add_hline(y=upper, line_dash="dash", line_color="orange", annotation_text="Upper Bound")
    st.plotly_chart(fig2, use_container_width=True)

    # ── Outlier brand distribution ────────────────────────────
    if len(outliers) > 0:
        st.subheader("3. Outlier Brand Distribution")
        brand_out = outliers["Brand"].value_counts().head(15).reset_index()
        brand_out.columns = ["Brand", "Count"]
        fig3 = px.bar(
            brand_out, x="Brand", y="Count",
            title="Most Common Brands Among Outliers",
            color="Count",
            color_continuous_scale="Reds",
        )
        st.plotly_chart(fig3, use_container_width=True)

    # ── Outlier table ─────────────────────────────────────────
    st.subheader("4. Outlier Records Table")
    st.write(f"Showing **{len(outliers)}** outlier records:")

    sort_col = st.selectbox(
        "Sort by", ["Selling_Price", "Year", "Kilometers_Driven", "Brand"],
        key="outlier_sort"
    )
    sort_asc = st.checkbox("Ascending order", value=False, key="outlier_asc")

    display_outliers = outliers.sort_values(sort_col, ascending=sort_asc)

    st.dataframe(
        display_outliers[[
            "Brand", "Model", "Year", "City",
            "Kilometers_Driven", "Fuel_Type", "Transmission", "Selling_Price", "Source"
        ]],
        use_container_width=True,
    )
