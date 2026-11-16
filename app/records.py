
import streamlit as st
import plotly.express as px
import pandas as pd


def show_records(df):
    st.header("📋 All Records")
    st.caption("Search, filter, and explore every record in the dataset.")

    # ── Filters ───────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            brands = ["All"] + sorted(df["Brand"].dropna().unique().tolist())
            sel_brand = st.selectbox("Brand", brands, key="rec_brand")

        with col2:
            fuels = ["All"] + sorted(df["Fuel_Type"].dropna().unique().tolist())
            sel_fuel = st.selectbox("Fuel Type", fuels, key="rec_fuel")

        with col3:
            cities = ["All"] + sorted(df["City"].dropna().unique().tolist())
            sel_city = st.selectbox("City", cities, key="rec_city")

        col4, col5 = st.columns(2)
        with col4:
            yr_min = int(df["Year"].min())
            yr_max = int(df["Year"].max())
            yr_range = st.slider(
                "Year Range", yr_min, yr_max, (yr_min, yr_max), key="rec_year"
            )
        with col5:
            pr_min = int(df["Selling_Price"].min())
            pr_max = int(df["Selling_Price"].max())
            pr_range = st.slider(
                "Price Range (₹)",
                pr_min, pr_max, (pr_min, pr_max),
                step=10000, key="rec_price"
            )

        search = st.text_input(
            "🔍 Search (Brand / Model / City)", placeholder="e.g. Maruti, Delhi…"
        )

    # ── Apply filters ─────────────────────────────────────────
    fdf = df.copy()
    if sel_brand != "All":
        fdf = fdf[fdf["Brand"] == sel_brand]
    if sel_fuel != "All":
        fdf = fdf[fdf["Fuel_Type"] == sel_fuel]
    if sel_city != "All":
        fdf = fdf[fdf["City"] == sel_city]
    fdf = fdf[(fdf["Year"] >= yr_range[0]) & (fdf["Year"] <= yr_range[1])]
    fdf = fdf[(fdf["Selling_Price"] >= pr_range[0]) & (fdf["Selling_Price"] <= pr_range[1])]

    if search:
        mask = (
            fdf["Brand"].str.contains(search, case=False, na=False) |
            fdf["Model"].str.contains(search, case=False, na=False) |
            fdf["City"].str.contains(search, case=False, na=False)
        )
        fdf = fdf[mask]

    st.info(f"**{len(fdf):,}** records match your filters.")

    # ── Sort ──────────────────────────────────────────────────
    sort_col = st.selectbox(
        "Sort by",
        ["Selling_Price", "Year", "Kilometers_Driven", "Brand", "City"],
        key="rec_sort"
    )
    sort_asc = st.checkbox("Ascending", value=False, key="rec_asc")

    display_df = fdf.sort_values(sort_col, ascending=sort_asc)

    st.dataframe(
        display_df[[
            "Record_ID", "Brand", "Model", "Year", "City",
            "Kilometers_Driven", "Fuel_Type", "Transmission",
            "Selling_Price", "Source"
        ]],
        use_container_width=True,
    )

    # ── Download ──────────────────────────────────────────────
    csv = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Filtered Data as CSV",
        csv,
        "filtered_cars.csv",
        "text/csv",
        use_container_width=True,
    )
