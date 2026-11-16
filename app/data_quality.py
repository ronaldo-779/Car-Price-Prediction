
import streamlit as st
import pandas as pd


def show_data_quality(df):

    st.header("Data Quality Analysis")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Total Columns", len(df.columns))
    col3.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )
    col4.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )

    st.divider()

    # Missing values
    st.subheader("1. Missing Values Analysis")

    missing = df.isnull().sum().reset_index()
    missing.columns = ["Column", "Missing Values"]

    st.dataframe(
        missing,
        use_container_width=True,
        hide_index=True
    )

    # Data types
    st.subheader("2. Column Information")

    info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Non-Null Count": df.notna().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        info,
        use_container_width=True,
        hide_index=True
    )

    # Source comparison
    st.subheader("3. Source-wise Dataset Summary")

    summary = df.groupby("Source").agg(
        Total_Records=("Selling_Price", "count"),
        Average_Price=("Selling_Price", "mean"),
        Median_Price=("Selling_Price", "median"),
        Average_Kilometers=("Kilometers_Driven", "mean")
    ).reset_index()

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True
    )

    # Category analysis
    st.subheader("4. Categorical Feature Analysis")

    category = st.selectbox(
        "Select a column",
        [
            "Brand",
            "Model",
            "City",
            "Fuel_Type",
            "Transmission",
            "Source"
        ]
    )

    counts = (
        df[category]
        .value_counts()
        .head(20)
        .reset_index()
    )

    counts.columns = [category, "Count"]

    st.dataframe(
        counts,
        use_container_width=True,
        hide_index=True
    )
