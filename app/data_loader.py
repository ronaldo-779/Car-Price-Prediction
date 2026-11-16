
import pandas as pd
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "processed_data"
    / "combined_used_cars_2000.csv"
)

TRAIN_PATH = PROJECT_ROOT / "original_data" / "Car_details_train.csv"

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )
    df = pd.read_csv(DATA_PATH)
    return df


@st.cache_data
def load_train_data():
    """Load the original training CSV for extended ML experiments."""
    if not TRAIN_PATH.exists():
        return None
    df = pd.read_csv(TRAIN_PATH)
    # Rename columns to match our schema
    df = df.rename(columns={
        "name": "Model",
        "year": "Year",
        "selling_price": "Selling_Price",
        "km_driven": "Kilometers_Driven",
        "fuel": "Fuel_Type",
        "transmission": "Transmission",
        "seller_type": "Seller_Type",
        "owner": "Owner",
        "mileage": "Mileage",
        "engine_cc": "Engine_CC",
        "max_power_bhp": "Max_Power_BHP",
        "seats": "Seats",
    })
    # Extract brand from Model name (first word)
    df["Brand"] = df["Model"].str.split().str[0]
    df = df.dropna(subset=["Selling_Price", "Year", "Kilometers_Driven"])
    df = df[df["Selling_Price"] > 0]
    return df
