"""
Data Processing & Preprocessing Module
Smart Business Intelligence & Sales Analytics Platform

This module handles:
1. Loading raw dataset with proper encoding
2. Splitting and merging auxiliary tables (Orders and Returns)
3. Handling missing values with business justifications
4. Type conversions (Dates, Zip codes, numerics)
5. Feature engineering (Margins, Delivery times, Date decomposition)
6. Exporting clean dataset for EDA, Modeling, and Dashboard
"""

from pathlib import Path
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_raw_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load raw Superstore dataset handling ISO-8859-1 encoding.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found at: {path}")
    
    logging.info(f"Loading raw data from: {path}")
    df = pd.read_csv(path, encoding="ISO-8859-1")
    return df


def clean_superstore_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Executes end-to-end data cleaning and transformation pipeline.

    Key Decisions:
    - Slice the first 9,994 rows: Separates the main Orders table from appended
      auxiliary tables (People and Returns).
    - Map Returns: Extracts order IDs from the Returns section and creates a
      binary 'Returned' flag ('Yes'/'No').
    - Impute Postal Code: Fills missing zip codes for Burlington, VT with '05401'
      (lost leading zero in spreadsheet exports).
    - Feature Engineering:
        * Order Date features: Year, Month, Month Name, YearMonth, Day of Week.
        * Delivery Days: Ship Date minus Order Date.
        * Profit Margin (%): (Profit / Sales) * 100.
        * Unit Price: Sales / Quantity.
    """
    logging.info("Starting data cleaning pipeline...")

    # 1. Separate Orders table (first 9,994 rows)
    df_orders = df_raw.iloc[:9994].copy()

    # 2. Extract Returns table to flag returned orders
    df_returns = df_raw.iloc[10000:].dropna(subset=["Order ID"]).copy()
    returned_order_ids = set(df_returns["Order ID"].unique())
    df_orders["Returned"] = df_orders["Order ID"].isin(returned_order_ids).map({True: "Yes", False: "No"})
    logging.info(f"Identified {len(returned_order_ids)} returned orders affecting {df_orders['Returned'].value_counts().get('Yes', 0)} items.")

    # 3. Handle missing Postal Code (Burlington, Vermont = 05401)
    burlington_mask = df_orders["Postal Code"].isnull() & (df_orders["City"] == "Burlington")
    df_orders.loc[burlington_mask, "Postal Code"] = 5401
    df_orders["Postal Code"] = df_orders["Postal Code"].astype(int).astype(str).str.zfill(5)

    # 4. Standardize dates
    df_orders["Order Date"] = pd.to_datetime(df_orders["Order Date"], format="%m/%d/%Y")
    df_orders["Ship Date"] = pd.to_datetime(df_orders["Ship Date"], format="%m/%d/%Y")

    # 5. Feature Engineering
    df_orders["Order Year"] = df_orders["Order Date"].dt.year
    df_orders["Order Month"] = df_orders["Order Date"].dt.month
    df_orders["Order Month Name"] = df_orders["Order Date"].dt.month_name()
    df_orders["Order YearMonth"] = df_orders["Order Date"].dt.strftime("%Y-%m")
    df_orders["Order DayOfWeek"] = df_orders["Order Date"].dt.day_name()
    df_orders["Order Quarter"] = df_orders["Order Date"].dt.to_period("Q").astype(str)

    # Delivery performance
    df_orders["Delivery Days"] = (df_orders["Ship Date"] - df_orders["Order Date"]).dt.days

    # Financial KPIs
    df_orders["Profit Margin"] = ((df_orders["Profit"] / df_orders["Sales"]) * 100).round(2)
    df_orders["Unit Price"] = (df_orders["Sales"] / df_orders["Quantity"]).round(2)
    df_orders["Discount Amount"] = ((df_orders["Sales"] / (1 - df_orders["Discount"])) - df_orders["Sales"]).round(2)

    # Convert Row ID to integer
    if "Row ID" in df_orders.columns:
        df_orders["Row ID"] = df_orders["Row ID"].astype(int)

    logging.info(f"Cleaning complete. Output shape: {df_orders.shape}, Nulls remaining: {df_orders.isnull().sum().sum()}")
    return df_orders


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    """
    Saves processed dataframe to CSV.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    logging.info(f"Processed dataset saved successfully to: {path}")


def run_pipeline(
    raw_filepath: str | Path = "data/raw/sample_superstore.csv",
    output_filepath: str | Path = "data/processed/cleaned_superstore.csv"
) -> pd.DataFrame:
    """
    Runs the full ingestion and cleaning pipeline.
    """
    df_raw = load_raw_data(raw_filepath)
    df_clean = clean_superstore_data(df_raw)
    save_processed_data(df_clean, output_filepath)
    return df_clean


if __name__ == "__main__":
    run_pipeline()
