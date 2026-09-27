"""
Customer Segmentation Module
Smart Business Intelligence & Sales Analytics Platform

Implements RFM (Recency, Frequency, Monetary) analysis and K-Means Clustering
to segment enterprise customers into actionable marketing cohorts.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def calculate_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Computes RFM metrics per Customer ID:
    - Recency: Days since last order
    - Frequency: Count of unique orders
    - Monetary: Total sales revenue
    - Profit: Total net profit
    """
    df_clean = df.copy()
    df_clean["Order Date"] = pd.to_datetime(df_clean["Order Date"])
    
    if snapshot_date is None:
        snapshot_date = df_clean["Order Date"].max() + pd.Timedelta(days=1)
        
    logging.info(f"Computing RFM metrics with snapshot date: {snapshot_date.date()}")
    
    rfm = df_clean.groupby("Customer ID").agg({
        "Order Date": lambda x: (snapshot_date - x.max()).days,
        "Order ID": "nunique",
        "Sales": "sum",
        "Profit": "sum",
        "Customer Name": "first",
        "Segment": "first"
    }).rename(columns={
        "Order Date": "Recency",
        "Order ID": "Frequency",
        "Sales": "Monetary"
    }).reset_index()
    
    rfm["Monetary"] = rfm["Monetary"].round(2)
    rfm["Profit"] = rfm["Profit"].round(2)
    return rfm


def train_kmeans_segmentation(
    rfm: pd.DataFrame,
    n_clusters: int = 4,
    random_state: int = 42
) -> tuple[pd.DataFrame, KMeans, StandardScaler]:
    """
    Applies Log transformation, StandardScaler, and fits K-Means clustering.
    Returns enriched dataframe, fitted KMeans model, and fitted scaler.
    """
    logging.info(f"Training K-Means with k={n_clusters}...")
    
    # 1. Log transform to handle right-skewness of financial data
    features = ["Recency", "Frequency", "Monetary"]
    rfm_log = np.log1p(rfm[features])
    
    # 2. StandardScaler to give equal weight to all 3 dimensions
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    
    # 3. Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    rfm_df = rfm.copy()
    rfm_df["Cluster"] = kmeans.fit_predict(rfm_scaled)
    
    # Evaluate silhouette
    score = silhouette_score(rfm_scaled, rfm_df["Cluster"])
    logging.info(f"K-Means training finished. Silhouette Score: {score:.4f}")
    
    # 4. Map Clusters to Human-Readable Business Personas
    # Identify clusters based on average characteristics
    cluster_means = rfm_df.groupby("Cluster")[features].mean()
    
    # Rank clusters by a combined value index (High Monetary + High Frequency - High Recency)
    # The cluster with highest Monetary is Champions
    champions_id = cluster_means["Monetary"].idxmax()
    # The cluster with highest Recency and lowest Monetary is Lost
    lost_id = cluster_means["Recency"].idxmax()
    
    remaining = [c for c in cluster_means.index if c not in [champions_id, lost_id]]
    # Among remaining, the one with lower recency is Active/Loyal
    if cluster_means.loc[remaining[0], "Recency"] < cluster_means.loc[remaining[1], "Recency"]:
        active_id = remaining[0]
        at_risk_id = remaining[1]
    else:
        active_id = remaining[1]
        at_risk_id = remaining[0]
        
    label_map = {
        champions_id: "Champions / High Value",
        active_id: "Active & Loyal",
        at_risk_id: "At Risk / Slipping",
        lost_id: "Lost / Dormant"
    }
    
    rfm_df["Segment_Name"] = rfm_df["Cluster"].map(label_map)
    return rfm_df, kmeans, scaler


def save_segmentation_artifacts(
    rfm_df: pd.DataFrame,
    kmeans: KMeans,
    scaler: StandardScaler,
    data_output: str | Path = "data/processed/customer_segments.csv",
    models_dir: str | Path = "models"
) -> None:
    """
    Saves segmented data CSV and serialized models.
    """
    # Save CSV
    out_csv = Path(data_output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rfm_df.to_csv(out_csv, index=False, encoding="utf-8")
    logging.info(f"Segmented customer data saved to: {out_csv}")
    
    # Save Models
    m_dir = Path(models_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(kmeans, m_dir / "kmeans_model.pkl")
    joblib.dump(scaler, m_dir / "kmeans_scaler.pkl")
    logging.info(f"K-Means model and Scaler saved in: {m_dir}")


def run_customer_segmentation_pipeline() -> pd.DataFrame:
    """
    Executes full customer segmentation pipeline.
    """
    data_path = Path("data/processed/cleaned_superstore.csv")
    df = pd.read_csv(data_path)
    rfm = calculate_rfm(df)
    rfm_segmented, kmeans, scaler = train_kmeans_segmentation(rfm, n_clusters=4)
    save_segmentation_artifacts(rfm_segmented, kmeans, scaler)
    return rfm_segmented


if __name__ == "__main__":
    run_customer_segmentation_pipeline()
