"""
Anomaly Detection Module
Smart Business Intelligence & Sales Analytics Platform

Implements Isolation Forest unsupervised learning to detect statistically unusual
transactions (severe financial loss, excessive discount, or blockbuster orders).
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_anomaly_detector(
    df: pd.DataFrame,
    contamination: float = 0.03,
    random_state: int = 42
) -> tuple[pd.DataFrame, IsolationForest]:
    """
    Trains an Isolation Forest model on key financial features.
    Flags transactions as 'Normal' or 'Anomalie' and categorizes them by business risk.
    """
    logging.info(f"Training Isolation Forest with contamination={contamination}...")
    
    features = ["Sales", "Profit", "Quantity", "Discount", "Profit Margin"]
    X = df[features].copy()
    
    iso_model = IsolationForest(
        n_estimators=150,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    
    df_result = df.copy()
    raw_preds = iso_model.fit_predict(X)
    df_result["Anomaly_Flag"] = raw_preds
    df_result["Is_Anomaly"] = df_result["Anomaly_Flag"].map({1: "Normal", -1: "Anomalie"})
    df_result["Anomaly_Score"] = iso_model.decision_function(X).round(4)
    
    # Categorize anomaly type for business actionability
    def categorize_anomaly(row):
        if row["Is_Anomaly"] == "Normal":
            return "Normal"
        if row["Profit"] < -500:
            return "Perte Critique (Marge Négative)"
        elif row["Discount"] >= 0.5:
            return "Remise Excessive (>= 50%)"
        elif row["Sales"] > 3000 and row["Profit"] > 1000:
            return "Vente Exceptionnelle (Blockbuster)"
        else:
            return "Profil Transactionnel Atypique"
            
    df_result["Anomaly_Type"] = df_result.apply(categorize_anomaly, axis=1)
    
    anomaly_count = (df_result["Is_Anomaly"] == "Anomalie").sum()
    logging.info(f"Detected {anomaly_count} anomalies ({anomaly_count/len(df_result)*100:.1f}% of total).")
    
    return df_result, iso_model


def save_anomaly_artifacts(
    df_result: pd.DataFrame,
    model: IsolationForest,
    models_dir: str | Path = "models",
    output_csv: str | Path = "data/processed/anomalies_detected.csv"
) -> None:
    """
    Saves serialized Isolation Forest model and anomalies dataset.
    """
    m_dir = Path(models_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, m_dir / "isolation_forest_model.pkl")
    
    out_csv = Path(output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(out_csv, index=False, encoding="utf-8")
    logging.info(f"Anomaly detection model and dataset saved in {m_dir} and {out_csv}.")


def run_anomaly_detection_pipeline() -> pd.DataFrame:
    """
    Executes full anomaly detection pipeline.
    """
    data_path = Path("data/processed/cleaned_superstore.csv")
    df = pd.read_csv(data_path)
    df_result, model = train_anomaly_detector(df)
    save_anomaly_artifacts(df_result, model)
    return df_result


if __name__ == "__main__":
    run_anomaly_detection_pipeline()
