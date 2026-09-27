"""
Sales Prediction & Forecasting Module
Smart Business Intelligence & Sales Analytics Platform

Implements time series feature engineering and machine learning regression
(Linear Regression, Random Forest, Gradient Boosting) to forecast future sales.
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def prepare_monthly_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates transactions into monthly sales time series and engineers features:
    - Temporal features: Month, Quarter
    - Lags: Lag 1, Lag 2, Lag 12 (same month previous year)
    - Rolling window: Rolling mean of past 3 months
    """
    df_clean = df.copy()
    df_clean["Order Date"] = pd.to_datetime(df_clean["Order Date"])
    
    # Resample to month start
    df_monthly = df_clean.set_index("Order Date").resample("MS")["Sales"].sum().reset_index()
    df_monthly.columns = ["Date", "Sales"]
    
    # Features
    df_monthly["Year"] = df_monthly["Date"].dt.year
    df_monthly["Month"] = df_monthly["Date"].dt.month
    df_monthly["Quarter"] = df_monthly["Date"].dt.quarter
    
    # Lags (avoiding future data leakage by shifting)
    df_monthly["Lag_1"] = df_monthly["Sales"].shift(1)
    df_monthly["Lag_2"] = df_monthly["Sales"].shift(2)
    df_monthly["Lag_12"] = df_monthly["Sales"].shift(12)  # Annual seasonality
    df_monthly["Rolling_Mean_3"] = df_monthly["Sales"].shift(1).rolling(3).mean()
    
    return df_monthly


def evaluate_sales_models(
    df_monthly: pd.DataFrame,
    test_year: int = 2018
) -> tuple[pd.DataFrame, dict, pd.DataFrame, str]:
    """
    Performs temporal train/test split (no data leakage).
    Trains Linear Regression, Random Forest, and Gradient Boosting.
    Returns comparison dataframe, fitted models dictionary, test predictions df, and best model name.
    """
    df_clean = df_monthly.dropna().copy()
    
    train = df_clean[df_clean["Year"] < test_year]
    test = df_clean[df_clean["Year"] == test_year].copy()
    
    feature_cols = ["Month", "Quarter", "Lag_1", "Lag_2", "Lag_12", "Rolling_Mean_3"]
    X_train, y_train = train[feature_cols], train["Sales"]
    X_test, y_test = test[feature_cols], test["Sales"]
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=3, learning_rate=0.05)
    }
    
    comparison = []
    fitted_models = {}
    preds_dict = {"Date": test["Date"].values, "Actual": y_test.values}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        comparison.append({
            "Model": name,
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        })
        fitted_models[name] = model
        preds_dict[name] = preds
        logging.info(f"{name} -> MAE: ${mae:,.2f} | RMSE: ${rmse:,.2f} | R²: {r2:.4f}")
        
    comp_df = pd.DataFrame(comparison).sort_values("R2", ascending=False)
    best_model_name = comp_df.iloc[0]["Model"]
    test_preds_df = pd.DataFrame(preds_dict)
    
    return comp_df, fitted_models, test_preds_df, best_model_name


def save_prediction_artifacts(
    best_model,
    best_model_name: str,
    test_preds_df: pd.DataFrame,
    df_monthly: pd.DataFrame,
    models_dir: str | Path = "models",
    output_csv: str | Path = "data/processed/sales_forecast.csv"
) -> None:
    """
    Saves the best forecasting model and forecast dataset.
    """
    m_dir = Path(models_dir)
    m_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, m_dir / "best_sales_model.pkl")
    
    # Save forecast dataset
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Merge history with test predictions
    forecast_df = df_monthly[["Date", "Sales"]].copy().rename(columns={"Sales": "Actual_Sales"})
    forecast_df = forecast_df.merge(
        test_preds_df.rename(columns={best_model_name: "Predicted_Sales"})[["Date", "Predicted_Sales"]],
        on="Date",
        how="left"
    )
    forecast_df.to_csv(out_path, index=False)
    logging.info(f"Forecast model ({best_model_name}) and dataset saved successfully.")


def run_sales_prediction_pipeline() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runs end-to-end sales forecasting pipeline.
    """
    data_path = Path("data/processed/cleaned_superstore.csv")
    df = pd.read_csv(data_path)
    df_monthly = prepare_monthly_sales_data(df)
    comp_df, fitted_models, test_preds_df, best_model_name = evaluate_sales_models(df_monthly)
    save_prediction_artifacts(fitted_models[best_model_name], best_model_name, test_preds_df, df_monthly)
    return comp_df, test_preds_df


if __name__ == "__main__":
    run_sales_prediction_pipeline()
