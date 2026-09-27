"""
Sales Prediction Page
Smart Business Intelligence Platform
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

st.set_page_config(page_title="Sales Prediction - Smart BI", page_icon="📈", layout="wide")


@st.cache_data
def load_forecast_data():
    path = ROOT_DIR / "data" / "processed" / "sales_forecast.csv"
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


forecast_df = load_forecast_data()

st.title("📈 Prédiction des Ventes & Modélisation Prédictive")
st.markdown("Prévision du chiffre d'affaires mensuel grâce à un modèle de **Gradient Boosting** entraîné sur l'historique chronologique.")

# Model Performance Cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Modèle Retenu", "Gradient Boosting", help="Meilleur score R² comparé à Linear Regression et Random Forest")
with col2:
    st.metric("Score R² (Variance Expliquée)", "71.2 %", delta="+6.2% vs baseline")
with col3:
    st.metric("Erreur Moyenne (MAE)", "$12,056", delta="-1,800$ vs Random Forest")
with col4:
    st.metric("RMSE", "$13,837")

st.markdown("---")

# Historical vs Predicted Chart
st.subheader("📊 Comparaison : Ventes Historiques vs Prédictions Modèle (2018)")

fig = go.Figure()

# Actual History
fig.add_trace(go.Scatter(
    x=forecast_df["Date"],
    y=forecast_df["Actual_Sales"],
    name="Ventes Réelles (Historique)",
    mode="lines+markers",
    line=dict(color="#1E3A8A", width=2.5)
))

# Predictions in 2018
preds_subset = forecast_df.dropna(subset=["Predicted_Sales"])
fig.add_trace(go.Scatter(
    x=preds_subset["Date"],
    y=preds_subset["Predicted_Sales"],
    name="Prédictions du Modèle (Validation 2018)",
    mode="lines+markers",
    line=dict(color="#10B981", width=3, dash="dash")
))

fig.update_layout(
    template="plotly_white",
    xaxis_title="Date (Mois)",
    yaxis_title="Ventes Mensuelles ($)",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.08, x=0.6)
)

st.plotly_chart(fig, use_container_width=True)

# Forecast Simulator for future business planning
st.markdown("---")
st.subheader("🔮 Simulateur Prévisionnel pour les Décideurs")
st.markdown("Projetez les ventes attendues selon différents scénarios de croissance de marché.")

col_sim1, col_sim2 = st.columns([4, 8])

with col_sim1:
    growth_rate = st.slider("Hypothèse de croissance annuelle (%)", min_value=-20, max_value=30, value=10, step=1)
    q4_boost = st.checkbox("Intégrer le pic saisonnier de fin d'année (Q4)", value=True)
    
    # Calculate baseline projection for 2019
    sales_2018 = forecast_df[forecast_df["Date"].dt.year == 2018]["Actual_Sales"].sum()
    projected_2019 = sales_2018 * (1 + growth_rate / 100)
    
    st.metric("Chiffre d'Affaires 2018 Réel", f"${sales_2018:,.0f}")
    st.metric("Chiffre d'Affaires 2019 Projeté", f"${projected_2019:,.0f}", delta=f"{growth_rate:+d}%")

with col_sim2:
    months_2019 = pd.date_range(start="2019-01-01", periods=12, freq="MS")
    # Base monthly seasonality pattern based on 2018
    monthly_pattern = forecast_df[forecast_df["Date"].dt.year == 2018]["Actual_Sales"].values / sales_2018
    if q4_boost:
        # slightly reinforce Nov & Dec
        monthly_pattern[10] *= 1.05
        monthly_pattern[11] *= 1.05
        monthly_pattern = monthly_pattern / monthly_pattern.sum()
        
    sim_sales = projected_2019 * monthly_pattern
    
    fig_sim = go.Figure()
    fig_sim.add_bar(
        x=[d.strftime("%b") for d in months_2019],
        y=sim_sales,
        marker_color="#3B82F6",
        text=[f"${v:,.0f}" for v in sim_sales],
        textposition="outside"
    )
    fig_sim.update_layout(
        title=f"<b>Projection Mensuelle 2019 (Scénario : {growth_rate:+d}%)</b>",
        template="plotly_white",
        yaxis_title="Ventes Estimées ($)"
    )
    st.plotly_chart(fig_sim, use_container_width=True)
