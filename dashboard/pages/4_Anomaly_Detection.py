"""
Anomaly Detection Page
Smart Business Intelligence Platform
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

st.set_page_config(page_title="Anomaly Detection - Smart BI", page_icon="🚨", layout="wide")


@st.cache_data
def load_anomalies_data():
    path = ROOT_DIR / "data" / "processed" / "anomalies_detected.csv"
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    return df


df = load_anomalies_data()
anomalies_df = df[df["Is_Anomaly"] == "Anomalie"]

st.title("🚨 Détection des Anomalies Commerciales (Isolation Forest)")
st.markdown("Surveillance algorithmique des transactions inhabituelles : pertes extrêmes, remises disproportionnées et commandes records.")

# Banner
st.info("ℹ️ **Précision Méthodologique :** Une anomalie statistique n'est pas synonyme de fraude. Il s'agit d'un événement qui s'écarte significativement de la norme, nécessitant soit une investigation pour risque financier, soit une valorisation commerciale.")

# Top KPIs
col1, col2, col3, col4 = st.columns(4)

nb_total_anom = len(anomalies_df)
nb_losses = len(df[df["Anomaly_Type"] == "Perte Critique (Marge Négative)"])
nb_discounts = len(df[df["Anomaly_Type"] == "Remise Excessive (>= 50%)"])
nb_blockbusters = len(df[df["Anomaly_Type"] == "Vente Exceptionnelle (Blockbuster)"])

col1.metric("Anomalies Détectées", f"{nb_total_anom}", delta="3.0% du volume")
col2.metric("Pertes Critiques", f"{nb_losses}", delta="-Marge sévère", delta_color="inverse")
col3.metric("Remises >= 50%", f"{nb_discounts}", delta="Risque marge", delta_color="inverse")
col4.metric("Ventes Blockbusters", f"{nb_blockbusters}", delta="+Gros CA")

st.markdown("---")

# Chart: Scatter Sales vs Profit
st.subheader("📊 Cartographie Algorithmique des Transactions")

fig = px.scatter(
    df,
    x="Sales",
    y="Profit",
    color="Anomaly_Type",
    hover_data=["Order ID", "Customer Name", "City", "Sub-Category", "Discount", "Profit Margin"],
    color_discrete_map={
        "Normal": "#9CA3AF",
        "Perte Critique (Marge N\u00e9gative)": "#DC2626",
        "Remise Excessive (>= 50%)": "#F59E0B",
        "Vente Exceptionnelle (Blockbuster)": "#10B981",
        "Profil Transactionnel Atypique": "#8B5CF6"
    },
    title="<b>Distribution des Transactions : Normales vs Types d'Anomalies</b>"
)
fig.add_hline(y=0, line_dash="dash", line_color="black", opacity=0.4)
fig.update_layout(template="plotly_white", height=500)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Data Table with Filters
st.subheader("📋 Audit des Transactions Signalées")

col_f1, col_f2 = st.columns(2)
with col_f1:
    type_options = ["Toutes les anomalies"] + sorted(anomalies_df["Anomaly_Type"].unique().tolist())
    selected_type = st.selectbox("Filtrer par type de signalement :", type_options)
with col_f2:
    state_options = ["Tous les États"] + sorted(anomalies_df["State"].unique().tolist())
    selected_state = st.selectbox("Filtrer par État géographique :", state_options)

table_df = anomalies_df.copy()
if selected_type != "Toutes les anomalies":
    table_df = table_df[table_df["Anomaly_Type"] == selected_type]
if selected_state != "Tous les États":
    table_df = table_df[table_df["State"] == selected_state]

cols_to_show = [
    "Order Date", "Order ID", "Customer Name", "City", "State",
    "Category", "Sub-Category", "Sales", "Profit", "Discount", "Profit Margin", "Anomaly_Type"
]

st.dataframe(
    table_df[cols_to_show].sort_values("Profit"),
    use_container_width=True,
    height=400
)

# Download CSV
csv = table_df[cols_to_show].to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Exporter la liste d'audit en CSV",
    data=csv,
    file_name="audit_anomalies_commerciales.csv",
    mime="text/csv"
)
