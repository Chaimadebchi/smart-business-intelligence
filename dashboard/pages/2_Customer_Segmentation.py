"""
Customer Segmentation Page
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

st.set_page_config(page_title="Customer Segmentation - Smart BI", page_icon="👥", layout="wide")


@st.cache_data
def load_rfm_data():
    path = ROOT_DIR / "data" / "processed" / "customer_segments.csv"
    return pd.read_csv(path)


rfm = load_rfm_data()

st.title("👥 Segmentation Client & Stratégie Marketing (RFM + K-Means)")
st.markdown("Identification automatique des personas clients à haute valeur ajoutée et des clients à risque d'attrition.")

# Segment distribution KPIs
col1, col2, col3, col4 = st.columns(4)

segments_info = [
    ("Champions / High Value", col1, "#10B981", "👑"),
    ("Active & Loyal", col2, "#3B82F6", "⭐"),
    ("At Risk / Slipping", col3, "#F59E0B", "⚠️"),
    ("Lost / Dormant", col4, "#EF4444", "💤")
]

for name, col, color, icon in segments_info:
    sub = rfm[rfm["Segment_Name"] == name]
    cnt = len(sub)
    pct = (cnt / len(rfm)) * 100
    rev = sub["Monetary"].sum()
    with col:
        st.markdown(f"""
        <div style="background-color: white; border-top: 4px solid {color}; border-radius: 8px; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
            <div style="font-size: 0.95rem; font-weight: 700; color: #1F2937;">{icon} {name}</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: {color}; margin: 4px 0;">{cnt} <span style="font-size: 0.9rem; font-weight: 500; color: #6B7280;">({pct:.1f}%)</span></div>
            <div style="font-size: 0.8rem; color: #4B5563;">CA Total : <b>${rev:,.0f}</b></div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Visualizations Row
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("📊 Récence vs Montant Dépensé (Log)")
    fig_scatter = px.scatter(
        rfm,
        x="Recency",
        y="Monetary",
        color="Segment_Name",
        size="Frequency",
        hover_name="Customer Name",
        hover_data={"Customer ID": True, "Frequency": True, "Monetary": ":$,.2f", "Recency": True},
        log_y=True,
        labels={"Recency": "Récence (Jours)", "Monetary": "Dépenses Totales ($)", "Segment_Name": "Segment"},
        color_discrete_map={
            "Champions / High Value": "#10B981",
            "Active & Loyal": "#3B82F6",
            "At Risk / Slipping": "#F59E0B",
            "Lost / Dormant": "#EF4444"
        }
    )
    fig_scatter.update_layout(template="plotly_white", height=450)
    st.plotly_chart(fig_scatter, use_container_width=True)

with col_right:
    st.subheader("🎯 Profil Moyen par Segment")
    summary = rfm.groupby("Segment_Name").agg({
        "Recency": "mean",
        "Frequency": "mean",
        "Monetary": "mean",
        "Profit": "mean"
    }).reset_index()
    
    st.dataframe(
        summary.style.format({
            "Recency": "{:.1f} j",
            "Frequency": "{:.1f} cmd",
            "Monetary": "${:,.0f}",
            "Profit": "${:,.0f}"
        }),
        use_container_width=True,
        height=220
    )
    
    st.markdown("""
    **💡 Plan d'Action Recommandé :**
    - **Champions :** Récompenser avec un accès VIP et contact commercial direct.
    - **Active & Loyal :** Proposer des offres personnalisées de cross-selling.
    - **At Risk :** Relancer d'urgence avec une réduction limitée à 15-20%.
    - **Lost :** Envoyer une enquête de reconquête ou limiter les dépenses d'acquisition.
    """)

st.markdown("---")
st.subheader("📋 Liste des Clients par Segment")

# Filter by segment
selected_seg = st.selectbox("Filtrer par Persona Client :", ["Tous"] + sorted(rfm["Segment_Name"].unique().tolist()))
search_query = st.text_input("Rechercher un client (Nom ou ID) :", "")

display_rfm = rfm.copy()
if selected_seg != "Tous":
    display_rfm = display_rfm[display_rfm["Segment_Name"] == selected_seg]
if search_query:
    display_rfm = display_rfm[
        display_rfm["Customer Name"].str.contains(search_query, case=False, na=False) |
        display_rfm["Customer ID"].str.contains(search_query, case=False, na=False)
    ]

st.dataframe(
    display_rfm[["Customer ID", "Customer Name", "Segment_Name", "Recency", "Frequency", "Monetary", "Profit"]].sort_values("Monetary", ascending=False),
    use_container_width=True,
    height=350
)
