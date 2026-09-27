"""
Sales Analysis Page
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

st.set_page_config(page_title="Sales Analysis - Smart BI", page_icon="🛍️", layout="wide")


@st.cache_data
def load_data():
    data_path = ROOT_DIR / "data" / "processed" / "cleaned_superstore.csv"
    df = pd.read_csv(data_path)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    return df


df = load_data()

st.title("🛍️ Analyse Approfondie des Ventes & Rentabilité")
st.markdown("Explorez les transactions avec des filtres multicritères dynamiques.")

# Sidebar Filters
st.sidebar.header("🔍 Filtres Interactifs")

# Date Filter
min_date = df["Order Date"].min().date()
max_date = df["Order Date"].max().date()
date_range = st.sidebar.date_input("Période de commande", [min_date, max_date], min_value=min_date, max_value=max_date)

# Region Filter
regions = ["Toutes"] + sorted(df["Region"].unique().tolist())
selected_region = st.sidebar.selectbox("Région", regions)

# Category Filter
categories = ["Toutes"] + sorted(df["Category"].unique().tolist())
selected_cat = st.sidebar.selectbox("Catégorie", categories)

# Sub-Category Filter
subcats_available = df["Sub-Category"].unique().tolist()
if selected_cat != "Toutes":
    subcats_available = df[df["Category"] == selected_cat]["Sub-Category"].unique().tolist()
subcats = ["Toutes"] + sorted(subcats_available)
selected_subcat = st.sidebar.selectbox("Sous-Catégorie", subcats)

# Apply Filters
filtered_df = df.copy()
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["Order Date"].dt.date >= date_range[0]) &
        (filtered_df["Order Date"].dt.date <= date_range[1])
    ]
if selected_region != "Toutes":
    filtered_df = filtered_df[filtered_df["Region"] == selected_region]
if selected_cat != "Toutes":
    filtered_df = filtered_df[filtered_df["Category"] == selected_cat]
if selected_subcat != "Toutes":
    filtered_df = filtered_df[filtered_df["Sub-Category"] == selected_subcat]

# KPIs for filtered selection
f_sales = filtered_df["Sales"].sum()
f_profit = filtered_df["Profit"].sum()
f_margin = (f_profit / f_sales * 100) if f_sales > 0 else 0
f_orders = filtered_df["Order ID"].nunique()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Ventes Filtrées", f"${f_sales:,.0f}")
col2.metric("Bénéfice Net", f"${f_profit:,.0f}", delta=f"{f_margin:.1f}% marge")
col3.metric("Commandes", f"{f_orders:,}")
col4.metric("Articles Vendus", f"{filtered_df['Quantity'].sum():,.0f}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Rentabilité par Sous-Catégorie", "🗺️ Analyse par État", "📋 Données Détaillées"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        sub_profit = filtered_df.groupby("Sub-Category")["Profit"].sum().reset_index().sort_values("Profit")
        colors = ["#EF4444" if p < 0 else "#10B981" for p in sub_profit["Profit"]]
        fig_sub = go.Figure(go.Bar(
            x=sub_profit["Profit"],
            y=sub_profit["Sub-Category"],
            orientation="h",
            marker_color=colors,
            text=sub_profit["Profit"].apply(lambda v: f"${v:,.0f}"),
            textposition="outside"
        ))
        fig_sub.update_layout(title="<b>Bénéfice par Sous-Catégorie ($)</b>", template="plotly_white", height=500)
        st.plotly_chart(fig_sub, use_container_width=True)

    with col_b:
        fig_scatter = px.scatter(
            filtered_df,
            x="Sales",
            y="Profit",
            color="Category",
            size="Quantity",
            hover_data=["Product Name", "Customer Name", "Discount"],
            title="<b>Distribution Ventes vs Bénéfice par Transaction</b>",
            color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B"]
        )
        fig_scatter.add_hline(y=0, line_dash="dash", line_color="red")
        fig_scatter.update_layout(template="plotly_white", height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

with tab2:
    state_perf = filtered_df.groupby("State").agg({"Sales": "sum", "Profit": "sum"}).reset_index().sort_values("Sales", ascending=False)
    st.dataframe(
        state_perf.style.format({"Sales": "${:,.2f}", "Profit": "${:,.2f}"}),
        use_container_width=True,
        height=400
    )

with tab3:
    display_cols = ["Order Date", "Order ID", "Customer Name", "Segment", "City", "State", "Category", "Sub-Category", "Sales", "Quantity", "Discount", "Profit"]
    st.dataframe(
        filtered_df[display_cols].sort_values("Order Date", ascending=False),
        use_container_width=True,
        height=450
    )
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger la sélection en CSV",
        data=csv,
        file_name="ventes_filtrees.csv",
        mime="text/csv"
    )
