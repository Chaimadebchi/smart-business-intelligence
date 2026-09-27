"""
Smart Business Intelligence & Sales Analytics Platform
Main Entry Point & Overview Dashboard
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Page configuration
st.set_page_config(
    page_title="Smart BI & Sales Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern enterprise look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #111827;
        margin: 4px 0;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #059669;
        font-weight: 500;
    }
    .loss-text {
        color: #dc2626 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data_path = ROOT_DIR / "data" / "processed" / "cleaned_superstore.csv"
    df = pd.read_csv(data_path)
    df["Order Date"] = pd.to_datetime(df["Order Date"])
    df["Ship Date"] = pd.to_datetime(df["Ship Date"])
    return df


def main():
    df = load_data()

    # Sidebar
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=70)
    st.sidebar.title("Smart BI Platform")
    st.sidebar.markdown("**Auteur :** Chaima Debchi")
    st.sidebar.markdown("---")
    st.sidebar.info("💡 Utilisez le menu ci-dessus ou les pages latérales pour naviguer entre les modules d'analyses, de Machine Learning et de détection d'anomalies.")

    # Header
    st.markdown('<div class="main-header">📊 Smart Business Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Tableau de bord exécutif & indicateurs clés de performance commerciale (2015 - 2018)</div>', unsafe_allow_html=True)

    # 1. KPI Cards Row
    total_sales = df["Sales"].sum()
    total_profit = df["Profit"].sum()
    total_orders = df["Order ID"].nunique()
    total_customers = df["Customer ID"].nunique()
    avg_order_value = total_sales / total_orders
    profit_margin = (total_profit / total_sales) * 100

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Chiffre d'Affaires</div>
            <div class="kpi-value">${total_sales:,.0f}</div>
            <div class="kpi-sub">Total 4 ans</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Bénéfice Net</div>
            <div class="kpi-value">${total_profit:,.0f}</div>
            <div class="kpi-sub">Marge : {profit_margin:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Commandes</div>
            <div class="kpi-value">{total_orders:,}</div>
            <div class="kpi-sub">Transactions uniques</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Clients Actifs</div>
            <div class="kpi-value">{total_customers:,}</div>
            <div class="kpi-sub">B2B & Particuliers</div>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Panier Moyen (AOV)</div>
            <div class="kpi-value">${avg_order_value:.0f}</div>
            <div class="kpi-sub">Par commande</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Charts Row 1: Sales & Profit Evolution + Profit by Category
    row1_col1, row1_col2 = st.columns([7, 5])

    with row1_col1:
        st.subheader("📈 Évolution Mensuelle des Ventes et Bénéfices")
        monthly = df.groupby("Order YearMonth").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=monthly["Order YearMonth"], y=monthly["Sales"],
            name="Ventes ($)", mode="lines+markers", line=dict(color="#2563EB", width=3)
        ))
        fig_trend.add_trace(go.Scatter(
            x=monthly["Order YearMonth"], y=monthly["Profit"],
            name="Bénéfice ($)", mode="lines+markers", line=dict(color="#10B981", width=2, dash="dot")
        ))
        fig_trend.update_layout(
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=1.1, x=0.8)
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with row1_col2:
        st.subheader("📦 Ventes vs Profit par Catégorie")
        cat_summary = df.groupby("Category").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
        fig_cat = go.Figure(data=[
            go.Bar(name="Ventes ($)", x=cat_summary["Category"], y=cat_summary["Sales"], marker_color="#3B82F6"),
            go.Bar(name="Bénéfice ($)", x=cat_summary["Category"], y=cat_summary["Profit"], marker_color="#10B981")
        ])
        fig_cat.update_layout(
            barmode="group",
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", y=1.1, x=0.7)
        )
        st.plotly_chart(fig_cat, use_container_width=True)

    # 3. Charts Row 2: Revenue by Region + Top 10 Products by Revenue
    row2_col1, row2_col2 = st.columns([5, 7])

    with row2_col1:
        st.subheader("🗺️ Répartition du CA par Région")
        region_sales = df.groupby("Region")["Sales"].sum().reset_index()
        fig_region = px.pie(
            region_sales,
            values="Sales",
            names="Region",
            hole=0.45,
            color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#EF4444"]
        )
        fig_region.update_layout(margin=dict(l=20, r=20, t=30, b=20), template="plotly_white")
        st.plotly_chart(fig_region, use_container_width=True)

    with row2_col2:
        st.subheader("🏆 Top 10 Produits Générateurs de Revenus")
        top_products = df.groupby("Product Name").agg({"Sales": "sum", "Profit": "sum"}).reset_index().nlargest(10, "Sales")
        fig_top = px.bar(
            top_products,
            x="Sales",
            y="Product Name",
            orientation="h",
            color="Profit",
            color_continuous_scale="Viridis",
            labels={"Sales": "Ventes ($)", "Product Name": "Produit"},
        )
        fig_top.update_layout(
            yaxis=dict(autorange="reversed"),
            template="plotly_white",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_top, use_container_width=True)


if __name__ == "__main__":
    main()
