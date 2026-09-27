"""
Visualization Module
Smart Business Intelligence & Sales Analytics Platform

Provides reusable, professional visualization functions using:
- Plotly (interactive charts for notebooks and Streamlit)
- Matplotlib / Seaborn (static publication-quality plots)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns


# Color palette constants
PRIMARY_COLOR = "#1f77b4"
SECONDARY_COLOR = "#ff7f0e"
SUCCESS_COLOR = "#2ca02c"
DANGER_COLOR = "#d62728"
PALETTE = ["#2b5c8f", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]


def plot_sales_profit_by_category(df: pd.DataFrame) -> go.Figure:
    """
    Creates a grouped bar chart comparing Sales and Profit by Category.
    Reveals profitability disparities across categories.
    """
    cat_summary = df.groupby("Category").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
    
    fig = go.Figure(data=[
        go.Bar(name="Ventes ($)", x=cat_summary["Category"], y=cat_summary["Sales"], marker_color="#2b5c8f"),
        go.Bar(name="Bénéfice ($)", x=cat_summary["Category"], y=cat_summary["Profit"], marker_color="#2ca02c")
    ])
    fig.update_layout(
        title="<b>Ventes vs Bénéfice par Catégorie</b>",
        xaxis_title="Catégorie",
        yaxis_title="Montant ($)",
        barmode="group",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def plot_subcategories_profitability(df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart of Sub-Category profits sorted from worst to best.
    Highlights loss-making sub-categories in red.
    """
    sub_summary = df.groupby("Sub-Category")["Profit"].sum().reset_index().sort_values("Profit")
    colors = [DANGER_COLOR if p < 0 else SUCCESS_COLOR for p in sub_summary["Profit"]]
    
    fig = go.Figure(go.Bar(
        x=sub_summary["Profit"],
        y=sub_summary["Sub-Category"],
        orientation="h",
        marker_color=colors,
        text=sub_summary["Profit"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside"
    ))
    fig.update_layout(
        title="<b>Rentabilité par Sous-Catégorie (Bénéfice net en $)</b>",
        xaxis_title="Bénéfice net ($)",
        yaxis_title="Sous-Catégorie",
        template="plotly_white",
        height=600
    )
    return fig


def plot_sales_trend(df: pd.DataFrame) -> go.Figure:
    """
    Monthly time series of Sales and Profit.
    """
    monthly = df.groupby("Order YearMonth").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["Order YearMonth"],
        y=monthly["Sales"],
        mode="lines+markers",
        name="Ventes ($)",
        line=dict(color="#1f77b4", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=monthly["Order YearMonth"],
        y=monthly["Profit"],
        mode="lines+markers",
        name="Bénéfice ($)",
        line=dict(color="#2ca02c", width=2, dash="dot")
    ))
    fig.update_layout(
        title="<b>Évolution Mensuelle des Ventes et du Bénéfice (2015 - 2018)</b>",
        xaxis_title="Mois",
        yaxis_title="Montant ($)",
        template="plotly_white",
        hovermode="x unified"
    )
    return fig


def plot_profit_by_state_map(df: pd.DataFrame) -> go.Figure:
    """
    US Choropleth map showing Profit by State.
    """
    # Mapping state names to 2-letter codes for US choropleth
    state_codes = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA',
        'Kansas': 'KS', 'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
        'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
        'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH',
        'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY', 'North Carolina': 'NC',
        'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK', 'Oregon': 'OR', 'Pennsylvania': 'PA',
        'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD', 'Tennessee': 'TN',
        'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
        'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
    }
    
    state_summary = df.groupby("State").agg({"Sales": "sum", "Profit": "sum"}).reset_index()
    state_summary["Code"] = state_summary["State"].map(state_codes)
    
    fig = px.choropleth(
        state_summary,
        locations="Code",
        locationmode="USA-states",
        color="Profit",
        scope="usa",
        color_continuous_scale="RdYlGn",
        hover_name="State",
        hover_data={"Sales": ":$,.2f", "Profit": ":$,.2f", "Code": False},
        title="<b>Distribution Géographique du Bénéfice par État (USA)</b>"
    )
    fig.update_layout(template="plotly_white")
    return fig


def plot_discount_vs_profit_impact(df: pd.DataFrame) -> go.Figure:
    """
    Shows how discount levels directly impact average profit margins.
    """
    df_temp = df.copy()
    df_temp["Discount Bracket"] = pd.cut(
        df_temp["Discount"],
        bins=[-0.01, 0.0, 0.1, 0.2, 0.3, 0.5, 0.85],
        labels=["0%", "1-10%", "11-20%", "21-30%", "31-50%", "> 50%"]
    )
    bracket_perf = df_temp.groupby("Discount Bracket", observed=False).agg(
        {"Sales": "sum", "Profit": "sum", "Profit Margin": "mean"}
    ).reset_index()
    
    fig = go.Figure(go.Bar(
        x=bracket_perf["Discount Bracket"],
        y=bracket_perf["Profit"],
        marker_color=[DANGER_COLOR if p < 0 else SUCCESS_COLOR for p in bracket_perf["Profit"]],
        text=bracket_perf["Profit"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside"
    ))
    fig.update_layout(
        title="<b>Impact du Niveau de Remise sur le Bénéfice Total ($)</b>",
        xaxis_title="Tranche de Remise",
        yaxis_title="Bénéfice Net ($)",
        template="plotly_white"
    )
    return fig
