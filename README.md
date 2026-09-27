# 📊 Smart Business Intelligence & Sales Analytics Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-F7931E.svg)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18%2B-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A full end-to-end Business Intelligence and Predictive Analytics platform built in Python, designed to help businesses analyze sales performance, segment their customer base, forecast future revenue, and automatically detect financial anomalies.

**Author:** [Chaima Debchi](https://github.com/Chaimadebchi)  
**Repository:** [smart-business-intelligence](https://github.com/Chaimadebchi/smart-business-intelligence)

---

## 📌 Table of Contents
- [Business Problem & Objectives](#-business-problem--objectives)
- [Features](#-features)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Machine Learning](#-machine-learning)
- [Verified Results](#-verified-results)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🎯 Business Problem & Objectives

In B2B/B2C commerce and distribution, businesses face four recurring challenges:

1. **Hidden profitability gaps** — high revenue can mask deeply unprofitable products and regions, often caused by uncontrolled discount policies.
2. **Uniform customer treatment** — treating a high-value loyal customer the same way as a dormant one destroys value.
3. **Unpredictable cash flows** — inability to anticipate seasonal peaks (e.g., Q4 / fiscal year-end).
4. **Invisible financial leaks** — anomalous orders with destructive margins that escape standard financial controllers.

### This platform addresses all four by providing:

| Module | Objective |
|:---|:---|
| **Data Cleaning & EDA** | Ingest, clean, and enrich raw transactional data with business-relevant features |
| **Sales Analytics** | Interactive KPI dashboards with geographic and temporal breakdowns |
| **Customer Segmentation (RFM + K-Means)** | Automatically identify 4 customer personas for targeted marketing |
| **Sales Forecasting (Gradient Boosting)** | Predict future monthly revenue without data leakage |
| **Anomaly Detection (Isolation Forest)** | Surface financially unusual transactions for audit and investigation |
| **Interactive Dashboard (Streamlit)** | Deliver all insights through a multi-page, filterable web application |

---

## ✨ Features

- 📊 **Executive Dashboard** — real-time KPIs, monthly sales & profit trends, category breakdown, top-10 products
- 🛍️ **Sales Analysis** — multi-criteria filters (date, region, category, sub-category), profitability charts, state-level performance table, CSV export
- 👥 **Customer Segmentation** — RFM scatter plots, segment profile summary, filterable customer directory
- 📈 **Sales Prediction** — actual vs. predicted chart with shaded test period, model comparison table, growth scenario simulator
- 🚨 **Anomaly Detection** — algorithmic transaction map, categorized anomaly audit table, CSV export for controllers

---

## 🏗️ Architecture

```text
Raw Data (sample_superstore.csv)
         ↓
  Data Cleaning (src/data_processing.py)
         ↓
  Cleaned Dataset (9,994 rows)
         ↓
 ┌─────────────────┬──────────────────┬────────────────────┐
 ↓                 ↓                  ↓
RFM + K-Means   Sales Prediction   Anomaly Detection
(segmentation)  (prediction.py)    (anomaly_detection.py)
 ↓                 ↓                  ↓
 └─────────────────┴──────────────────┴────────────────────┘
                        ↓
              Streamlit Dashboard (dashboard/app.py)
```

**Data flow:**
```
data/raw/sample_superstore.csv
  → src/data_processing.py → data/processed/cleaned_superstore.csv
  → src/segmentation.py    → data/processed/customer_segments.csv + models/kmeans_*.pkl
  → src/prediction.py      → data/processed/sales_forecast.csv + models/best_sales_model.pkl
  → src/anomaly_detection.py → data/processed/anomalies_detected.csv + models/isolation_forest_model.pkl
  → dashboard/app.py       → Streamlit multi-page web application
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|:---|:---|
| Language | Python 3.10+ |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn (K-Means, GradientBoosting, IsolationForest, StandardScaler) |
| Visualization | Plotly Express/Graph Objects, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Environment | Virtualenv (`venv`), Jupyter / ipykernel |
| Serialization | Joblib |

---

## 📦 Dataset

**Source:** [Superstore Sales Dataset](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final) — a widely used benchmark dataset in business analytics.

**Raw file:** `data/raw/sample_superstore.csv`

The raw CSV contains 10,800 rows across 21 columns, including an embedded `Returns` auxiliary table appended after the main orders table. The data cleaning pipeline separates these, creates a binary `Returned` flag, imputes 11 missing postal codes (Burlington, VT — leading zero truncated by Excel), and adds 11 derived features.

| Attribute | Value |
|:---|:---|
| **Cleaned rows** | 9,994 transactions |
| **Date range** | 2015-01-03 → 2018-12-30 |
| **Total columns after cleaning** | 32 |
| **Missing values after cleaning** | 0 |
| **Duplicate rows** | 0 |

---

## 🤖 Machine Learning

### 👥 Customer Segmentation — RFM + K-Means

**RFM metrics** are computed per customer on the full transaction history:
- **Recency (R):** Days since the customer's last order (lower = more active)
- **Frequency (F):** Number of unique orders placed
- **Monetary (M):** Total revenue generated by the customer

**Why log transformation?** RFM distributions are heavily right-skewed (a few customers spend orders of magnitude more). `np.log1p` compresses the scale and makes K-Means distance meaningful.

**Why StandardScaler?** After log-transform, each RFM dimension still has a different scale. StandardScaler brings all three to zero-mean / unit-variance so that no single dimension dominates the Euclidean distance.

**Why K=4?** Validated by the elbow method (within-cluster inertia) and confirmed by the Silhouette Score.

**Cluster naming:** Labels are assigned programmatically based on actual cluster characteristics — not arbitrarily:
- Cluster with highest average Monetary → **Champions / High Value**
- Cluster with highest average Recency (most inactive) → **Lost / Dormant**
- Of the remaining two, lower Recency → **Active & Loyal**, higher Recency → **At Risk / Slipping**

### 📈 Sales Forecasting — Gradient Boosting

**Strict temporal split** (no data leakage):
- Train: 2015–2017 (36 monthly observations)
- Test: 2018 (12 monthly observations — fully out-of-sample)

**Features engineered on the monthly aggregated series:**
- `Month`, `Quarter` — calendar seasonality
- `Lag_1`, `Lag_2` — short-term momentum
- `Lag_12` — annual seasonality (same month, prior year)
- `Rolling_Mean_3` — smoothed recent trend (computed on lagged values to prevent leakage)

### 🚨 Anomaly Detection — Isolation Forest

Isolation Forest isolates anomalies by randomly partitioning the feature space. Anomalous points require fewer partitions to isolate (shorter path length).

**Features used:** `Sales`, `Profit`, `Quantity`, `Discount`, `Profit Margin`

**Contamination rate:** 3% → 300 flagged transactions out of 9,994.

**Business categorization of detected anomalies:**
- **Critical Loss** — `Profit < -500 $` (destructive discount or pricing error)
- **Excessive Discount** — `Discount ≥ 50%` (margin killer)
- **Blockbuster Sale** — `Sales > 3,000 $ AND Profit > 1,000 $` (positive outlier)
- **Atypical Profile** — other statistically unusual transactions

---

## 📊 Verified Results

> ⚠️ All metrics below were computed directly by running the project pipelines on the real dataset. No value is estimated or invented.

### Dataset

| Metric | Value |
|:---|:---|
| Total Revenue (2015-2018) | $2,297,200.86 |
| Total Profit | $286,397.02 |
| Overall Profit Margin | 12.47% |
| Unique Customers | 793 |
| Unique Orders | 5,009 |
| Unique Products | 1,862 |
| Transactions with Negative Profit | 1,871 (18.7%) |

### Category Performance

| Category | Sales | Profit | Margin |
|:---|:---:|:---:|:---:|
| Technology | $836,154 | $145,455 | 17.4% |
| Office Supplies | $719,047 | $122,491 | 17.0% |
| Furniture | $742,000 | $18,451 | 2.5% |

### Customer Segmentation (RFM + K-Means, K=4, Silhouette=0.2553)

| Segment | Customers | Avg Recency (days) | Avg Frequency (orders) | Avg Monetary ($) |
|:---|:---:|:---:|:---:|:---:|
| Champions / High Value | 258 | 103.3 | 8.6 | $4,773 |
| Active & Loyal | 199 | 19.7 | 6.8 | $2,713 |
| At Risk / Slipping | 253 | 235.2 | 4.9 | $1,937 |
| Lost / Dormant | 83 | 326.5 | 2.6 | $432 |

### Sales Prediction (Gradient Boosting — Test: 2018, 12 months)

| Model | MAE ($) | RMSE ($) | R² |
|:---|:---:|:---:|:---:|
| **Gradient Boosting** ✅ | **12,056** | **13,838** | **0.7117** |
| Linear Regression | 12,348 | 15,240 | 0.6504 |
| Random Forest | 14,291 | 16,342 | 0.5980 |

### Anomaly Detection (Isolation Forest, contamination=3%)

| Anomaly Type | Count |
|:---|:---:|
| Excessive Discount (≥ 50%) | 117 |
| Atypical Transactional Profile | 101 |
| Critical Loss (Profit < -500$) | 49 |
| Blockbuster Sale | 33 |
| **Total flagged** | **300** |

---

## 📁 Project Structure

```text
smart-business-intelligence/
│
├── data/
│   ├── raw/
│   │   └── sample_superstore.csv          # Raw Superstore dataset (10,800 rows)
│   └── processed/
│       ├── cleaned_superstore.csv         # Cleaned dataset (9,994 rows, 32 columns)
│       ├── customer_segments.csv          # RFM + cluster labels per customer
│       ├── sales_forecast.csv             # Monthly actuals + 2018 predictions
│       └── anomalies_detected.csv         # Full dataset with anomaly flags
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb             # Data ingestion, cleaning, feature engineering
│   ├── 02_exploratory_analysis.ipynb      # EDA, business insights
│   ├── 03_customer_segmentation.ipynb     # RFM + K-Means clustering
│   ├── 04_sales_prediction.ipynb          # Time series ML forecasting
│   └── 05_anomaly_detection.ipynb         # Isolation Forest anomaly detection
│
├── src/
│   ├── data_processing.py                 # Modular cleaning pipeline
│   ├── visualization.py                   # Reusable Plotly/Seaborn chart functions
│   ├── segmentation.py                    # RFM computation + K-Means pipeline
│   ├── prediction.py                      # Monthly feature engineering + model training
│   └── anomaly_detection.py               # Isolation Forest pipeline
│
├── dashboard/
│   ├── app.py                             # Streamlit main entry point (Overview)
│   └── pages/
│       ├── 1_Sales_Analysis.py            # Detailed sales & profitability analysis
│       ├── 2_Customer_Segmentation.py     # RFM cluster explorer
│       ├── 3_Sales_Prediction.py          # Forecast viewer & growth simulator
│       └── 4_Anomaly_Detection.py         # Anomaly audit table
│
├── models/
│   ├── kmeans_model.pkl                   # Fitted K-Means model (k=4)
│   ├── kmeans_scaler.pkl                  # Fitted StandardScaler for RFM
│   ├── best_sales_model.pkl               # Fitted Gradient Boosting regressor
│   └── isolation_forest_model.pkl         # Fitted Isolation Forest
│
├── screenshots/                           # Dashboard screenshots (for GitHub preview)
├── requirements.txt                       # Python dependencies with minimum versions
├── .gitignore                             # Excludes venv, __pycache__, .env, etc.
└── README.md                              # This file
```

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.10 or higher
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Chaimadebchi/smart-business-intelligence.git
cd smart-business-intelligence
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> ⚠️ **PowerShell execution policy issue?** If you see `cannot be loaded because running scripts is disabled`, run:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```
> Alternatively, use CMD instead:
> ```cmd
> venv\Scripts\activate.bat
> ```
> Or run everything directly with the venv Python:
> ```bash
> venv\Scripts\python.exe your_script.py
> ```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the data and ML pipelines

These steps generate all processed data and trained models. They only need to be run once — processed outputs are already included in the repository.

```bash
python src/data_processing.py
python src/segmentation.py
python src/prediction.py
python src/anomaly_detection.py
```

### 5. Launch the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

Or if using the venv directly (no activation):
```bash
venv\Scripts\streamlit.exe run dashboard/app.py   # Windows
venv/bin/streamlit run dashboard/app.py            # Linux/macOS
```

The application opens automatically in your browser at `http://localhost:8501`.

---

## ⚠️ Limitations

This project is honest about what it can and cannot claim:

- **Static historical dataset** — the Superstore dataset covers 2015-2018 only. Insights do not automatically transfer to different time periods, markets, or industries.
- **Small training window for forecasting** — the prediction model is trained on 36 monthly points (2015-2017), which limits its statistical reliability. A real production forecaster would require significantly more history.
- **Segmentation based on historical behavior** — RFM clusters reflect past purchasing patterns. A customer labeled "Lost" may have churned for reasons not captured in the data.
- **Anomalies require business validation** — a statistical anomaly is not evidence of fraud. Each flagged transaction requires human review and domain knowledge before any action.
- **No production deployment** — the dashboard runs locally. It is not deployed, authenticated, or connected to a live database.
- **No real-time data ingestion** — the pipeline is batch-based. Data must be re-run manually when new data arrives.

---

## 🔭 Future Improvements

| Area | Improvement |
|:---|:---|
| **Data** | Connect to a live SQL/NoSQL database instead of static CSV |
| **Ingestion** | Build an automated ETL pipeline (e.g., Apache Airflow or Prefect) |
| **Forecasting** | Explore Prophet or LSTM for longer-horizon, multi-step forecasting |
| **Backend** | Expose predictions via a FastAPI REST API |
| **Deployment** | Deploy on Streamlit Cloud, Heroku, or a Docker container |
| **Security** | Add authentication to the dashboard (Streamlit `st.login`) |
| **Monitoring** | Track model drift over time with evidently.ai or similar |
| **Testing** | Add unit tests for `src/` pipeline functions with pytest |

---

## 📜 License

Distributed under the MIT License. Free to use for academic and professional portfolio purposes.
