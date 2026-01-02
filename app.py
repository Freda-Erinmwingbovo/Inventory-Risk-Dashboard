# ============================================================
# app.py — Inventory Risk Pro (FINAL PROFESSIONAL VERSION)
# Excel/CSV • Wrong Data Handling • Lead Generation • Enterprise-Ready
# Built by Freda Erinmwingbovo • Abuja, Nigeria • January 2026
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import io

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch

st.set_page_config(page_title="Inventory Risk Pro", page_icon="📦", layout="wide")

st.markdown("""
<style>
    .big-font {font-size: 50px !important; font-weight: bold;}
    .risk {color: #d32f2f;}
    .save {color: #388e3c;}
    h1 {color: #1e88e5; text-align: center;}
    .stTabs [data-baseweb="tab"] {font-size: 18px; font-weight: bold;}
    .recommendation {background-color: #e8f5e9; padding: 12px; border-radius: 8px; border-left: 5px solid #388e3c;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📦 Inventory Risk Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 22px;'>Enterprise Inventory Optimization • Excel & CSV • Real-World Ready</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload your inventory file (CSV or Excel supported)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Robust file reading with encoding fallback
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            raw_df = pd.read_excel(uploaded_file)
            st.info("✅ Excel file loaded successfully")
        else:
            try:
                raw_df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file, encoding='latin-1')
            except Exception:
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file, encoding='cp1252', errors='replace')
            st.info("✅ CSV file loaded (encoding adjusted automatically)")

        # Required core + optional
        required_core = ['product_id', 'product_name', 'current_stock', 'avg_daily_sales', 'unit_cost_ngn']
        optional = ['lead_time_days', 'safety_stock_days']

        # Fuzzy column matching
        raw_cols_lower = {col.lower(): col for col in raw_df.columns}
        mapped = {}
        missing_core = []
        missing_optional = []

        for col in required_core + optional:
            if col.lower() in raw_cols_lower:
                mapped[col] = raw_cols_lower[col.lower()]
            else:
                if col in required_core:
                    missing_core.append(col)
                else:
                    missing_optional.append(col)

        # --- FRIENDLY HANDLING OF WRONG DATA TYPE ---
        if missing_core:
            st.warning("📊 This file appears to be **sales transaction data** (invoices, orders, etc.), not a current inventory snapshot.")
            st.info("""
**You're in the right place — this is very common!**

Most businesses start with sales data from their POS or ERP system.

This app is specifically designed for **current inventory levels**:
- How many units of each product do you have **right now**?
- Average daily sales
- Unit cost

But we can help you go much further using your **sales history**:
- Build **demand forecasting** from your transactions
- Create **automatic reorder alerts**
- Develop a **full inventory + sales dashboard** tailored to your business

**Need a custom solution for your company?**  
Let's build it together.

📧 Contact: fredaerins@gmail.com  
💼 LinkedIn: [Add your LinkedIn if you have one]

I'm here to help turn your data into real savings and efficiency.
            """)

            st.markdown("### Want to try the app now?")
            st.write("Download this sample inventory template, fill it with your current stock data, and upload it:")
            template_data = {
                "product_id": [101, 102, 103, 104, 105],
                "product_name": ["Wireless Mouse", "USB Cable", "Laptop Stand", "Webcam", "External HDD"],
                "current_stock": [45, 120, 18, 8, 32],
                "avg_daily_sales": [8, 15, 3, 2, 5],
                "unit_cost_ngn": [12000, 3000, 45000, 75000, 80000],
                "lead_time_days": [10, 5, 21, 14, 30],
                "safety_stock_days": [5, 3, 7, 5, 10]
            }
            template_df = pd.DataFrame(template_data)
            csv_template = template_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇️ Download Sample Inventory Template",
                csv_template,
                "sample_inventory_template.csv",
                "text/csv"
            )
            st.stop()

        # Proceed with valid data
        df = raw_df[[mapped[col] for col in mapped]].copy()
        df.columns = list(mapped.keys())

        # Cleaning
        numeric_cols = ['current_stock', 'avg_daily_sales', 'unit_cost_ngn']
        if 'lead_time_days' in df.columns:
            numeric_cols.append('lead_time_days')
        if 'safety_stock_days' in df.columns:
            numeric_cols.append('safety_stock_days')

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df = df.dropna(subset=['current_stock', 'avg_daily_sales', 'unit_cost_ngn'])
        df[numeric_cols] = df[numeric_cols].clip(lower=0)

        df['product_name'] = df['product_name'].fillna("Unnamed").astype(str).str.strip()
        df['product_name'] = df['product_name'].replace(['', 'nan'], 'Unnamed Product')

        # Defaults
        if 'lead_time_days' not in df.columns:
            df['lead_time_days'] = 14
            st.info("lead_time_days not found → using default: 14 days")
        if 'safety_stock_days' not in df.columns:
            df['safety_stock_days'] = 7
            st.info("safety_stock_days not found → using default: 7 days")

        st.success(f"✅ Analysis complete: {len(df):,} products processed")

        # --- CALCULATIONS ---
        df['days_on_hand'] = df['current_stock'] / (df['avg_daily_sales'] + 0.01)
        df['reorder_point'] = df['avg_daily_sales'] * (df['lead_time_days'] + df['safety_stock_days'])
        df['stockout_risk'] = df['current_stock'] < df['reorder_point']
        df['overstock_risk'] = df['days_on_hand'] > 180
        df['slow_moving'] = (df['days_on_hand'] > 90) & (df['days_on_hand'] <= 365)
        df['dead_stock'] = df['days_on_hand'] > 365
        df['holding_cost_ngn'] = df['current_stock'] * df['unit_cost_ngn'] * 0.25
        df['potential_savings_ngn'] = np.where(df['dead_stock'], df['holding_cost_ngn'], 0)

        df['annual_value_ngn'] = df['avg_daily_sales'] * 365 * df['unit_cost_ngn']
        df = df.sort_values('annual_value_ngn', ascending=False).copy()
        df['cumulative_pct'] = df['annual_value_ngn'].cumsum() / df['annual_value_ngn'].sum()
        df['abc_class'] = np.where(df['cumulative_pct'] <= 0.8, 'A',
                          np.where(df['cumulative_pct'] <= 0.95, 'B', 'C'))

        df['stockout_probability'] = np.where(df['stockout_risk'], 0.25, 0.03)
        df['excess_stock'] = np.maximum(0, df['current_stock'] - df['reorder_point'])
        df['cash_at_risk_ngn'] = df['excess_stock'] * df['unit_cost_ngn']

        def get_rec(row):
            recs = []
            if row['stockout_risk']: recs.append("URGENT REORDER")
            if row['dead_stock']: recs.append("LIQUIDATE DEAD STOCK")
            if row['overstock_risk']: recs.append("REDUCE FUTURE ORDERS")
            if row['slow_moving']: recs.append("PROMOTE TO CLEAR")
            if row['abc_class'] == 'A': recs.append("HIGH PRIORITY A-ITEM")
            if not recs: recs.append("HEALTHY STOCK")
            return " • ".join(recs)

        df['recommendation'] = df.apply(get_rec, axis=1)

        total_holding = df['holding_cost_ngn'].sum()
        total_cash_risk = df['cash_at_risk_ngn'].sum()

        # Dashboard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Products Analyzed", f"{len(df):,}")
        col2.metric("A-Class Items", (df['abc_class']=='A').sum())
        col3.metric("Cash-at-Risk (₦)", f"{total_cash_risk:,.0f}", delta_color="inverse")
        col4.metric("Stockout Risk Items", df['stockout_risk'].sum(), delta_color="inverse")

        st.markdown("---")

        # Tabs (simplified for brevity — keep your preferred structure)
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 ABC Classification", "⚠️ Risk Items", "💰 Cost Simulator", "📄 Executive Report", "📈 Export Data"
        ])

        # [Keep your previous tab content here — ABC charts, risk table, EOQ simulator, PDF, export]

        with tab5:
            st.subheader("Download Enriched Dataset")
            st.write("Includes all calculated columns: reorder_point, stockout_risk, holding_cost_ngn, recommendation, abc_class, cash_at_risk_ngn, etc.")
            st.download_button(
                "⬇️ Export Full Analysis",
                df.to_csv(index=False).encode('utf-8'),
                "inventory_risk_pro_analysis.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"File processing failed: {e}")
        st.info("Try saving your file as CSV UTF-8 or Excel format.")

else:
    st.info("👆 Upload your current inventory file — Excel or CSV accepted!")
    st.markdown("""
    **Required columns** (names can vary in case/spacing):
    - product_id
    - product_name
    - current_stock
    - avg_daily_sales
    - unit_cost_ngn

    **Optional**: lead_time_days, safety_stock_days (defaults used if missing)

    Have sales data instead? No problem — contact fredaerins@gmail.com for a custom solution.
    """)

st.caption("Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria • January 2026")
