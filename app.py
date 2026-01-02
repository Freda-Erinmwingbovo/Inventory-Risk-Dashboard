# ============================================================
# app.py — Inventory Risk Pro (FINAL FULL & CLEAN)
# All Tabs Restored • Clean Landing Page • Production-Ready
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
        # Load file with encoding fallback
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            raw_df = pd.read_excel(uploaded_file)
        else:
            try:
                raw_df = pd.read_csv(uploaded_file)
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file, encoding='latin-1')
            except:
                uploaded_file.seek(0)
                raw_df = pd.read_csv(uploaded_file, encoding='cp1252', errors='replace')

        # Column mapping
        required_core = ['product_id', 'product_name', 'current_stock', 'avg_daily_sales', 'unit_cost_ngn']
        optional = ['lead_time_days', 'safety_stock_days']

        raw_cols_lower = {col.lower(): col for col in raw_df.columns}
        mapped = {}
        missing_core = []

        for col in required_core + optional:
            if col.lower() in raw_cols_lower:
                mapped[col] = raw_cols_lower[col.lower()]
            else:
                if col in required_core:
                    missing_core.append(col)

        # Only show disclaimer if wrong data type
        if missing_core:
            st.warning("📊 This file appears to be **sales transaction data** (invoices, orders, etc.), not current inventory levels.")
            st.info("""
**Good news — you're not doing anything wrong!**

Most businesses start with sales data from their POS or ERP.

This app is designed for **current stock snapshots**.

But we can build a **custom tool** using your sales history for:
- Demand forecasting
- Automatic reorder alerts
- Full inventory + sales dashboard

**Need a tailored solution?**  
📧 fredaerins@gmail.com
            """)

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
            st.download_button("⬇️ Download Sample Template", csv_template, "sample_inventory_template.csv", "text/csv")
            st.stop()

        # Correct data — proceed
        df = raw_df[[mapped[col] for col in mapped]].copy()
        df.columns = list(mapped.keys())

        # Cleaning & defaults
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

        if 'lead_time_days' not in df.columns:
            df['lead_time_days'] = 14
        if 'safety_stock_days' not in df.columns:
            df['safety_stock_days'] = 7

        st.success(f"✅ Analysis ready: {len(df):,} products")

        # Calculations
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
            if row['overstock_risk']: recs.append("REDUCE ORDERS")
            if row['slow_moving']: recs.append("PROMOTE TO CLEAR")
            if row['abc_class'] == 'A': recs.append("HIGH PRIORITY")
            if not recs: recs.append("HEALTHY")
            return " • ".join(recs)

        df['recommendation'] = df.apply(get_rec, axis=1)

        total_holding = df['holding_cost_ngn'].sum()
        total_cash_risk = df['cash_at_risk_ngn'].sum()

        # Dashboard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Products Analyzed", f"{len(df):,}")
        col2.metric("A-Class Items", (df['abc_class']=='A').sum())
        col3.metric("Cash-at-Risk (₦)", f"{total_cash_risk:,.0f}")
        col4.metric("Stockout Risk Items", df['stockout_risk'].sum())

        st.markdown("---")

        # ALL TABS RESTORED
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 ABC Classification", "⚠️ Risk Items", "💰 Cost Simulator", "📄 Executive Report", "📈 Export Data"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ABC Distribution")
                fig, ax = plt.subplots()
                df['abc_class'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
                st.pyplot(fig)
            with col2:
                st.subheader("Risk Breakdown")
                stockout = df['stockout_risk'].sum()
                over = df['overstock_risk'].sum()
                slow = df['slow_moving'].sum()
                dead = df['dead_stock'].sum()
                healthy = len(df) - (stockout + over + slow + dead)
                fig, ax = plt.subplots()
                ax.pie([healthy, stockout, over, slow, dead], labels=['Healthy', 'Stockout', 'Overstock', 'Slow', 'Dead'], autopct='%1.1f%%')
                st.pyplot(fig)

        with tab2:
            st.subheader("Items Needing Action")
            action_items = df[df['stockout_risk'] | df['dead_stock'] | df['overstock_risk'] | df['slow_moving'] | (df['abc_class']=='A')]
            display = action_items[['product_name', 'days_on_hand', 'reorder_point', 'current_stock',
                                    'holding_cost_ngn', 'recommendation']].copy()
            display['holding_cost_ngn'] = display['holding_cost_ngn'].apply(lambda x: f"₦{x:,.0f}")
            st.dataframe(display.head(50), use_container_width=True)

        with tab3:
            st.subheader("EOQ Cost Optimization Simulator")
            order_cost = st.slider("Average Cost per Order (₦)", 500, 10000, 3000)
            eoq = np.sqrt(2 * df['avg_daily_sales'] * 365 * order_cost / (df['unit_cost_ngn'] * 0.25 + 1e-6))
            optimized_holding = (eoq / 2 * df['unit_cost_ngn'] * 0.25).sum()
            savings = total_holding - optimized_holding
            st.markdown(f"<p class='save big-font'>Potential Savings: ₦{savings:,.0f}/year</p>", unsafe_allow_html=True)

        with tab4:
            if st.button("Generate Executive PDF Report"):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                story = [Paragraph("Inventory Risk Pro Report", styles['Title']),
                         Spacer(1, 20),
                         Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal'])]
                data = [["Metric", "Value"],
                        ["Products", len(df)],
                        ["Cash-at-Risk", f"₦{total_cash_risk:,.0f}"],
                        ["Potential Savings", f"₦{savings:,.0f}"]]
                story.append(Table(data))
                doc.build(story)
                buffer.seek(0)
                st.download_button("Download Report", buffer, "inventory_report.pdf", "application/pdf")

        with tab5:
            st.subheader("Download Enriched Dataset")
            st.download_button(
                "⬇️ Export Full Analysis",
                df.to_csv(index=False).encode('utf-8'),
                "inventory_risk_pro_analysis.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"File error: {e}")

else:
    # Clean landing page
    st.info("👆 Upload your current inventory file — Excel or CSV accepted!")
    
    st.markdown("### 📋 Required Columns")
    st.markdown("""
    • product_id  
    • product_name  
    • current_stock  
    • avg_daily_sales  
    • unit_cost_ngn
    """)
    
    st.markdown("### ⚙️ Optional")
    st.markdown("""
    • lead_time_days (default: 14)  
    • safety_stock_days (default: 7)
    """)
    
    st.markdown("### 🚀 Test the App Now")
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
    st.download_button("⬇️ Download Sample Template", csv_template, "sample_inventory_template.csv", "text/csv")

st.caption("Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria • January 2026")
