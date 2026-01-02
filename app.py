# ============================================================
# app.py — Inventory Risk Pro (FINAL PROFESSIONAL LANDING PAGE)
# Clean, Minimalist, Enterprise-Ready • No Clutter • Single Uploader
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

# Clean, professional styling
st.markdown("""
<style>
    .big-font {font-size: 50px !important; font-weight: bold;}
    .risk {color: #d32f2f;}
    .save {color: #388e3c;}
    h1 {color: #1e88e5; text-align: center;}
    .stTabs [data-baseweb="tab"] {font-size: 18px; font-weight: bold;}
    .recommendation {background-color: #e8f5e9; padding: 12px; border-radius: 8px; border-left: 5px solid #388e3c;}
    .main-container {max-width: 800px; margin: auto; padding-top: 60px;}
    .title {font-size: 48px; font-weight: 600; color: #1e88e5; margin-bottom: 10px;}
    .subtitle {font-size: 22px; color: #666; margin-bottom: 50px;}
</style>
""", unsafe_allow_html=True)

# Professional landing page — only shown when no file uploaded
if 'df' not in st.session_state:
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.markdown("<h1 class='title'>📦 Inventory Risk Pro</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Enterprise-grade inventory optimization powered by AI</p>", unsafe_allow_html=True)

    # Single, elegant uploader
    uploaded_file = st.file_uploader(
        "Upload your current inventory file to begin analysis",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
        help="Supported formats: CSV, Excel (.xlsx, .xls) • Max 200MB"
    )

    # Subtle help and sample
    col1, col2 = st.columns([3, 1])
    with col1:
        with st.expander("📋 View required data format"):
            st.markdown("""
**Required columns** (case-insensitive):
- `product_id` – Unique product identifier
- `product_name` – Product description
- `current_stock` – Units currently in stock
- `avg_daily_sales` – Average units sold per day
- `unit_cost_ngn` – Cost per unit in Naira (₦)

**Optional columns** (defaults used if missing):
- `lead_time_days` → default 14 days
- `safety_stock_days` → default 7 days
            """)

    with col2:
        with st.expander("🚀 Try with sample data"):
            if st.button("Load sample & explore"):
                sample_data = {
                    "product_id": [101, 102, 103, 104, 105, 106, 107, 108],
                    "product_name": ["Wireless Mouse", "USB Cable", "Laptop Stand", "Webcam", "External HDD", "Keyboard", "Monitor", "Printer Ink"],
                    "current_stock": [45, 120, 18, 8, 32, 65, 12, 200],
                    "avg_daily_sales": [8, 15, 3, 2, 5, 10, 1.5, 20],
                    "unit_cost_ngn": [12000, 3000, 45000, 75000, 80000, 25000, 150000, 5000],
                    "lead_time_days": [10, 5, 21, 14, 30, 7, 28, 3],
                    "safety_stock_days": [5, 3, 7, 5, 10, 4, 10, 2]
                }
                st.session_state.df = pd.DataFrame(sample_data)
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

else:
    df = st.session_state.df

# If file uploaded or sample loaded — run analysis
if uploaded_file is not None or 'df' in st.session_state:
    if uploaded_file is not None:
        try:
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

            if missing_core:
                st.warning("This file appears to be sales transaction data, not current inventory levels.")
                st.info("""
This app requires a **current stock snapshot**.

We can build a custom solution using your sales data for forecasting, reorders, or full analytics.

**Interested?**  
📧 fredaerins@gmail.com
                """)

                template_data = {
                    "product_id": [101, 102, 103],
                    "product_name": ["Wireless Mouse", "USB Cable", "Laptop Stand"],
                    "current_stock": [45, 120, 18],
                    "avg_daily_sales": [8, 15, 3],
                    "unit_cost_ngn": [12000, 3000, 45000],
                    "lead_time_days": [10, 5, 21],
                    "safety_stock_days": [5, 3, 7]
                }
                template_df = pd.DataFrame(template_data)
                csv_template = template_df.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download Sample Template", csv_template, "sample_inventory_template.csv", "text/csv")
                st.stop()

            df = raw_df[[mapped[col] for col in mapped]].copy()
            df.columns = list(mapped.keys())
            st.session_state.df = df  # Save for persistence

        except Exception as e:
            st.error(f"File processing error: {e}")
            st.stop()

    # Use df from session state (either uploaded or sample)
    df = st.session_state.df

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

    st.success(f"Analysis ready — {len(df):,} products")

    # Calculations (same as before)
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

    # Tabs
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
        st.subheader("Export Enriched Dataset")
        st.download_button(
            "⬇️ Export Full Analysis",
            df.to_csv(index=False).encode('utf-8'),
            "inventory_risk_pro_analysis.csv",
            "text/csv"
        )

st.caption("Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria • January 2026")
