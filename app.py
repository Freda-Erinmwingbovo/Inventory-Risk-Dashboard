# ============================================================
# app.py — Inventory Risk Pro (FINAL PRODUCTION VERSION)
# Excel + CSV Support • Optional Columns • Bulletproof • Enterprise-Grade
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
st.markdown("<p style='text-align: center; font-size: 22px;'>Enterprise Optimization • Excel & CSV • Smart Defaults • Real-World Ready</p>", unsafe_allow_html=True)

# --- SUPPORTS BOTH CSV AND EXCEL ---
uploaded_file = st.file_uploader("📁 Upload your inventory file (CSV or Excel supported)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Read file based on extension
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            raw_df = pd.read_excel(uploaded_file)
            st.info("✅ Excel file loaded successfully")
        else:
            raw_df = pd.read_csv(uploaded_file)
            st.info("✅ CSV file loaded successfully")

        initial_rows = len(raw_df)

        # Required + Optional columns
        required_core = ['product_id', 'product_name', 'current_stock', 'avg_daily_sales', 'unit_cost_ngn']
        optional = ['lead_time_days', 'safety_stock_days']

        # Fuzzy case-insensitive matching
        raw_cols_lower = {col.lower(): col for col in raw_df.columns}
        mapped_cols = {}
        missing = []

        for col in required_core + optional:
            if col in raw_df.columns:
                mapped_cols[col] = col
            elif col.lower() in raw_cols_lower:
                mapped_cols[col] = raw_cols_lower[col.lower()]
            else:
                missing.append(col)

        if missing and set(missing).issubset(optional):
            st.warning(f"Optional columns not found: {', '.join(missing)}. Using defaults.")
        elif any(m in required_core for m in missing):
            st.error(f"Required columns missing: {', '.join([m for m in missing if m in required_core])}. Please check your file.")
            st.stop()

        # Select and rename columns
        df = raw_df[[mapped_cols[col] for col in mapped_cols]].copy()
        df.columns = list(mapped_cols.keys())

        # --- BULLETPROOF CLEANING ---
        numeric_cols = ['current_stock', 'avg_daily_sales', 'unit_cost_ngn']
        if 'lead_time_days' in df.columns:
            numeric_cols.append('lead_time_days')
        if 'safety_stock_days' in df.columns:
            numeric_cols.append('safety_stock_days')

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        before_drop = len(df)
        df = df.dropna(subset=['current_stock', 'avg_daily_sales', 'unit_cost_ngn'])
        dropped_rows = before_drop - len(df)

        df[numeric_cols] = df[numeric_cols].clip(lower=0)

        df['product_name'] = df['product_name'].fillna("Unnamed").astype(str).str.strip()
        df['product_name'] = df['product_name'].replace(['', 'nan'], 'Unnamed Product')

        # --- OPTIONAL COLUMNS WITH SMART DEFAULTS ---
        if 'lead_time_days' not in df.columns:
            df['lead_time_days'] = 14  # Standard default
            st.info("lead_time_days not provided → using default: 14 days")
        if 'safety_stock_days' not in df.columns:
            df['safety_stock_days'] = 7   # Standard default
            st.info("safety_stock_days not provided → using default: 7 days")

        final_rows = len(df)
        st.success(f"✅ Analysis ready: {final_rows:,} valid products")

        if dropped_rows > 0:
            with st.expander("🧹 Data cleaning applied"):
                st.info(f"Removed {dropped_rows} rows with invalid/missing core data")

        # --- ALL CALCULATED COLUMNS ---
        df['days_on_hand'] = df['current_stock'] / (df['avg_daily_sales'] + 0.01)
        df['reorder_point'] = df['avg_daily_sales'] * (df['lead_time_days'] + df['safety_stock_days'])
        df['stockout_risk'] = df['current_stock'] < df['reorder_point']
        df['overstock_risk'] = df['days_on_hand'] > 180
        df['slow_moving'] = (df['days_on_hand'] > 90) & (df['days_on_hand'] <= 365)
        df['dead_stock'] = df['days_on_hand'] > 365
        df['holding_cost_ngn'] = df['current_stock'] * df['unit_cost_ngn'] * 0.25
        df['potential_savings_ngn'] = np.where(df['dead_stock'], df['holding_cost_ngn'], 0)

        # Enterprise features
        df['annual_value_ngn'] = df['avg_daily_sales'] * 365 * df['unit_cost_ngn']
        df = df.sort_values('annual_value_ngn', ascending=False).copy()
        df['cumulative_pct'] = df['annual_value_ngn'].cumsum() / df['annual_value_ngn'].sum()
        df['abc_class'] = np.where(df['cumulative_pct'] <= 0.8, 'A',
                          np.where(df['cumulative_pct'] <= 0.95, 'B', 'C'))

        df['stockout_probability'] = np.where(df['stockout_risk'], 0.25, 0.03)
        df['excess_stock'] = np.maximum(0, df['current_stock'] - df['reorder_point'])
        df['cash_at_risk_ngn'] = df['excess_stock'] * df['unit_cost_ngn']

        # Recommendation
        def get_rec(row):
            recs = []
            if row['stockout_risk']:
                recs.append("URGENT REORDER")
            if row['dead_stock']:
                recs.append("LIQUIDATE DEAD STOCK")
            if row['overstock_risk']:
                recs.append("REDUCE FUTURE ORDERS")
            if row['slow_moving']:
                recs.append("PROMOTE TO CLEAR")
            if row['abc_class'] == 'A':
                recs.append("HIGH PRIORITY A-ITEM")
            if not recs:
                recs.append("HEALTHY STOCK")
            return " • ".join(recs)

        df['recommendation'] = df.apply(get_rec, axis=1)

        total_holding = df['holding_cost_ngn'].sum()
        total_cash_risk = df['cash_at_risk_ngn'].sum()

        # Dashboard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valid Products", f"{len(df):,}")
        col2.metric("A-Class Items", (df['abc_class']=='A').sum())
        col3.metric("Cash-at-Risk (₦)", f"{total_cash_risk:,.0f}", delta_color="inverse")
        col4.metric("Stockout Risk Items", df['stockout_risk'].sum(), delta_color="inverse")

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 ABC-XYZ", "⚠️ Risk Items", "💰 Optimizer", "📄 Report", "📈 Export"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ABC Classification")
                fig, ax = plt.subplots()
                df['abc_class'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#4caf50', '#ff9800', '#f44336'])
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
                                    'holding_cost_ngn', 'stockout_risk', 'dead_stock', 'recommendation']].copy()
            display['holding_cost_ngn'] = display['holding_cost_ngn'].apply(lambda x: f"₦{x:,.0f}")
            st.dataframe(display.head(50), use_container_width=True)

        with tab3:
            st.subheader("EOQ Optimization")
            order_cost = st.slider("Cost per Order (₦)", 500, 10000, 3000)
            eoq = np.sqrt(2 * df['avg_daily_sales'] * 365 * order_cost / (df['unit_cost_ngn'] * 0.25 + 1e-6))
            optimized_holding = (eoq / 2 * df['unit_cost_ngn'] * 0.25).sum()
            savings = total_holding - optimized_holding
            st.markdown(f"<p class='save big-font'>Potential Savings: ₦{savings:,.0f}/year</p>", unsafe_allow_html=True)

        with tab4:
            if st.button("Generate Executive PDF"):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                story = [Paragraph("Inventory Risk Pro Report", getSampleStyleSheet()['Title']),
                         Spacer(1, 20),
                         Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", getSampleStyleSheet()['Normal'])]
                data = [["Metric", "Value"],
                        ["Products", len(df)],
                        ["Cash-at-Risk", f"₦{total_cash_risk:,.0f}"],
                        ["Potential Savings", f"₦{savings:,.0f}"]]
                story.append(Table(data))
                doc.build(story)
                buffer.seek(0)
                st.download_button("Download PDF", buffer, "inventory_report.pdf", "application/pdf")

        with tab5:
            st.subheader("Download Full Enriched Dataset")
            st.write("Includes all original + calculated columns: reorder_point, stockout_risk, holding_cost_ngn, recommendation, abc_class, cash_at_risk_ngn, etc.")
            st.download_button(
                "⬇️ Export Complete CSV",
                df.to_csv(index=False).encode('utf-8'),
                "inventory_full_analysis.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"File processing error: {e}. Check format and try again.")
else:
    st.info("Upload your inventory file — Excel (.xlsx) or CSV supported!")
    st.markdown("**Core required**: product_id, product_name, current_stock, avg_daily_sales, unit_cost_ngn  \n**Optional**: lead_time_days (default 14), safety_stock_days (default 7)")

st.caption("Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria • January 2026")
