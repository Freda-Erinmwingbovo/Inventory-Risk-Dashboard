# ============================================================
# app.py — Inventory Risk Pro (BULLETPROOF PRODUCTION-GRADE)
# Handles Dirty Data • ABC-XYZ • Cash-at-Risk • EOQ Optimization
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
    .high-impact {background-color: #fff3cd; padding: 10px; border-radius: 8px; border-left: 5px solid #ffc107;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>📦 Inventory Risk Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 22px;'>Bulletproof Enterprise Optimization • Handles Real-World Data • ABC-XYZ • Cash-at-Risk</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload your inventory CSV (even if messy!)", type="csv")

if uploaded_file is not None:
    try:
        # Load raw data
        raw_df = pd.read_csv(uploaded_file)
        initial_rows = len(raw_df)
        st.info(f"Raw upload: {initial_rows:,} rows detected")

        # Required columns
        required = ['product_id', 'product_name', 'current_stock', 'avg_daily_sales',
                    'unit_cost_ngn', 'lead_time_days', 'safety_stock_days']

        # Check for missing columns (case-insensitive fuzzy match)
        raw_cols_lower = {col.lower(): col for col in raw_df.columns}
        mapped_cols = {}
        missing = []
        for req in required:
            if req in raw_df.columns:
                mapped_cols[req] = req
            elif req.lower() in raw_cols_lower:
                mapped_cols[req] = raw_cols_lower[req.lower()]
            else:
                missing.append(req)

        if missing:
            st.error(f"Cannot find columns: {', '.join(missing)}. Please check spelling.")
            st.stop()

        df = raw_df[[mapped_cols[col] for col in required]].copy()
        df.columns = required  # Standardize names

        # --- BULLETPROOF DATA CLEANING ---
        numeric_cols = ['current_stock', 'avg_daily_sales', 'unit_cost_ngn', 'lead_time_days', 'safety_stock_days']
        cleaned_count = 0

        for col in numeric_cols:
            original = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors='coerce')  # "N/A", text → NaN
            invalid = original.notna() & df[col].isna()
            cleaned_count += invalid.sum()

        # Remove rows with any critical NaN after cleaning
        before_drop = len(df)
        df = df.dropna(subset=numeric_cols)
        dropped_rows = before_drop - len(df)

        # Clip negatives to zero
        negatives_fixed = (df[numeric_cols] < 0).sum().sum()
        df[numeric_cols] = df[numeric_cols].clip(lower=0)

        # Clean product names
        df['product_name'] = df['product_name'].fillna("Unnamed")
        df['product_name'] = df['product_name'].astype(str).str.strip()
        df['product_name'] = df['product_name'].replace(['', 'nan', 'None'], 'Unnamed Product')

        # Final clean dataset
        final_rows = len(df)
        st.success(f"✅ Cleaned data ready: {final_rows:,} valid products")

        if cleaned_count > 0 or dropped_rows > 0 or negatives_fixed > 0:
            with st.expander("🧹 Data cleaning summary (click to view)"):
                if cleaned_count > 0:
                    st.warning(f"Fixed {cleaned_count} non-numeric entries (converted to valid numbers or removed)")
                if dropped_rows > 0:
                    st.warning(f"Removed {dropped_rows} rows with missing critical data")
                if negatives_fixed > 0:
                    st.info(f"Corrected {negatives_fixed} negative values to zero")

        # ---------------- ENTERPRISE CALCULATIONS ----------------
        df['annual_demand'] = df['avg_daily_sales'] * 365
        df['annual_value_ngn'] = df['annual_demand'] * df['unit_cost_ngn']
        df['days_on_hand'] = df['current_stock'] / (df['avg_daily_sales'] + 0.01)
        df['reorder_point'] = df['avg_daily_sales'] * (df['lead_time_days'] + df['safety_stock_days'])
        df['holding_cost_ngn'] = df['current_stock'] * df['unit_cost_ngn'] * 0.25

        # ABC Classification
        df = df.sort_values('annual_value_ngn', ascending=False).copy()
        df['cumulative_pct'] = df['annual_value_ngn'].cumsum() / df['annual_value_ngn'].sum()
        df['abc_class'] = np.where(df['cumulative_pct'] <= 0.80, 'A',
                          np.where(df['cumulative_pct'] <= 0.95, 'B', 'C'))

        # XYZ (simplified using CV proxy)
        df['demand_cv'] = df['avg_daily_sales'].rolling(window=10, min_periods=1).std() / (df['avg_daily_sales'] + 0.01)
        df['xyz_class'] = np.where(df['demand_cv'] < 0.3, 'X',
                          np.where(df['demand_cv'] < 0.7, 'Y', 'Z'))
        df['abc_xyz'] = df['abc_class'] + df['xyz_class']

        # Risk & Financials
        df['stockout_risk'] = df['current_stock'] < df['reorder_point']
        df['stockout_probability'] = np.where(df['stockout_risk'], 0.25, 0.03)
        df['excess_stock'] = np.maximum(0, df['current_stock'] - df['reorder_point'])
        df['cash_at_risk_ngn'] = df['excess_stock'] * df['unit_cost_ngn']

        total_cash_at_risk = df['cash_at_risk_ngn'].sum()
        total_holding = df['holding_cost_ngn'].sum()
        a_items = (df['abc_class'] == 'A').sum()

        # Dashboard
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Valid SKUs", f"{len(df):,}")
        col2.metric("A-Class (80% value)", a_items)
        col3.metric("Cash-at-Risk (₦)", f"{total_cash_at_risk:,.0f}", delta_color="inverse")
        col4.metric("High Stockout Risk Items", df[df['stockout_probability'] > 0.1].shape[0], delta_color="inverse")

        st.markdown("---")

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 ABC-XYZ Classification", "⚠️ Risk & Financial Impact", "💰 Optimization Simulator", "📄 Executive Report", "📈 Export Data"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ABC Distribution")
                fig, ax = plt.subplots()
                abc_counts = df['abc_class'].value_counts()
                ax.pie(abc_counts, labels=abc_counts.index, autopct='%1.1f%%', colors=['#4caf50', '#ff9800', '#f44336'])
                st.pyplot(fig)
            with col2:
                st.subheader("ABC-XYZ Matrix")
                matrix = df.pivot_table(values='product_id', index='abc_class', columns='xyz_class', aggfunc='count', fill_value=0)
                fig, ax = plt.subplots()
                sns.heatmap(matrix, annot=True, fmt='d', cmap="YlGnBu", ax=ax)
                st.pyplot(fig)

        with tab2:
            st.subheader("Critical Items (A-Class or High Risk)")
            critical = df[(df['abc_class']=='A') | (df['stockout_probability'] > 0.1) | (df['cash_at_risk_ngn'] > df['cash_at_risk_ngn'].quantile(0.75))]
            critical = critical.sort_values('annual_value_ngn', ascending=False)

            display = critical[['product_name', 'abc_xyz', 'stockout_probability', 'cash_at_risk_ngn', 'days_on_hand']].copy()
            display['stockout_probability'] = (display['stockout_probability']*100).round(0).astype(str) + "%"
            display['cash_at_risk_ngn'] = display['cash_at_risk_ngn'].apply(lambda x: f"₦{x:,.0f}")
            st.dataframe(display.head(50), use_container_width=True)

            for _, row in critical.head(10).iterrows():
                with st.expander(f"🔴 {row['product_name']} | {row['abc_xyz']} | Risk: {row['stockout_probability']}"):
                    st.markdown(f"<div class='high-impact'>Cash-at-Risk: ₦{row['cash_at_risk_ngn']:,.0f} | Days on Hand: {row['days_on_hand']:.0f}</div>", unsafe_allow_html=True)

        with tab3:
            st.subheader("EOQ & Reorder Cost Optimization")
            order_cost = st.slider("Avg Cost per Order (₦)", 500, 10000, 2500, 500)
            eoq = np.sqrt((2 * df['annual_demand'] * order_cost) / (df['unit_cost_ngn'] * 0.25 + 1e-6))
            optimized_holding = (eoq / 2 * df['unit_cost_ngn'] * 0.25).sum()
            ordering_cost = (df['annual_demand'] / eoq * order_cost).sum()
            total_optimized = optimized_holding + ordering_cost
            savings = total_holding - optimized_holding

            st.markdown(f"<p class='save big-font'>Potential Savings: ₦{savings:,.0f}/year</p>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.metric("Current Holding Cost", f"₦{total_holding:,.0f}")
            col2.metric("Optimized Annual Cost", f"₦{total_optimized:,.0f}")

        with tab4:
            if st.button("Generate Executive Report"):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4)
                styles = getSampleStyleSheet()
                story = [Paragraph("Inventory Risk Pro – Executive Report", styles['Title']),
                         Spacer(1, 20),
                         Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']),
                         Spacer(1, 30)]

                data = [["Metric", "Value"],
                        ["Valid SKUs Analyzed", f"{len(df):,}"],
                        ["A-Class Items", a_items],
                        ["Cash-at-Risk", f"₦{total_cash_at_risk:,.0f}"],
                        ["Potential Savings", f"₦{savings:,.0f}"]]
                story.append(Table(data))
                doc.build(story)
                buffer.seek(0)
                st.download_button("Download Report", buffer, "inventory_executive_report.pdf", "application/pdf")

        with tab5:
            st.download_button("Export Cleaned & Enriched Data", df.to_csv(index=False).encode(), "inventory_clean_optimized.csv", "text/csv")

    except Exception as e:
        st.error(f"Upload failed: {e}. Check file format and try again.")
else:
    st.info("Upload your inventory file — even messy Excel exports work!")

st.caption("Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria • January 2026")
