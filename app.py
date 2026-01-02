# ============================================================
# app.py — Inventory Risk Pro (FINAL PROFESSIONAL & CLEAN)
# Clean Landing Page • Tabs Visible • Production-Ready
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

# Main uploader — clean and prominent
uploaded_file = st.file_uploader("📁 Upload your inventory file (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        # Load file with robust encoding
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

        # Friendly handling for wrong data
        if missing_core:
            st.warning("📊 This file appears to be sales transaction data (invoices, orders), not current inventory levels.")
            st.info("""
**No problem — this is very common!**

This app needs a snapshot of your **current stock** (how many units you have now).

We can build a custom tool using your sales data for forecasting, profitability, or full dashboard.

**Interested in a tailored solution?**  
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

        # Proceed with correct data
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

        st.success(f"✅ Analysis complete — {len(df):,} products processed")

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

        # Dashboard metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Products Analyzed", f"{len(df):,}")
        col2.metric("A-Class Items", (df['abc_class']=='A').sum())
        col3.metric("Cash-at-Risk (₦)", f"{total_cash_risk:,.0f}")
        col4.metric("Stockout Risk Items", df['stockout_risk'].sum())

        st.markdown("---")

        # TABS — FULLY RESTORED
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 ABC Classification", "⚠️ Risk Items", "💰 Cost Simulator", "📄 Executive Report", "📈 Export Data"
        ])

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("ABC Distribution")
                fig, ax = plt.subplots(figsize=(6, 5))
                df['abc_class'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=['#4caf50', '#ff9800', '#f44336'])
                ax.set_ylabel('')
                st.pyplot(fig)
            with col2:
                st.subheader("Risk Breakdown")
                stockout = df['stockout_risk'].sum()
                over = df['overstock_risk'].sum()
                slow = df['slow_moving'].sum()
                dead = df['dead_stock'].sum()
                healthy = len(df) - (stockout + over + slow + dead)
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.pie([healthy, stockout, over, slow, dead], labels=['Healthy', 'Stockout', 'Overstock', 'Slow', 'Dead'], autopct='%1.1f%%', colors=sns.color_palette("viridis", 5))
                ax.set_title("Inventory Health")
                st.pyplot(fig)

        with tab2:
            st.subheader("Items Needing Immediate Action")
            action_items = df[df['stockout_risk'] | df['dead_stock'] | df['overstock_risk'] | df['slow_moving'] | (df['abc_class']=='A')]
            action_items = action_items.sort_values('holding_cost_ngn', ascending=False)

            display_cols = ['product_name', 'current_stock', 'days_on_hand', 'reorder_point', 'holding_cost_ngn', 'recommendation']
            display = action_items[display_cols].copy()
            display['holding_cost_ngn'] = display['holding_cost_ngn'].apply(lambda x: f"₦{x:,.0f}")
            st.dataframe(display.head(50), use_container_width=True)

            st.write("**Top 10 Priority Items**")
            for _, row in action_items.head(10).iterrows():
                with st.expander(f"{row['product_name']} — {'🔴 Urgent' if row['stockout_risk'] else '🟡 Review'}"):
                    st.markdown(f"<div class='recommendation'>{row['recommendation']}</div>", unsafe_allow_html=True)

        with tab3:
            st.subheader("EOQ & Reorder Cost Optimization Simulator")
            order_cost = st.slider("Average Cost per Purchase Order (₦)", 500, 10000, 3000, 500)
            eoq = np.sqrt(2 * df['avg_daily_sales'] * 365 * order_cost / (df['unit_cost_ngn'] * 0.25 + 1e-6))
            optimized_holding = (eoq / 2 * df['unit_cost_ngn'] * 0.25).sum()
            ordering_cost = (df['avg_daily_sales'] * 365 / eoq * order_cost).sum()
            total_optimized = optimized_holding + ordering_cost
            savings = total_holding - optimized_holding

            st.markdown(f"<p class='save big-font'>Potential Annual Savings: ₦{savings:,.0f}</p>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Holding Cost", f"₦{total_holding:,.0f}")
            col2.metric("Optimized Total Cost", f"₦{total_optimized:,.0f}")
            col3.metric("Savings", f"₦{savings:,.0f}")

        with tab4:
            st.subheader("Executive Report")
            if st.button("Generate PDF Report"):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=inch)
                styles = getSampleStyleSheet()
                story = []

                story.append(Paragraph("Inventory Risk Pro – Executive Report", styles['Title']))
                story.append(Spacer(1, 20))
                story.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
                story.append(Paragraph("Prepared by: Freda Erinmwingbovo", styles['Normal']))
                story.append(Spacer(1, 30))

                metrics_data = [
                    ["Metric", "Value"],
                    ["Products Analyzed", f"{len(df):,}"],
                    ["A-Class Items", (df['abc_class']=='A').sum()],
                    ["Cash-at-Risk", f"₦{total_cash_risk:,.0f}"],
                    ["Potential Annual Savings", f"₦{savings:,.0f}"]
                ]
                t = Table(metrics_data, colWidths=[3*inch, 2.5*inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e88e5")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige)
                ]))
                story.append(t)

                doc.build(story)
                buffer.seek(0)
                st.download_button("⬇️ Download Executive Report", buffer, "inventory_executive_report.pdf", "application/pdf")

        with tab5:
            st.subheader("Export Enriched Dataset")
            st.write("All original + calculated columns included (reorder_point, risks, holding_cost, recommendation, abc_class, cash_at_risk, etc.)")
            st.download_button(
                "⬇️ Download Full Analysis CSV",
                df.to_csv(index=False).encode('utf-8'),
                "inventory_risk_pro_complete_analysis.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"File processing error: {e}")

else:
    # Clean, professional landing page
    st.info("👆 Drag and drop your inventory file here — Excel or CSV supported")

    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        uploaded_file = st.file_uploader("", type=["csv", "xlsx", "xls"], label_visibility="collapsed")

    with st.expander("📋 Show data requirements"):
        st.markdown("""
**Required columns** (case-insensitive):
- `product_id`  
- `product_name`  
- `current_stock`  
- `avg_daily_sales`  
- `unit_cost_ngn`

**Optional** (defaults used if missing):
- `lead_time_days` → 14 days  
- `safety_stock_days` → 7 days
        """)

    with st.expander("🚀 Try the app instantly with sample data"):
        st.write("Load sample inventory to explore all features right now:")
        if st.button("Load Sample Data & Run Analysis"):
            sample_data = {
                "product_id": [101, 102, 103, 104, 105],
                "product_name": ["Wireless Mouse", "USB Cable", "Laptop Stand", "Webcam", "External HDD"],
                "current_stock": [45, 120, 18, 8, 32],
                "avg_daily_sales": [8, 15, 3, 2, 5],
                "unit_cost_ngn": [12000, 3000, 45000, 75000, 80000],
                "lead_time_days": [10, 5, 21, 14, 30],
                "safety_stock_days": [5, 3, 7, 5, 10]
            }
            df = pd.DataFrame(sample_data)
            st.session_state.sample_loaded = True
            st.rerun()

st.caption("Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria • January 2026")
