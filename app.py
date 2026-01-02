# ============================================================
# app.py — Inventory Risk Pro
# Real Inventory Data • Stock Optimization • ₦ Impact
# Built by Freda Erinmwingbovo • Abuja, Nigeria • January 2026
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from io import BytesIO

st.set_page_config(page_title="Inventory Risk Pro", page_icon="📦", layout="wide")

st.markdown("""
<style>
    .big-font {font-size: 50px !important; font-weight: bold;}
    .risk {color: #d32f2f;}
    .save {color: #388e3c;}
    .metric-card {background-color: #f8f9fa; padding: 20px; border-radius: 15px; box-shadow: 0 6px 12px rgba(0,0,0,0.1);}
    h1 {color: #1e88e5;}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center;'>📦 Inventory Risk Pro</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 22px;'>Optimize Stock • Reduce Costs • Prevent Stockouts</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload your inventory CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} products")

        required = ['product_id', 'product_name', 'current_stock', 'avg_daily_sales', 'unit_cost_ngn', 'lead_time_days', 'safety_stock_days']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        # Calculations
        df['days_on_hand'] = df['current_stock'] / (df['avg_daily_sales'] + 0.01)
        df['reorder_point'] = df['avg_daily_sales'] * df['lead_time_days'] + df['avg_daily_sales'] * df['safety_stock_days']
        df['stockout_risk'] = df['current_stock'] < df['reorder_point']
        df['slow_moving'] = df['days_on_hand'] > 90
        df['dead_stock'] = df['days_on_hand'] > 365
        df['holding_cost_ngn'] = df['current_stock'] * df['unit_cost_ngn'] * 0.25  # 25% annual holding rate

        total_holding_cost = df['holding_cost_ngn'].sum()
        dead_stock_cost = df[df['dead_stock']]['holding_cost_ngn'].sum()
        stockout_products = df[df['stockout_risk']]['product_name'].nunique()

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Products", len(df))
        col2.metric("Dead Stock Cost (₦)", f"{dead_stock_cost:,.0f}", delta_color="inverse")
        col3.metric("Total Holding Cost (₦)", f"{total_holding_cost:,.0f}", delta_color="inverse")
        col4.metric("Products at Stockout Risk", stockout_products, delta_color="inverse")

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            sns.histplot(df['days_on_hand'], bins=50, kde=True, ax=ax)
            ax.set_title("Distribution of Days on Hand")
            st.pyplot(fig)
        with col2:
            status = ['Normal', 'Slow-Moving', 'Dead Stock', 'Stockout Risk']
            counts = [
                len(df) - df['slow_moving'].sum() - df['dead_stock'].sum() - df['stockout_risk'].sum(),
                df['slow_moving'].sum(),
                df['dead_stock'].sum(),
                df['stockout_risk'].sum()
            ]
            fig, ax = plt.subplots()
            ax.pie(counts, labels=status, autopct='%1.1f%%', colors=sns.color_palette("viridis"))
            ax.set_title("Inventory Status Breakdown")
            st.pyplot(fig)

        # Reorder Recommendations
        st.subheader("Auto-Reorder Recommendations")
        reorder = df[df['current_stock'] < df['reorder_point']].copy()
        reorder['reorder_qty'] = reorder['reorder_point'] - reorder['current_stock']
        if not reorder.empty:
            st.dataframe(reorder[['product_name', 'current_stock', 'reorder_point', 'reorder_qty', 'unit_cost_ngn']].head(20))
            st.write(f"**Recommend reordering {len(reorder)} products** to avoid stockouts")
        else:
            st.success("No immediate reorder needed — stock levels healthy")

        # What-If Simulator
        st.subheader("What-If Simulator")
        col1, col2 = st.columns(2)
        with col1:
            new_lead_time = st.slider("Change Lead Time (days)", 1, 60, int(df['lead_time_days'].mean()))
        with col2:
            new_safety = st.slider("Change Safety Stock (days)", 1, 30, int(df['safety_stock_days'].mean()))

        simulated = df.copy()
        simulated['reorder_point'] = simulated['avg_daily_sales'] * new_lead_time + simulated['avg_daily_sales'] * new_safety
        simulated['stockout_risk'] = simulated['current_stock'] < simulated['reorder_point']
        new_risk_count = simulated['stockout_risk'].sum()
        st.metric("Projected Stockout Risk Products", new_risk_count, delta=new_risk_count - stockout_products)

        # PDF Report
        st.subheader("Download Report")
        if st.button("Generate PDF Report"):
            buffer = BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []

            elements.append(Paragraph("Inventory Risk Report", styles['Title']))
            elements.append(Spacer(1, 20))
            elements.append(Paragraph(f"Date: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
            elements.append(Spacer(1, 20))

            data = [["Metric", "Value"],
                    ["Total Products", len(df)],
                    ["Dead Stock Cost", f"₦{dead_stock_cost:,.0f}"],
                    ["Total Holding Cost", f"₦{total_holding_cost:,.0f}"],
                    ["Stockout Risk Products", stockout_products]]
            t = Table(data)
            t.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('GRID', (0,0), (-1,-1), 1, colors.black)]))
            elements.append(t)

            doc.build(elements)
            buffer.seek(0)
            st.download_button("Download PDF", buffer, "inventory_report.pdf", "application/pdf")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("Upload your inventory CSV to begin")
    st.markdown("**Required columns**: product_id, product_name, current_stock, avg_daily_sales, unit_cost_ngn, lead_time_days, safety_stock_days")

st.caption("Built by Freda Erinmwingbovo • Abuja, Nigeria • January 2026")
