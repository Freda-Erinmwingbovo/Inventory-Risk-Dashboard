# ============================================================
# app.py — Inventory Risk Pro (FLAGSHIP VERSION)
# Optimize Stock • Reduce Costs • Prevent Stockouts & Overstock
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

# ---------------- CONFIG & STYLE ----------------
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
st.markdown("<p style='text-align: center; font-size: 22px;'>AI-Powered Inventory Optimization • Cost Savings • Risk Prevention</p>", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("📁 Upload your inventory CSV", type="csv")

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.success(f"✅ Successfully loaded {len(df):,} products")

        required = ['product_id', 'product_name', 'current_stock', 'avg_daily_sales',
                    'unit_cost_ngn', 'lead_time_days', 'safety_stock_days']
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"🚫 Missing required columns: {', '.join(missing)}")
            st.stop()

        # ---------------- CALCULATIONS ----------------
        df['days_on_hand'] = df['current_stock'] / (df['avg_daily_sales'] + 0.01)
        df['reorder_point'] = df['avg_daily_sales'] * (df['lead_time_days'] + df['safety_stock_days'])
        df['stockout_risk'] = df['current_stock'] < df['reorder_point']
        df['overstock_risk'] = df['days_on_hand'] > 180
        df['slow_moving'] = (df['days_on_hand'] > 90) & (df['days_on_hand'] <= 365)
        df['dead_stock'] = df['days_on_hand'] > 365

        df['holding_cost_ngn'] = df['current_stock'] * df['unit_cost_ngn'] * 0.25  # 25% annual holding rate
        df['potential_savings_ngn'] = np.where(df['dead_stock'], df['holding_cost_ngn'], 0)

        total_holding_cost = df['holding_cost_ngn'].sum()
        dead_stock_cost = df[df['dead_stock']]['holding_cost_ngn'].sum()
        stockout_count = df['stockout_risk'].sum()
        overstock_count = df['overstock_risk'].sum()

        # Personalized Recommendations
        def generate_recommendation(row):
            recs = []
            if row['stockout_risk']:
                reorder_qty = row['reorder_point'] - row['current_stock']
                recs.append(f"**URGENT:** Reorder {reorder_qty:.0f} units immediately to avoid stockout")
            if row['dead_stock']:
                recs.append("**Liquidate or discount** – dead stock tying up capital")
            if row['overstock_risk']:
                recs.append("**Reduce future orders** – high overstock risk")
            if row['slow_moving']:
                recs.append("**Promote or bundle** to clear slow-moving inventory")
            if not recs:
                recs.append("Stock level healthy – maintain current policy")
            return "<br>".join(recs)

        df['recommendation'] = df.apply(generate_recommendation, axis=1)

        # ---------------- DASHBOARD METRICS ----------------
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Products", f"{len(df):,}")
        col2.metric("At Stockout Risk", f"{stockout_count}", delta_color="inverse")
        col3.metric("Dead Stock Cost (₦)", f"{dead_stock_cost:,.0f}", delta_color="inverse")
        col4.metric("Total Annual Holding Cost (₦)", f"{total_holding_cost:,.0f}", delta_color="inverse")

        col5, col6 = st.columns(2)
        col5.metric("Overstock Risk Items", f"{overstock_count}")
        col6.metric("Potential Savings from Dead Stock (₦)", f"{dead_stock_cost:,.0f}", delta_color="normal")

        st.markdown("---")

        # ---------------- TABS ----------------
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "⚠️ Risk Overview", "🧠 Insights & Recommendations", "💰 What-If Simulator", "📄 Executive Report", "📊 Download Data"
        ])

        # TAB 1: Risk Overview
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Days on Hand Distribution")
                fig, ax = plt.subplots(figsize=(8, 5))
                sns.histplot(df['days_on_hand'], bins=50, kde=True, color="#1e88e5", ax=ax)
                ax.axvline(90, color='orange', linestyle='--', label="Slow-Moving Threshold")
                ax.axvline(365, color='red', linestyle='--', label="Dead Stock Threshold")
                ax.legend()
                st.pyplot(fig)

            with col2:
                st.subheader("Inventory Health Breakdown")
                labels = ['Healthy', 'Slow-Moving', 'Dead Stock', 'Stockout Risk', 'Overstock Risk']
                sizes = [
                    len(df) - df['slow_moving'].sum() - df['dead_stock'].sum() - df['stockout_risk'].sum() - df['overstock_risk'].sum(),
                    df['slow_moving'].sum(),
                    df['dead_stock'].sum(),
                    df['stockout_risk'].sum(),
                    df['overstock_risk'].sum()
                ]
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=sns.color_palette("viridis", 5))
                ax.set_title("Inventory Status")
                st.pyplot(fig)

        # TAB 2: Insights & Recommendations
        with tab2:
            st.subheader("Products Needing Attention")
            risk_df = df[df['stockout_risk'] | df['dead_stock'] | df['overstock_risk'] | df['slow_moving']].copy()
            risk_df = risk_df.sort_values('holding_cost_ngn', ascending=False)

            display_cols = ['product_name', 'current_stock', 'days_on_hand', 'holding_cost_ngn', 'recommendation']
            display_df = risk_df[display_cols].copy()
            display_df['holding_cost_ngn'] = display_df['holding_cost_ngn'].apply(lambda x: f"₦{x:,.0f}")
            display_df['days_on_hand'] = display_df['days_on_hand'].round(1)

            st.dataframe(display_df.head(50), use_container_width=True, height=600)

            for _, row in risk_df.head(10).iterrows():
                with st.expander(f"🔴 {row['product_name']} – Holding Cost: ₦{row['holding_cost_ngn']:,.0f}"):
                    st.write(f"**Current Stock:** {row['current_stock']:,} | **Days on Hand:** {row['days_on_hand']:.0f}")
                    st.markdown(f"<div class='recommendation'>{row['recommendation']}</div>", unsafe_allow_html=True)

        # TAB 3: What-If Simulator
        with tab3:
            st.subheader("Policy Change Simulator")
            col1, col2 = st.columns(2)
            with col1:
                new_lead_time = st.slider("Average Lead Time (days)", 1, 90, int(df['lead_time_days'].mean()))
            with col2:
                new_safety_days = st.slider("Safety Stock Buffer (days)", 1, 60, int(df['safety_stock_days'].mean()))

            simulated = df.copy()
            simulated['reorder_point'] = simulated['avg_daily_sales'] * (new_lead_time + new_safety_days)
            simulated['stockout_risk'] = simulated['current_stock'] < simulated['reorder_point']
            new_stockout_count = simulated['stockout_risk'].sum()

            delta_risk = new_stockout_count - stockout_count
            st.markdown(f"<p class='big-font {"save" if delta_risk < 0 else "risk"}'>Projected Stockout Risk: {new_stockout_count} products ({'↓' if delta_risk < 0 else '↑'}{abs(delta_risk)})</p>", unsafe_allow_html=True)

            if delta_risk < 0:
                st.success(f"Safer policy – reduces stockout risk by {abs(delta_risk)} items")
            elif delta_risk > 0:
                st.warning(f"Riskier policy – increases stockouts by {delta_risk} items")
            else:
                st.info("No change in stockout risk")

        # TAB 4: Executive Report
        with tab4:
            if st.button("📄 Generate Executive PDF Report"):
                buffer = io.BytesIO()
                doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=inch)
                styles = getSampleStyleSheet()
                story = []

                story.append(Paragraph("Inventory Risk Pro – Executive Report", styles['Title']))
                story.append(Spacer(1, 20))
                story.append(Paragraph(f"Generated: {datetime.now().strftime('%B %d, %Y')}", styles['Normal']))
                story.append(Paragraph("Prepared by: Freda Erinmwingbovo", styles['Normal']))
                story.append(Spacer(1, 30))

                # Metrics Table
                metrics_data = [
                    ["Metric", "Value"],
                    ["Total Products", f"{len(df):,}"],
                    ["Products at Stockout Risk", f"{stockout_count}"],
                    ["Dead Stock Holding Cost", f"₦{dead_stock_cost:,.0f}"],
                    ["Total Annual Holding Cost", f"₦{total_holding_cost:,.0f}"],
                    ["Potential Savings (Dead Stock)", f"₦{dead_stock_cost:,.0f}"]
                ]
                t = Table(metrics_data, colWidths=[3*inch, 2.5*inch])
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#1e88e5")),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('GRID', (0,0), (-1,-1), 1, colors.black),
                    ('BACKGROUND', (0,1), (-1,-1), colors.beige)
                ]))
                story.append(t)
                story.append(Spacer(1, 30))

                # Top 10 Risk Items
                story.append(Paragraph("Top 10 Highest-Risk Products", styles['Heading2']))
                top10 = risk_df.head(10)[['product_name', 'days_on_hand', 'holding_cost_ngn', 'recommendation']].copy()
                top10['holding_cost_ngn'] = top10['holding_cost_ngn'].apply(lambda x: f"₦{x:,.0f}")

                risk_data = [["Product", "Days on Hand", "Holding Cost", "Action"]]
                for _, r in top10.iterrows():
                    action = r['recommendation'].replace("<br>", ": ").replace("**", "").strip()
                    risk_data.append([r['product_name'], f"{r['days_on_hand']:.0f}", r['holding_cost_ngn'], action[:80] + ("..." if len(action) > 80 else "")])

                rt = Table(risk_data, colWidths=[2*inch, 1*inch, 1.2*inch, 2*inch])
                rt.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,0), colors.grey),
                    ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                    ('GRID', (0,0), (-1,-1), 0.5, colors.black),
                    ('FONTSIZE', (0,0), (-1,-1), 9)
                ]))
                story.append(rt)

                doc.build(story)
                buffer.seek(0)
                st.download_button("⬇️ Download Executive PDF", buffer, "inventory_risk_report.pdf", "application/pdf")

        # TAB 5: Download Data
        with tab5:
            st.subheader("Download Enriched Dataset")
            st.download_button(
                "⬇️ Download Full Analysis CSV (with risks, costs & recommendations)",
                df.to_csv(index=False).encode(),
                "inventory_optimized_analysis.csv",
                "text/csv"
            )

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("👆 Upload your inventory CSV to begin optimization.")
    st.markdown("""
    **Required columns**:
    - `product_id`
    - `product_name`
    - `current_stock`
    - `avg_daily_sales`
    - `unit_cost_ngn`
    - `lead_time_days`
    - `safety_stock_days`
    """)

st.caption("Built with ❤️ by Freda Erinmwingbovo • Abuja, Nigeria • January 2026")
