#!/usr/bin/env python3
"""
SmartContainer Risk Engine - Streamlit Dashboard App
Interactive UI for running the risk engine and viewing results.

Usage:
    streamlit run app.py
"""
import streamlit as st
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="SmartContainer Risk Engine",
    page_icon="🚢",
    layout="wide",
)

st.title("🚢 SmartContainer Risk Engine")
st.caption("AI-powered container risk assessment for customs inspection")

with st.sidebar:
    st.header("Controls")
    uploaded_hist = st.file_uploader("Upload Historical CSV", type='csv', key='hist')
    uploaded_rt = st.file_uploader("Upload Real-Time CSV", type='csv', key='rt')
    run_btn = st.button("Run Risk Assessment", type="primary")
    st.divider()
    st.subheader("Model Info")
    st.metric("Model", "RF + IsolationForest")
    st.metric("Features", "26")
    st.metric("Training Data", "~45,519 containers")
    st.divider()
    st.subheader("Risk Thresholds")
    st.write("- **Low Risk**: 0-30")
    st.write("- **Medium Risk**: 31-60")
    st.write("- **Critical**: 61-100")

if run_btn:
    with st.spinner("Running full pipeline... this may take a minute."):
        hist_path = None
        rt_path = None
        if uploaded_hist is not None:
            hist_path = 'data/uploaded_historical.csv'
            os.makedirs('data', exist_ok=True)
            with open(hist_path, 'wb') as f:
                f.write(uploaded_hist.getvalue())
        if uploaded_rt is not None:
            rt_path = 'data/uploaded_realtime.csv'
            os.makedirs('data', exist_ok=True)
            with open(rt_path, 'wb') as f:
                f.write(uploaded_rt.getvalue())
        from pipeline import run_pipeline
        try:
            results = run_pipeline(hist_path, rt_path)
        except Exception as e:
            st.error("Pipeline failed: " + str(e))
            st.stop()

    st.success("Pipeline completed successfully!")
    results = pd.read_csv('outputs/predictions.csv')

    col1, col2, col3, col4 = st.columns(4)
    total = len(results)
    critical_n = int((results['Risk_Level'] == 'Critical').sum())
    medium_n = int((results['Risk_Level'] == 'Medium Risk').sum())
    low_n = int((results['Risk_Level'] == 'Low Risk').sum())
    col1.metric("Total Processed", str(total))
    col2.metric("🔴 Critical", str(critical_n))
    col3.metric("🟡 Medium Risk", str(medium_n))
    col4.metric("🟢 Low Risk", str(low_n))

    st.subheader("Container Risk Predictions")
    def color_risk(val):
        colors = {
            'Critical': 'background-color: #ffcccc; color: #cc0000; font-weight: bold',
            'Medium Risk': 'background-color: #fff3cd; color: #856404; font-weight: bold',
            'Low Risk': 'background-color: #d4edda; color: #155724; font-weight: bold',
        }
        return colors.get(val, '')
    styled = results.style.applymap(color_risk, subset=['Risk_Level'])
    st.dataframe(styled, use_container_width=True, height=400)

    csv_data = results.to_csv(index=False)
    st.download_button("📥 Download predictions.csv", csv_data, "predictions.csv", "text/csv")

    st.subheader("Dashboard")
    dashboard_path = 'outputs/summary_report.png'
    if os.path.exists(dashboard_path):
        st.image(dashboard_path, use_container_width=True)
    else:
        st.warning("Dashboard image not found at " + dashboard_path)

    st.subheader("Top 10 Riskiest Containers")
    top10 = results.nlargest(10, 'Risk_Score')
    st.dataframe(top10, use_container_width=True)

else:
    st.info("Click **Run Risk Assessment** in the sidebar to start the pipeline. Upload custom CSVs or use the default data files in the repository.")
    if os.path.exists('outputs/predictions.csv'):
        st.subheader("Previous Results")
        prev = pd.read_csv('outputs/predictions.csv')
        st.write("Found " + str(len(prev)) + " predictions from a previous run.")
        def color_risk_prev(val):
            colors = {
                'Critical': 'background-color: #ffcccc',
                'Medium Risk': 'background-color: #fff3cd',
                'Low Risk': 'background-color: #d4edda',
            }
            return colors.get(val, '')
        st.dataframe(prev.style.applymap(color_risk_prev, subset=['Risk_Level']), use_container_width=True, height=300)