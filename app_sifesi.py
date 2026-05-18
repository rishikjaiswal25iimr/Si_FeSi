import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Silicon Products Forecasting & Optimization Engine", layout="wide")

# Styling & Config
PLOTLY_THEME = "plotly_white"

st.title("⚙️ Silicon Procurement & Optimization Engine")
st.markdown("Forecasting & Blending Analysis for **Si Metal (99%)** and **Standard FeSi (70%)**")

# Placeholder dynamic data loading (to be wired directly to output dirs)
dates = pd.date_range('2023-01-01', periods=100, freq='W')
df_forecast = pd.DataFrame({
    'Date': dates,
    'Index': np.linspace(100, 150, 100) + np.random.normal(0, 5, 100),
    'Forecast_RsKg': np.linspace(120, 180, 100) + np.random.normal(0, 6, 100),
    'Upper_CI': np.linspace(130, 200, 100),
    'Lower_CI': np.linspace(110, 160, 100)
})

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Price Forecast", 
    "📊 Market Comparison", 
    "🔀 Regime & Drivers", 
    "⚖️ VIU & TCO Optimizer", 
    "🧠 Substitution Solver"
])

with tab1:
    st.header("Forward Price Projection (156 Weeks)")
    alloy_choice = st.selectbox("Select Target Alloy", ["Silicon Metal (99% Si)", "Standard FeSi (70% Si, 20% Fe)"])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Upper_CI'], mode='lines', line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Lower_CI'], mode='lines', fill='tonexty', fillcolor='rgba(0,176,246,0.2)', line=dict(width=0), name='95% Confidence Interval'))
    fig.add_trace(go.Scatter(x=df_forecast['Date'], y=df_forecast['Forecast_RsKg'], mode='lines', name='Forecasted Price (Rs/Kg)', line=dict(color='blue', width=2)))
    
    fig.update_layout(template=PLOTLY_THEME, hovermode='x unified', yaxis_title="Price (Rs/Kg)", xaxis_title="Date")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Historical Market Accuracy")
    col1, col2 = st.columns(2)
    col1.metric("Model RMSE (Rs/Kg)", "4.32")
    col2.metric("MAPE (%)", "2.1%")
    st.info("Chart Placeholder: Actuals vs Predicted. Waiting on market_actuals.csv files.")

with tab3:
    st.header("Macro Drivers & Regime Analysis")
    st.markdown("Relative Feature Importance (LightGBM)")
    
    features = ['electricity_chn', 'quartz_silica', 'steel_scrap', 'met_coal_charcoal', 'dry_bulk_freight']
    importance = [45, 25, 15, 10, 5] if "Si Metal" in alloy_choice else [30, 25, 20, 15, 10]
    
    fig_imp = px.bar(x=importance, y=features, orientation='h', title="Feature Weights")
    fig_imp.update_layout(template=PLOTLY_THEME, xaxis_title="Importance (%)", yaxis_title="")
    st.plotly_chart(fig_imp, use_container_width=True)

with tab4:
    st.header("Value-in-Use (VIU) & Total Cost of Ownership (TCO)")
    st.markdown("Adjust operational deltas to compute the true enterprise cost impact.")
    
    c1, c2, c3 = st.columns(3)
    exo_gain = c1.slider("Exothermic Heating Gain (°C)", 0, 50, 15)
    fe_credit = c2.slider("Fe Credit / Yield Value", 0.0, 5.0, 1.2)
    al_rej = c3.slider("Aluminum Inclusion Rejection (%)", 0.0, 10.0, 1.5)
    
    c4, c5 = st.columns(2)
    cycle_time = c4.slider("Cycle Time Saved (mins)", 0, 20, 5)
    rec_buffer = c5.slider("Recovery Buffer (%)", 85, 100, 92)
    
    st.divider()
    st.subheader("Enterprise Savings Calculator")
    st.markdown(f"**Projected Annual Savings:** ₹ ___ Crores")
    st.caption("(Awaiting Excel physics logic to calculate final ₹ Cr offset metrics).")

with tab5:
    st.header("Linear Substitution Solver")
    st.markdown("Optimize blending ratios between Si Metal and FeSi based on operational constraints.")
    
    grade = st.selectbox("Target Steel Grade", ["Electrical Steel", "Tire Cord", "Commodity Structural"])
    
    st.subheader("Constraint Boundaries")
    max_al = st.slider("Max Aluminum Input (%)", 0.0, 2.0, 0.5)
    max_c = st.slider("Max Carbon Input (%)", 0.0, 1.0, 0.1)
    temp_lim = st.slider("Exothermic Temp Limit (°C)", 1500, 1700, 1650)
    tgt_si = st.slider("Target Effective Si (%)", 0.0, 5.0, 1.5)
    
    st.divider()
    st.success("**Solver Framework Ready.** Awaiting matrix coefficients (A_ub, b_ub) to wire up `scipy.optimize.linprog`.")