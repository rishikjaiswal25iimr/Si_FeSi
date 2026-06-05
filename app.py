"""
INTEGRATED DASHBOARD – FeMn & FeSi Substitutions
================================================
Tab 1: Value-in-Use (VIU) Dashboard
Tab 2: Substitution Solver (Linear Programming)

Combines:
1. LC FeMn vs Mn Briquette
2. MC FeMn vs Mn Briquette
3. FeSi70 vs Si Metal

All formulas, calculations, limits, and optimization equations 
are sourced strictly from their original logic and kept completely unmodified.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import linprog

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VIU & Solver Dashboard",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Combined Colour palette
C_LCFEMN   = "#2196F3"   # blue  – LC FeMn
C_EMM      = "#4CAF50"   # green – Mn Briquette / EMM
C_MCFEMN   = "#2196F3"   # blue  – MC FeMn
C_BRIQ     = "#4CAF50"   # green – Mn Briquette
C_FESI     = "#607D8B"   # blue-grey  – FeSi70
C_SIMETAL  = "#009688"   # teal       – Si Metal
C_FESI_SOLVER     = "#2196F3"   # blue  – FeSi70 (Solver)
C_SIMETAL_SOLVER  = "#4CAF50"   # green – Si Metal (Solver)
C_DELTA    = "#FF9800"   # amber – delta / benefit
C_NEG      = "#F44336"   # red   – penalties / negative
C_GRID     = "#EEEEEE"
C_BG       = "#FAFAFA"
C_TEXT     = "#333333"
C_FEV      = "#2196F3"   # blue  – FeV80
C_NV       = "#4CAF50"   # green – Nitrovan
C_FEV_SOLVER = "#1976D2" # dark blue – FeV80 Solver
C_NV_SOLVER  = "#E64A19" # deep orange – Nitrovan Solver
C_CARD_BG  = "#FFFFFF"

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- page background ---------- */
.stApp { background: #F0F4F8; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A237E 0%, #283593 40%, #1565C0 100%);
}
[data-testid="stSidebar"] * { color: #E8EAF6 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #5C6BC0 !important; }
[data-testid="stSidebar"] hr { border-color: #3949AB; }
[data-testid="stSidebar"] .stNumberInput input { background: #283593; border-color: #5C6BC0; color: #fff !important; }
[data-testid="stSidebar"] .stSelectbox select { background: #283593; color: #fff; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label { color: #fff !important; }

/* ---------- KPI cards ---------- */
.kpi-card {
    background: #FFFFFF;
    border-radius: 12px;
    padding: 18px 22px 14px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    border-left: 5px solid #2196F3;
    margin-bottom: 8px;
}
.kpi-card-green  { border-left-color: #4CAF50; }
.kpi-card-teal   { border-left-color: #009688; }
.kpi-card-amber  { border-left-color: #FF9800; }
.kpi-card-red    { border-left-color: #F44336; }
.kpi-card-purple { border-left-color: #9C27B0; }
.kpi-label { font-size: 12px; font-weight: 600; color: #78909C; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.kpi-value { font-size: 26px; font-weight: 700; color: #1A237E; line-height: 1.15; }
.kpi-sub   { font-size: 12px; color: #90A4AE; margin-top: 3px; }

/* ---------- section headers ---------- */
.section-header {
    font-size: 20px; font-weight: 800; color: #1A237E;
    text-transform: uppercase; letter-spacing: 0.05em;
    border-bottom: 3px solid #2196F3;
    padding-bottom: 8px; margin-bottom: 24px; margin-top: 32px;
}

/* ---------- info boxes ---------- */
.info-box {
    background: #E3F2FD; border-radius: 8px;
    padding: 12px 16px; font-size: 13px; color: #1565C0;
    border-left: 4px solid #2196F3; margin-bottom: 10px;
}
.warn-box {
    background: #FFF3E0; border-radius: 8px;
    padding: 12px 16px; font-size: 13px; color: #E65100;
    border-left: 4px solid #FF9800; margin-bottom: 10px;
}
.success-box {
    background: #E8F5E9; border-radius: 8px;
    padding: 12px 16px; font-size: 13px; color: #1B5E20;
    border-left: 4px solid #4CAF50; margin-bottom: 10px;
}
.solver-kpi-box {
    background: #FFFFFF; border-radius: 12px;
    padding: 18px 22px; border: 1px solid #E2E8F0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}

/* Tabs Styling overrides */
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 60px;
    white-space: pre-wrap;
    padding-top: 10px;
    padding-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Plotly layout templates & KPIs
# ══════════════════════════════════════════════════════════════════════════════
def _layout(title: str, y_title: str = "", height: int = 420) -> dict:
    return dict(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor=C_BG,
        font=dict(family="Inter, sans-serif", size=12, color=C_TEXT),
        title=dict(text=title, font=dict(size=15, color="#1A237E"), x=0.01),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#DDD", borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False, title=y_title),
        hovermode="x unified",
        height=height,
        margin=dict(l=60, r=30, t=55, b=45),
    )

def _layout_viu(title: str, y_title: str = "", height: int = 420) -> dict:
    return _layout(title, y_title, height)

def _layout_solver(title: str, y_title: str = "", height: int = 380) -> dict:
    return dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color=C_TEXT),
        title=dict(text=title, font=dict(size=14, color="#1A237E"), x=0.01),
        xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False, title=y_title),
        hovermode="closest",
        height=height,
        margin=dict(l=50, r=20, t=40, b=40),
    )

def _layout_viu_fesi(title: str, y_title: str = "", height: int = 420) -> dict:
    return dict(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor=C_BG,
        font=dict(family="Inter, sans-serif", size=12, color=C_TEXT),
        title=dict(text=title, font=dict(size=15, color="#263238"), x=0.01),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#DDD", borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False, title=y_title),
        hovermode="x unified",
        height=height,
        margin=dict(l=60, r=30, t=55, b=45),
    )

def _layout_solver_fesi(title: str, y_title: str = "", height: int = 420) -> dict:
    return dict(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor=C_BG,
        font=dict(family="Inter, sans-serif", size=12, color=C_TEXT),
        title=dict(text=title, font=dict(size=15, color="#1A237E"), x=0.01),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#DDD", borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False, title=y_title),
        hovermode="x unified",
        height=height,
        margin=dict(l=60, r=30, t=55, b=45),
    )

def kpi(label: str, value: str, sub: str = "", colour: str = "") -> str:
    cls = f"kpi-card {colour}"
    return f"""
    <div class="{cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""

# Colour palette for Al Ingot dashboard
C_PRI      = "#2196F3"   # blue  – Primary Al
C_SEC      = "#009688"   # teal  – Secondary Al
C_NOTCH    = "#FF9800"   # amber – Al Notch Bar
C_SOLVER_PRIM   = "#2196F3"
C_SOLVER_SEC    = "#9C27B0"
C_SOLVER_NOTCH  = "#FF9800"

def _layout_viu_al(title: str, y_title: str = "", height: int = 440) -> dict:
    return dict(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Inter, sans-serif", size=11, color="#333333"),
        title=dict(text=title, font=dict(size=14, color="#263238"), x=0.01),
        legend=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5
        ),
        xaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False, title=y_title),
        hovermode="x unified",
        height=height,
        margin=dict(l=60, r=30, t=55, b=95),
    )

def _layout_solver_al(title: str, y_title: str = "", height: int = 420) -> dict:
    return dict(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor="#FAFAFA",
        font=dict(family="Inter, sans-serif", size=12, color="#333333"),
        title=dict(text=title, font=dict(size=15, color="#1A237E"), x=0.01),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#DDD", borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#EEEEEE", zeroline=False, title=y_title),
        hovermode="x unified", height=height, margin=dict(l=60, r=30, t=55, b=45),
    )

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – ALL INPUT PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Dashboard Controls")
    st.divider()

    st.markdown("### A. Comparison Selection")
    comparison_selection = st.radio(
        "Select Analysis:",
        ["Not selected", "LC FeMn vs Mn Briquette", "MC FeMn vs Mn Briquette", "FeSi vs Si Metal", "Primary Al Ingot vs Secondary Al Ingot vs Al Notch Bar", "FeV80 vs Nitrovan"],
        index=0
    )
    
    if comparison_selection == "LC FeMn vs Mn Briquette":
        st.divider()
        st.markdown("### B. Financial Parameters")
        P_LCFeMn_Price       = st.number_input("LC FeMn Price (₹/MT)",        value=145000, step=1000, min_value=50000, max_value=400000, key="lc_p_lcfemn")
        P_EMM_Price          = st.number_input("Mn Briquette Price (₹/MT)", value=240000, step=1000, min_value=50000, max_value=600000, key="lc_p_emm")
        P_Power_Tariff       = st.number_input("Power Tariff (₹/kWh)",         value=6.5,   step=0.1, min_value=1.0,   max_value=20.0, format="%.2f", key="lc_tariff")
        P_Electrode_Cost     = st.number_input("Electrode Cost (₹/kg)",        value=240,   step=10,  min_value=50,    max_value=800, key="lc_elec")
        P_Steel_Value        = st.number_input("Steel Value (₹/MT)",           value=60000, step=1000, min_value=20000, max_value=200000, key="lc_steel")
        P_Margin_Steel       = st.number_input("Throughput Margin (₹/MT)",     value=2800,  step=100, min_value=500,   max_value=10000, key="lc_margin")
        P_LF_Retreatment_Cost= st.number_input("LF Re-treatment Cost (₹/heat)",value=15000, step=500, min_value=2000,  max_value=50000, key="lc_ret_cost")
        P_RH_Minute_Cost     = st.number_input("RH Cost per Minute (₹/min)",   value=2500,  step=100, min_value=500,   max_value=10000, key="lc_rh_min")
        P_Ladle_Reline_Cost  = st.number_input("Ladle Reline Cost (₹)",        value=1500000,step=50000,min_value=200000,max_value=5000000, key="lc_ladle_cost")
        P_Scrap_Price        = st.number_input("Scrap / Fe Credit (₹/MT)",     value=35000, step=500, min_value=5000,  max_value=80000, key="lc_scrap")

        st.divider()
        st.markdown("### C. Technical Parameters")
        P_LCFeMn_Mn  = st.slider("LC FeMn Mn Content (%)",   60.0, 95.0, 80.0, 0.5, key="lc_mn_pct") / 100
        P_EMM_Mn     = st.slider("EMM Mn Content (%)",        95.0, 100.0, 99.7, 0.1, key="lc_emm_pct") / 100
        P_LCFeMn_Rec = st.slider("LC FeMn Recovery (%)",      70.0, 99.0, 90.0, 0.5, key="lc_rec") / 100
        P_EMM_Mn_Rec = st.slider("EMM Recovery (%)",          80.0, 99.9, 97.0, 0.5, key="lc_emm_rec") / 100
        P_LCFeMn_Fe  = st.slider("LC FeMn Fe Content (%)",    5.0,  35.0, 15.0, 0.5, key="lc_fe") / 100
        P_LCFeMn_C   = st.slider("LC FeMn Carbon (%)",        0.1,  2.0,  0.5,  0.1, key="lc_c") / 100
        P_SpHeat_Steel  = st.slider("Steel Specific Heat (MJ/T/°C)", 0.5, 1.0, 0.75, 0.01, key="lc_heat")
        P_Chill_LCFeMn  = st.slider("LC FeMn Chill Factor (°C/kg/t)", 1.0, 4.0, 2.057, 0.001, key="lc_chill_lc")
        P_Chill_EMM     = st.slider("EMM Chill Factor (°C/kg/t)",     0.5, 2.5, 1.0,  0.05, key="lc_chill_emm")
        H2_Degas_Rate   = st.slider("H₂ Degas Rate (ppm/min)",      0.02, 0.10, 0.045, 0.005, key="lc_h2_deg")

        st.divider()
        st.markdown("### D. Operational Parameters")
        P_Heat_Size  = st.slider("Heat Size (MT)",            100,  350,  190,  5, key="lc_heat_sz")
        P_Cycle_Time = st.slider("LF Cycle Time (min)",        30,   90,   53,  1, key="lc_cycle")
        P_Ladle_Life = st.slider("Ladle Life (heats)",         50,  200,  100,  5, key="lc_ladle_life")
        Active_Mn    = st.number_input("Mn Addition Target (%)", value=0.36, step=0.01, min_value=0.01, max_value=5.0, format="%.2f", key="lc_active")
        P_LF_Efficiency = st.slider("LF Efficiency (%)",       25.0, 80.0, 45.0, 1.0, key="lc_lf_eff") / 100
        P_Arc_Duty      = st.slider("Arc Duty Cycle (%)",      30.0, 90.0, 60.0, 1.0, key="lc_arc") / 100
        P_Reheat_Rate   = st.slider("Reheat Rate (°C/min)",     2.0,  6.0,  3.5,  0.1, key="lc_reh")
        P_Graphite_Factor = st.slider("Electrode Wear (kg/kWh)", 0.005, 0.020, 0.010, 0.001, key="lc_graphite")
        LCFeMn_Overdose    = st.slider("LC FeMn Overdose Buffer (%)",  0.5,  5.0,  2.0,  0.1, key="lc_od") / 100
        EMM_Overdose       = st.slider("EMM Overdose Buffer (%)",       0.1,  2.0,  0.5,  0.1, key="lc_emm_od") / 100
        LCFeMn_Rec_Var     = st.slider("LC FeMn Recovery Std-Dev (%)", 0.5,  6.0,  3.0,  0.1, key="lc_rec_var") / 100
        EMM_Rec_Var        = st.slider("EMM Recovery Std-Dev (%)",      0.5,  3.0,  1.5,  0.1, key="lc_emm_rec_var") / 100
        Reject_LCFeMn      = st.number_input("LC FeMn Rejection Rate", value=0.0005, format="%.5f", step=0.0001, key="lc_rej")
        Reject_EMM         = st.number_input("EMM Rejection Rate",      value=0.00035, format="%.5f", step=0.0001, key="lc_emm_rej")
        Retreatment_LCFeMn = st.slider("LC FeMn Re-treatment Rate (%)",1.0,  8.0,  3.0,  0.1, key="lc_ret") / 100
        Retreatment_EMM    = st.slider("EMM Re-treatment Rate (%)",     0.5,  5.0,  2.5,  0.1, key="lc_emm_ret") / 100
        C_Corr_Freq_LCFeMn = st.slider("Carbon Correction Frequency", 0.02, 0.30, 0.10, 0.01, key="lc_c_corr")
        RH_Corr_Time       = st.slider("RH Carbon Corr. Time (min)",   2,   15,    5,    1, key="lc_rh_time")
        H2_Pickup_EMM      = st.slider("H₂ Pickup EMM (ppm)",          0.01, 0.15, 0.045, 0.005, key="lc_h2_pick")
        Refractory_Wear_Drop = st.slider("Refractory Wear Reduction (%)", 0.5, 8.0, 2.0, 0.5, key="lc_wear") / 100

        st.divider()
        st.markdown("### E. Realization Factors")
        R_Power       = st.slider("Power Realization",       0.50, 1.00, 0.90, 0.01, key="lc_r_pow")
        R_Electrode   = st.slider("Electrode Realization",   0.50, 1.00, 0.90, 0.01, key="lc_r_elec")
        R_Throughput  = st.slider("Throughput Realization",  0.10, 0.80, 0.40, 0.01, key="lc_r_thr")
        R_Stability   = st.slider("Stability Realization",   0.20, 1.00, 0.50, 0.01, key="lc_r_sta")
        R_Reblow      = st.slider("Reblow Realization",      0.30, 1.00, 0.75, 0.01, key="lc_r_reb")
        R_Cleanliness = st.slider("Cleanliness Realization", 0.10, 0.70, 0.30, 0.01, key="lc_r_cln")
        R_Yield       = st.slider("Yield Realization",       0.05, 0.50, 0.20, 0.01, key="lc_r_yld")

        st.divider()
        st.markdown("### F. Enterprise Savings")
        EMM_Consumption_FY = st.number_input("Consumption (MT)", value=8300, step=100, min_value=100, max_value=100000, key="lc_cons")
        Substitution_Pct   = st.slider("% Substitution", 0.0, 1.0, 0.50, 0.05, key="lc_sub")

    elif comparison_selection == "MC FeMn vs Mn Briquette":
        st.divider()
        st.markdown("### B. Financial Parameters")
        P_MCFeMn_Price       = st.number_input("MC FeMn Price (₹/MT)",        value=130000, step=1000, min_value=50000, max_value=400000, key="mc_p_mcfemn")
        P_Briq_Price         = st.number_input("Mn Briquette Price (₹/MT)",   value=175000, step=1000, min_value=50000, max_value=600000, key="mc_p_briq")
        P_Power_Tariff       = st.number_input("Power Tariff (₹/kWh)",        value=6.5,   step=0.1, min_value=1.0,   max_value=20.0, format="%.2f", key="mc_tariff")
        P_Electrode_Cost     = st.number_input("Electrode Cost (₹/kg)",       value=240,   step=10,  min_value=50,    max_value=800, key="mc_elec")
        P_Steel_Value        = st.number_input("Steel Value (₹/MT)",          value=60000, step=1000, min_value=20000, max_value=200000, key="mc_steel")
        P_Margin_Steel       = st.number_input("Throughput Margin (₹/MT)",    value=2800,  step=100, min_value=500,   max_value=10000, key="mc_margin")
        P_LF_Retreatment_Cost= st.number_input("LF Re-treatment Cost (₹/heat)",value=15000, step=500, min_value=2000,  max_value=50000, key="mc_ret_cost")
        P_RH_Corr_Cost       = st.number_input("RH Correction Cost (₹/heat)", value=2500,  step=100, min_value=500,   max_value=10000, key="mc_rh_corr")
        P_Ladle_Reline_Cost  = st.number_input("Ladle Reline Cost (₹)",       value=1500000,step=50000,min_value=200000,max_value=5000000, key="mc_ladle_cost")
        P_Scrap_Price        = st.number_input("Scrap / Fe Credit (₹/MT)",    value=35000, step=500, min_value=5000,  max_value=80000, key="mc_scrap")

        st.divider()
        st.markdown("### C. Technical Parameters")
        P_MCFeMn_Mn  = st.slider("MC FeMn Mn Content (%)",  60.0, 85.0, 70.0, 0.5, key="mc_mn_pct") / 100
        P_Briq_Mn    = st.slider("Mn Briquette Mn Content (%)", 90.0, 100.0, 99.0, 0.1, key="mc_briq_pct") / 100
        P_MCFeMn_Rec = st.slider("MC FeMn Recovery (%)",    70.0, 99.0, 85.0, 0.5, key="mc_rec") / 100
        P_Briq_Rec   = st.slider("Mn Briquette Recovery (%)", 80.0, 99.9, 95.0, 0.5, key="mc_briq_rec") / 100
        P_MCFeMn_Fe  = st.slider("MC FeMn Fe Content (%)",  5.0,  35.0, 20.0, 0.5, key="mc_fe") / 100
        P_MCFeMn_C   = st.slider("MC FeMn Carbon (%)",      0.1,  2.5,  1.5,  0.1, key="mc_c") / 100
        P_Briq_C     = st.slider("Mn Briquette Carbon (%)", 0.01, 0.5,  0.1,  0.01, key="mc_briq_c") / 100
        P_SpHeat_Steel  = st.slider("Steel Specific Heat (MJ/T/°C)", 0.5, 1.0, 0.75, 0.01, key="mc_heat")
        P_Chill_MCFeMn  = st.slider("MC FeMn Chill (°C/heat)", 1.0, 10.0, 5.0, 0.1, key="mc_chill_mc")
        P_Chill_Briq    = st.slider("Briq Chill (°C/heat)", 0.5, 5.0, 1.0, 0.1, key="mc_chill_briq")
        H2_Degas_Rate   = st.slider("H₂ Degas Rate (ppm/min)",    0.02, 0.10, 0.045, 0.005, key="mc_h2_deg")

        st.divider()
        st.markdown("### D. Operational Parameters")
        P_Heat_Size  = st.slider("Heat Size (MT)",            100,  350,  190,  5, key="mc_heat_sz")
        P_Cycle_Time = st.slider("LF Cycle Time (min)",        30,   90,   53,  1, key="mc_cycle")
        P_Ladle_Life = st.slider("Ladle Life (heats)",         50,  200,  100,  5, key="mc_ladle_life")
        P_Alloy_Target = st.number_input("Mn Addition Target (%)", value=0.4, step=0.01, min_value=0.1, max_value=2.0, format="%.2f", key="mc_active")
        P_LF_Efficiency = st.slider("LF Efficiency (%)",       25.0, 80.0, 45.0, 1.0, key="mc_lf_eff") / 100
        P_Arc_Duty      = st.slider("Arc Duty Cycle (%)",      30.0, 90.0, 60.0, 1.0, key="mc_arc") / 100
        P_Reheat_Rate   = st.slider("Reheat Rate (°C/min)",     2.0,  6.0,  3.5,  0.1, key="mc_reh")
        P_Graphite_Factor = st.slider("Electrode Wear (kg/kWh)", 0.002, 0.020, 0.010, 0.001, key="mc_graphite")
        MCFeMn_Overdose    = st.slider("MC FeMn Overdose Buffer (%)",  0.5,  8.0,  5.0,  0.1, key="mc_od") / 100
        Briq_Overdose      = st.slider("Mn Briq Overdose Buffer (%)",   0.1,  4.0,  1.5,  0.1, key="mc_briq_od") / 100
        MCFeMn_Rec_Var     = st.slider("MC FeMn Recovery Std-Dev (%)", 0.5,  8.0,  5.0,  0.1, key="mc_rec_var") / 100
        Briq_Rec_Var       = st.slider("Mn Briq Recovery Std-Dev (%)",  0.5,  4.0,  1.5,  0.1, key="mc_briq_rec_var") / 100
        Reject_MCFeMn      = st.number_input("MC FeMn Rejection Rate", value=0.0002, format="%.5f", step=0.0001, key="mc_rej")
        Reject_Briq        = st.number_input("Mn Briq Rejection Rate", value=0.0000, format="%.5f", step=0.0001, key="mc_briq_rej")
        Retreatment_MCFeMn = st.slider("MC FeMn Re-treatment Rate (%)",1.0,  10.0, 4.0,  0.1, key="mc_ret") / 100
        Retreatment_Briq   = st.slider("Mn Briq Re-treatment Rate (%)", 0.5,  5.0,  2.0,  0.1, key="mc_briq_ret") / 100
        C_Corr_Freq_MCFeMn = st.slider("Carbon Correction Frequency", 0.02, 0.30, 0.10, 0.01, key="mc_c_corr")
        H2_Pickup_Briq     = st.slider("H₂ Pickup Briq (ppm)",         0.01, 0.20, 0.09, 0.01, key="mc_h2_pick")
        P_Yield_Factor     = st.number_input("Yield Improvement Factor", value=0.0003, format="%.5f", step=0.0001, key="mc_yield_fac")
        Refractory_Wear_Drop = st.slider("Refractory Wear Reduction (%)", 0.5, 8.0, 2.0, 0.5, key="mc_wear") / 100

        st.divider()
        st.markdown("### E. Realization Factors")
        R_Power       = st.slider("Power Realization",       0.50, 1.00, 0.90, 0.01, key="mc_r_pow")
        R_Electrode   = st.slider("Electrode Realization",   0.50, 1.00, 0.90, 0.01, key="mc_r_elec")
        R_Throughput  = st.slider("Throughput Realization",  0.10, 0.80, 0.25, 0.01, key="mc_r_thr")
        R_Stability   = st.slider("Stability Realization",   0.20, 1.00, 0.50, 0.01, key="mc_r_sta")
        R_Reblow      = st.slider("Reblow Realization",      0.30, 1.00, 0.50, 0.01, key="mc_r_reb")
        R_Cleanliness = st.slider("Cleanliness Realization", 0.10, 1.00, 1.00, 0.01, key="mc_r_cln")
        R_Yield       = st.slider("Yield Realization",       0.05, 0.50, 0.25, 0.01, key="mc_r_yld")
        R_Carbon      = st.slider("Carbon Corr. Realization",0.20, 1.00, 1.00, 0.01, key="mc_r_c")
        R_Hydrogen    = st.slider("Hydrogen Penalty Realization", 0.20, 1.00, 0.50, 0.01, key="mc_r_h2")
        R_Refractory  = st.slider("Refractory Realization",  0.10, 1.00, 1.00, 0.01, key="mc_r_ref")

        st.divider()
        st.markdown("### F. Enterprise Savings")
        Briq_Consumption_FY = st.number_input("Consumption (MT)", value=24000, step=100, min_value=100, max_value=100000, key="mc_cons")
        Substitution_Pct    = st.slider("% Substitution", 0.0, 1.0, 0.05, 0.05, key="mc_sub")
        
    elif comparison_selection == "FeSi vs Si Metal":
        st.divider()
        st.markdown("### B. Financial Parameters")
        P_FeSi_Price         = st.number_input("FeSi70 Price (₹/MT)",         value=111500, step=1000, min_value=50000, max_value=300000, key="fesi_p_fesi")
        P_SiMetal_Price      = st.number_input("Si Metal Price (₹/MT)",       value=143000, step=1000, min_value=50000, max_value=400000, key="fesi_p_simetal")
        P_Power_Tariff       = st.number_input("Power Cost (₹/kWh)",          value=6.5,    step=0.1,  min_value=1.0,   max_value=20.0, format="%.2f", key="fesi_power")
        P_Electrode_Cost     = st.number_input("Electrode Cost (₹/kg)",       value=240,    step=10,   min_value=50,    max_value=800, key="fesi_elec")
        P_Steel_Value        = st.number_input("Steel Value (₹/MT)",          value=60000,  step=1000, min_value=20000, max_value=200000, key="fesi_steel")
        P_Margin_Steel       = st.number_input("Throughput Margin (₹/MT)",    value=2800,   step=100,  min_value=500,   max_value=10000, key="fesi_margin")
        P_LF_Retreatment_Cost= st.number_input("LF Re-treatment Cost (₹/heat)",value=15000, step=500,  min_value=2000,  max_value=50000, key="fesi_ret_cost")
        P_Slag_Handling_Cost = st.number_input("Slag Handling Cost (₹/MT)",   value=600,    step=50,   min_value=100,   max_value=5000, key="fesi_slag_cost")
        P_CaWire_Cost        = st.number_input("Ca-Wire Cost (₹/kg)",         value=120,    step=5,    min_value=20,    max_value=500, key="fesi_cawire_cost")
        P_Scrap_Price        = st.number_input("Scrap / Fe Credit (₹/MT)",    value=35000,  step=500,  min_value=5000,  max_value=80000, key="fesi_scrap")
        P_Safety_Compliance_Cost = st.number_input("Safety & Storage Benefit (₹/MT)", value=500, step=50, min_value=0, max_value=5000, key="fesi_safety_cost")

        st.divider()
        st.markdown("### C. Technical Parameters")
        P_FeSi_Si            = st.slider("FeSi70 Si Content (%)",         60.0, 80.0, 70.0, 0.5, key="fesi_si_pct") / 100
        P_SiMetal_Si         = st.slider("Si Metal Si Content (%)",       95.0, 99.9, 98.0, 0.1, key="fesi_simetal_pct") / 100
        P_FeSi_Rec           = st.slider("FeSi70 Recovery (%)",           70.0, 99.0, 90.0, 0.5, key="fesi_rec") / 100
        P_SiMetal_Rec        = st.slider("Si Metal Recovery (%)",         80.0, 99.9, 93.0, 0.5, key="fesi_simetal_rec") / 100
        P_FeSi_Fe            = st.slider("FeSi70 Fe Content (%)",         5.0,  35.0, 25.0, 0.5, key="fesi_fe") / 100
        P_SpHeat_Steel       = st.slider("Steel Specific Heat (MJ/T/°C)", 0.5,  1.0,  0.75, 0.01, key="fesi_heat")
        P_Temp_Rise_FeSi     = st.slider("FeSi Temp Rise (°C/kg Si)",     0.5,  3.0,  1.38, 0.01, key="fesi_temp_fesi")
        P_Temp_Rise_SiMetal  = st.slider("Si Metal Temp Rise (°C/kg Si)", 1.0,  4.0,  1.95, 0.01, key="fesi_temp_si")

        st.divider()
        st.markdown("### D. Operational Parameters")
        Active_Si            = st.number_input("Target Active Si (%)",   value=0.35, step=0.01, format="%.3f", key="fesi_active")
        P_Heat_Size          = st.slider("Heat Size (MT)",               100,  350,  190,  5, key="fesi_heat_sz")
        P_Cycle_Time         = st.slider("LF Cycle Time (min)",          30,   90,   53,   1, key="fesi_cycle")
        P_LF_Efficiency      = st.slider("LF Heating Efficiency (%)",    25.0, 80.0, 45.0, 1.0, key="fesi_lf_eff") / 100
        P_Graphite_Factor    = st.number_input("Electrode Wear (kg/kWh)",value=0.0012, step=0.0001, format="%.4f", key="fesi_graphite")
        Time_Saved_SiMetal   = st.slider("Time Saved w/ Si Metal (min)", 0.0,  15.0, 2.0,  0.5, key="fesi_time_saved")
        FeSi_Overdose        = st.slider("FeSi Overdose Buffer (%)",     0.5,  5.0,  2.0,  0.1, key="fesi_od") / 100
        SiMetal_Overdose     = st.slider("Si Metal Overdose Buffer (%)", 0.1,  2.0,  0.5,  0.1, key="fesi_si_od") / 100
        Slag_Reduction       = st.slider("Slag Reduction (kg/T steel)",  0.0,  2.0,  0.35, 0.05, key="fesi_slag_red")
        Reject_FeSi          = st.number_input("FeSi Rejection Rate",    value=0.0005, format="%.5f", step=0.0001, key="fesi_rej")
        Reject_SiMetal       = st.number_input("Si Metal Rejection Rate",value=0.00035, format="%.5f", step=0.0001, key="fesi_si_rej")
        Yield_Gain_SiMetal   = st.slider("Yield Gain w/ Si Metal (%)",   0.01, 0.10, 0.03, 0.01, key="fesi_yield_gain") / 100
        CaWire_FeSi          = st.slider("Ca-Wire FeSi (kg/T)",          0.2,  2.0,  1.0,  0.05, key="fesi_cawire_fesi")
        CaWire_SiMetal       = st.slider("Ca-Wire Si Metal (kg/T)",      0.1,  1.5,  0.65, 0.05, key="fesi_cawire_si")
        Retreatment_FeSi     = st.slider("Re-treatment Rate FeSi (%)",   0.5,  8.0,  2.5,  0.1, key="fesi_ret") / 100
        Retreatment_SiMetal  = st.slider("Re-treatment Si Metal (%)",    0.1,  5.0,  1.0,  0.1, key="fesi_si_ret") / 100

        st.divider()
        st.markdown("### E. Realization Factors")
        R_Power       = st.slider("Power Realization",       0.50, 1.00, 0.90, 0.01, key="fesi_r_pow")
        R_Electrode   = st.slider("Electrode Realization",   0.50, 1.00, 0.90, 0.01, key="fesi_r_elec")
        R_Throughput  = st.slider("Throughput Realization",  0.10, 0.80, 0.30, 0.01, key="fesi_r_thr")
        R_Stability   = st.slider("Stability Realization",   0.20, 1.00, 0.80, 0.01, key="fesi_r_sta")
        R_Slag        = st.slider("Slag Handling Realization",0.10, 1.00, 0.50, 0.01, key="fesi_r_slag")
        R_Cleanliness = st.slider("Cleanliness Realization", 0.10, 0.70, 0.30, 0.01, key="fesi_r_cln")
        R_Yield       = st.slider("Yield Realization",       0.10, 1.00, 0.60, 0.01, key="fesi_r_yld")
        R_CaWire      = st.slider("Ca-Wire Realization",     0.10, 1.00, 0.30, 0.01, key="fesi_r_cawire")
        R_Retreatment = st.slider("Re-treatment Realization",0.30, 1.00, 0.75, 0.01, key="fesi_r_ret")
        R_Safety      = st.slider("Safety Realization",      0.10, 1.00, 1.00, 0.01, key="fesi_r_safe")

        st.divider()
        st.markdown("### F. Enterprise Savings")
        SiMetal_Consumption_FY = st.number_input("Consumption Baseline (MT)", value=11800, step=100, min_value=100, max_value=100000, key="fesi_cons")
        Substitution_Pct       = st.slider("% Substitution", 0.0, 1.0, 0.40, 0.05, key="fesi_sub")

    elif comparison_selection == "Primary Al Ingot vs Secondary Al Ingot vs Al Notch Bar":
        st.divider()
        st.markdown("### B. Financial Parameters")
        P_Pri_Price          = st.number_input("Primary Al Price (₹/MT)",      value=380000, step=1000, min_value=100000, key="al_pri_price")
        P_Sec_Price          = st.number_input("Secondary Al Price (₹/MT)",    value=340000, step=1000, min_value=100000, key="al_sec_price")
        P_Notch_Price        = st.number_input("Al Notch Bar Price (₹/MT)",    value=335000, step=1000, min_value=100000, key="al_notch_price")
        P_Power_Cost         = st.number_input("Power Cost (₹/kWh)",           value=6.5,    step=0.1,  format="%.2f", key="al_power_cost")
        P_Electrode_Cost     = st.number_input("Electrode Cost (₹/kg)",        value=240,    step=10, key="al_elec_cost")
        P_Steel_Value        = st.number_input("Steel Value (₹/MT)",           value=60000,  step=1000, key="al_steel_val")
        P_Slag_Cost          = st.number_input("Ladle Slag Handling (₹/MT)",   value=800,    step=50, key="al_slag_cost")
        P_Margin_Steel       = st.number_input("Throughput Margin (₹/MT)",     value=2800,   step=100, key="al_margin")
        P_LF_Retreatment     = st.number_input("LF Re-treatment Cost (₹/heat)",value=15000,  step=500, key="al_lf_ret")

        st.divider()
        st.markdown("### C. Technical Parameters")
        P_Pri_Purity   = st.slider("Primary Al Purity (%)",     95.0, 99.9, 99.0, 0.1, key="al_pri_pur") / 100
        P_Sec_Purity   = st.slider("Secondary Al Purity (%)",   90.0, 99.0, 97.0, 0.1, key="al_sec_pur") / 100
        P_Notch_Purity = st.slider("Al Notch Bar Purity (%)",   85.0, 98.0, 95.0, 0.1, key="al_notch_pur") / 100

        P_Pri_Rec      = st.slider("Primary Al Recovery (%)",   30.0, 70.0, 49.0, 0.5, key="al_pri_rec") / 100
        P_Sec_Rec      = st.slider("Secondary Al Recovery (%)", 30.0, 70.0, 46.0, 0.5, key="al_sec_rec") / 100
        P_Notch_Rec    = st.slider("Notch Bar Recovery (%)",    30.0, 70.0, 46.0, 0.5, key="al_notch_rec") / 100

        P_Pri_Yield    = st.slider("Primary Metallic Yield (%)",   99.0, 100.0, 99.89, 0.01, key="al_pri_yld") / 100
        P_Sec_Yield    = st.slider("Secondary Metallic Yield (%)", 99.0, 100.0, 99.88, 0.01, key="al_sec_yld") / 100
        P_Notch_Yield  = st.slider("Notch Bar Metallic Yield (%)", 99.0, 100.0, 99.87, 0.01, key="al_notch_yld") / 100

        P_LF_Efficiency= st.slider("LF Thermal Efficiency (%)", 20.0, 80.0, 40.0, 1.0, key="al_lf_eff") / 100
        P_SpHeat       = st.slider("Specific Heat Impurities (MJ/kg)", 1.0, 5.0, 2.5, 0.1, key="al_spheat")

        st.divider()
        st.markdown("### D. Operational Parameters")
        P_Heat_Size    = st.slider("Heat Size (MT)",            100,  350,  190,  5, key="al_heat_size")
        P_Cycle_Time   = st.slider("LF Cycle Time (min)",        30,   90,   53,  1, key="al_cycle")
        Active_Al      = st.number_input("Al Addition Target (%)", value=0.2, step=0.01, format="%.3f", key="al_active")

        P_Pri_Overdose   = st.number_input("Primary Overdose Buffer",   value=0.02, format="%.3f", key="al_pri_od")
        P_Sec_Overdose   = st.number_input("Secondary Overdose Buffer", value=0.03, format="%.3f", key="al_sec_od")
        P_Notch_Overdose = st.number_input("Notch Bar Overdose Buffer", value=0.035, format="%.3f", key="al_notch_od")

        P_Pri_Reject     = st.number_input("Primary Rejection Rate",    value=0.0003, format="%.5f", key="al_pri_rej")
        P_Sec_Reject     = st.number_input("Secondary Rejection Rate",  value=0.00045, format="%.5f", key="al_sec_rej")
        P_Notch_Reject   = st.number_input("Notch Bar Rejection Rate",  value=0.00055, format="%.5f", key="al_notch_rej")

        P_Pri_Retreat    = st.number_input("Primary Re-treatment Rate",   value=0.015, format="%.3f", key="al_pri_rtr")
        P_Sec_Retreat    = st.number_input("Secondary Re-treatment Rate", value=0.020, format="%.3f", key="al_sec_rtr")
        P_Notch_Retreat  = st.number_input("Notch Bar Re-treatment Rate", value=0.025, format="%.3f", key="al_notch_rtr")

        Extra_Time_Sec   = st.slider("Extra Time - Sec (min)",   0.0, 5.0, 0.5, 0.1, key="al_xt_sec")
        Extra_Time_Notch = st.slider("Extra Time - Notch (min)", 0.0, 5.0, 1.0, 0.1, key="al_xt_notch")

        P_Elec_Wear      = st.number_input("Electrode Wear (kg/kWh)", value=0.0015, format="%.4f", key="al_elec_wear")

        Dross_Pri_Sec    = st.slider("Dross Diff Pri vs Sec (kg/MT)",   0.0, 50.0, 20.0, 1.0, key="al_dross_sec")
        Dross_Pri_Notch  = st.slider("Dross Diff Pri vs Notch (kg/MT)", 0.0, 80.0, 40.0, 1.0, key="al_dross_notch")

        Slag_Pri_Sec     = st.slider("Slag Diff Pri vs Sec (kg/T)",     0.0, 100.0, 55.4, 0.1, key="al_slag_sec")
        Slag_Pri_Notch   = st.slider("Slag Diff Pri vs Notch (kg/T)",   0.0, 100.0, 55.0, 0.1, key="al_slag_notch")

        st.divider()
        st.markdown("### E. Realization Factors")
        R_Power       = st.slider("Power Realization",       0.0, 1.0, 1.00, 0.05, key="al_r_power")
        R_Electrode   = st.slider("Electrode Realization",   0.0, 1.0, 1.00, 0.05, key="al_r_elec")
        R_Throughput  = st.slider("Throughput Realization",  0.0, 1.0, 0.10, 0.05, key="al_r_thru")
        R_Stability   = st.slider("Stability Realization",   0.0, 1.0, 0.50, 0.05, key="al_r_stab")
        R_Slag        = st.slider("Slag Handling Realization",0.0, 1.0, 0.50, 0.05, key="al_r_slag")
        R_Cleanliness = st.slider("Cleanliness Realization", 0.0, 1.0, 0.40, 0.05, key="al_r_clean")
        R_Yield       = st.slider("Yield Realization",       0.0, 1.0, 0.50, 0.05, key="al_r_yield")
        R_Reblow      = st.slider("Reblow Realization",      0.0, 1.0, 1.00, 0.05, key="al_r_reblow")

        st.divider()
        st.markdown("### F. Enterprise Savings")
        Al_Consumption_FY = st.number_input("Base Al Consumption (MT)", value=4325, step=100, min_value=100, key="al_cons")
        Substitution_Pct  = st.slider("% Substitution", 0.0, 1.0, 0.50, 0.05, key="al_sub_pct")



    elif comparison_selection == "FeV80 vs Nitrovan":
        st.divider()
        st.markdown("### B. Financial Parameters")
        P_FeV_Price           = st.number_input("FeV80 Price (₹/MT)", value=2950000.0, step=10000.0, key="fev_price")
        P_NV_Price            = st.number_input("Nitrovan Price (₹/MT)", value=2500000.0, step=10000.0, key="nv_price")
        P_Power_Tariff        = st.number_input("Power Cost (₹/kWh)", value=6.5, step=0.1, format="%.2f", key="fev_power")
        P_Electrode_Cost      = st.number_input("Electrode Cost (₹/kg)", value=240.0, step=5.0, key="fev_elec")
        P_Steel_Value         = st.number_input("Steel Value (₹/MT)", value=60000.0, step=1000.0, key="fev_steel")
        P_LF_Minute_Cost      = st.number_input("LF Fixed Operating Cost (₹/min)", value=850.0, step=50.0, key="fev_lf_min")
        P_LF_Retreatment_Cost = st.number_input("Re-treatment / Reblow Cost (₹/event)", value=15000.0, step=1000.0, key="fev_ret_cost")

        st.divider()
        st.markdown("### C. Technical & Thermodynamic")
        P_FeV_V        = st.number_input("FeV80 V Content (V %)", value=0.80, format="%.2f", key="fev_v")
        P_NV_V         = st.number_input("Nitrovan V Content (V %)", value=0.77, format="%.2f", key="nv_v")
        P_NV_N         = st.number_input("Nitrovan N Content (N %)", value=0.16, format="%.2f", key="nv_n")
        P_FeV_Rec      = st.number_input("FeV80 Recovery (%)", value=0.95, format="%.2f", key="fev_rec")
        P_NV_Rec       = st.number_input("Nitrovan Recovery (%)", value=0.92, format="%.2f", key="nv_rec")
        P_FeV_Eff      = st.number_input("Strengthening Eff. – FeV80", value=1.0, format="%.1f", key="fev_eff")
        P_NV_Eff       = st.number_input("Strengthening Eff. – Nitrovan", value=1.1, format="%.1f", key="nv_eff")
        P_Chill_FeV    = st.number_input("FeV80 Chill Factor (°C/kg)", value=-1.8, format="%.1f", key="fev_chill")
        P_Chill_NV     = st.number_input("Nitrovan Chill Factor (°C/kg)", value=-3.0, format="%.1f", key="nv_chill")
        P_LF_Efficiency= st.number_input("LF Heating Efficiency (%)", value=0.60, format="%.2f", key="fev_lf_eff")
        P_SpHeat_Steel = st.number_input("Specific Heat of Steel (MJ/T/°C)", value=0.75, step=0.01, format="%.2f", key="fev_spheat")
        P_Graphite_Factor = st.number_input("LF Electrode Consumption (kg/kWh)", value=0.010, format="%.3f", key="fev_graphite")
        Conversion_MJ  = st.number_input("Conversion Factor (MJ/kWh)", value=3.6, format="%.1f", key="fev_conv")

        st.divider()
        st.markdown("### D. Operational Parameters")
        P_Heat_Size              = st.number_input("Heat Size (MT steel/heat)", value=190.0, step=5.0, key="fev_heat_size")
        Active_V                 = st.number_input("Target Vanadium Addition Rate (%)", value=0.20, format="%.2f", key="fev_active_v")
        P_Dissolution_Time_Saved = st.number_input("LF Cycle Time Reduction (min/heat)", value=2.5, format="%.1f", key="fev_diss")
        FeV_Overdose             = st.number_input("FeV Overdose Buffer (%)", value=0.0001, format="%.5f", key="fev_od")
        NV_Overdose              = st.number_input("NV Overdose Buffer (%)", value=0.00015, format="%.5f", key="nv_od")
        Retreatment_FeV          = st.number_input("LF Re-treatment Rate FeV80 (%)", value=0.010, format="%.4f", key="fev_retrt")
        Retreatment_NV           = st.number_input("LF Re-treatment Rate Nitrovan (%)", value=0.025, format="%.4f", key="nv_retrt")
        Reject_FeV               = st.number_input("Inclusion Rejection Rate FeV80 (%)", value=0.00008, format="%.5f", key="fev_rej")
        Reject_NV                = st.number_input("Inclusion Rejection Rate Nitrovan (%)", value=0.00015, format="%.5f", key="nv_rej")
        Yield_Loss_FeV           = st.number_input("Oxidation Yield Loss FeV80 (%)", value=0.00003, format="%.5f", key="fev_yld")
        Yield_Loss_NV            = st.number_input("Oxidation Yield Loss Nitrovan (%)", value=0.00005, format="%.5f", key="nv_yld")

        st.markdown("#### Solver-Specific Constraints")
        P_FeV_Rec_Var  = st.slider("FeV80 Rec Var (%) [Solver]",        1.0, 5.0, 2.5, 0.1, key="fev_rec_var")
        P_NV_Rec_Var   = st.slider("Nitrovan Rec Var (%) [Solver]",     1.0, 6.0, 3.8, 0.1, key="nv_rec_var")
        P_FeV_Retrt    = st.slider("FeV80 Re-treat Risk (%) [Solver]",  0.5, 5.0, 1.2, 0.1, key="fev_retrt_slv")
        P_NV_Retrt     = st.slider("Nitrovan Re-treat Risk (%) [Solver]", 1.0, 5.0, 2.8, 0.1, key="nv_retrt_slv")
        P_FeV_Inc      = st.slider("FeV Cleanliness Idx (%) [Solver]",  0.005, 0.030, 0.015, 0.001, format="%.3f", key="fev_inc")
        P_NV_Inc       = st.slider("NV Cleanliness Idx (%) [Solver]",   0.005, 0.030, 0.010, 0.001, format="%.3f", key="nv_inc")
        P_FeV_Yield    = st.slider("FeV80 Yield Loss (%) [Solver]",     0.001, 0.010, 0.003, 0.001, format="%.3f", key="fev_yield_slv")
        P_NV_Yield     = st.slider("Nitrovan Yield Loss (%) [Solver]",  0.001, 0.010, 0.005, 0.001, format="%.3f", key="nv_yield_slv")

        st.divider()
        st.markdown("### E. Realization Factors")
        R_Power       = st.number_input("Power Savings Realization (%)", value=0.40, format="%.2f", key="fev_r_power")
        R_Electrode   = st.number_input("Electrode Savings Realization (%)", value=0.50, format="%.2f", key="fev_r_elec")
        R_Throughput  = st.number_input("Throughput Realization (%)", value=0.40, format="%.2f", key="fev_r_thru")
        R_Stability   = st.number_input("Recovery Stability Realization (%)", value=0.10, format="%.2f", key="fev_r_stab")
        R_Reblow      = st.number_input("Re-treatment Realization (%)", value=0.80, format="%.2f", key="fev_r_reblow")
        R_Yield       = st.number_input("Yield Improvement Realization (%)", value=0.50, format="%.2f", key="fev_r_yield")
        R_Cleanliness = st.number_input("Inclusion Cleanliness Realization (%)", value=0.40, format="%.2f", key="fev_r_clean")

        st.divider()
        st.markdown("### F. Enterprise Savings")
        NV_Consumption_FY = st.number_input("Baseline Volume FY (MT)", value=80.0, step=10.0, key="fev_cons")
        Substitution_Pct  = st.slider("Substitution Percentage (%)", 0.0, 1.0, 1.0, 0.05, key="fev_sub_pct")

        # Link mapped variables for the solver
        P_NV_N_pct = P_NV_N
        P_FeV_ESU_Fac = P_FeV_Eff
        P_NV_ESU_Fac = P_NV_Eff
        P_FeV_Chill_Solver = abs(P_Chill_FeV)
        P_NV_Chill_Solver = abs(P_Chill_NV)
        Consump_FY = NV_Consumption_FY
        Sub_Pct = Substitution_Pct

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT GUARD
# ══════════════════════════════════════════════════════════════════════════════
if comparison_selection == "Not selected":
    st.info("Please select a substitution combination from the sidebar to run the analysis.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# CORE CALCULATIONS 
# ══════════════════════════════════════════════════════════════════════════════

if comparison_selection == "LC FeMn vs Mn Briquette":
    # Calculate true mass balance for substitutions (Excel Power Calc Engine basis)
    # P_Alloy_Target (LC FeMn addition rate in kg/T) is calculated based on Active_Mn target
    P_Alloy_Target = (Active_Mn / 100.0) * 1000.0 / (P_LCFeMn_Mn * P_LCFeMn_Rec)
    Alloy_LC = P_Alloy_Target

    # Calculate the effective target mass in kg/T to compute EMM parity
    Active_Mn_kg = (Active_Mn / 100.0) * 1000.0
    Alloy_EMM = Active_Mn_kg / (P_EMM_Mn * P_EMM_Mn_Rec)

    Steel_Per_MT_EMM = 1000.0 / Alloy_EMM
    kWh_MJ = 3.6

    # --- Power Saving (Rigorous Mass Balance Basis per MT EMM) ---
    Temp_Drop_LC = P_Chill_LCFeMn * Alloy_LC
    Temp_Drop_EMM = P_Chill_EMM * Alloy_EMM
    Delta_Temp_Rigorous = Temp_Drop_LC - Temp_Drop_EMM

    Energy_Saved_per_T_steel = (Delta_Temp_Rigorous * P_SpHeat_Steel) / (kWh_MJ * P_LF_Efficiency)
    Power_kWh_Saved_Per_MT = Energy_Saved_per_T_steel * Steel_Per_MT_EMM
    Benefit_Power = Power_kWh_Saved_Per_MT * P_Power_Tariff * R_Power

    # --- Electrode Saving ---
    Benefit_Electrode = Power_kWh_Saved_Per_MT * P_Graphite_Factor * P_Electrode_Cost * R_Electrode

    # --- Throughput Gain (Legacy Text Formula Logic per MT LC FeMn) ---
    Delta_Chill_Simple = P_Chill_LCFeMn - P_Chill_EMM
    Thermal_Gain_Total = Delta_Chill_Simple * P_Alloy_Target 
    Time_Saved_Min = Thermal_Gain_Total / P_Reheat_Rate
    Benefit_Throughput = (Time_Saved_Min / P_Cycle_Time) * P_Heat_Size * P_Margin_Steel * R_Throughput * (1000.0 / (P_Alloy_Target * P_Heat_Size))

    # --- Recovery Stability ---
    Benefit_Stability = (LCFeMn_Overdose - EMM_Overdose) * P_LCFeMn_Price * R_Stability

    # --- Re-treatment Reduction ---
    Benefit_Retreatment = (Retreatment_LCFeMn - Retreatment_EMM) * P_LF_Retreatment_Cost * (1000.0 / (P_Alloy_Target * P_Heat_Size)) * R_Reblow

    # --- Cleanliness ---
    Benefit_Cleanliness = (Reject_LCFeMn - Reject_EMM) * P_Steel_Value * (1000.0 / P_Alloy_Target) * R_Cleanliness

    # --- Yield Improvement ---
    P_Yield_Factor = 2.5e-05
    Benefit_Yield = P_Yield_Factor * P_Steel_Value * (1000.0 / P_Alloy_Target) * R_Yield

    # --- Carbon Correction Avoidance ---
    Benefit_Carbon = C_Corr_Freq_LCFeMn * RH_Corr_Time * P_RH_Minute_Cost * (1000.0 / (P_Alloy_Target * P_Heat_Size))

    # --- Hydrogen Penalty ---
    Benefit_Hydrogen = -(H2_Pickup_EMM / H2_Degas_Rate) * P_RH_Minute_Cost * (1000.0 / (P_Alloy_Target * P_Heat_Size))

    # --- Refractory Life ---
    Benefit_Refractory = (P_Ladle_Reline_Cost / P_Ladle_Life) * Refractory_Wear_Drop * (1000.0 / (P_Alloy_Target * P_Heat_Size))

    # --- Gross Operational Credits ---
    Gross_Op_Benefits = (
        Benefit_Power + Benefit_Electrode + Benefit_Throughput +
        Benefit_Stability + Benefit_Retreatment + Benefit_Cleanliness +
        Benefit_Yield + Benefit_Carbon + Benefit_Hydrogen + Benefit_Refractory
    )

    # ══ VIU SUMMARY EXACT LOGIC ═══════════════════════════════════════════════════
    Alloy_Per_MT_Mn_LC  = 1.0 / (P_LCFeMn_Mn * P_LCFeMn_Rec)
    Alloy_Per_MT_Mn_EMM = 1.0 / (P_EMM_Mn   * P_EMM_Mn_Rec)

    Cost_Per_Mn_LC  = Alloy_Per_MT_Mn_LC  * P_LCFeMn_Price
    Cost_Per_Mn_EMM = Alloy_Per_MT_Mn_EMM * P_EMM_Price
    Iron_Credit_LC  = P_LCFeMn_Fe * P_Scrap_Price

    # Direct cost delta is strictly the normalized Cost per Active Mn difference
    Cost_Per_Mn_Delta = Cost_Per_Mn_EMM - Cost_Per_Mn_LC

    Total_Op_Credits = Gross_Op_Benefits - Iron_Credit_LC
    Net_VIU_Advantage = Cost_Per_Mn_Delta - Total_Op_Credits
    Savings_Per_MT = Total_Op_Credits - Cost_Per_Mn_Delta
    Annual_Savings_Rs = EMM_Consumption_FY * Substitution_Pct * abs(Savings_Per_MT)
    Annual_Savings_Cr = Annual_Savings_Rs / 1e7

elif comparison_selection == "MC FeMn vs Mn Briquette":
    # Calculate true mass balance for substitutions (Excel Power Calc Engine basis)
    Active_Mn_per_Heat = P_Heat_Size * (P_Alloy_Target / 100.0 * 1000.0)
    MCFeMn_per_Heat = Active_Mn_per_Heat / (P_MCFeMn_Mn * P_MCFeMn_Rec)
    Briq_per_Heat = Active_Mn_per_Heat / (P_Briq_Mn * P_Briq_Rec)
    
    Scale_Factor = 1000.0 / Briq_per_Heat  # Converts ₹/heat to ₹/MT Briquette
    kWh_MJ = 3.6
    
    # --- Power Saving (Rigorous Mass Balance Basis per MT Briquette) ---
    Delta_Temp_Rigorous = P_Chill_MCFeMn - P_Chill_Briq
    Energy_Saved_per_Heat = (P_Heat_Size * P_SpHeat_Steel * Delta_Temp_Rigorous) / (kWh_MJ * P_LF_Efficiency)
    Benefit_Power = Energy_Saved_per_Heat * P_Power_Tariff * R_Power * Scale_Factor
    
    # --- Electrode Saving ---
    Benefit_Electrode = Energy_Saved_per_Heat * P_Graphite_Factor * P_Electrode_Cost * R_Electrode * Scale_Factor
    
    # --- Throughput Gain ---
    Time_Saved_Min = Delta_Temp_Rigorous / P_Reheat_Rate
    Benefit_Throughput = (Time_Saved_Min / P_Cycle_Time) * P_Heat_Size * P_Margin_Steel * R_Throughput * Scale_Factor
    
    # --- Recovery Stability ---
    Cost_MC_Overdose = MCFeMn_per_Heat * MCFeMn_Overdose * (P_MCFeMn_Price / 1000.0)
    Cost_Briq_Overdose = Briq_per_Heat * Briq_Overdose * (P_Briq_Price / 1000.0)
    Benefit_Stability = (Cost_MC_Overdose - Cost_Briq_Overdose) * R_Stability * Scale_Factor
    
    # --- Re-treatment Reduction ---
    Benefit_Retreatment = (Retreatment_MCFeMn - Retreatment_Briq) * P_LF_Retreatment_Cost * R_Reblow * Scale_Factor
    
    # --- Cleanliness ---
    Benefit_Cleanliness = (Reject_MCFeMn - Reject_Briq) * P_Steel_Value * P_Heat_Size * R_Cleanliness * Scale_Factor
    
    # --- Yield Improvement ---
    Benefit_Yield = P_Yield_Factor * P_Steel_Value * P_Heat_Size * R_Yield * Scale_Factor
    
    # --- Carbon Correction Avoidance ---
    Benefit_Carbon = C_Corr_Freq_MCFeMn * P_RH_Corr_Cost * R_Carbon * Scale_Factor
    
    # --- Hydrogen Penalty ---
    Benefit_Hydrogen = -(H2_Pickup_Briq / H2_Degas_Rate) * P_RH_Corr_Cost * R_Hydrogen * Scale_Factor
    
    # --- Refractory Life ---
    Benefit_Refractory = (P_Ladle_Reline_Cost / P_Ladle_Life) * Refractory_Wear_Drop * R_Refractory * Scale_Factor
    
    # --- Gross Operational Credits ---
    Gross_Op_Benefits = (
        Benefit_Power + Benefit_Electrode + Benefit_Throughput +
        Benefit_Stability + Benefit_Retreatment + Benefit_Cleanliness +
        Benefit_Yield + Benefit_Carbon + Benefit_Hydrogen + Benefit_Refractory
    )
    
    # ══ VIU SUMMARY EXACT LOGIC ═══════════════════════════════════════════════════
    Alloy_Per_MT_Mn_MC   = 1.0 / (P_MCFeMn_Mn * P_MCFeMn_Rec)
    Alloy_Per_MT_Mn_Briq = 1.0 / (P_Briq_Mn   * P_Briq_Rec)
    
    Cost_Per_Mn_MC   = Alloy_Per_MT_Mn_MC   * P_MCFeMn_Price
    Cost_Per_Mn_Briq = Alloy_Per_MT_Mn_Briq * P_Briq_Price
    Iron_Credit_MC   = P_MCFeMn_Fe * P_Scrap_Price
    
    Cost_Per_Mn_Delta = Cost_Per_Mn_Briq - Cost_Per_Mn_MC
    Lost_Iron_Credit_per_MT_Briq = Iron_Credit_MC
    Total_Op_Credits = Gross_Op_Benefits - Lost_Iron_Credit_per_MT_Briq
    Savings_Per_MT = Total_Op_Credits - Cost_Per_Mn_Delta
    Annual_Savings_Rs = Briq_Consumption_FY * Substitution_Pct * Savings_Per_MT
    Annual_Savings_Cr = Annual_Savings_Rs / 1e7

elif comparison_selection == "FeSi vs Si Metal":
    # 1. Active Si targets and Mass Balance
    Active_Si_kg = (Active_Si / 100.0) * 1000.0
    Alloy_SiMetal_kg_per_T = Active_Si_kg / (P_SiMetal_Si * P_SiMetal_Rec)
    Steel_Per_MT_SiMetal = 1000.0 / Alloy_SiMetal_kg_per_T
    Heats_per_MT_SiMetal = Steel_Per_MT_SiMetal / P_Heat_Size

    # 2. Power Saving 
    Delta_Temp_Rise = P_Temp_Rise_SiMetal - P_Temp_Rise_FeSi
    Energy_Saved_kJ_per_kg_Si = Delta_Temp_Rise * (P_SpHeat_Steel * 1000.0)
    Power_kWh_Saved_Per_MT = (Energy_Saved_kJ_per_kg_Si * P_SiMetal_Si * 1000.0) / 3600.0 / P_LF_Efficiency
    Benefit_Power = Power_kWh_Saved_Per_MT * P_Power_Tariff * R_Power

    # 3. Electrode Saving
    Benefit_Electrode = Power_kWh_Saved_Per_MT * P_Graphite_Factor * P_Electrode_Cost * R_Electrode

    # 4. Throughput Gain
    Benefit_Throughput = (Time_Saved_SiMetal / P_Cycle_Time) * P_Heat_Size * P_Margin_Steel * R_Throughput * Heats_per_MT_SiMetal

    # 5. Recovery Stability Benefit
    Benefit_Stability = (FeSi_Overdose - SiMetal_Overdose) * P_FeSi_Price * R_Stability

    # 6. Slag Handling Benefit
    Benefit_Slag = Slag_Reduction * Steel_Per_MT_SiMetal * (P_Slag_Handling_Cost / 1000.0) * R_Slag

    # 7. Inclusion Cleanliness Benefit
    Benefit_Cleanliness = (Reject_FeSi - Reject_SiMetal) * P_Steel_Value * Steel_Per_MT_SiMetal * R_Cleanliness

    # 8. Yield Improvement
    Benefit_Yield = Yield_Gain_SiMetal * P_Steel_Value * Steel_Per_MT_SiMetal * R_Yield

    # 9. Ca-Wire Reduction
    Benefit_CaWire = (CaWire_FeSi - CaWire_SiMetal) * P_CaWire_Cost * Steel_Per_MT_SiMetal * R_CaWire

    # 10. Re-treatment Reduction
    Benefit_Retreatment = (Retreatment_FeSi - Retreatment_SiMetal) * P_LF_Retreatment_Cost * Heats_per_MT_SiMetal * R_Retreatment

    # 11. Safety & Storage Benefit
    Benefit_Safety = P_Safety_Compliance_Cost * R_Safety

    # Total Gross Operational Credits
    Gross_Op_Benefits = (
        Benefit_Power + Benefit_Electrode + Benefit_Throughput +
        Benefit_Stability + Benefit_Slag + Benefit_Cleanliness +
        Benefit_Yield + Benefit_CaWire + Benefit_Retreatment + Benefit_Safety
    )

    # 12. Lost Iron Credit Penalty
    Iron_Credit_FeSi = P_FeSi_Fe * P_Scrap_Price
    Total_Op_Credits = Gross_Op_Benefits - Iron_Credit_FeSi

    # 13. Active Silicon Cost Math & Base Price Delta
    Alloy_Per_MT_Si_FeSi = 1.0 / (P_FeSi_Si * P_FeSi_Rec)
    Alloy_Per_MT_Si_SiMetal = 1.0 / (P_SiMetal_Si * P_SiMetal_Rec)

    Cost_Per_Si_FeSi = Alloy_Per_MT_Si_FeSi * P_FeSi_Price
    Cost_Per_Si_SiMetal = Alloy_Per_MT_Si_SiMetal * P_SiMetal_Price

    Cost_Per_Si_Delta = Cost_Per_Si_FeSi - Cost_Per_Si_SiMetal
    Direct_Cost_Saving_Per_MT_SiMetal = Cost_Per_Si_Delta
    Savings_Per_MT = Direct_Cost_Saving_Per_MT_SiMetal + Total_Op_Credits

    # 14. Enterprise Level
    Annual_Savings_Rs = SiMetal_Consumption_FY * Substitution_Pct * abs(Savings_Per_MT)
    Annual_Savings_Cr = Annual_Savings_Rs / 1e7

# ══════════════════════════════════════════════════════════════════════════════
# TABS SETUP
# ══════════════════════════════════════════════════════════════════════════════
if comparison_selection != "Primary Al Ingot vs Secondary Al Ingot vs Al Notch Bar":
    tab1, tab2 = st.tabs(["⚗️ VIU Dashboard", "🧠 Substitution Solver"])
else:
    tab1, tab2 = st.tabs(["📊 VIU Dashboard", "🧠 Substitution Solver"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: VIU DASHBOARD 
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if comparison_selection == "LC FeMn vs Mn Briquette":
        st.markdown("""
        <div style="background: linear-gradient(135deg,#1A237E 0%,#1565C0 60%,#0277BD 100%);
                    padding:22px 28px 18px 28px; border-radius:14px; margin-bottom:20px;
                    box-shadow:0 4px 24px rgba(26,35,126,0.25);">
          <h1 style="color:#FFFFFF;margin:0;font-size:26px;font-weight:800;letter-spacing:0.02em;">
            ⚗️ VIU Dashboard — LC FeMn vs Mn Briquette
          </h1>
          <p style="color:#90CAF9;margin:6px 0 0 0;font-size:13px;">
            Value-In-Use Economic Analysis &nbsp;|&nbsp; Low-Carbon Ferromanganese (80% Mn) 
            vs Electrolytic Manganese Metal / Mn Briquette (99.7% Mn)
          </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(kpi("LC FeMn Price", f"₹{P_LCFeMn_Price:,.0f}", "per MT alloy", ""), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("Mn Briquette Price", f"₹{P_EMM_Price:,.0f}", "per MT alloy", "kpi-card-green"), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi("Mn Cost Gap", f"₹{Cost_Per_Mn_Delta:,.0f}", "per MT Active Mn", "kpi-card-amber"), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi("Total VIU Credits", f"₹{Total_Op_Credits:,.0f}", "net benefit / MT alloy", "kpi-card-green"), unsafe_allow_html=True)
        with c5:
            col = "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-red"
            lbl = "Net Savings / MT Alloy"
            st.markdown(kpi(lbl, f"₹{Savings_Per_MT:+,.0f}", "EMM advantage (positive = better)", col), unsafe_allow_html=True)
        with c6:
            col_yr = "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-red"
            st.markdown(kpi("Annual Savings FY26", f"₹{abs(Annual_Savings_Cr):.2f} Cr", f"@ {Substitution_Pct*100:.0f}% Substitution", col_yr), unsafe_allow_html=True)

        st.markdown('<div class="section-header">VIU Economic Synthesis</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("#### Cost per Active Manganese (₹/MT Mn)")
            km1, km2 = st.columns(2)
            with km1:
                st.markdown(kpi("LC FeMn Cost/MT Mn", f"₹{Cost_Per_Mn_LC:,.0f}", f"@ {P_LCFeMn_Mn*100:.1f}% Mn × {P_LCFeMn_Rec*100:.0f}% rec", ""), unsafe_allow_html=True)
            with km2:
                st.markdown(kpi("EMM Cost/MT Mn", f"₹{Cost_Per_Mn_EMM:,.0f}", f"@ {P_EMM_Mn*100:.1f}% Mn × {P_EMM_Mn_Rec*100:.0f}% rec", "kpi-card-green"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### VIU Components")
            data_summary = {
                "Component": [
                    "Cost per MT Active Mn",
                    "Direct Cost Delta (EMM premium)",
                    "Gross Operational Credits",
                    "Lost Iron Credit Penalty",
                    "Total Net Credits",
                    "Net VIU Advantage (Credits − Delta)",
                ],
                "LC FeMn (₹/MT)": [
                    f"₹{Cost_Per_Mn_LC:,.0f}", "—",
                    "—", "—", "—", "—",
                ],
                "EMM (₹/MT)": [
                    f"₹{Cost_Per_Mn_EMM:,.0f}", f"₹{Cost_Per_Mn_Delta:,.0f}",
                    f"₹{Gross_Op_Benefits:,.0f}", f"-₹{Iron_Credit_LC:,.0f}",
                    f"₹{Total_Op_Credits:,.0f}", f"₹{Savings_Per_MT:+,.0f}",
                ],
            }
            df_sum = pd.DataFrame(data_summary).set_index("Component")
            st.dataframe(df_sum, use_container_width=True)

            if Savings_Per_MT > 0:
                st.markdown(f"""
                <div class="success-box">
                ✅ <b>Mn Briquette (EMM) offers a net advantage of ₹{Savings_Per_MT:,.0f}/MT alloy.</b><br>
                Operational credits exceed the price premium, making EMM the economically superior choice.
                </div>""", unsafe_allow_html=True)
            elif Savings_Per_MT < -2000:
                st.markdown(f"""
                <div class="warn-box">
                ⚠️ <b>LC FeMn is currently more cost-effective by ₹{abs(Savings_Per_MT):,.0f}/MT.</b><br>
                At current prices and parameters, the LC FeMn price advantage outweighs operational credits.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                ℹ️ <b>Near economic parity.</b> Net VIU: ₹{Savings_Per_MT:+,.0f}/MT alloy.
                Consider plant-specific factors and grade-specific requirements.
                </div>""", unsafe_allow_html=True)

        with col_r:
            benefit_names = [
                "Power Saving", "Electrode Saving", "Throughput Gain",
                "Recovery Stability", "Re-treatment Reduction",
                "Cleanliness Benefit", "Yield Improvement",
                "Carbon Avoidance", "Refractory Benefit",
            ]
            benefit_values = [
                Benefit_Power, Benefit_Electrode, Benefit_Throughput,
                Benefit_Stability, Benefit_Retreatment, Benefit_Cleanliness,
                Benefit_Yield, Benefit_Carbon, Benefit_Refractory,
            ]
            pos_names  = [n for n, v in zip(benefit_names, benefit_values) if v > 0]
            pos_values = [v for v in benefit_values if v > 0]

            colours_donut = [
                "#2196F3", "#1565C0", "#42A5F5",
                "#4CAF50", "#66BB6A", "#81C784",
                "#FF9800", "#FFA726", "#FFC107",
            ]

            fig_donut = go.Figure(data=[go.Pie(
                labels=pos_names, values=pos_values,
                hole=0.52,
                marker=dict(colors=colours_donut[:len(pos_names)], line=dict(color="#fff", width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}/MT<extra></extra>",
            )])
            fig_donut.add_annotation(
                text=f"<b>₹{Gross_Op_Benefits:,.0f}</b><br><span style='font-size:10px'>Gross Credits</span>",
                x=0.5, y=0.5, font_size=14, showarrow=False,
            )
            fig_donut.update_layout(
                title="Gross Operational Credit Composition (₹/MT Alloy)",
                template="plotly_white", height=420,
                margin=dict(l=20, r=20, t=55, b=20),
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            k1, k2 = st.columns(2)
            with k1:
                st.markdown(kpi("Alloy/MT Active Mn (LC)", f"{Alloy_Per_MT_Mn_LC:.3f} MT", "LC FeMn required", ""), unsafe_allow_html=True)
            with k2:
                st.markdown(kpi("Alloy/MT Active Mn (EMM)", f"{Alloy_Per_MT_Mn_EMM:.3f} MT", "Mn Briquette required", "kpi-card-green"), unsafe_allow_html=True)

        st.markdown('<div class="section-header">Detailed Benefit Breakdown</div>', unsafe_allow_html=True)

        all_benefit_names = [
            "Power Saving", "Electrode Saving", "Throughput Gain",
            "Recovery Stability", "Re-treatment Reduction", "Cleanliness Benefit",
            "Yield Improvement", "Carbon Avoidance", "Hydrogen Penalty",
            "Refractory Benefit",
        ]
        all_benefit_values = [
            Benefit_Power, Benefit_Electrode, Benefit_Throughput,
            Benefit_Stability, Benefit_Retreatment, Benefit_Cleanliness,
            Benefit_Yield, Benefit_Carbon, Benefit_Hydrogen, Benefit_Refractory,
        ]
        all_benefit_basis = [
            f"ΔT={Delta_Temp_Rigorous:.3f}°C/t steel, {P_LF_Efficiency*100:.0f}% LF eff, {R_Power*100:.0f}% real.",
            f"P_kWh_saved={Power_kWh_Saved_Per_MT:.1f} kWh/MT EMM, {P_Graphite_Factor*1000:.0f}g/kWh, {R_Electrode*100:.0f}% real.",
            f"Time saved={Time_Saved_Min:.2f} min/heat, {R_Throughput*100:.0f}% real.",
            f"Overdose Δ={(LCFeMn_Overdose-EMM_Overdose)*100:.1f}%, {R_Stability*100:.0f}% real.",
            f"Miss Δ={(Retreatment_LCFeMn-Retreatment_EMM)*100:.1f}%, {R_Reblow*100:.0f}% real.",
            f"Reject Δ={(Reject_LCFeMn-Reject_EMM)*100:.4f}%, {R_Cleanliness*100:.0f}% real.",
            f"Yield factor={P_Yield_Factor*1e6:.1f}ppm, {R_Yield*100:.0f}% real.",
            f"C-corr freq={C_Corr_Freq_LCFeMn*100:.0f}%, {RH_Corr_Time}min, ₹{P_RH_Minute_Cost}/min.",
            f"H₂ pickup={H2_Pickup_EMM:.3f}ppm, degas={H2_Degas_Rate:.3f}ppm/min.",
            f"Wear drop={Refractory_Wear_Drop*100:.1f}%, ladle cost=₹{P_Ladle_Reline_Cost:,}.",
        ]

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            bar_colors = [C_DELTA if v >= 0 else C_NEG for v in all_benefit_values]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=all_benefit_names[::-1], x=all_benefit_values[::-1], orientation="h",
                marker=dict(color=bar_colors[::-1], line=dict(color="white", width=1)),
                text=[f"₹{v:+,.0f}" for v in all_benefit_values[::-1]], textposition="outside",
                hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}/MT alloy<extra></extra>",
            ))
            fig_bar.add_vline(x=0, line_dash="solid", line_color="#333", line_width=1.5)
            fig_bar.update_layout(**_layout("Gross Benefit Contribution per MT Alloy (₹/MT)", "₹/MT Alloy", 460))
            fig_bar.update_layout(xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_table:
            df_breakdown = pd.DataFrame({
                "Benefit Component": all_benefit_names,
                "₹/MT Alloy": [f"₹{v:+,.0f}" for v in all_benefit_values],
                "Basis & Assumptions": all_benefit_basis,
            }).set_index("Benefit Component")

            def color_values(val):
                num = float(val.replace("₹", "").replace(",", "").replace("+", ""))
                if num > 0: return "color: #1B5E20; font-weight: 600"
                elif num < 0: return "color: #B71C1C; font-weight: 600"
                return ""

            st.dataframe(df_breakdown.style.map(color_values, subset=["₹/MT Alloy"]), use_container_width=True, height=460)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Benefit Sensitivity Heatmap (₹/MT at varying Realization Factors)")
        real_range = np.arange(0.1, 1.05, 0.1)
        heat_names = [
            "Power Saving", "Electrode Saving", "Throughput Gain",
            "Recovery Stability", "Re-treatment Reduction",
            "Cleanliness", "Yield", "Carbon Avoidance",
        ]
        base_heat_values = [
            Power_kWh_Saved_Per_MT * P_Power_Tariff,
            Power_kWh_Saved_Per_MT * P_Graphite_Factor * P_Electrode_Cost,
            (Time_Saved_Min / P_Cycle_Time) * P_Heat_Size * P_Margin_Steel * (1000.0 / (P_Alloy_Target * P_Heat_Size)),
            (LCFeMn_Overdose - EMM_Overdose) * P_LCFeMn_Price,
            (Retreatment_LCFeMn - Retreatment_EMM) * P_LF_Retreatment_Cost * (1000.0 / (P_Alloy_Target * P_Heat_Size)),
            (Reject_LCFeMn - Reject_EMM) * P_Steel_Value * (1000.0 / P_Alloy_Target),
            P_Yield_Factor * P_Steel_Value * (1000.0 / P_Alloy_Target),
            C_Corr_Freq_LCFeMn * RH_Corr_Time * P_RH_Minute_Cost * (1000.0 / (P_Alloy_Target * P_Heat_Size)),
        ]
        heat_matrix = np.array([[bv * r for r in real_range] for bv in base_heat_values])

        fig_heat = go.Figure(go.Heatmap(
            z=heat_matrix, x=[f"{r*100:.0f}%" for r in real_range], y=heat_names,
            colorscale="Blues", text=np.round(heat_matrix, 0).astype(int), texttemplate="₹%{text}",
            textfont=dict(size=10), hovertemplate="<b>%{y}</b><br>Realization: %{x}<br>₹%{z:,.0f}/MT<extra></extra>",
        ))
        fig_heat.update_layout(**_layout("VIU Benefit Heatmap — Realization Factor Sensitivity", "", 380))
        fig_heat.update_layout(xaxis_title="Realization Factor", yaxis_title="")
        st.plotly_chart(fig_heat, use_container_width=True)

        st.markdown('<div class="section-header">VIU Waterfall Analysis</div>', unsafe_allow_html=True)
        wf_labels = [
            "LC FeMn Active Mn Cost", "Power Saving", "Electrode Saving",
            "Throughput Gain", "Recovery Stability", "Re-treatment Reduction",
            "Cleanliness", "Yield", "Carbon Avoidance", "Hydrogen Penalty",
            "Refractory Life", "Lost Iron Credit", "EMM Active Mn Cost",
        ]
        wf_values = [
            Cost_Per_Mn_LC, Benefit_Power, Benefit_Electrode,
            Benefit_Throughput, Benefit_Stability, Benefit_Retreatment,
            Benefit_Cleanliness, Benefit_Yield, Benefit_Carbon,
            Benefit_Hydrogen, Benefit_Refractory, -Iron_Credit_LC, 0,
        ]

        measures = ["absolute"] + ["relative"] * (len(wf_labels) - 2) + ["total"]
        wf_text = [f"₹{abs(v):,.0f}" for v in wf_values[:-1]] + [f"₹{Cost_Per_Mn_EMM:,.0f}"]
        wf_values_display = wf_values[:-1] + [Cost_Per_Mn_EMM]

        wf_colors = ["#1A237E"]
        for v in wf_values[1:-1]: wf_colors.append(C_DELTA if v > 0 else C_NEG)
        wf_colors.append("#4CAF50")

        fig_wf = go.Figure(go.Waterfall(
            name="VIU Waterfall", orientation="v", measure=measures,
            x=wf_labels, y=wf_values_display, text=wf_text, textposition="outside",
            connector=dict(line=dict(color="#BDBDBD", width=1.5, dash="dot")),
            increasing=dict(marker=dict(color=C_DELTA)), decreasing=dict(marker=dict(color=C_NEG)),
            totals=dict(marker=dict(color="#4CAF50" if Cost_Per_Mn_EMM <= Cost_Per_Mn_LC + Total_Op_Credits else C_NEG)),
            hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
        ))
        fig_wf.add_hline(
            y=Cost_Per_Mn_EMM, line_dash="dash", line_color="#4CAF50", line_width=1.5,
            annotation_text=f"EMM Cost/MT Mn ₹{Cost_Per_Mn_EMM:,.0f}", annotation_position="right",
        )
        fig_wf.update_layout(**_layout("VIU Waterfall: Active Mn Cost & Operational Adjustments (₹/MT)", "₹/MT", 520))
        fig_wf.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <b>How to read this waterfall:</b> Visualizes the synthesis algorithm exactly as formulated in the Excel model. 
        Starting from the Base Cost per MT Active Mn of LC FeMn, we add the operational 
        advantage benefits (Power, Electrode, Throughput, etc.) as credits mapping up towards the EMM Active Mn market price. 
        The Hydrogen Penalty pushes the threshold back down. Finally, the Iron Credit is applied as a penalty deduction 
        (because EMM lacks the free iron found in LC FeMn). 
        The final bar is the market cost per MT of Active Mn for EMM. If the total height of LC FeMn + Benefits - Penalties 
        exceeds the EMM bar, EMM is more cost effective.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Cost Comparison & Sensitivity Analysis</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)

        with col_a:
            fig_stack = go.Figure()
            categories = ["LC FeMn", "Mn Briquette (EMM)"]

            fig_stack.add_trace(go.Bar(
                name="Cost per MT Active Mn", x=categories, y=[Cost_Per_Mn_LC, Cost_Per_Mn_EMM],
                marker_color=[C_LCFEMN, C_EMM], text=[f"₹{Cost_Per_Mn_LC:,.0f}", f"₹{Cost_Per_Mn_EMM:,.0f}"],
                textposition="inside",
            ))
            fig_stack.add_trace(go.Bar(
                name="Gross Operational Credits (deduct)", x=categories, y=[0, -Gross_Op_Benefits],
                marker_color=["rgba(0,0,0,0)", "#FFC107"], text=["", f"-₹{Gross_Op_Benefits:,.0f}"],
                textposition="inside",
            ))
            fig_stack.add_trace(go.Bar(
                name="Lost Iron Credit Penalty (add)", x=categories, y=[0, Iron_Credit_LC],
                marker_color=["rgba(0,0,0,0)", "#FF7043"], text=["", f"+₹{Iron_Credit_LC:,.0f}"],
                textposition="inside",
            ))
            fig_stack.update_layout(barmode="relative", **_layout("Effective Cost Components (₹/MT Active Mn)", "₹/MT", 420))
            st.plotly_chart(fig_stack, use_container_width=True)

        with col_b:
            emm_prices  = np.linspace(P_LCFeMn_Price * 0.8, P_LCFeMn_Price * 2.5, 80)
            cost_mn_emms = (1.0 / (P_EMM_Mn * P_EMM_Mn_Rec)) * emm_prices
            net_viuss   = Total_Op_Credits - (cost_mn_emms - Cost_Per_Mn_LC)
            breakeven   = (Cost_Per_Mn_LC + Total_Op_Credits) * (P_EMM_Mn * P_EMM_Mn_Rec)

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=emm_prices, y=net_viuss, mode="lines", name="Net VIU Advantage",
                line=dict(color=C_DELTA, width=3), fill="tozeroy", fillcolor="rgba(76,175,80,0.1)",
                hovertemplate="EMM Price: ₹%{x:,.0f}<br>Net Advantage: ₹%{y:,.0f}/MT<extra></extra>",
            ))
            fig_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
            fig_sens.add_vline(x=P_EMM_Price, line_dash="dot", line_color=C_EMM, line_width=2,
                               annotation_text=f"Current ₹{P_EMM_Price:,}", annotation_position="top right")
            fig_sens.add_vline(x=breakeven, line_dash="dot", line_color=C_NEG, line_width=2,
                               annotation_text=f"Break-even ₹{breakeven:,.0f}", annotation_position="top left")
            fig_sens.update_layout(**_layout("EMM Price Sensitivity – Net VIU Advantage (₹/MT)", "Net Advantage (₹/MT)", 420))
            st.plotly_chart(fig_sens, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_c, col_d = st.columns(2)

        with col_c:
            lc_prices   = np.linspace(P_EMM_Price * 0.3, P_EMM_Price * 1.2, 80)
            cost_mn_lcs = (1.0 / (P_LCFeMn_Mn * P_LCFeMn_Rec)) * lc_prices
            net_lc_sens = Total_Op_Credits - (Cost_Per_Mn_EMM - cost_mn_lcs)
            
            fig_lc_sens = go.Figure()
            fig_lc_sens.add_trace(go.Scatter(
                x=lc_prices, y=net_lc_sens, mode="lines", name="Net VIU (varying LC FeMn price)",
                line=dict(color=C_LCFEMN, width=3), fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
                hovertemplate="LC FeMn: ₹%{x:,.0f}<br>Net Advantage: ₹%{y:,.0f}/MT<extra></extra>",
            ))
            fig_lc_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
            fig_lc_sens.add_vline(x=P_LCFeMn_Price, line_dash="dot", line_color=C_LCFEMN, line_width=2,
                                  annotation_text=f"Current ₹{P_LCFeMn_Price:,}", annotation_position="top right")
            fig_lc_sens.update_layout(**_layout("LC FeMn Price Sensitivity – Net VIU Advantage (₹/MT)", "Net Advantage (₹/MT)", 380))
            st.plotly_chart(fig_lc_sens, use_container_width=True)

        with col_d:
            tornado_names  = ["Power Saving", "Electrode Saving", "Throughput Gain",
                              "Recovery Stability", "Re-treatment", "Cleanliness",
                              "Carbon Avoidance", "Refractory"]
            tornado_base   = [Benefit_Power, Benefit_Electrode, Benefit_Throughput,
                              Benefit_Stability, Benefit_Retreatment, Benefit_Cleanliness,
                              Benefit_Carbon, Benefit_Refractory]
            tornado_low    = [v * 0.80 for v in tornado_base]
            tornado_high   = [v * 1.20 for v in tornado_base]

            fig_tornado = go.Figure()
            fig_tornado.add_trace(go.Bar(
                y=tornado_names[::-1], x=[h - b for h, b in zip(tornado_high[::-1], tornado_base[::-1])],
                orientation="h", name="+20%", marker_color=C_DELTA, base=[b for b in tornado_base[::-1]],
            ))
            fig_tornado.add_trace(go.Bar(
                y=tornado_names[::-1], x=[l - b for l, b in zip(tornado_low[::-1], tornado_base[::-1])],
                orientation="h", name="−20%", marker_color="#EF9A9A", base=[b for b in tornado_base[::-1]],
            ))
            fig_tornado.update_layout(barmode="overlay", **_layout("Sensitivity Tornado (±20% Realization)", "₹/MT Alloy", 380))
            st.plotly_chart(fig_tornado, use_container_width=True)

        st.markdown("#### Side-by-Side Cost per Active Manganese Summary")
        df_cmp = pd.DataFrame({
            "Metric": [
                "Market Price (₹/MT alloy)", "Active Mn Content (%)", "Mn Recovery (%)",
                "Effective Mn Efficiency (%)", "Alloy Needed per MT Active Mn (MT)",
                "Raw Cost per MT Active Mn (₹)", "Gross Operational Credits (₹/MT alloy)",
                "Lost Iron Credit Penalty (₹/MT alloy)", "Net Adjusted Cost per MT Active Mn (₹)",
            ],
            "LC FeMn": [
                f"₹{P_LCFeMn_Price:,}", f"{P_LCFeMn_Mn*100:.1f}%", f"{P_LCFeMn_Rec*100:.1f}%",
                f"{P_LCFeMn_Mn*P_LCFeMn_Rec*100:.1f}%", f"{Alloy_Per_MT_Mn_LC:.3f} MT",
                f"₹{Cost_Per_Mn_LC:,.0f}", "—", "—", f"₹{Cost_Per_Mn_LC:,.0f}",
            ],
            "Mn Briquette (EMM)": [
                f"₹{P_EMM_Price:,}", f"{P_EMM_Mn*100:.1f}%", f"{P_EMM_Mn_Rec*100:.1f}%",
                f"{P_EMM_Mn*P_EMM_Mn_Rec*100:.1f}%", f"{Alloy_Per_MT_Mn_EMM:.3f} MT",
                f"₹{Cost_Per_Mn_EMM:,.0f}", f"₹{Gross_Op_Benefits:,.0f}",
                f"₹{Iron_Credit_LC:,.0f}", f"₹{Cost_Per_Mn_EMM - Total_Op_Credits:,.0f}",
            ],
        }).set_index("Metric")
        st.dataframe(df_cmp, use_container_width=True)

        st.markdown('<div class="section-header">Enterprise Savings Calculator</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(kpi("Substituted Volume", f"{EMM_Consumption_FY * Substitution_Pct:,.0f} MT", f"at {Substitution_Pct*100:.0f}% substitution", ""), unsafe_allow_html=True)
        with s2:
            st.markdown(kpi("Savings / MT Alloy", f"₹{abs(Savings_Per_MT):,.0f}", "Magnitude of net advantage", "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-amber"), unsafe_allow_html=True)
        with s3:
            abs_savings_yr = abs(Annual_Savings_Cr)
            st.markdown(kpi("Annual Savings FY26", f"₹{abs_savings_yr:.2f} Cr", "at stated volume", "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-amber"), unsafe_allow_html=True)
        with s4:
            monthly = Annual_Savings_Cr * 1e7 / 12 / 1e5
            st.markdown(kpi("Monthly Savings", f"₹{abs(monthly):.1f} L", "per month average", "kpi-card-purple"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_lft, col_rgt = st.columns([2, 1])

        with col_lft:
            vol_range = np.arange(1000, EMM_Consumption_FY * 2.5, 500)
            savings_cr = (abs(Savings_Per_MT) * vol_range * Substitution_Pct) / 1e7

            fig_sav = go.Figure()
            fig_sav.add_trace(go.Scatter(
                x=vol_range, y=savings_cr, mode="lines", name="Annual Savings (₹ Cr)",
                line=dict(color=C_DELTA if Savings_Per_MT > 0 else C_NEG, width=3),
                fill="tozeroy", fillcolor="rgba(76,175,80,0.12)" if Savings_Per_MT > 0 else "rgba(244,67,54,0.12)",
                hovertemplate="Consumption: %{x:,.0f} MT<br>Savings: ₹%{y:.2f} Cr<extra></extra>",
            ))
            fig_sav.add_vline(
                x=EMM_Consumption_FY, line_dash="dash", line_color="#1A237E", line_width=2,
                annotation_text=f"Total: {EMM_Consumption_FY:,} MT → ₹{Annual_Savings_Cr:.2f} Cr (@ {Substitution_Pct*100:.0f}% Sub)",
                annotation_position="top right",
            )
            fig_sav.add_hline(y=0, line_dash="solid", line_color="#333", line_width=1.5)
            fig_sav.update_layout(**_layout(f"Enterprise Savings vs Total Consumption Volume (at {Substitution_Pct*100:.0f}% Sub)", "Savings (₹ Crore)", 400))
            st.plotly_chart(fig_sav, use_container_width=True)

            st.markdown(f"#### 3-Year Savings Projection (5% annual price escalation)")
            years = ["FY 2026", "FY 2027", "FY 2028"]
            escalation = [1.0, 1.05, 1.1025]
            proj_savings = [Annual_Savings_Cr * e for e in escalation]
            cumulative_cr = np.cumsum(proj_savings)

            fig_3yr = go.Figure()
            fig_3yr.add_trace(go.Bar(
                x=years, y=proj_savings, name="Annual Savings (₹ Cr)",
                marker_color=[C_DELTA if s > 0 else C_NEG for s in proj_savings],
                text=[f"₹{v:.2f} Cr" for v in proj_savings], textposition="outside",
            ))
            fig_3yr.add_trace(go.Scatter(
                x=years, y=cumulative_cr, mode="lines+markers+text", name="Cumulative (₹ Cr)",
                line=dict(color="#9C27B0", width=2.5, dash="dash"),
                marker=dict(size=9, color="#9C27B0"), text=[f"₹{v:.2f} Cr" for v in cumulative_cr],
                textposition="top center",
            ))
            fig_3yr.update_layout(**_layout("3-Year Enterprise Savings Projection (₹ Crore)", "₹ Crore", 380))
            st.plotly_chart(fig_3yr, use_container_width=True)

        with col_rgt:
            st.markdown("#### Per-Benefit Annual Savings (₹ Cr)")
            benefits_annual = {
                n: (v * EMM_Consumption_FY * Substitution_Pct) / 1e7
                for n, v in zip(all_benefit_names, all_benefit_values)
            }
            df_bens = pd.DataFrame({
                "Benefit": list(benefits_annual.keys()),
                "₹ Crore / Year": [round(v, 3) for v in benefits_annual.values()],
            }).sort_values("₹ Crore / Year", ascending=False).set_index("Benefit")

            def style_ben(val):
                return "color:#1B5E20;font-weight:600" if val > 0 else "color:#B71C1C;font-weight:600"

            st.dataframe(df_bens.style.map(style_ben, subset=["₹ Crore / Year"]), use_container_width=True, height=350)

            st.markdown("#### Savings Components Sunburst")
            pos_bens  = [(n, (v * EMM_Consumption_FY * Substitution_Pct) / 1e7) for n, v in zip(all_benefit_names, all_benefit_values) if v > 0]
            sun_labels = ["Gross VIU Credits"] + [p[0] for p in pos_bens]
            sun_parents = [""] + ["Gross VIU Credits"] * len(pos_bens)
            sun_values = [sum(p[1] for p in pos_bens)] + [p[1] for p in pos_bens]

            fig_sun = go.Figure(go.Sunburst(
                labels=sun_labels, parents=sun_parents, values=sun_values, branchvalues="total",
                hovertemplate="<b>%{label}</b><br>₹%{value:.3f} Cr<extra></extra>",
                marker=dict(colors=["#1A237E"] + colours_donut[:len(pos_bens)]),
            ))
            fig_sun.update_layout(title="Savings Sunburst (₹ Cr)", template="plotly_white", height=380, margin=dict(l=5, r=5, t=40, b=5))
            st.plotly_chart(fig_sun, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Break-Even Price Analysis")
        be1, be2, be3 = st.columns(3)
        emm_eff = P_EMM_Mn * P_EMM_Mn_Rec
        lc_eff = P_LCFeMn_Mn * P_LCFeMn_Rec

        emm_breakeven_price = (Cost_Per_Mn_LC + Total_Op_Credits) * emm_eff
        lc_breakeven_price  = (Cost_Per_Mn_EMM - Total_Op_Credits) * lc_eff
        min_credits_needed  = Cost_Per_Mn_Delta

        with be1:
            st.markdown(kpi("EMM Break-Even Price", f"₹{emm_breakeven_price:,.0f}",
                            f"Current EMM: ₹{P_EMM_Price:,} | {'BELOW' if P_EMM_Price < emm_breakeven_price else 'ABOVE'} break-even",
                            "kpi-card-green" if P_EMM_Price <= emm_breakeven_price else "kpi-card-amber"), unsafe_allow_html=True)
        with be2:
            st.markdown(kpi("LC FeMn Break-Even Price", f"₹{lc_breakeven_price:,.0f}",
                            f"Current LC: ₹{P_LCFeMn_Price:,} | {'BELOW' if P_LCFeMn_Price < lc_breakeven_price else 'ABOVE'} break-even",
                            "kpi-card-amber"), unsafe_allow_html=True)
        with be3:
            st.markdown(kpi("Min. Credits Needed", f"₹{min_credits_needed:,.0f}",
                            f"Current credits: ₹{Total_Op_Credits:,.0f} | {'✅ Sufficient' if Total_Op_Credits >= min_credits_needed else '❌ Insufficient'}",
                            "kpi-card-green" if Total_Op_Credits >= min_credits_needed else "kpi-card-red"), unsafe_allow_html=True)

        st.markdown('<div class="section-header">Final Recommendation</div>', unsafe_allow_html=True)

        if Savings_Per_MT > 0:
            st.markdown(f"""
            <div style="background:#E8F5E9; border-left:6px solid #4CAF50; padding:24px 32px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
                <h2 style="color:#1B5E20; margin-top:0; font-size:28px;">🏆 Mn Briquette Preferred</h2>
                <p style="font-size:16px; color:#2E7D32; line-height:1.6; margin-bottom:0;">
                    <b>Projected Annual Savings: ₹{Annual_Savings_Cr:.2f} Crore</b><br>
                    By shifting {Substitution_Pct*100:.0f}% of your {EMM_Consumption_FY:,} MT baseline consumption to Mn Briquette (EMM), 
                    you realize a net advantage of <b>₹{Savings_Per_MT:,.0f}/MT alloy</b>. 
                    The operational credits (₹{Total_Op_Credits:,.0f}/MT) effectively overcome the 
                    ₹{Cost_Per_Mn_Delta:,.0f}/MT Active Mn cost premium.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#FFF3E0; border-left:6px solid #FF9800; padding:24px 32px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
                <h2 style="color:#E65100; margin-top:0; font-size:28px;">🏆 LC FeMn Preferred</h2>
                <p style="font-size:16px; color:#EF6C00; line-height:1.6; margin-bottom:0;">
                    <b>LC FeMn Cost Efficiency: ₹{abs(Savings_Per_MT):,.0f}/MT alloy</b><br>
                    At current input parameters, LC FeMn remains the more cost-effective option, yielding a projected <b>₹{Annual_Savings_Cr:.2f} Crore</b> in savings vs switching. 
                    The EMM operational credits (₹{Total_Op_Credits:,.0f}/MT) 
                    do not fully offset the ₹{abs(Cost_Per_Mn_Delta):,.0f}/MT Active Mn cost premium for Mn Briquette. 
                    Adjust substitution strategies or renegotiate market pricing to break-even.
                </p>
            </div>
            """, unsafe_allow_html=True)

    elif comparison_selection == "MC FeMn vs Mn Briquette":
        st.markdown("""
        <div style="background: linear-gradient(135deg,#1A237E 0%,#1565C0 60%,#0277BD 100%);
                    padding:22px 28px 18px 28px; border-radius:14px; margin-bottom:20px;
                    box-shadow:0 4px 24px rgba(26,35,126,0.25);">
          <h1 style="color:#FFFFFF;margin:0;font-size:26px;font-weight:800;letter-spacing:0.02em;">
            ⚗️ VIU Dashboard — MC FeMn vs Mn Briquette
          </h1>
          <p style="color:#90CAF9;margin:6px 0 0 0;font-size:13px;">
            Value-In-Use Economic Analysis &nbsp;|&nbsp; Medium-Carbon Ferromanganese (70% Mn) 
            vs Manganese Metal Briquette (99% Mn)
          </p>
        </div>
        """, unsafe_allow_html=True)
        
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(kpi("MC FeMn Price", f"₹{P_MCFeMn_Price:,.0f}", "per MT alloy", ""), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("Mn Briquette Price", f"₹{P_Briq_Price:,.0f}", "per MT alloy", "kpi-card-green"), unsafe_allow_html=True)
        with c3:
            col_gap = "kpi-card-green" if Cost_Per_Mn_Delta <= 0 else "kpi-card-amber"
            st.markdown(kpi("Mn Cost Gap", f"₹{Cost_Per_Mn_Delta:,.0f}", "per MT Active Mn", col_gap), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi("Total VIU Credits", f"₹{Total_Op_Credits:,.0f}", "net benefit / MT alloy", "kpi-card-green"), unsafe_allow_html=True)
        with c5:
            col = "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-red"
            lbl = "Net Savings / MT Alloy"
            st.markdown(kpi(lbl, f"₹{Savings_Per_MT:+,.0f}", "Briquette advantage", col), unsafe_allow_html=True)
        with c6:
            col_yr = "kpi-card-green" if Annual_Savings_Cr > 0 else "kpi-card-red"
            st.markdown(kpi("Annual Savings FY26", f"₹{abs(Annual_Savings_Cr):.2f} Cr", f"@ {Substitution_Pct*100:.0f}% Substitution", col_yr), unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">VIU Economic Synthesis</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("#### Cost per Active Manganese (₹/MT Mn)")
            km1, km2 = st.columns(2)
            with km1:
                st.markdown(kpi("MC FeMn Cost/MT Mn", f"₹{Cost_Per_Mn_MC:,.0f}", f"@ {P_MCFeMn_Mn*100:.1f}% Mn × {P_MCFeMn_Rec*100:.0f}% rec", ""), unsafe_allow_html=True)
            with km2:
                st.markdown(kpi("Mn Briq Cost/MT Mn", f"₹{Cost_Per_Mn_Briq:,.0f}", f"@ {P_Briq_Mn*100:.1f}% Mn × {P_Briq_Rec*100:.0f}% rec", "kpi-card-green"), unsafe_allow_html=True)
        
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### VIU Components")
            data_summary = {
                "Component": [
                    "Cost per MT Active Mn",
                    "Direct Cost Delta (Briquette vs MC)",
                    "Gross Operational Credits",
                    "Lost Iron Credit Penalty",
                    "Total Net Credits",
                    "Net VIU Advantage (Credits − Delta)",
                ],
                "MC FeMn (₹/MT)": [
                    f"₹{Cost_Per_Mn_MC:,.0f}", "—",
                    "—", "—", "—", "—",
                ],
                "Mn Briquette (₹/MT)": [
                    f"₹{Cost_Per_Mn_Briq:,.0f}", f"₹{Cost_Per_Mn_Delta:,.0f}",
                    f"₹{Gross_Op_Benefits:,.0f}", f"-₹{Lost_Iron_Credit_per_MT_Briq:,.0f}",
                    f"₹{Total_Op_Credits:,.0f}", f"₹{Savings_Per_MT:+,.0f}",
                ],
            }
            df_sum = pd.DataFrame(data_summary).set_index("Component")
            st.dataframe(df_sum, use_container_width=True)
        
            if Savings_Per_MT > 0:
                st.markdown(f"""
                <div class="success-box">
                ✅ <b>Mn Briquette offers a net advantage of ₹{Savings_Per_MT:,.0f}/MT alloy.</b><br>
                Favorable active Mn pricing coupled with operational credits makes Briquettes the economically superior choice.
                </div>""", unsafe_allow_html=True)
            elif Savings_Per_MT < -2000:
                st.markdown(f"""
                <div class="warn-box">
                ⚠️ <b>MC FeMn is currently more cost-effective by ₹{abs(Savings_Per_MT):,.0f}/MT.</b><br>
                At current prices and parameters, the MC FeMn price advantage outweighs Briquette operational credits.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                ℹ️ <b>Near economic parity.</b> Net VIU: ₹{Savings_Per_MT:+,.0f}/MT alloy.
                Consider plant-specific factors and grade-specific requirements.
                </div>""", unsafe_allow_html=True)
        
        with col_r:
            benefit_names = [
                "Power Saving", "Electrode Saving", "Throughput Gain",
                "Recovery Stability", "Re-treatment Reduction",
                "Cleanliness Benefit", "Yield Improvement",
                "Carbon Avoidance", "Refractory Benefit",
            ]
            benefit_values = [
                Benefit_Power, Benefit_Electrode, Benefit_Throughput,
                Benefit_Stability, Benefit_Retreatment, Benefit_Cleanliness,
                Benefit_Yield, Benefit_Carbon, Benefit_Refractory,
            ]
            pos_names  = [n for n, v in zip(benefit_names, benefit_values) if v > 0]
            pos_values = [v for v in benefit_values if v > 0]
        
            colours_donut = [
                "#2196F3", "#1565C0", "#42A5F5",
                "#4CAF50", "#66BB6A", "#81C784",
                "#FF9800", "#FFA726", "#FFC107",
            ]
        
            fig_donut = go.Figure(data=[go.Pie(
                labels=pos_names, values=pos_values,
                hole=0.52,
                marker=dict(colors=colours_donut[:len(pos_names)], line=dict(color="#fff", width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}/MT<extra></extra>",
            )])
            fig_donut.add_annotation(
                text=f"<b>₹{Gross_Op_Benefits:,.0f}</b><br><span style='font-size:10px'>Gross Credits</span>",
                x=0.5, y=0.5, font_size=14, showarrow=False,
            )
            fig_donut.update_layout(
                title="Gross Operational Credit Composition (₹/MT Alloy)",
                template="plotly_white", height=420,
                margin=dict(l=20, r=20, t=55, b=20),
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        
            k1, k2 = st.columns(2)
            with k1:
                st.markdown(kpi("Alloy/MT Active Mn (MC)", f"{Alloy_Per_MT_Mn_MC:.3f} MT", "MC FeMn required", ""), unsafe_allow_html=True)
            with k2:
                st.markdown(kpi("Alloy/MT Active Mn (Briq)", f"{Alloy_Per_MT_Mn_Briq:.3f} MT", "Mn Briquette required", "kpi-card-green"), unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Detailed Benefit Breakdown</div>', unsafe_allow_html=True)
        
        all_benefit_names = [
            "Power Saving", "Electrode Saving", "Throughput Gain",
            "Recovery Stability", "Re-treatment Reduction",
            "Cleanliness Benefit", "Yield Improvement",
            "Carbon Avoidance", "Hydrogen Penalty",
            "Refractory Benefit",
        ]
        all_benefit_values = [
            Benefit_Power, Benefit_Electrode, Benefit_Throughput,
            Benefit_Stability, Benefit_Retreatment, Benefit_Cleanliness,
            Benefit_Yield, Benefit_Carbon, Benefit_Hydrogen, Benefit_Refractory,
        ]
        all_benefit_basis = [
            f"ΔT={Delta_Temp_Rigorous:.2f}°C/heat, {P_LF_Efficiency*100:.0f}% LF eff, {R_Power*100:.0f}% real.",
            f"E_saved={Energy_Saved_per_Heat:.1f} kWh/heat, {P_Graphite_Factor*1000:.0f}g/kWh, {R_Electrode*100:.0f}% real.",
            f"Time saved={Time_Saved_Min:.2f} min/heat, {R_Throughput*100:.0f}% real.",
            f"Overdose cost Δ=₹{Cost_MC_Overdose - Cost_Briq_Overdose:,.0f}/heat, {R_Stability*100:.0f}% real.",
            f"Miss Δ={(Retreatment_MCFeMn-Retreatment_Briq)*100:.1f}%, {R_Reblow*100:.0f}% real.",
            f"Reject Δ={(Reject_MCFeMn-Reject_Briq)*100:.4f}%, {R_Cleanliness*100:.0f}% real.",
            f"Yield factor={P_Yield_Factor*1e6:.1f}ppm, {R_Yield*100:.0f}% real.",
            f"C-corr freq={C_Corr_Freq_MCFeMn*100:.0f}%, ₹{P_RH_Corr_Cost}/heat, {R_Carbon*100:.0f}% real.",
            f"H₂ pickup={H2_Pickup_Briq:.3f}ppm, degas={H2_Degas_Rate:.3f}ppm/min.",
            f"Wear drop={Refractory_Wear_Drop*100:.1f}%, ladle cost=₹{P_Ladle_Reline_Cost:,}.",
        ]
        
        col_chart, col_table = st.columns([3, 2])
        
        with col_chart:
            bar_colors = [C_DELTA if v >= 0 else C_NEG for v in all_benefit_values]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=all_benefit_names[::-1], x=all_benefit_values[::-1], orientation="h",
                marker=dict(color=bar_colors[::-1], line=dict(color="white", width=1)),
                text=[f"₹{v:+,.0f}" for v in all_benefit_values[::-1]], textposition="outside",
                hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}/MT alloy<extra></extra>",
            ))
            fig_bar.add_vline(x=0, line_dash="solid", line_color="#333", line_width=1.5)
            fig_bar.update_layout(**_layout_viu("Gross Benefit Contribution per MT Alloy (₹/MT)", "₹/MT Alloy", 460))
            fig_bar.update_layout(xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False))
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_table:
            df_breakdown = pd.DataFrame({
                "Benefit Component": all_benefit_names,
                "₹/MT Alloy": [f"₹{v:+,.0f}" for v in all_benefit_values],
                "Basis & Assumptions": all_benefit_basis,
            }).set_index("Benefit Component")
        
            def color_values(val):
                num = float(val.replace("₹", "").replace(",", "").replace("+", ""))
                if num > 0: return "color: #1B5E20; font-weight: 600"
                elif num < 0: return "color: #B71C1C; font-weight: 600"
                return ""
        
            st.dataframe(df_breakdown.style.map(color_values, subset=["₹/MT Alloy"]), use_container_width=True, height=460)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Benefit Sensitivity Heatmap (₹/MT at varying Realization Factors)")
        real_range = np.arange(0.1, 1.05, 0.1)
        heat_names = [
            "Power Saving", "Electrode Saving", "Throughput Gain",
            "Recovery Stability", "Re-treatment Reduction",
            "Cleanliness", "Yield", "Carbon Avoidance",
        ]
        base_heat_values = [
            Energy_Saved_per_Heat * P_Power_Tariff * Scale_Factor,
            Energy_Saved_per_Heat * P_Graphite_Factor * P_Electrode_Cost * Scale_Factor,
            (Time_Saved_Min / P_Cycle_Time) * P_Heat_Size * P_Margin_Steel * Scale_Factor,
            (Cost_MC_Overdose - Cost_Briq_Overdose) * Scale_Factor,
            (Retreatment_MCFeMn - Retreatment_Briq) * P_LF_Retreatment_Cost * Scale_Factor,
            (Reject_MCFeMn - Reject_Briq) * P_Steel_Value * P_Heat_Size * Scale_Factor,
            P_Yield_Factor * P_Steel_Value * P_Heat_Size * Scale_Factor,
            C_Corr_Freq_MCFeMn * P_RH_Corr_Cost * Scale_Factor,
        ]
        heat_matrix = np.array([[bv * r for r in real_range] for bv in base_heat_values])
        
        fig_heat = go.Figure(go.Heatmap(
            z=heat_matrix, x=[f"{r*100:.0f}%" for r in real_range], y=heat_names,
            colorscale="Blues", text=np.round(heat_matrix, 0).astype(int), texttemplate="₹%{text}",
            textfont=dict(size=10), hovertemplate="<b>%{y}</b><br>Realization: %{x}<br>₹%{z:,.0f}/MT<extra></extra>",
        ))
        fig_heat.update_layout(**_layout_viu("VIU Benefit Heatmap — Realization Factor Sensitivity", "", 380))
        fig_heat.update_layout(xaxis_title="Realization Factor", yaxis_title="")
        st.plotly_chart(fig_heat, use_container_width=True)
        
        st.markdown('<div class="section-header">VIU Waterfall Analysis</div>', unsafe_allow_html=True)
        wf_labels = [
            "MC FeMn Active Mn Cost", "Power Saving", "Electrode Saving",
            "Throughput Gain", "Recovery Stability", "Re-treatment Reduction",
            "Cleanliness", "Yield", "Carbon Avoidance", "Hydrogen Penalty",
            "Refractory Life", "Lost Iron Credit", "Mn Briquette Active Mn Cost",
        ]
        wf_values = [
            Cost_Per_Mn_MC, Benefit_Power, Benefit_Electrode,
            Benefit_Throughput, Benefit_Stability, Benefit_Retreatment,
            Benefit_Cleanliness, Benefit_Yield, Benefit_Carbon,
            Benefit_Hydrogen, Benefit_Refractory, -Lost_Iron_Credit_per_MT_Briq, 0,
        ]
        
        measures = ["absolute"] + ["relative"] * (len(wf_labels) - 2) + ["total"]
        wf_text = [f"₹{abs(v):,.0f}" for v in wf_values[:-1]] + [f"₹{Cost_Per_Mn_Briq:,.0f}"]
        wf_values_display = wf_values[:-1] + [Cost_Per_Mn_Briq]
        
        wf_colors = ["#1A237E"]
        for v in wf_values[1:-1]: wf_colors.append(C_DELTA if v > 0 else C_NEG)
        wf_colors.append("#4CAF50")
        
        fig_wf = go.Figure(go.Waterfall(
            name="VIU Waterfall", orientation="v", measure=measures,
            x=wf_labels, y=wf_values_display, text=wf_text, textposition="outside",
            connector=dict(line=dict(color="#BDBDBD", width=1.5, dash="dot")),
            increasing=dict(marker=dict(color=C_DELTA)), decreasing=dict(marker=dict(color=C_NEG)),
            totals=dict(marker=dict(color="#4CAF50" if Cost_Per_Mn_Briq <= Cost_Per_Mn_MC + Total_Op_Credits else C_NEG)),
            hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
        ))
        fig_wf.add_hline(
            y=Cost_Per_Mn_Briq, line_dash="dash", line_color="#4CAF50", line_width=1.5,
            annotation_text=f"Mn Briq Cost/MT Mn ₹{Cost_Per_Mn_Briq:,.0f}", annotation_position="right",
        )
        fig_wf.update_layout(**_layout_viu("VIU Waterfall: Active Mn Cost & Operational Adjustments (₹/MT)", "₹/MT", 520))
        fig_wf.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_wf, use_container_width=True)
        
        st.markdown("""
        <div class="info-box">
        <b>How to read this waterfall:</b> Visualizes the synthesis mapping the Base Cost per MT Active Mn of MC FeMn, adding the operational 
        advantage benefits (Power, Electrode, Throughput, etc.) as credits against the Mn Briquette target price. 
        The Iron Credit is applied as a penalty deduction (because Mn Briquettes lack the free iron found in MC FeMn). 
        The final bar is the market cost per MT of Active Mn for Mn Briquettes. If the total height of MC FeMn + Benefits - Penalties 
        exceeds the Mn Briquette bar, Briquettes are the more cost-effective choice.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Cost Comparison & Sensitivity Analysis</div>', unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        
        with col_a:
            fig_stack = go.Figure()
            categories = ["MC FeMn", "Mn Briquette"]
        
            fig_stack.add_trace(go.Bar(
                name="Cost per MT Active Mn", x=categories, y=[Cost_Per_Mn_MC, Cost_Per_Mn_Briq],
                marker_color=[C_MCFEMN, C_BRIQ], text=[f"₹{Cost_Per_Mn_MC:,.0f}", f"₹{Cost_Per_Mn_Briq:,.0f}"],
                textposition="inside",
            ))
            fig_stack.add_trace(go.Bar(
                name="Gross Operational Credits (deduct)", x=categories, y=[0, -Gross_Op_Benefits],
                marker_color=["rgba(0,0,0,0)", "#FFC107"], text=["", f"-₹{Gross_Op_Benefits:,.0f}"],
                textposition="inside",
            ))
            fig_stack.add_trace(go.Bar(
                name="Lost Iron Credit Penalty (add)", x=categories, y=[0, Lost_Iron_Credit_per_MT_Briq],
                marker_color=["rgba(0,0,0,0)", "#FF7043"], text=["", f"+₹{Lost_Iron_Credit_per_MT_Briq:,.0f}"],
                textposition="inside",
            ))
            fig_stack.update_layout(barmode="relative", **_layout_viu("Effective Cost Components (₹/MT Active Mn)", "₹/MT", 420))
            st.plotly_chart(fig_stack, use_container_width=True)
        
        with col_b:
            briq_prices  = np.linspace(P_MCFeMn_Price * 0.8, P_MCFeMn_Price * 2.5, 80)
            cost_mn_briqs = (1.0 / (P_Briq_Mn * P_Briq_Rec)) * briq_prices
            net_viuss   = Total_Op_Credits - (cost_mn_briqs - Cost_Per_Mn_MC)
            breakeven   = (Cost_Per_Mn_MC + Total_Op_Credits) * (P_Briq_Mn * P_Briq_Rec)
        
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=briq_prices, y=net_viuss, mode="lines", name="Net VIU Advantage",
                line=dict(color=C_DELTA, width=3), fill="tozeroy", fillcolor="rgba(76,175,80,0.1)",
                hovertemplate="Briquette Price: ₹%{x:,.0f}<br>Net Advantage: ₹%{y:,.0f}/MT<extra></extra>",
            ))
            fig_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
            fig_sens.add_vline(x=P_Briq_Price, line_dash="dot", line_color=C_BRIQ, line_width=2,
                               annotation_text=f"Current ₹{P_Briq_Price:,}", annotation_position="top right")
            fig_sens.add_vline(x=breakeven, line_dash="dot", line_color=C_NEG, line_width=2,
                               annotation_text=f"Break-even ₹{breakeven:,.0f}", annotation_position="top left")
            fig_sens.update_layout(**_layout_viu("Mn Briquette Price Sensitivity – Net VIU Advantage (₹/MT)", "Net Advantage (₹/MT)", 420))
            st.plotly_chart(fig_sens, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_c, col_d = st.columns(2)
        
        with col_c:
            mc_prices   = np.linspace(P_Briq_Price * 0.3, P_Briq_Price * 1.2, 80)
            cost_mn_mcs = (1.0 / (P_MCFeMn_Mn * P_MCFeMn_Rec)) * mc_prices
            net_mc_sens = Total_Op_Credits - (Cost_Per_Mn_Briq - cost_mn_mcs)
            
            fig_mc_sens = go.Figure()
            fig_mc_sens.add_trace(go.Scatter(
                x=mc_prices, y=net_mc_sens, mode="lines", name="Net VIU (varying MC FeMn price)",
                line=dict(color=C_MCFEMN, width=3), fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
                hovertemplate="MC FeMn: ₹%{x:,.0f}<br>Net Advantage: ₹%{y:,.0f}/MT<extra></extra>",
            ))
            fig_mc_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
            fig_mc_sens.add_vline(x=P_MCFeMn_Price, line_dash="dot", line_color=C_MCFEMN, line_width=2,
                                  annotation_text=f"Current ₹{P_MCFeMn_Price:,}", annotation_position="top right")
            fig_mc_sens.update_layout(**_layout_viu("MC FeMn Price Sensitivity – Net VIU Advantage (₹/MT)", "Net Advantage (₹/MT)", 380))
            st.plotly_chart(fig_mc_sens, use_container_width=True)
        
        with col_d:
            tornado_names  = ["Power Saving", "Electrode Saving", "Throughput Gain",
                              "Recovery Stability", "Re-treatment", "Cleanliness",
                              "Carbon Avoidance", "Refractory"]
            tornado_base   = [Benefit_Power, Benefit_Electrode, Benefit_Throughput,
                              Benefit_Stability, Benefit_Retreatment, Benefit_Cleanliness,
                              Benefit_Carbon, Benefit_Refractory]
            tornado_low    = [v * 0.80 for v in tornado_base]
            tornado_high   = [v * 1.20 for v in tornado_base]
        
            fig_tornado = go.Figure()
            fig_tornado.add_trace(go.Bar(
                y=tornado_names[::-1], x=[h - b for h, b in zip(tornado_high[::-1], tornado_base[::-1])],
                orientation="h", name="+20%", marker_color=C_DELTA, base=[b for b in tornado_base[::-1]],
            ))
            fig_tornado.add_trace(go.Bar(
                y=tornado_names[::-1], x=[l - b for l, b in zip(tornado_low[::-1], tornado_base[::-1])],
                orientation="h", name="−20%", marker_color="#EF9A9A", base=[b for b in tornado_base[::-1]],
            ))
            fig_tornado.update_layout(barmode="overlay", **_layout_viu("Sensitivity Tornado (±20% Realization)", "₹/MT Alloy", 380))
            st.plotly_chart(fig_tornado, use_container_width=True)
        
        st.markdown("#### Side-by-Side Cost per Active Manganese Summary")
        df_cmp = pd.DataFrame({
            "Metric": [
                "Market Price (₹/MT alloy)", "Active Mn Content (%)", "Mn Recovery (%)",
                "Effective Mn Efficiency (%)", "Alloy Needed per MT Active Mn (MT)",
                "Raw Cost per MT Active Mn (₹)", "Gross Operational Credits (₹/MT alloy)",
                "Lost Iron Credit Penalty (₹/MT alloy)", "Net Adjusted Cost per MT Active Mn (₹)",
            ],
            "MC FeMn": [
                f"₹{P_MCFeMn_Price:,}", f"{P_MCFeMn_Mn*100:.1f}%", f"{P_MCFeMn_Rec*100:.1f}%",
                f"{P_MCFeMn_Mn*P_MCFeMn_Rec*100:.1f}%", f"{Alloy_Per_MT_Mn_MC:.3f} MT",
                f"₹{Cost_Per_Mn_MC:,.0f}", "—", "—", f"₹{Cost_Per_Mn_MC:,.0f}",
            ],
            "Mn Briquette": [
                f"₹{P_Briq_Price:,}", f"{P_Briq_Mn*100:.1f}%", f"{P_Briq_Rec*100:.1f}%",
                f"{P_Briq_Mn*P_Briq_Rec*100:.1f}%", f"{Alloy_Per_MT_Mn_Briq:.3f} MT",
                f"₹{Cost_Per_Mn_Briq:,.0f}", f"₹{Gross_Op_Benefits:,.0f}",
                f"₹{Lost_Iron_Credit_per_MT_Briq:,.0f}", f"₹{Cost_Per_Mn_Briq - Total_Op_Credits:,.0f}",
            ],
        }).set_index("Metric")
        st.dataframe(df_cmp, use_container_width=True)
        
        st.markdown('<div class="section-header">Enterprise Savings Calculator</div>', unsafe_allow_html=True)
        
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(kpi("Substituted Volume", f"{Briq_Consumption_FY * Substitution_Pct:,.0f} MT", f"at {Substitution_Pct*100:.0f}% substitution", ""), unsafe_allow_html=True)
        with s2:
            st.markdown(kpi("Savings / MT Alloy", f"₹{abs(Savings_Per_MT):,.0f}", "Magnitude of net advantage", "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-amber"), unsafe_allow_html=True)
        with s3:
            abs_savings_yr = abs(Annual_Savings_Cr)
            st.markdown(kpi("Annual Savings FY26", f"₹{abs_savings_yr:.2f} Cr", "at stated volume", "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-amber"), unsafe_allow_html=True)
        with s4:
            monthly = Annual_Savings_Cr * 1e7 / 12 / 1e5
            st.markdown(kpi("Monthly Savings", f"₹{abs(monthly):.1f} L", "per month average", "kpi-card-purple"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        col_lft, col_rgt = st.columns([2, 1])
        
        with col_lft:
            vol_range = np.arange(1000, Briq_Consumption_FY * 2.5, 500)
            savings_cr = (Savings_Per_MT * vol_range * Substitution_Pct) / 1e7
        
            fig_sav = go.Figure()
            fig_sav.add_trace(go.Scatter(
                x=vol_range, y=savings_cr, mode="lines", name="Annual Savings (₹ Cr)",
                line=dict(color=C_DELTA if Savings_Per_MT > 0 else C_NEG, width=3),
                fill="tozeroy", fillcolor="rgba(76,175,80,0.12)" if Savings_Per_MT > 0 else "rgba(244,67,54,0.12)",
                hovertemplate="Consumption: %{x:,.0f} MT<br>Savings: ₹%{y:.2f} Cr<extra></extra>",
            ))
            fig_sav.add_vline(
                x=Briq_Consumption_FY, line_dash="dash", line_color="#1A237E", line_width=2,
                annotation_text=f"Total: {Briq_Consumption_FY:,} MT → ₹{Annual_Savings_Cr:.2f} Cr (@ {Substitution_Pct*100:.0f}% Sub)",
                annotation_position="top right",
            )
            fig_sav.add_hline(y=0, line_dash="solid", line_color="#333", line_width=1.5)
            fig_sav.update_layout(**_layout_viu(f"Enterprise Savings vs Total Consumption Volume (at {Substitution_Pct*100:.0f}% Sub)", "Savings (₹ Crore)", 400))
            st.plotly_chart(fig_sav, use_container_width=True)
        
            st.markdown(f"#### 3-Year Savings Projection (5% annual price escalation)")
            years = ["FY 2026", "FY 2027", "FY 2028"]
            escalation = [1.0, 1.05, 1.1025]
            proj_savings = [Annual_Savings_Cr * e for e in escalation]
            cumulative_cr = np.cumsum(proj_savings)
        
            fig_3yr = go.Figure()
            fig_3yr.add_trace(go.Bar(
                x=years, y=proj_savings, name="Annual Savings (₹ Cr)",
                marker_color=[C_DELTA if s > 0 else C_NEG for s in proj_savings],
                text=[f"₹{v:.2f} Cr" for v in proj_savings], textposition="outside",
            ))
            fig_3yr.add_trace(go.Scatter(
                x=years, y=cumulative_cr, mode="lines+markers+text", name="Cumulative (₹ Cr)",
                line=dict(color="#9C27B0", width=2.5, dash="dash"),
                marker=dict(size=9, color="#9C27B0"), text=[f"₹{v:.2f} Cr" for v in cumulative_cr],
                textposition="top center",
            ))
            fig_3yr.update_layout(**_layout_viu("3-Year Enterprise Savings Projection (₹ Crore)", "₹ Crore", 380))
            st.plotly_chart(fig_3yr, use_container_width=True)
        
        with col_rgt:
            st.markdown("#### Per-Benefit Annual Savings (₹ Cr)")
            benefits_annual = {
                n: (v * Briq_Consumption_FY * Substitution_Pct) / 1e7
                for n, v in zip(all_benefit_names, all_benefit_values)
            }
            df_bens = pd.DataFrame({
                "Benefit": list(benefits_annual.keys()),
                "₹ Crore / Year": [round(v, 3) for v in benefits_annual.values()],
            }).sort_values("₹ Crore / Year", ascending=False).set_index("Benefit")
        
            def style_ben(val):
                return "color:#1B5E20;font-weight:600" if val > 0 else "color:#B71C1C;font-weight:600"
        
            st.dataframe(df_bens.style.map(style_ben, subset=["₹ Crore / Year"]), use_container_width=True, height=350)
        
            st.markdown("#### Savings Components Sunburst")
            pos_bens  = [(n, (v * Briq_Consumption_FY * Substitution_Pct) / 1e7) for n, v in zip(all_benefit_names, all_benefit_values) if v > 0]
            sun_labels = ["Gross VIU Credits"] + [p[0] for p in pos_bens]
            sun_parents = [""] + ["Gross VIU Credits"] * len(pos_bens)
            sun_values = [sum(p[1] for p in pos_bens)] + [p[1] for p in pos_bens]
        
            fig_sun = go.Figure(go.Sunburst(
                labels=sun_labels, parents=sun_parents, values=sun_values,
                branchvalues="total", hovertemplate="<b>%{label}</b><br>₹%{value:.3f} Cr<extra></extra>",
                marker=dict(colors=["#1A237E"] + colours_donut[:len(pos_bens)]),
            ))
            fig_sun.update_layout(title="Savings Sunburst (₹ Cr)", template="plotly_white", height=380, margin=dict(l=5, r=5, t=40, b=5))
            st.plotly_chart(fig_sun, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Break-Even Price Analysis")
        be1, be2, be3 = st.columns(3)
        
        briq_eff = P_Briq_Mn * P_Briq_Rec
        mc_eff = P_MCFeMn_Mn * P_MCFeMn_Rec
        
        briq_breakeven_price = (Cost_Per_Mn_MC + Total_Op_Credits) * briq_eff
        mc_breakeven_price  = (Cost_Per_Mn_Briq - Total_Op_Credits) * mc_eff
        min_credits_needed  = Cost_Per_Mn_Delta
        
        with be1:
            st.markdown(kpi("Mn Briq Break-Even Price", f"₹{briq_breakeven_price:,.0f}",
                            f"Current: ₹{P_Briq_Price:,} | {'BELOW' if P_Briq_Price <= briq_breakeven_price else 'ABOVE'} break-even",
                            "kpi-card-green" if P_Briq_Price <= briq_breakeven_price else "kpi-card-amber"), unsafe_allow_html=True)
        with be2:
            st.markdown(kpi("MC FeMn Break-Even Price", f"₹{mc_breakeven_price:,.0f}",
                            f"Current: ₹{P_MCFeMn_Price:,} | {'BELOW' if P_MCFeMn_Price <= mc_breakeven_price else 'ABOVE'} break-even",
                            "kpi-card-green" if P_MCFeMn_Price <= mc_breakeven_price else "kpi-card-amber"), unsafe_allow_html=True)
        with be3:
            st.markdown(kpi("Min. Credits Needed", f"₹{min_credits_needed:,.0f}",
                            f"Current credits: ₹{Total_Op_Credits:,.0f} | {'✅ Sufficient' if Total_Op_Credits >= min_credits_needed else '❌ Insufficient'}",
                            "kpi-card-green" if Total_Op_Credits >= min_credits_needed else "kpi-card-red"), unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Final Recommendation</div>', unsafe_allow_html=True)
        
        if Savings_Per_MT > 0:
            st.markdown(f"""
            <div style="background:#E8F5E9; border-left:6px solid #4CAF50; padding:24px 32px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
                <h2 style="color:#1B5E20; margin-top:0; font-size:28px;">🏆 Mn Briquette Preferred</h2>
                <p style="font-size:16px; color:#2E7D32; line-height:1.6; margin-bottom:0;">
                    <b>Projected Annual Savings: ₹{Annual_Savings_Cr:.2f} Crore</b><br>
                    By shifting {Substitution_Pct*100:.0f}% of your {Briq_Consumption_FY:,} MT baseline consumption to Mn Briquettes, 
                    you realize a net advantage of <b>₹{Savings_Per_MT:,.0f}/MT alloy</b>. 
                    The base price efficiency and operational credits (₹{Total_Op_Credits:,.0f}/MT) make it a highly 
                    economic choice.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#FFF3E0; border-left:6px solid #FF9800; padding:24px 32px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
                <h2 style="color:#E65100; margin-top:0; font-size:28px;">🏆 MC FeMn Preferred</h2>
                <p style="font-size:16px; color:#EF6C00; line-height:1.6; margin-bottom:0;">
                    <b>MC FeMn Cost Efficiency: ₹{abs(Savings_Per_MT):,.0f}/MT alloy</b><br>
                    At current input parameters, MC FeMn remains the more cost-effective option, yielding a projected <b>₹{abs(Annual_Savings_Cr):.2f} Crore</b> in savings vs switching. 
                    The Briquette operational credits (₹{Total_Op_Credits:,.0f}/MT) 
                    do not fully offset the active Mn cost dynamics. 
                    Adjust substitution strategies or renegotiate market pricing to break-even.
                </p>
            </div>
            """, unsafe_allow_html=True)

    elif comparison_selection == "FeSi vs Si Metal":
        # ── SECTION 1: DASHBOARD HEADER ──
        st.markdown("""
        <div style="background: linear-gradient(135deg,#263238 0%,#37474F 60%,#00838F 100%);
                    padding:22px 28px 18px 28px; border-radius:14px; margin-bottom:20px;
                    box-shadow:0 4px 24px rgba(38,50,56,0.25);">
          <h1 style="color:#FFFFFF;margin:0;font-size:26px;font-weight:800;letter-spacing:0.02em;">
            🔥 VIU Dashboard — FeSi70 vs Si Metal
          </h1>
          <p style="color:#B2EBF2;margin:6px 0 0 0;font-size:13px;">
            Value-In-Use Economic Analysis &nbsp;|&nbsp; Standard Ferrosilicon (70% Si) 
            vs High-Purity Silicon Metal (98% Si)
          </p>
        </div>
        """, unsafe_allow_html=True)

        # ── SECTION 2: TOP KPI CARDS ──
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(kpi("FeSi70 Price", f"₹{P_FeSi_Price:,.0f}", "per MT alloy", ""), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("Si Metal Price", f"₹{P_SiMetal_Price:,.0f}", "per MT alloy", "kpi-card-teal"), unsafe_allow_html=True)
        with c3:
            lbl_gap = "Si Cost Advantage" if Cost_Per_Si_Delta >= 0 else "Si Cost Premium"
            col_gap = "kpi-card-teal" if Cost_Per_Si_Delta >= 0 else "kpi-card-amber"
            st.markdown(kpi(lbl_gap, f"₹{abs(Cost_Per_Si_Delta):,.0f}", "per MT Active Si", col_gap), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi("Total VIU Credits", f"₹{Total_Op_Credits:,.0f}", "net benefit / MT alloy", "kpi-card-teal"), unsafe_allow_html=True)
        with c5:
            col = "kpi-card-teal" if Savings_Per_MT > 0 else "kpi-card-red"
            st.markdown(kpi("Net Savings / MT", f"₹{Savings_Per_MT:+,.0f}", "Si Metal advantage", col), unsafe_allow_html=True)
        with c6:
            col_yr = "kpi-card-teal" if Savings_Per_MT > 0 else "kpi-card-red"
            st.markdown(kpi("Annual Savings FY", f"₹{abs(Annual_Savings_Cr):.2f} Cr", f"@ {Substitution_Pct*100:.0f}% Substitution", col_yr), unsafe_allow_html=True)

        # ── SECTION 3: VIU SUMMARY ──
        st.markdown('<div class="section-header">VIU Economic Synthesis</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns([1, 1])
        with col_l:
            st.markdown("#### Cost per Active Silicon (₹/MT Si)")
            km1, km2 = st.columns(2)
            with km1:
                st.markdown(kpi("FeSi Cost/MT Si", f"₹{Cost_Per_Si_FeSi:,.0f}", f"@ {P_FeSi_Si*100:.1f}% Si × {P_FeSi_Rec*100:.0f}% rec", ""), unsafe_allow_html=True)
            with km2:
                st.markdown(kpi("Si Metal Cost/MT Si", f"₹{Cost_Per_Si_SiMetal:,.0f}", f"@ {P_SiMetal_Si*100:.1f}% Si × {P_SiMetal_Rec*100:.0f}% rec", "kpi-card-teal"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### VIU Components (Per MT of Si Metal)")
            data_summary = {
                "Component": [
                    "FeSi Cost/MT Si",
                    "Si Metal Cost/MT Si",
                    "Si Cost Advantage",
                    "Gross Operational Credits",
                    "Lost Iron Credit Penalty",
                    "Total Net Credits",
                    "Total Net Advantage",
                ],
                "Value (₹/MT Alloy)": [
                    f"₹{Cost_Per_Si_FeSi:,.0f}", 
                    f"₹{Cost_Per_Si_SiMetal:,.0f}", 
                    f"₹{Cost_Per_Si_Delta:,.0f}",
                    f"₹{Gross_Op_Benefits:,.0f}", 
                    f"-₹{Iron_Credit_FeSi:,.0f}",
                    f"₹{Total_Op_Credits:,.0f}", 
                    f"₹{Savings_Per_MT:+,.0f}",
                ],
            }
            df_sum = pd.DataFrame(data_summary).set_index("Component")
            st.dataframe(df_sum, use_container_width=True)

            # Verdict
            if Savings_Per_MT > 0:
                st.markdown(f"""
                <div class="success-box">
                ✅ <b>Si Metal offers a net advantage of ₹{Savings_Per_MT:,.0f}/MT alloy.</b><br>
                Direct cost efficiencies combined with strong operational credits make Si Metal the economically superior choice.
                </div>""", unsafe_allow_html=True)
            elif Savings_Per_MT < -2000:
                st.markdown(f"""
                <div class="warn-box">
                ⚠️ <b>FeSi70 is currently more cost-effective by ₹{abs(Savings_Per_MT):,.0f}/MT.</b><br>
                At current prices and parameters, the FeSi price advantage outweighs operational credits of Si Metal.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="info-box">
                ℹ️ <b>Near economic parity.</b> Net VIU: ₹{Savings_Per_MT:+,.0f}/MT alloy.
                Consider plant-specific factors and specific grade cleanliness requirements.
                </div>""", unsafe_allow_html=True)

        with col_r:
            # --- VIU Donut: Credit composition ---
            benefit_names = [
                "Power Saving", "Electrode Saving", "Throughput Gain",
                "Recovery Stability", "Slag Handling", "Inclusion Cleanliness",
                "Yield Improvement", "Ca-Wire Reduction", "Re-treatment Reduction", "Safety & Storage"
            ]
            benefit_values = [
                Benefit_Power, Benefit_Electrode, Benefit_Throughput,
                Benefit_Stability, Benefit_Slag, Benefit_Cleanliness,
                Benefit_Yield, Benefit_CaWire, Benefit_Retreatment, Benefit_Safety
            ]
            # Only positive credits for the donut
            pos_names  = [n for n, v in zip(benefit_names, benefit_values) if v > 0]
            pos_values = [v for v in benefit_values if v > 0]

            colours_donut = [
                "#00BCD4", "#009688", "#4CAF50", "#8BC34A", "#CDDC39", 
                "#FFC107", "#FF9800", "#FF5722", "#795548", "#607D8B"
            ]

            fig_donut = go.Figure(data=[go.Pie(
                labels=pos_names, values=pos_values,
                hole=0.52,
                marker=dict(colors=colours_donut[:len(pos_names)], line=dict(color="#fff", width=2)),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}/MT<extra></extra>",
            )])
            fig_donut.add_annotation(
                text=f"<b>₹{Gross_Op_Benefits:,.0f}</b><br><span style='font-size:10px'>Gross Credits</span>",
                x=0.5, y=0.5, font_size=14, showarrow=False,
            )
            fig_donut.update_layout(
                title="Gross Operational Credit Composition (₹/MT Si Metal)",
                template="plotly_white", height=420,
                margin=dict(l=20, r=20, t=55, b=20),
                legend=dict(font=dict(size=11)),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

            # Summary KPIs
            k1, k2 = st.columns(2)
            with k1:
                st.markdown(kpi("FeSi Needed/MT Act. Si", f"{Alloy_Per_MT_Si_FeSi:.3f} MT", "FeSi70 mass to buy", ""), unsafe_allow_html=True)
            with k2:
                st.markdown(kpi("Si Metal Needed/MT Act. Si", f"{Alloy_Per_MT_Si_SiMetal:.3f} MT", "Si Metal mass to buy", "kpi-card-teal"), unsafe_allow_html=True)

        # ── SECTION 4: BENEFIT BREAKDOWN ──
        st.markdown('<div class="section-header">Detailed Benefit Breakdown</div>', unsafe_allow_html=True)

        all_benefit_names = benefit_names
        all_benefit_values = benefit_values
        all_benefit_basis = [
            f"ΔT={Delta_Temp_Rise:.2f}°C/kg Si, {P_LF_Efficiency*100:.0f}% LF eff, {R_Power*100:.0f}% real.",
            f"P_kWh_saved={Power_kWh_Saved_Per_MT:.1f} kWh/MT, {P_Graphite_Factor*1000:.1f}g/kWh, {R_Electrode*100:.0f}% real.",
            f"Time saved={Time_Saved_SiMetal:.1f} min/heat, {R_Throughput*100:.0f}% real.",
            f"Overdose Δ={(FeSi_Overdose-SiMetal_Overdose)*100:.1f}%, {R_Stability*100:.0f}% real.",
            f"Slag drop={Slag_Reduction}kg/T, {Steel_Per_MT_SiMetal:.0f}T support, {R_Slag*100:.0f}% real.",
            f"Reject Δ={(Reject_FeSi-Reject_SiMetal)*100:.4f}%, {R_Cleanliness*100:.0f}% real.",
            f"Yield gain={Yield_Gain_SiMetal*100:.2f}%, {R_Yield*100:.0f}% real.",
            f"Ca-wire Δ={CaWire_FeSi - CaWire_SiMetal:.2f}kg/T, {R_CaWire*100:.0f}% real.",
            f"Miss freq Δ={(Retreatment_FeSi-Retreatment_SiMetal)*100:.1f}%, {R_Retreatment*100:.0f}% real.",
            f"Avoided compliance and gas hazard cost, {R_Safety*100:.0f}% real.",
        ]

        col_chart, col_table = st.columns([3, 2])

        with col_chart:
            bar_colors = [C_DELTA if v >= 0 else C_NEG for v in all_benefit_values]
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=all_benefit_names[::-1],
                x=all_benefit_values[::-1],
                orientation="h",
                marker=dict(color=bar_colors[::-1], line=dict(color="white", width=1)),
                text=[f"₹{v:+,.0f}" for v in all_benefit_values[::-1]],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}/MT alloy<extra></extra>",
            ))
            fig_bar.add_vline(x=0, line_dash="solid", line_color="#333", line_width=1.5)
            fig_bar.update_layout(
                **_layout_viu_fesi("Gross Benefit Contribution per MT Si Metal (₹/MT)", "₹/MT Alloy", 460)
            )
            fig_bar.update_layout(xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False))
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_table:
            df_breakdown = pd.DataFrame({
                "Benefit Component": all_benefit_names,
                "₹/MT Alloy": [f"₹{v:+,.0f}" for v in all_benefit_values],
                "Basis & Assumptions": all_benefit_basis,
            }).set_index("Benefit Component")

            def color_values(val):
                num = float(val.replace("₹", "").replace(",", "").replace("+", ""))
                if num > 0:
                    return "color: #1B5E20; font-weight: 600"
                elif num < 0:
                    return "color: #B71C1C; font-weight: 600"
                return ""

            st.dataframe(
                df_breakdown.style.map(color_values, subset=["₹/MT Alloy"]),
                use_container_width=True, height=460,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Heatmap of benefits by realization factor sensitivity
        st.markdown("#### Benefit Sensitivity Heatmap (₹/MT at varying Realization Factors)")
        real_range = np.arange(0.1, 1.05, 0.1)

        # Back out realization to get raw theoretical maximum value
        raw_heat_values = [
            Benefit_Power / R_Power,
            Benefit_Electrode / R_Electrode,
            Benefit_Throughput / R_Throughput,
            Benefit_Stability / R_Stability,
            Benefit_Slag / R_Slag,
            Benefit_Cleanliness / R_Cleanliness,
            Benefit_Yield / R_Yield,
            Benefit_CaWire / R_CaWire,
            Benefit_Retreatment / R_Retreatment,
            Benefit_Safety / (R_Safety if R_Safety > 0 else 1),
        ]
        heat_matrix = np.array([[raw_val * r for r in real_range] for raw_val in raw_heat_values])

        fig_heat = go.Figure(go.Heatmap(
            z=heat_matrix,
            x=[f"{r*100:.0f}%" for r in real_range],
            y=all_benefit_names,
            colorscale="Teal",
            text=np.round(heat_matrix, 0).astype(int),
            texttemplate="₹%{text}",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>Realization: %{x}<br>₹%{z:,.0f}/MT<extra></extra>",
        ))
        fig_heat.update_layout(
            **_layout_viu_fesi("VIU Benefit Heatmap — Realization Factor Sensitivity", "", 380)
        )
        fig_heat.update_layout(xaxis_title="Realization Factor", yaxis_title="")
        st.plotly_chart(fig_heat, use_container_width=True)

        # ── SECTION 5: WATERFALL ANALYSIS ──
        st.markdown('<div class="section-header">VIU Waterfall Analysis</div>', unsafe_allow_html=True)

        Equivalent_FeSi_Cost = Direct_Cost_Saving_Per_MT_SiMetal + P_SiMetal_Price

        wf_labels = [
            "Equivalent FeSi Job Cost",
            "Power Saving",
            "Electrode Saving",
            "Throughput Gain",
            "Recovery Stability",
            "Slag Handling",
            "Inclusion Cleanliness",
            "Yield Improvement",
            "Ca-Wire Reduction",
            "Re-treatment Reduction",
            "Safety & Storage",
            "Lost Iron Credit",
            "Breakeven Si Metal Value",
        ]
        # Start from equivalent replacement cost, subtract benefits to find the effective target price (breakeven)
        wf_values = [
            Equivalent_FeSi_Cost,     # base equivalent cost
            -Benefit_Power,           # benefits drive DOWN the effective cost to replace
            -Benefit_Electrode,
            -Benefit_Throughput,
            -Benefit_Stability,
            -Benefit_Slag,
            -Benefit_Cleanliness,
            -Benefit_Yield,
            -Benefit_CaWire,
            -Benefit_Retreatment,
            -Benefit_Safety,
            Iron_Credit_FeSi,         # iron credit loss drives UP the effective cost
            0,                        # total placeholder
        ]

        measures = ["absolute"] + ["relative"] * (len(wf_labels) - 2) + ["total"]
        breakeven_value = Equivalent_FeSi_Cost - Total_Op_Credits
        wf_text = [f"₹{abs(v):,.0f}" for v in wf_values[:-1]] + [f"₹{breakeven_value:,.0f}"]

        wf_values_display = wf_values[:-1] + [breakeven_value]

        fig_wf = go.Figure(go.Waterfall(
            name="VIU Waterfall",
            orientation="v",
            measure=measures,
            x=wf_labels,
            y=wf_values_display,
            text=wf_text,
            textposition="outside",
            connector=dict(line=dict(color="#BDBDBD", width=1.5, dash="dot")),
            increasing=dict(marker=dict(color=C_NEG)),    # Costs/Penalties making it harder to justify
            decreasing=dict(marker=dict(color=C_DELTA)),  # Benefits making Si Metal effectively cheaper
            totals=dict(marker=dict(color=C_SIMETAL if breakeven_value >= P_SiMetal_Price else C_NEG)),
            hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
        ))

        fig_wf.add_hline(
            y=P_SiMetal_Price, line_dash="dash", line_color=C_SIMETAL, line_width=2,
            annotation_text=f"Market Price: ₹{P_SiMetal_Price:,.0f}", annotation_position="right",
        )
        fig_wf.update_layout(
            **_layout_viu_fesi("VIU Waterfall: Finding the Breakeven Value of Si Metal (₹/MT Alloy)", "₹/MT Si Metal", 520)
        )
        fig_wf.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_wf, use_container_width=True)

        st.markdown("""
        <div class="info-box">
        <b>How to read this waterfall:</b> We start with the <b>Equivalent FeSi Job Cost</b> (what it would cost in FeSi to achieve the same active Si mass as 1 MT of Si Metal). 
        We then subtract the operational savings Si Metal provides. Finally, we add back the penalty of lost iron credits. 
        The final bar is the <b>Breakeven Value</b> (the maximum you should theoretically pay for 1 MT of Si Metal). 
        If this Breakeven Value sits <b>above</b> the dashed Market Price line, switching to Si Metal captures net savings.
        </div>
        """, unsafe_allow_html=True)

        # ── SECTION 6: COST COMPARISON ──
        st.markdown('<div class="section-header">Cost Comparison & Sensitivity Analysis</div>', unsafe_allow_html=True)

        col_a, col_b = st.columns(2)

        with col_a:
            # Stacked bar: cost components per MT Active Si
            fig_stack = go.Figure()
            categories = ["FeSi70", "Si Metal"]

            # For Si Metal, we apply credits to the Active Si cost
            # Factor to convert MT Si Metal to MT Active Si
            factor = 1.0 / (P_SiMetal_Si * P_SiMetal_Rec)

            fig_stack.add_trace(go.Bar(
                name="Cost per MT Active Si", x=categories,
                y=[Cost_Per_Si_FeSi, Cost_Per_Si_SiMetal],
                marker_color=[C_FESI, C_SIMETAL],
                text=[f"₹{Cost_Per_Si_FeSi:,.0f}", f"₹{Cost_Per_Si_SiMetal:,.0f}"],
                textposition="inside",
            ))
            fig_stack.add_trace(go.Bar(
                name="Gross Op. Credits (deduct)", x=categories,
                y=[0, -(Gross_Op_Benefits * factor)],
                marker_color=["rgba(0,0,0,0)", "#FFC107"],
                text=["", f"-₹{(Gross_Op_Benefits * factor):,.0f}"],
                textposition="inside",
            ))
            fig_stack.add_trace(go.Bar(
                name="Lost Fe Credit Penalty (add)", x=categories,
                y=[0, (Iron_Credit_FeSi * factor)],
                marker_color=["rgba(0,0,0,0)", "#FF7043"],
                text=["", f"+₹{(Iron_Credit_FeSi * factor):,.0f}"],
                textposition="inside",
            ))
            fig_stack.update_layout(
                barmode="relative",
                **_layout_viu_fesi("Effective Cost Components (₹/MT Active Silicon)", "₹/MT", 420),
            )
            st.plotly_chart(fig_stack, use_container_width=True)

        with col_b:
            # Si Metal Price sensitivity on Net VIU
            si_prices  = np.linspace(P_FeSi_Price * 0.8, P_FeSi_Price * 1.8, 80)
            
            # Recalculate net savings dynamically across array of si_prices
            cost_si_array = (1.0 / (P_SiMetal_Si * P_SiMetal_Rec)) * si_prices
            delta_si_array = Cost_Per_Si_FeSi - cost_si_array
            direct_saving_array = delta_si_array * (P_SiMetal_Si * P_SiMetal_Rec)
            net_viuss = direct_saving_array + Total_Op_Credits
            
            breakeven_si = breakeven_value # Corrected variable map

            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=si_prices, y=net_viuss,
                mode="lines", name="Net VIU Advantage",
                line=dict(color=C_SIMETAL, width=3),
                fill="tozeroy",
                fillcolor="rgba(0,150,136,0.1)",
                hovertemplate="Si Metal Price: ₹%{x:,.0f}<br>Net Advantage: ₹%{y:,.0f}/MT<extra></extra>",
            ))
            fig_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
            fig_sens.add_vline(x=P_SiMetal_Price, line_dash="dot", line_color=C_SIMETAL, line_width=2,
                               annotation_text=f"Current ₹{P_SiMetal_Price:,}", annotation_position="top right")
            fig_sens.add_vline(x=breakeven_si, line_dash="dot", line_color=C_NEG, line_width=2,
                               annotation_text=f"Break-even ₹{breakeven_si:,.0f}", annotation_position="top left")
            fig_sens.update_layout(
                **_layout_viu_fesi("Si Metal Price Sensitivity – Net VIU Advantage (₹/MT)", "Net Advantage (₹/MT Alloy)", 420)
            )
            st.plotly_chart(fig_sens, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            # FeSi Price Sensitivity
            fesi_prices = np.linspace(P_SiMetal_Price * 0.5, P_SiMetal_Price * 1.1, 80)
            cost_fe_array = (1.0 / (P_FeSi_Si * P_FeSi_Rec)) * fesi_prices
            delta_fe_array = cost_fe_array - Cost_Per_Si_SiMetal
            direct_fe_saving = delta_fe_array * (P_SiMetal_Si * P_SiMetal_Rec)
            net_fesi_sens = direct_fe_saving + Total_Op_Credits
            
            fig_fesi_sens = go.Figure()
            fig_fesi_sens.add_trace(go.Scatter(
                x=fesi_prices, y=net_fesi_sens,
                mode="lines", name="Net VIU (varying FeSi price)",
                line=dict(color=C_FESI, width=3),
                fill="tozeroy",
                fillcolor="rgba(96,125,139,0.1)",
                hovertemplate="FeSi Price: ₹%{x:,.0f}<br>Net Advantage: ₹%{y:,.0f}/MT<extra></extra>",
            ))
            fig_fesi_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
            fig_fesi_sens.add_vline(x=P_FeSi_Price, line_dash="dot", line_color=C_FESI, line_width=2,
                                  annotation_text=f"Current ₹{P_FeSi_Price:,}", annotation_position="top right")
            fig_fesi_sens.update_layout(
                **_layout_viu_fesi("FeSi70 Price Sensitivity – Net VIU Advantage (₹/MT)", "Net Advantage (₹/MT Alloy)", 380)
            )
            st.plotly_chart(fig_fesi_sens, use_container_width=True)

        with col_d:
            # Tornado chart: individual benefit sensitivity (±20%)
            tornado_names  = all_benefit_names
            tornado_base   = all_benefit_values
            tornado_low    = [v * 0.80 for v in tornado_base]
            tornado_high   = [v * 1.20 for v in tornado_base]

            fig_tornado = go.Figure()
            fig_tornado.add_trace(go.Bar(
                y=tornado_names[::-1], x=[h - b for h, b in zip(tornado_high[::-1], tornado_base[::-1])],
                orientation="h", name="+20%", marker_color=C_SIMETAL,
                base=[b for b in tornado_base[::-1]],
            ))
            fig_tornado.add_trace(go.Bar(
                y=tornado_names[::-1], x=[l - b for l, b in zip(tornado_low[::-1], tornado_base[::-1])],
                orientation="h", name="−20%", marker_color="#EF9A9A",
                base=[b for b in tornado_base[::-1]],
            ))
            fig_tornado.update_layout(
                barmode="overlay",
                **_layout_viu_fesi("Sensitivity Tornado (±20% Realization)", "₹/MT Alloy", 380),
            )
            st.plotly_chart(fig_tornado, use_container_width=True)

        # ── SECTION 7: ENTERPRISE SAVINGS ──
        st.markdown('<div class="section-header">Enterprise Savings Calculator</div>', unsafe_allow_html=True)

        s1, s2, s3, s4 = st.columns(4)
        with s1:
            st.markdown(kpi("Substituted Volume", f"{SiMetal_Consumption_FY * Substitution_Pct:,.0f} MT", f"at {Substitution_Pct*100:.0f}% substitution", ""), unsafe_allow_html=True)
        with s2:
            st.markdown(kpi("Savings / MT Alloy", f"₹{abs(Savings_Per_MT):,.0f}", "Magnitude of net advantage", "kpi-card-teal" if Savings_Per_MT > 0 else "kpi-card-amber"), unsafe_allow_html=True)
        with s3:
            abs_savings_yr = abs(Annual_Savings_Cr)
            st.markdown(kpi("Annual Savings FY", f"₹{abs_savings_yr:.2f} Cr", "at stated volume", "kpi-card-teal" if Savings_Per_MT > 0 else "kpi-card-amber"), unsafe_allow_html=True)
        with s4:
            monthly = Annual_Savings_Cr * 1e7 / 12 / 1e5
            st.markdown(kpi("Monthly Savings", f"₹{abs(monthly):.1f} L", "per month average", "kpi-card-purple"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_lft, col_rgt = st.columns([2, 1])

        with col_lft:
            # Savings vs consumption volume chart
            vol_range = np.arange(1000, SiMetal_Consumption_FY * 2.5, 500)
            savings_cr = (Savings_Per_MT * vol_range * Substitution_Pct) / 1e7

            fig_sav = go.Figure()
            fig_sav.add_trace(go.Scatter(
                x=vol_range, y=savings_cr,
                mode="lines", name="Annual Savings (₹ Cr)",
                line=dict(color=C_SIMETAL if Savings_Per_MT > 0 else C_NEG, width=3),
                fill="tozeroy",
                fillcolor="rgba(0,150,136,0.12)" if Savings_Per_MT > 0 else "rgba(244,67,54,0.12)",
                hovertemplate="Consumption: %{x:,.0f} MT<br>Savings: ₹%{y:.2f} Cr<extra></extra>",
            ))
            fig_sav.add_vline(
                x=SiMetal_Consumption_FY, line_dash="dash", line_color="#263238", line_width=2,
                annotation_text=f"Total: {SiMetal_Consumption_FY:,} MT → ₹{Annual_Savings_Cr:.2f} Cr",
                annotation_position="top right",
            )
            fig_sav.add_hline(y=0, line_dash="solid", line_color="#333", line_width=1.5)
            fig_sav.update_layout(
                **_layout_viu_fesi(f"Enterprise Savings vs Baseline Volume (at {Substitution_Pct*100:.0f}% Sub)", "Savings (₹ Crore)", 400)
            )
            st.plotly_chart(fig_sav, use_container_width=True)

        with col_rgt:
            # Savings breakdown by benefit
            st.markdown("#### Operational Component Values (₹ Cr)")
            benefits_annual = {
                n: (v * SiMetal_Consumption_FY * Substitution_Pct) / 1e7
                for n, v in zip(all_benefit_names, all_benefit_values)
            }
            df_bens = pd.DataFrame({
                "Benefit": list(benefits_annual.keys()),
                "₹ Crore / Year": [round(v, 3) for v in benefits_annual.values()],
            }).sort_values("₹ Crore / Year", ascending=False).set_index("Benefit")

            def style_ben(val):
                return "color:#1B5E20;font-weight:600" if val > 0 else "color:#B71C1C;font-weight:600"

            st.dataframe(
                df_bens.style.map(style_ben, subset=["₹ Crore / Year"]),
                use_container_width=True, height=350,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown('<div class="section-header">Break-Even Price Analysis</div>', unsafe_allow_html=True)
        be1, be2, be3 = st.columns(3)
        
        # 1. Si Metal Break-Even
        # Break-Even point where Net Savings = 0
        breakeven_si = (Cost_Per_Si_FeSi + Total_Op_Credits) / Alloy_Per_MT_Si_SiMetal
        si_be_status = "BELOW break-even" if P_SiMetal_Price <= breakeven_si else "ABOVE break-even"
        si_be_color = "kpi-card-teal" if P_SiMetal_Price <= breakeven_si else "kpi-card-amber"
        with be1:
            st.markdown(kpi("Si Metal Break-Even Price", f"₹{breakeven_si:,.0f}", f"Current: ₹{P_SiMetal_Price:,.0f} | {si_be_status}", si_be_color), unsafe_allow_html=True)

        # 2. FeSi70 Break-Even
        # Break-Even point where Net Savings = 0
        breakeven_fesi = (Cost_Per_Si_SiMetal - Total_Op_Credits) / Alloy_Per_MT_Si_FeSi
        fesi_be_status = "ABOVE break-even" if P_FeSi_Price >= breakeven_fesi else "BELOW break-even"
        fesi_be_color = "kpi-card-teal" if P_FeSi_Price >= breakeven_fesi else "kpi-card-amber"
        with be2:
            st.markdown(kpi("FeSi70 Break-Even Price", f"₹{breakeven_fesi:,.0f}", f"Current: ₹{P_FeSi_Price:,.0f} | {fesi_be_status}", fesi_be_color), unsafe_allow_html=True)

        # 3. Min. Credits Needed
        min_credits = -Cost_Per_Si_Delta
        if min_credits <= 0:
            cred_stat = "Chemically cheaper (0 needed)"
            cred_col = "kpi-card-teal"
        else:
            cred_stat = "Credits offset premium" if Total_Op_Credits >= min_credits else "Shortfall in credits"
            cred_col = "kpi-card-teal" if Total_Op_Credits >= min_credits else "kpi-card-amber"
            
        with be3:
            st.markdown(kpi("Min. Credits Needed", f"₹{min_credits:,.0f}", f"Current Credits: ₹{Total_Op_Credits:,.0f} | {cred_stat}", cred_col), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SECTION 8: RECOMMENDATION ──
        st.markdown('<div class="section-header">Final Recommendation</div>', unsafe_allow_html=True)

        if Savings_Per_MT > 0:
            st.markdown(f"""
            <div style="background:#E0F2F1; border-left:6px solid #009688; padding:24px 32px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
                <h2 style="color:#004D40; margin-top:0; font-size:28px;">🏆 Silicon Metal Preferred</h2>
                <p style="font-size:16px; color:#00695C; line-height:1.6; margin-bottom:0;">
                    <b>Projected Annual Savings: ₹{Annual_Savings_Cr:.2f} Crore</b><br>
                    By shifting {Substitution_Pct*100:.0f}% of your {SiMetal_Consumption_FY:,} MT baseline to High-Purity Silicon Metal, 
                    you capture an enormous net advantage of <b>₹{Savings_Per_MT:,.0f}/MT alloy</b>. 
                    The combination of direct active silicon cost-competitiveness and substantial operational credits (₹{Total_Op_Credits:,.0f}/MT) 
                    makes Si Metal highly lucrative for this application.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#FFF3E0; border-left:6px solid #FF9800; padding:24px 32px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
                <h2 style="color:#E65100; margin-top:0; font-size:28px;">🏆 FeSi70 Preferred</h2>
                <p style="font-size:16px; color:#EF6C00; line-height:1.6; margin-bottom:0;">
                    <b>FeSi Cost Efficiency: ₹{abs(Savings_Per_MT):,.0f}/MT alloy</b><br>
                    At current input parameters, standard FeSi70 remains the more cost-effective option, yielding a projected <b>₹{Annual_Savings_Cr:.2f} Crore</b> in savings vs switching. 
                    The Si Metal operational credits (₹{Total_Op_Credits:,.0f}/MT) are not currently strong enough 
                    to justify the premium pricing required to match the chemical delivery of FeSi70.
                </p>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════

    elif comparison_selection == "Primary Al Ingot vs Secondary Al Ingot vs Al Notch Bar":
        # --- 1. Effective Al Delivery ---
        Eff_Pri   = P_Pri_Purity * P_Pri_Rec
        Eff_Sec   = P_Sec_Purity * P_Sec_Rec
        Eff_Notch = P_Notch_Purity * P_Notch_Rec
    
        Total_Eff_Al_Req = Al_Consumption_FY * Eff_Pri
        Req_Pri   = 1.0 / Eff_Pri
        Req_Sec   = 1.0 / Eff_Sec
        Req_Notch = 1.0 / Eff_Notch
    
        Raw_Cost_Pri   = P_Pri_Price * Req_Pri
        Raw_Cost_Sec   = P_Sec_Price * Req_Sec
        Raw_Cost_Notch = P_Notch_Price * Req_Notch
    
        Active_Al_kg_T = (Active_Al / 100.0) * 1000.0
        Alloy_T_Pri   = Active_Al_kg_T / Eff_Pri
        Alloy_T_Sec   = Active_Al_kg_T / Eff_Sec
        Alloy_T_Notch = Active_Al_kg_T / Eff_Notch
    
        Steel_Pri   = 1000.0 / Alloy_T_Pri
        Steel_Sec   = 1000.0 / Alloy_T_Sec
        Steel_Notch = 1000.0 / Alloy_T_Notch
    
        # --- 2. Operational Penalties (Relative to Primary Al Baseline) ---
        Extra_kWh_Sec   = (Dross_Pri_Sec * P_SpHeat) / 3.6 / P_LF_Efficiency
        Extra_kWh_Notch = (Dross_Pri_Notch * P_SpHeat) / 3.6 / P_LF_Efficiency
        Pen_Power_Sec   = Extra_kWh_Sec * P_Power_Cost * R_Power
        Pen_Power_Notch = Extra_kWh_Notch * P_Power_Cost * R_Power
    
        Pen_Elec_Sec   = Extra_kWh_Sec * P_Elec_Wear * P_Electrode_Cost * R_Electrode
        Pen_Elec_Notch = Extra_kWh_Notch * P_Elec_Wear * P_Electrode_Cost * R_Electrode
    
        Pen_Time_Sec   = (Extra_Time_Sec / P_Cycle_Time) * P_Heat_Size * P_Margin_Steel * (1000.0 / (Alloy_T_Sec * P_Heat_Size)) * R_Throughput
        Pen_Time_Notch = (Extra_Time_Notch / P_Cycle_Time) * P_Heat_Size * P_Margin_Steel * (1000.0 / (Alloy_T_Notch * P_Heat_Size)) * R_Throughput
    
        Pen_Stab_Sec   = (P_Sec_Overdose - P_Pri_Overdose) * (P_Sec_Price / 1000.0) * Steel_Sec * R_Stability
        Pen_Stab_Notch = (P_Notch_Overdose - P_Pri_Overdose) * (P_Notch_Price / 1000.0) * Steel_Notch * R_Stability
    
        Pen_Slag_Sec   = Slag_Pri_Sec * (P_Slag_Cost / 1000.0) * R_Slag
        Pen_Slag_Notch = Slag_Pri_Notch * (P_Slag_Cost / 1000.0) * R_Slag
    
        Pen_Clean_Sec   = (P_Sec_Reject - P_Pri_Reject) * P_Steel_Value * Steel_Sec * R_Cleanliness
        Pen_Clean_Notch = (P_Notch_Reject - P_Pri_Reject) * P_Steel_Value * Steel_Notch * R_Cleanliness
    
        Pen_Yield_Sec   = (P_Pri_Yield - P_Sec_Yield) * P_Steel_Value * Steel_Sec * R_Yield
        Pen_Yield_Notch = (P_Pri_Yield - P_Notch_Yield) * P_Steel_Value * Steel_Notch * R_Yield
    
        Pen_Reblow_Sec   = (P_Sec_Retreat - P_Pri_Retreat) * P_LF_Retreatment * (Steel_Sec / P_Heat_Size) * R_Reblow
        Pen_Reblow_Notch = (P_Notch_Retreat - P_Pri_Retreat) * P_LF_Retreatment * (Steel_Notch / P_Heat_Size) * R_Reblow
    
        penalty_names = [
            "Power Penalty", "Electrode Penalty", "Throughput Penalty",
            "Recovery Stability", "Slag Handling",
            "Cleanliness", "Yield Loss", "Re-treatment Risk"
        ]
        pen_sec_vals = [
            Pen_Power_Sec, Pen_Elec_Sec, Pen_Time_Sec,
            Pen_Stab_Sec, Pen_Slag_Sec, Pen_Clean_Sec,
            Pen_Yield_Sec, Pen_Reblow_Sec
        ]
        pen_notch_vals = [
            Pen_Power_Notch, Pen_Elec_Notch, Pen_Time_Notch,
            Pen_Stab_Notch, Pen_Slag_Notch, Pen_Clean_Notch,
            Pen_Yield_Notch, Pen_Reblow_Notch
        ]
    
        Total_Pen_Sec   = sum(pen_sec_vals)
        Total_Pen_Notch = sum(pen_notch_vals)
    
        Conv_Pen_Sec   = Total_Pen_Sec * Req_Sec
        Conv_Pen_Notch = Total_Pen_Notch * Req_Notch
    
        Adj_Cost_Pri   = Raw_Cost_Pri
        Adj_Cost_Sec   = Raw_Cost_Sec + Conv_Pen_Sec
        Adj_Cost_Notch = Raw_Cost_Notch + Conv_Pen_Notch
    
        costs = {"Primary Al Ingot": Adj_Cost_Pri, "Secondary Al Ingot": Adj_Cost_Sec, "Al Notch Bar": Adj_Cost_Notch}
        best_commodity = min(costs, key=costs.get)
        best_cost = costs[best_commodity]
    
        Savings_Per_MT_Eff_Al = Adj_Cost_Pri - best_cost
        Savings_Rs            = (Adj_Cost_Pri - best_cost) * Al_Consumption_FY * Substitution_Pct
        Annual_Savings_Cr     = Savings_Rs / 1e7
    
        Cost_Gap_Pri_Sec   = Raw_Cost_Pri - Raw_Cost_Sec
        Cost_Gap_Pri_Notch = Raw_Cost_Pri - Raw_Cost_Notch
    
        sec_costlier_text   = "Primary Costlier" if Cost_Gap_Pri_Sec > 0 else "Secondary Costlier"
        notch_costlier_text = "Primary Costlier" if Cost_Gap_Pri_Notch > 0 else "Notch Costlier"
    
        Total_VIU_Credits_Pri_Sec   = Conv_Pen_Sec
        Total_VIU_Credits_Pri_Notch = Conv_Pen_Notch
    
        Net_Savings_Pri_Sec   = Adj_Cost_Pri - Adj_Cost_Sec
        Net_Savings_Pri_Notch = Adj_Cost_Pri - Adj_Cost_Notch
    
        net_sav_label_sec   = "Secondary Favored" if Net_Savings_Pri_Sec > 0 else "Primary Favored"
        net_sav_color_sec   = "kpi-card-green" if Net_Savings_Pri_Sec > 0 else "kpi-card-red"
        net_sav_text_sec    = f"₹{abs(Net_Savings_Pri_Sec):,.0f} ({net_sav_label_sec})"
    
        net_sav_label_notch = "Notch Bar Favored" if Net_Savings_Pri_Notch > 0 else "Primary Favored"
        net_sav_color_notch = "kpi-card-green" if Net_Savings_Pri_Notch > 0 else "kpi-card-red"
        net_sav_text_notch  = f"₹{abs(Net_Savings_Pri_Notch):,.0f} ({net_sav_label_notch})"
    
        Annual_Savings_Pri_Sec_Cr   = (Net_Savings_Pri_Sec * Al_Consumption_FY * Substitution_Pct) / 1e7
        Annual_Savings_Pri_Notch_Cr = (Net_Savings_Pri_Notch * Al_Consumption_FY * Substitution_Pct) / 1e7
    
        annual_sav_label_sec   = "Secondary Favored" if Annual_Savings_Pri_Sec_Cr > 0 else "Primary Favored"
        annual_sav_color_sec   = "kpi-card-green" if Annual_Savings_Pri_Sec_Cr > 0 else "kpi-card-red"
        annual_sav_text_sec    = f"₹{abs(Annual_Savings_Pri_Sec_Cr):.2f} Cr ({annual_sav_label_sec})"
    
        annual_sav_label_notch = "Notch Bar Favored" if Annual_Savings_Pri_Notch_Cr > 0 else "Primary Favored"
        annual_sav_color_notch = "kpi-card-green" if Annual_Savings_Pri_Notch_Cr > 0 else "kpi-card-red"
        annual_sav_text_notch  = f"₹{abs(Annual_Savings_Pri_Notch_Cr):.2f} Cr ({annual_sav_label_notch})"
    
        # ── SECTION 1: DASHBOARD HEADER ──
        st.markdown("""
        <div style="background: linear-gradient(135deg,#263238 0%,#37474F 60%,#455A64 100%);
                    padding:22px 28px 18px 28px; border-radius:14px; margin-bottom:20px;
                    box-shadow:0 4px 24px rgba(38,50,56,0.25);">
          <h1 style="color:#FFFFFF;margin:0;font-size:26px;font-weight:800;letter-spacing:0.02em;">
            🏭 Aluminium VIU Dashboard — Primary vs Secondary vs Notch Bar
          </h1>
          <p style="color:#B0BEC5;margin:6px 0 0 0;font-size:13px;">
            Value-In-Use Economic Analysis &nbsp;|&nbsp; Quantifying the true metallurgical cost of impurities, physical form, and chemistry variability.
          </p>
        </div>
        """, unsafe_allow_html=True)
    
        # ── SECTION 2: TOP KPI CARDS ──
        st.markdown("### 📊 Enterprise KPI Dashboard Overview")
        st.markdown("##### 🌐 Global Portfolio Metrics")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.markdown(kpi("Primary Al Price", f"₹{P_Pri_Price:,.0f}", "per MT alloy", ""), unsafe_allow_html=True)
        with c2:
            st.markdown(kpi("Secondary Al Price", f"₹{P_Sec_Price:,.0f}", "per MT alloy", "kpi-card-teal"), unsafe_allow_html=True)
        with c3:
            st.markdown(kpi("Al Notch Bar Price", f"₹{P_Notch_Price:,.0f}", "per MT alloy", "kpi-card-amber"), unsafe_allow_html=True)
        with c4:
            st.markdown(kpi("Lowest VIU Cost", f"₹{best_cost:,.0f}", "per MT Effective Al", "kpi-card-green"), unsafe_allow_html=True)
        with c5:
            st.markdown(kpi("Optimal Choice", f"{best_commodity}", "Lowest total cost", "kpi-card-green"), unsafe_allow_html=True)
        with c6:
            st.markdown(kpi("Max Annual Savings", f"₹{Annual_Savings_Cr:.2f} Cr", f"vs Primary @ {Substitution_Pct*100:.0f}% Sub", "kpi-card-green" if Annual_Savings_Cr > 0 else "kpi-card-purple"), unsafe_allow_html=True)
    
        st.markdown("##### 🔬 Primary vs. Secondary Al Ingot Deep-Dive")
        cs1, cs2, cs3, cs4 = st.columns(4)
        with cs1:
            st.markdown(kpi("Al Cost Gap (Pri vs Sec)", f"₹{abs(Cost_Gap_Pri_Sec):,.0f}", f"per MT Effective Al ({sec_costlier_text})", "kpi-card-teal"), unsafe_allow_html=True)
        with cs2:
            st.markdown(kpi("Total VIU Credits (Pri over Sec)", f"₹{Total_VIU_Credits_Pri_Sec:,.0f}", "per MT Effective Al (Op Penalties applied)", "kpi-card-teal"), unsafe_allow_html=True)
        with cs3:
            st.markdown(kpi("Net Savings (Pri vs Sec)", net_sav_text_sec, "Adjusted Cost Delta per MT Effective Al", net_sav_color_sec), unsafe_allow_html=True)
        with cs4:
            st.markdown(kpi("Annual Savings (Pri vs Sec)", annual_sav_text_sec, f"@ {Substitution_Pct*100:.0f}% Substitution (Direct Vol)", annual_sav_color_sec), unsafe_allow_html=True)
    
        st.markdown("##### 📐 Primary vs. Aluminium Notch Bar Deep-Dive")
        cn1, cn2, cn3, cn4 = st.columns(4)
        with cn1:
            st.markdown(kpi("Al Cost Gap (Pri vs Notch)", f"₹{abs(Cost_Gap_Pri_Notch):,.0f}", f"per MT Effective Al ({notch_costlier_text})", "kpi-card-amber"), unsafe_allow_html=True)
        with cn2:
            st.markdown(kpi("Total VIU Credits (Pri over Notch)", f"₹{Total_VIU_Credits_Pri_Notch:,.0f}", "per MT Effective Al (Op Penalties applied)", "kpi-card-amber"), unsafe_allow_html=True)
        with cn3:
            st.markdown(kpi("Net Savings (Pri vs Notch)", net_sav_text_notch, "Adjusted Cost Delta per MT Effective Al", net_sav_color_notch), unsafe_allow_html=True)
        with cn4:
            st.markdown(kpi("Annual Savings (Pri vs Notch)", annual_sav_text_notch, f"@ {Substitution_Pct*100:.0f}% Substitution (Direct Vol)", annual_sav_color_notch), unsafe_allow_html=True)
    
        # ── SECTION 3: VIU SUMMARY ──
        st.markdown('<div class="section-header">VIU Economic Synthesis</div>', unsafe_allow_html=True)
        col_l, col_r = st.columns([1.1, 0.9])
        with col_l:
            st.markdown("#### VIU Components (per MT Effective Aluminium)")
            data_summary = {
                "Metric": [
                    "Market Price (₹/MT alloy)",
                    "Active Al Purity (%)",
                    "Physical Recovery (%)",
                    "Effective Al Efficiency (%)",
                    "Alloy Required per MT Eff. Al",
                    "Base Cost (₹/MT Eff. Al)",
                    "Operational Penalties (Converted)",
                    "Total VIU Adjusted Cost (₹/MT)",
                ],
                "Primary Al": [
                    f"₹{P_Pri_Price:,.0f}",
                    f"{P_Pri_Purity*100:.1f}%",
                    f"{P_Pri_Rec*100:.1f}%",
                    f"{Eff_Pri*100:.2f}%",
                    f"{Req_Pri:.3f} MT",
                    f"₹{Raw_Cost_Pri:,.0f}",
                    "— (Baseline)",
                    f"₹{Adj_Cost_Pri:,.0f}",
                ],
                "Secondary Al": [
                    f"₹{P_Sec_Price:,.0f}",
                    f"{P_Sec_Purity*100:.1f}%",
                    f"{P_Sec_Rec*100:.1f}%",
                    f"{Eff_Sec*100:.2f}%",
                    f"{Req_Sec:.3f} MT",
                    f"₹{Raw_Cost_Sec:,.0f}",
                    f"+₹{Conv_Pen_Sec:,.0f}",
                    f"₹{Adj_Cost_Sec:,.0f}",
                ],
                "Al Notch Bar": [
                    f"₹{P_Notch_Price:,.0f}",
                    f"{P_Notch_Purity*100:.1f}%",
                    f"{P_Notch_Rec*100:.1f}%",
                    f"{Eff_Notch*100:.2f}%",
                    f"{Req_Notch:.3f} MT",
                    f"₹{Raw_Cost_Notch:,.0f}",
                    f"+₹{Conv_Pen_Notch:,.0f}",
                    f"₹{Adj_Cost_Notch:,.0f}",
                ]
            }
            df_sum = pd.DataFrame(data_summary).set_index("Metric")
            st.dataframe(df_sum, use_container_width=True)
    
            if best_commodity == "Primary Al Ingot":
                st.markdown(f"""
                <div class="warn-box">
                ⚠️ <b>Primary Aluminium Ingot is the most cost-effective choice.</b><br>
                Despite higher market prices, the severe metallurgical and operational penalties (₹{Conv_Pen_Sec:,.0f} for Secondary, ₹{Conv_Pen_Notch:,.0f} for Notch) incurred by lower-tier alloys make them economically inferior on a Value-in-Use basis.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="success-box">
                ✅ <b>{best_commodity} offers a net VIU advantage.</b><br>
                The upfront price discount of {best_commodity} overcomes its operational penalties, resulting in an effective saving of ₹{Savings_Per_MT_Eff_Al:,.0f} per MT of Effective Aluminium compared to Primary Ingot.
                </div>""", unsafe_allow_html=True)
    
        with col_r:
            fig_pen = go.Figure()
            fig_pen.add_trace(go.Bar(name='Secondary Al', x=penalty_names, y=pen_sec_vals, marker_color=C_SEC))
            fig_pen.add_trace(go.Bar(name='Al Notch Bar', x=penalty_names, y=pen_notch_vals, marker_color=C_NOTCH))
    
            fig_pen.update_layout(
                barmode='group',
                **_layout_viu_al("Absolute Operational Penalties (₹ per MT Alloy)", "₹/MT Alloy", 380)
            )
            fig_pen.update_layout(
                legend=dict(
                    orientation="h", 
                    yanchor="top", 
                    y=-0.28, 
                    xanchor="center", 
                    x=0.5
                )
            )
            st.plotly_chart(fig_pen, use_container_width=True)
    
        # ── SECTION 4: BENEFIT BREAKDOWN ──
        st.markdown('<div class="section-header">Detailed Operational Penalty Breakdown</div>', unsafe_allow_html=True)
    
        col_chart, col_table = st.columns([2.5, 3])
    
        with col_chart:
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                y=penalty_names[::-1],
                x=pen_sec_vals[::-1],
                orientation="h",
                name="Secondary Al",
                marker_color=C_SEC,
            ))
            fig_bar.add_trace(go.Bar(
                y=penalty_names[::-1],
                x=pen_notch_vals[::-1],
                orientation="h",
                name="Notch Bar",
                marker_color=C_NOTCH,
            ))
            fig_bar.update_layout(
                barmode="group",
                **_layout_viu_al("Penalty Severity Ranking (₹/MT Alloy)", "₹/MT Alloy", 480)
            )
            fig_bar.update_layout(
                legend=dict(
                    orientation="h", 
                    yanchor="top", 
                    y=-0.16, 
                    xanchor="center", 
                    x=0.5
                )
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
        with col_table:
            breakdown_basis = [
                f"Extra Heat (Sec: {Extra_kWh_Sec:.1f} kWh, Notch: {Extra_kWh_Notch:.1f} kWh), LF Eff: {P_LF_Efficiency*100:.0f}%",
                f"Wear rate {P_Elec_Wear*1000:.1f}g/kWh, Cost: ₹{P_Electrode_Cost}/kg",
                f"Extra Time (Sec: {Extra_Time_Sec}m, Notch: {Extra_Time_Notch}m), Margin: ₹{P_Margin_Steel}/T",
                f"Overdose Diff (Sec: {(P_Sec_Overdose-P_Pri_Overdose)*100:.2f}%, Notch: {(P_Notch_Overdose-P_Pri_Overdose)*100:.2f}%)",
                f"Slag Vol Diff (Sec: {Slag_Pri_Sec:.1f} kg, Notch: {Slag_Pri_Notch:.1f} kg)",
                f"Reject Diff (Sec: {(P_Sec_Reject-P_Pri_Reject)*100:.3f}%, Notch: {(P_Notch_Reject-P_Pri_Reject)*100:.3f}%)",
                f"Yield Drop (Sec: {(P_Pri_Yield-P_Sec_Yield)*100:.2f}%, Notch: {(P_Pri_Yield-P_Notch_Yield)*100:.2f}%)",
                f"Reblow Diff (Sec: {(P_Sec_Retreat-P_Pri_Retreat)*100:.2f}%, Notch: {(P_Notch_Retreat-P_Pri_Retreat)*100:.2f}%)",
            ]
    
            df_breakdown = pd.DataFrame({
                "Penalty Component": penalty_names,
                "Secondary (₹/MT)": [f"₹{v:,.0f}" for v in pen_sec_vals],
                "Notch Bar (₹/MT)": [f"₹{v:,.0f}" for v in pen_notch_vals],
                "Basis & Assumptions": breakdown_basis,
            }).set_index("Penalty Component")
    
            def color_values(val):
                num = float(val.replace("₹", "").replace(",", ""))
                return "color: #B71C1C; font-weight: 600" if num > 0 else ""
    
            st.dataframe(
                df_breakdown.style.map(color_values, subset=["Secondary (₹/MT)", "Notch Bar (₹/MT)"]),
                use_container_width=True, height=480,
            )
    
        st.markdown("<br>", unsafe_allow_html=True)
    
        st.markdown("#### Penalty Sensitivity Heatmap (Secondary Al, ₹/MT at varying Realization Factors)")
        real_range = np.arange(0.1, 1.05, 0.1)
        heat_matrix = []
        base_heat_values = [
            Pen_Power_Sec / R_Power if R_Power > 0 else Extra_kWh_Sec * P_Power_Cost,
            Pen_Elec_Sec / R_Electrode if R_Electrode > 0 else Extra_kWh_Sec * P_Elec_Wear * P_Electrode_Cost,
            Pen_Time_Sec / R_Throughput if R_Throughput > 0 else (Extra_Time_Sec / P_Cycle_Time) * P_Heat_Size * P_Margin_Steel * (1000.0 / (Alloy_T_Sec * P_Heat_Size)),
            Pen_Stab_Sec / R_Stability if R_Stability > 0 else (P_Sec_Overdose - P_Pri_Overdose) * (P_Sec_Price / 1000.0) * Steel_Sec,
            Pen_Slag_Sec / R_Slag if R_Slag > 0 else Slag_Pri_Sec * (P_Slag_Cost / 1000.0),
            Pen_Clean_Sec / R_Cleanliness if R_Cleanliness > 0 else (P_Sec_Reject - P_Pri_Reject) * P_Steel_Value * Steel_Sec,
            Pen_Yield_Sec / R_Yield if R_Yield > 0 else (P_Pri_Yield - P_Sec_Yield) * P_Steel_Value * Steel_Sec,
            Pen_Reblow_Sec / R_Reblow if R_Reblow > 0 else (P_Sec_Retreat - P_Pri_Retreat) * P_LF_Retreatment * (Steel_Sec / P_Heat_Size),
        ]
        heat_matrix = np.array([[bv * r for r in real_range] for bv in base_heat_values])
    
        fig_heat = go.Figure(go.Heatmap(
            z=heat_matrix,
            x=[f"{r*100:.0f}%" for r in real_range],
            y=penalty_names,
            colorscale="YlOrRd",
            text=np.round(heat_matrix, 0).astype(int),
            texttemplate="₹%{text}",
            textfont=dict(size=10, color="#1E293B"),
            hovertemplate="<b>%{y}</b><br>Realization: %{x}<br>Penalty: ₹%{z:,.0f}/MT<extra></extra>",
        ))
        fig_heat.update_layout(
            **_layout_viu_al("VIU Penalty Heatmap — Realization Factor Sensitivity", "", 380)
        )
        fig_heat.update_layout(xaxis_title="Realization Factor", yaxis_title="")
        st.plotly_chart(fig_heat, use_container_width=True)
    
        # ── SECTION 5: WATERFALL ANALYSIS ──
        st.markdown('<div class="section-header">VIU Waterfall Analysis</div>', unsafe_allow_html=True)
    
        wf_labels = [
            "Base Secondary Cost", "Power Penalty", "Electrode Penalty", "Throughput Penalty",
            "Recovery Stability", "Slag Handling", "Cleanliness Risk", "Yield Loss",
            "Re-treatment Risk", "True Adjusted Cost (Sec)",
        ]
    
        wf_values = [
            Raw_Cost_Sec, Pen_Power_Sec * Req_Sec, Pen_Elec_Sec * Req_Sec, Pen_Time_Sec * Req_Sec,
            Pen_Stab_Sec * Req_Sec, Pen_Slag_Sec * Req_Sec, Pen_Clean_Sec * Req_Sec, Pen_Yield_Sec * Req_Sec,
            Pen_Reblow_Sec * Req_Sec, 0,
        ]
    
        measures = ["absolute"] + ["relative"] * (len(wf_labels) - 2) + ["total"]
        wf_text = [f"₹{v:,.0f}" for v in wf_values[:-1]] + [f"₹{Adj_Cost_Sec:,.0f}"]
        wf_values_display = wf_values[:-1] + [Adj_Cost_Sec]
    
        fig_wf = go.Figure(go.Waterfall(
            name="VIU Waterfall",
            orientation="v",
            measure=measures,
            x=wf_labels,
            y=wf_values_display,
            text=wf_text,
            textposition="outside",
            connector=dict(line=dict(color="#BDBDBD", width=1.5, dash="dot")),
            increasing=dict(marker=dict(color=C_NEG)),
            decreasing=dict(marker=dict(color=C_DELTA)),
            totals=dict(marker=dict(color=C_SEC)),
            hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}/MT Eff. Al<extra></extra>",
        ))
        fig_wf.add_hline(
            y=Adj_Cost_Pri, line_dash="dash", line_color=C_PRI, line_width=2,
            annotation_text=f"Primary Benchmark ₹{Adj_Cost_Pri:,.0f}", annotation_position="top left",
        )
        fig_wf.update_layout(
            **_layout_viu_al("VIU Waterfall: Secondary Al True Cost Escalation (₹/MT Effective Al)", "₹/MT Effective Al", 520)
        )
        fig_wf.update_layout(showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_wf, use_container_width=True)
    
        st.markdown("""
        <div class="info-box">
        <b>How to read this waterfall:</b> Visualizes the VIU synthesis on an <b>Effective Aluminium</b> basis. 
        Starting with the raw base cost of Secondary Aluminium, we sequentially add the metallurgical and operational penalties 
        (converted to the Effective Al basis). If the final bar (True Adjusted Cost) exceeds the dashed line (Primary Benchmark), 
        then Secondary Aluminium is economically unviable despite its cheaper upfront price.
        </div>
        """, unsafe_allow_html=True)
    
        # ── SECTION 6: COST COMPARISON ──
        st.markdown('<div class="section-header">Cost Comparison & Sensitivity Analysis</div>', unsafe_allow_html=True)
    
        col_a, col_b = st.columns(2)
    
        with col_a:
            fig_stack = go.Figure()
            categories = ["Primary Al", "Secondary Al", "Al Notch Bar"]
    
            fig_stack.add_trace(go.Bar(
                name="Raw Base Cost (₹/MT Eff Al)", x=categories,
                y=[Raw_Cost_Pri, Raw_Cost_Sec, Raw_Cost_Notch],
                marker_color=[C_PRI, C_SEC, C_NOTCH],
                text=[f"₹{Raw_Cost_Pri:,.0f}", f"₹{Raw_Cost_Sec:,.0f}", f"₹{Raw_Cost_Notch:,.0f}"],
                textposition="inside",
            ))
            fig_stack.add_trace(go.Bar(
                name="Operational Penalties (add)", x=categories,
                y=[0, Conv_Pen_Sec, Conv_Pen_Notch],
                marker_color=["rgba(0,0,0,0)", C_NEG, C_NEG],
                text=["", f"+₹{Conv_Pen_Sec:,.0f}", f"+₹{Conv_Pen_Notch:,.0f}"],
                textposition="inside",
            ))
            
            fig_stack.update_layout(
                barmode="stack",
                **_layout_viu_al("Total Effective Cost Composition (₹/MT Effective Al)", "₹/MT", 420),
            )
            fig_stack.update_layout(
                legend=dict(
                    orientation="h", 
                    yanchor="top", 
                    y=-0.22, 
                    xanchor="center", 
                    x=0.5
                )
            )
            st.plotly_chart(fig_stack, use_container_width=True)
    
        with col_b:
            sec_prices  = np.linspace(P_Pri_Price * 0.7, P_Pri_Price * 1.1, 80)
            raw_cost_secs = sec_prices * Req_Sec
            fixed_pen_sec = Pen_Power_Sec + Pen_Elec_Sec + Pen_Time_Sec + Pen_Slag_Sec + Pen_Clean_Sec + Pen_Yield_Sec + Pen_Reblow_Sec
            var_pen_sec_mult = (P_Sec_Overdose - P_Pri_Overdose) * (1 / 1000.0) * Steel_Sec * R_Stability
            
            adj_cost_secs = raw_cost_secs + (fixed_pen_sec + sec_prices * var_pen_sec_mult) * Req_Sec
            net_viu_secs = Adj_Cost_Pri - adj_cost_secs
    
            breakeven_idx = np.abs(net_viu_secs).argmin()
            breakeven_price = sec_prices[breakeven_idx]
    
            fig_sens = go.Figure()
            fig_sens.add_trace(go.Scatter(
                x=sec_prices, y=net_viu_secs,
                mode="lines", name="Secondary Net Advantage",
                line=dict(color=C_SEC, width=3),
                fill="tozeroy",
                fillcolor="rgba(0,150,136,0.15)",
                hovertemplate="Sec Price: ₹%{x:,.0f}<br>Net Advantage: ₹%{y:,.0f}/MT Eff Al<extra></extra>",
            ))
            fig_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
            fig_sens.add_vline(x=P_Sec_Price, line_dash="dot", line_color=C_SEC, line_width=2,
                               annotation_text=f"Current ₹{P_Sec_Price:,}", annotation_position="top right")
            fig_sens.add_vline(x=breakeven_price, line_dash="dot", line_color=C_NEG, line_width=2,
                               annotation_text=f"Break-even ≈ ₹{breakeven_price:,.0f}", annotation_position="top left")
            fig_sens.update_layout(
                **_layout_viu_al("Secondary Price Sensitivity – Net VIU Advantage (₹/MT Eff Al)", "Net Advantage (₹)", 420)
            )
            fig_sens.update_layout(
                legend=dict(
                    orientation="h", 
                    yanchor="top", 
                    y=-0.22, 
                    xanchor="center", 
                    x=0.5
                )
            )
            st.plotly_chart(fig_sens, use_container_width=True)
    
        st.markdown("<br>", unsafe_allow_html=True)
    
        # ── SECTION 7: ENTERPRISE SAVINGS ──
        st.markdown('<div class="section-header">Enterprise Savings Calculator</div>', unsafe_allow_html=True)
    
        s1, s2, s3, s4 = st.columns(4)
        with s1:
            substituted_eff = Total_Eff_Al_Req * Substitution_Pct
            st.markdown(kpi("Substituted Eff. Volume", f"{substituted_eff:,.0f} MT", f"Effective Al at {Substitution_Pct*100:.0f}% substitution", ""), unsafe_allow_html=True)
        with s2:
            st.markdown(kpi("Savings / MT Eff. Al", f"₹{abs(Savings_Per_MT_Eff_Al):,.0f}", "Magnitude of net advantage", "kpi-card-green" if Annual_Savings_Cr > 0 else "kpi-card-purple"), unsafe_allow_html=True)
        with s3:
            abs_savings_yr = abs(Annual_Savings_Cr)
            st.markdown(kpi("Annual Savings FY26", f"₹{abs_savings_yr:.2f} Cr", "at stated volume", "kpi-card-green" if Annual_Savings_Cr > 0 else "kpi-card-purple"), unsafe_allow_html=True)
        with s4:
            monthly = Annual_Savings_Cr * 1e7 / 12 / 1e5
            st.markdown(kpi("Monthly Savings", f"₹{abs(monthly):.1f} L", "per month average", "kpi-card-amber"), unsafe_allow_html=True)
    
        st.markdown("<br>", unsafe_allow_html=True)
    
        col_lft, col_rgt = st.columns([2, 1])
    
        with col_lft:
            vol_range = np.arange(1000, Al_Consumption_FY * 2.5, 500)
            savings_cr = ((Adj_Cost_Pri - best_cost) * vol_range * Substitution_Pct) / 1e7
    
            fig_sav = go.Figure()
            fig_sav.add_trace(go.Scatter(
                x=vol_range, y=savings_cr,
                mode="lines", name="Annual Savings (₹ Cr)",
                line=dict(color=C_DELTA if Savings_Per_MT_Eff_Al > 0 else C_PRI, width=3),
                fill="tozeroy",
                fillcolor="rgba(76,175,80,0.12)" if Savings_Per_MT_Eff_Al > 0 else "rgba(33,150,243,0.12)",
                hovertemplate="Consumption: %{x:,.0f} MT<br>Savings/Avoided Loss: ₹%{y:.2f} Cr<extra></extra>",
            ))
            fig_sav.add_vline(
                x=Al_Consumption_FY, line_dash="dash", line_color="#263238", line_width=2,
                annotation_text=f"Total: {Al_Consumption_FY:,} MT → ₹{Annual_Savings_Cr:.2f} Cr",
                annotation_position="top left",
            )
            fig_sav.add_hline(y=0, line_dash="solid", line_color="#333", line_width=1.5)
            
            title_text = f"Enterprise Optimization Value vs Consumption Volume (@ {Substitution_Pct*100:.0f}% Sub)"
            fig_sav.update_layout(
                **_layout_viu_al(title_text, "Value (₹ Crore)", 400)
            )
            fig_sav.update_layout(
                legend=dict(
                    orientation="h", 
                    yanchor="top", 
                    y=-0.22, 
                    xanchor="center", 
                    x=0.5
                )
            )
            st.plotly_chart(fig_sav, use_container_width=True)
    
            st.markdown(f"#### 3-Year Strategic Optimization Value (5% annual price escalation)")
            years = ["FY 2026", "FY 2027", "FY 2028"]
            escalation = [1.0, 1.05, 1.1025]
            proj_savings = [abs(Annual_Savings_Cr) * e for e in escalation]
            cumulative_cr = np.cumsum(proj_savings)
    
            fig_3yr = go.Figure()
            fig_3yr.add_trace(go.Bar(
                x=years, y=proj_savings,
                name="Annual Value (₹ Cr)",
                marker_color=[C_DELTA if Annual_Savings_Cr > 0 else C_PRI for _ in proj_savings],
                text=[f"₹{v:.2f} Cr" for v in proj_savings],
                textposition="outside",
            ))
            fig_3yr.add_trace(go.Scatter(
                x=years, y=cumulative_cr,
                mode="lines+markers+text", name="Cumulative (₹ Cr)",
                line=dict(color="#FF9800", width=2.5, dash="dash"),
                marker=dict(size=9, color="#FF9800"),
                text=[f"₹{v:.2f} Cr" for v in cumulative_cr],
                textposition="top center",
            ))
            fig_3yr.update_layout(
                **_layout_viu_al("3-Year Cumulative Value (₹ Crore)", "₹ Crore", 380)
            )
            fig_3yr.update_layout(
                legend=dict(
                    orientation="h", 
                    yanchor="top", 
                    y=-0.24, 
                    xanchor="center", 
                    x=0.5
                )
            )
            st.plotly_chart(fig_3yr, use_container_width=True)
    
        with col_rgt:
            st.markdown("#### Break-Even Price Analysis")
            sec_be = (Adj_Cost_Pri - (Conv_Pen_Sec)) / Req_Sec
            notch_be = (Adj_Cost_Pri - (Conv_Pen_Notch)) / Req_Notch
    
            st.markdown(kpi("Sec. Al Break-Even Price", f"₹{sec_be:,.0f}",
                            f"Current Sec: ₹{P_Sec_Price:,} | {'BELOW' if P_Sec_Price < sec_be else 'ABOVE'} break-even",
                            "kpi-card-green" if P_Sec_Price <= sec_be else "kpi-card-red"), unsafe_allow_html=True)
            
            st.markdown(kpi("Notch Bar Break-Even Price", f"₹{notch_be:,.0f}",
                            f"Current Notch: ₹{P_Notch_Price:,} | {'BELOW' if P_Notch_Price < notch_be else 'ABOVE'} break-even",
                            "kpi-card-green" if P_Notch_Price <= notch_be else "kpi-card-red"), unsafe_allow_html=True)
            
            st.markdown(kpi("Max Viable Penalty Limit", f"₹{(Adj_Cost_Pri - Raw_Cost_Sec) / Req_Sec:,.0f}",
                            f"Current Sec penalties must drop below this to justify ₹{P_Sec_Price:,}",
                            "kpi-card-amber"), unsafe_allow_html=True)
    
        # ── SECTION 8: RECOMMENDATION ──
        st.markdown('<div class="section-header">Final Recommendation</div>', unsafe_allow_html=True)
    
        if best_commodity == "Primary Al Ingot":
            st.markdown(f"""
            <div style="background:#FFF3E0; border-left:6px solid #FF9800; padding:24px 32px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
                <h2 style="color:#E65100; margin-top:0; font-size:28px;">🏆 Primary Aluminium Preferred</h2>
                <p style="font-size:16px; color:#EF6C00; line-height:1.6; margin-bottom:0;">
                    <b>Cost Protection Value: ₹{abs(Annual_Savings_Cr):.2f} Crore</b><br>
                    At current parameters, maintaining {Substitution_Pct*100:.0f}% of your consumption in <b>Primary Aluminium Ingot</b> 
                    protects the meltshop from severe operational penalties associated with lower-grade alternatives. 
                    The upfront price discounts of Secondary Ingot and Notch Bar fail to offset their massive VIU penalties 
                    (such as high inclusion rejection risks, slag handling, and recovery instability).<br><br>
                    <b>Action:</b> Do not substitute unless Secondary Al price drops below the break-even of <b>₹{sec_be:,.0f}/MT</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#E8F5E9; border-left:6px solid #4CAF50; padding:24px 32px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
                <h2 style="color:#1B5E20; margin-top:0; font-size:28px;">🏆 {best_commodity} Preferred</h2>
                <p style="font-size:16px; color:#2E7D32; line-height:1.6; margin-bottom:0;">
                    <b>Projected Annual Savings: ₹{Annual_Savings_Cr:.2f} Crore</b><br>
                    By shifting {Substitution_Pct*100:.0f}% of your baseline consumption to <b>{best_commodity}</b>, 
                    you realize a net Value-in-Use advantage of <b>₹{Savings_Per_MT_Eff_Al:,.0f} per MT of Effective Aluminium</b>. 
                    The market price discount of {best_commodity} is large enough to comfortably absorb the incurred operational penalties 
                    (₹{Conv_Pen_Sec if best_commodity == 'Secondary Al Ingot' else Conv_Pen_Notch:,.0f} per MT Effective Al).
                </p>
            </div>
            """, unsafe_allow_html=True)
    
        st.divider()
        st.markdown("""
        <div style="text-align:center; color:#90A4AE; font-size:12px; padding:8px 0;">
          Aluminium VIU Dashboard &nbsp;|&nbsp; Primary vs Secondary vs Notch Bar &nbsp;|&nbsp; 
          All metallurgical formulas sourced from standardized Aluminium VIU Workbook.
        </div>
        """, unsafe_allow_html=True)
    
    

    elif comparison_selection == "FeV80 vs Nitrovan":
    
            # ── EXACT BREAKDOWN CALCULATION REPLICATION FROM EXCEL ──
            # Intermediate Dependencies (INPUT_PARAMETERS!H1:H10)
            H2 = (Active_V / 100.0) * 1000.0  # Effective V in Heat (kg/T Steel)
            H4 = H2 / P_FeV_Eff / P_FeV_V / P_FeV_Rec # FeV Req /T Steel
            H6 = 1000.0 / H4 # MT of FeV support Steel Production (T)
            H8 = H2 / P_NV_Eff / P_NV_V / P_NV_Rec # NV Req /T Steel
            H10 = 1000.0 / H8 # MT of NV support Steel Production (T)

            # 1. Power Saving (Penalty)
            # ABS(H4*ABS(B19)-H8*ABS(B20))*B22/B39/B21*H6 -> Updated to H10
            kWh_Saved = abs(H4 * abs(P_Chill_FeV) - H8 * abs(P_Chill_NV)) * P_SpHeat_Steel / Conversion_MJ / P_LF_Efficiency * H10
            Benefit_Power = kWh_Saved * P_Power_Tariff * R_Power

            # 2. Electrode Saving
            Benefit_Electrode = kWh_Saved * P_Graphite_Factor * P_Electrode_Cost * R_Electrode

            # 3. Throughput Gain
            Benefit_Throughput = (P_Dissolution_Time_Saved / P_Heat_Size) * P_LF_Minute_Cost * H10 * R_Throughput

            # 4. Recovery Stability
            Benefit_Stability = (NV_Overdose - FeV_Overdose) * P_FeV_Price * R_Stability * H10

            # 5. Reblow Reduction
            Benefit_Retreatment = (Retreatment_NV - Retreatment_FeV) * P_LF_Retreatment_Cost / P_Heat_Size * H10 * R_Reblow

            # 6. Inclusion Cleanliness (Auto/API)
            Benefit_Cleanliness = (Reject_NV - Reject_FeV) * P_Steel_Value * H10 * R_Cleanliness

            # 7. Yield Improvement
            Benefit_Yield = (Yield_Loss_NV - Yield_Loss_FeV) * P_Steel_Value * H10 * R_Yield

            # Total Operational Credits (FeV Advantage over NV)
            Total_VIU_Credits = Benefit_Power + Benefit_Electrode + Benefit_Throughput + Benefit_Stability + Benefit_Retreatment + Benefit_Cleanliness + Benefit_Yield

            # --- Direct Cost Synthesis ---
            Eff_V_FeV_VIU = P_FeV_V * P_FeV_Rec * P_FeV_Eff
            Eff_V_NV_VIU = P_NV_V * P_NV_Rec * P_NV_Eff

            Cost_Per_Eff_V_FeV = P_FeV_Price / Eff_V_FeV_VIU
            Cost_Per_Eff_V_NV = P_NV_Price / Eff_V_NV_VIU

            # Direct Cost Delta = Material Premium of FeV80
            Cost_Gap = Cost_Per_Eff_V_FeV - Cost_Per_Eff_V_NV
    
            # Net Savings (Advantage of Nitrovan over FeV80)
            Net_Savings = Cost_Gap - Total_VIU_Credits

            # Enterprise Savings
            Annual_Savings_Rs = NV_Consumption_FY * Substitution_Pct * Net_Savings
            Annual_Savings_Cr = Annual_Savings_Rs / 1e7

            # Break-Even Pricing Calculation (Net Savings = 0)
            nv_breakeven_price = (Cost_Per_Eff_V_FeV - Total_VIU_Credits) * Eff_V_NV_VIU
            fev_breakeven_price = (Cost_Per_Eff_V_NV + Total_VIU_Credits) * Eff_V_FeV_VIU


            # ── SECTION 1: DASHBOARD HEADER ──
            st.markdown("""
            <div style="background: linear-gradient(135deg,#1A237E 0%,#1565C0 60%,#0277BD 100%);
                        padding:22px 28px 18px 28px; border-radius:14px; margin-bottom:20px;
                        box-shadow:0 4px 24px rgba(26,35,126,0.25);">
              <h1 style="color:#FFFFFF;margin:0;font-size:26px;font-weight:800;letter-spacing:0.02em;">
                ⚗️ VIU Dashboard — FeV80 vs Nitrovan
              </h1>
              <p style="color:#90CAF9;margin:6px 0 0 0;font-size:13px;">
                Value-In-Use Economic Analysis &nbsp;|&nbsp; Ferrovanadium (80% V) 
                vs Nitrovan (16% N, 77% V)
              </p>
            </div>
            """, unsafe_allow_html=True)

            # ── SECTION 2: TOP KPI CARDS ──
            r1_c1, r1_c2, r1_c3 = st.columns(3)
            with r1_c1: st.markdown(kpi("FeV80 Price", f"₹{P_FeV_Price:,.0f}", "per MT alloy", ""), unsafe_allow_html=True)
            with r1_c2: st.markdown(kpi("Nitrovan Price", f"₹{P_NV_Price:,.0f}", "per MT alloy", "kpi-card-green"), unsafe_allow_html=True)
            with r1_c3: st.markdown(kpi("Cost Gap", f"₹{Cost_Gap:,.0f}", "per MT Eff. V", "kpi-card-amber"), unsafe_allow_html=True)

            r2_c1, r2_c2, r2_c3 = st.columns(3)
            with r2_c1: st.markdown(kpi("Total VIU Credits", f"₹{Total_VIU_Credits:,.0f}", "net advantage of FeV80", "kpi-card-amber"), unsafe_allow_html=True)
            with r2_c2:
                col = "kpi-card-green" if Net_Savings > 0 else "kpi-card-red"
                st.markdown(kpi("Net Savings", f"₹{Net_Savings:+,.0f}", "Nitrovan adv (+ = better)", col), unsafe_allow_html=True)
            with r2_c3:
                col_yr = "kpi-card-green" if Net_Savings > 0 else "kpi-card-red"
                st.markdown(kpi("Annual Savings", f"₹{abs(Annual_Savings_Cr):.2f} Cr", f"@ {Substitution_Pct*100:.0f}% Substitution", col_yr), unsafe_allow_html=True)


            # ── SECTION 3: VIU SUMMARY ──
            st.markdown('<div class="section-header">VIU Economic Synthesis</div>', unsafe_allow_html=True)
            col_l, col_r = st.columns([1, 1])
    
            with col_l:
                st.markdown("#### Cost per MT Effective Vanadium (₹/MT Eff. V)")
                km1, km2 = st.columns(2)
                with km1: st.markdown(kpi("FeV80 Cost / MT Eff. V", f"₹{Cost_Per_Eff_V_FeV:,.0f}", f"@ {P_FeV_V*100:.0f}% V × {P_FeV_Rec*100:.0f}% rec", ""), unsafe_allow_html=True)
                with km2: st.markdown(kpi("Nitrovan Cost / MT Eff. V", f"₹{Cost_Per_Eff_V_NV:,.0f}", f"@ {P_NV_V*100:.0f}% V × {P_NV_Eff}x eff.", "kpi-card-green"), unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("#### VIU Components")
                data_summary = {
                    "Component": [
                        "Raw Cost per MT Effective V",
                        "Cost Gap (FeV Material Premium)",
                        "VIU Operational Credits (FeV Advantage)",
                        "Net Savings (Nitrovan Advantage)",
                    ],
                    "FeV80 (₹)": [f"₹{Cost_Per_Eff_V_FeV:,.0f}", "—", f"₹{Total_VIU_Credits:,.0f}", "—"],
                    "Nitrovan (₹)": [
                        f"₹{Cost_Per_Eff_V_NV:,.0f}", f"₹{Cost_Gap:,.0f}",
                        "—", f"₹{Net_Savings:+,.0f}",
                    ],
                }
                df_sum = pd.DataFrame(data_summary).set_index("Component")
                st.dataframe(df_sum, use_container_width=True)

                # Verdict
                if Net_Savings > 0:
                    st.markdown(f"""<div class="success-box">✅ <b>Nitrovan offers a net advantage of ₹{Net_Savings:,.0f}.</b><br>The material pricing gap overcomes the FeV operational credits, making Nitrovan the economically superior choice.</div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="warn-box">⚠️ <b>FeV80 is currently more cost-effective by ₹{abs(Net_Savings):,.0f}.</b><br>At current prices and parameters, FeV80's operational superiority overcomes its material premium.</div>""", unsafe_allow_html=True)

            with col_r:
                # --- VIU Donut: Credit composition ---
                benefit_names = ["Power Saving", "Electrode Saving", "Throughput Gain", "Recovery Stability", "Reblow Reduction", "Inclusion Cleanliness", "Yield Improvement"]
                benefit_values = [Benefit_Power, Benefit_Electrode, Benefit_Throughput, Benefit_Stability, Benefit_Retreatment, Benefit_Cleanliness, Benefit_Yield]
        
                pos_names  = [n for n, v in zip(benefit_names, benefit_values) if v > 0]
                pos_values = [v for v in benefit_values if v > 0]
                colours_donut = ["#2196F3", "#1565C0", "#42A5F5", "#4CAF50", "#66BB6A", "#81C784", "#FF9800", "#FFA726", "#FFC107"]

                fig_donut = go.Figure(data=[go.Pie(
                    labels=pos_names, values=pos_values, hole=0.52,
                    marker=dict(colors=colours_donut[:len(pos_names)], line=dict(color="#fff", width=2)),
                    textinfo="label+percent", hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<extra></extra>",
                )])
                fig_donut.add_annotation(text=f"<b>₹{sum(pos_values):,.0f}</b><br><span style='font-size:10px'>FeV Op Credits</span>", x=0.5, y=0.5, font_size=14, showarrow=False)
                fig_donut.update_layout(title="FeV Operational Credit Composition (₹)", template="plotly_white", height=420, margin=dict(l=20, r=20, t=55, b=20), legend=dict(font=dict(size=11)))
                st.plotly_chart(fig_donut, use_container_width=True)

            # ── SECTION 4: BENEFIT BREAKDOWN ──
            st.markdown('<div class="section-header">Detailed Benefit Breakdown</div>', unsafe_allow_html=True)

            all_benefit_names = ["Power Saving", "Electrode Saving", "Throughput Gain", "Recovery Stability", "Reblow Reduction", "Inclusion Cleanliness", "Yield Improvement"]
            all_benefit_values = [Benefit_Power, Benefit_Electrode, Benefit_Throughput, Benefit_Stability, Benefit_Retreatment, Benefit_Cleanliness, Benefit_Yield]

            col_chart, col_table = st.columns([3, 2])

            with col_chart:
                bar_colors = [C_DELTA if v >= 0 else C_NEG for v in all_benefit_values]
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    y=all_benefit_names[::-1], x=all_benefit_values[::-1], orientation="h",
                    marker=dict(color=bar_colors[::-1], line=dict(color="white", width=1)),
                    text=[f"₹{v:+,.0f}" for v in all_benefit_values[::-1]], textposition="auto",
                    hovertemplate="<b>%{y}</b><br>₹%{x:,.0f}<extra></extra>",
                ))
                fig_bar.add_vline(x=0, line_dash="solid", line_color="#333", line_width=1.5)
                fig_bar.update_layout(**_layout("Benefit Contribution (Advantage of FeV over NV)", "₹ Value", 450))
                st.plotly_chart(fig_bar, use_container_width=True)

            with col_table:
                df_breakdown = pd.DataFrame({
                    "Benefit Component": all_benefit_names,
                    "Advantage (₹)": [f"₹{v:+,.0f}" for v in all_benefit_values],
                }).set_index("Benefit Component")

                def color_values(val):
                    num = float(val.replace("₹", "").replace(",", "").replace("+", ""))
                    if num > 0: return "color: #1B5E20; font-weight: 600"
                    elif num < 0: return "color: #B71C1C; font-weight: 600"
                    return ""

                st.dataframe(df_breakdown.style.map(color_values, subset=["Advantage (₹)"]), use_container_width=True, height=450)

            st.markdown("<br>", unsafe_allow_html=True)

            # Heatmap of benefits by realization factor sensitivity
            st.markdown("#### Benefit Sensitivity Heatmap (₹ at varying Realization Factors)")
            real_range = np.arange(0.1, 1.05, 0.1)
            base_heat_values = [
                Benefit_Power / R_Power if R_Power > 0 else 0,
                Benefit_Electrode / R_Electrode if R_Electrode > 0 else 0,
                Benefit_Throughput / R_Throughput if R_Throughput > 0 else 0,
                Benefit_Stability / R_Stability if R_Stability > 0 else 0,
                Benefit_Retreatment / R_Reblow if R_Reblow > 0 else 0,
                Benefit_Cleanliness / R_Cleanliness if R_Cleanliness > 0 else 0,
                Benefit_Yield / R_Yield if R_Yield > 0 else 0,
            ]
            heat_matrix = np.array([[bv * r for r in real_range] for bv in base_heat_values])

            fig_heat = go.Figure(go.Heatmap(
                z=heat_matrix, x=[f"{r*100:.0f}%" for r in real_range], y=all_benefit_names, colorscale="Blues",
                text=np.round(heat_matrix, 0).astype(int), texttemplate="₹%{text}", textfont=dict(size=10),
                hovertemplate="<b>%{y}</b><br>Realization: %{x}<br>₹%{z:,.0f}<extra></extra>",
            ))
            fig_heat.update_layout(**_layout("VIU Benefit Heatmap — Realization Factor Sensitivity", "", 380))
            fig_heat.update_layout(xaxis_title="Realization Factor", yaxis_title="")
            st.plotly_chart(fig_heat, use_container_width=True)

            # ── SECTION 5: WATERFALL ANALYSIS ──
            st.markdown('<div class="section-header">VIU Waterfall Analysis</div>', unsafe_allow_html=True)
    
            wf_labels = ["FeV80 Cost per Eff. V", "Power Saving", "Electrode Saving", "Throughput Gain", "Recovery Stability", "Reblow Reduction", "Cleanliness", "Yield", "True FeV Effective Cost"]
            wf_values = [Cost_Per_Eff_V_FeV, -Benefit_Power, -Benefit_Electrode, -Benefit_Throughput, -Benefit_Stability, -Benefit_Retreatment, -Benefit_Cleanliness, -Benefit_Yield, 0]
    
            measures = ["absolute"] + ["relative"] * 7 + ["total"]
            wf_text = [f"₹{abs(v):,.0f}" for v in wf_values[:-1]] + [f"₹{Cost_Per_Eff_V_FeV - Total_VIU_Credits:,.0f}"]
            wf_values_display = wf_values[:-1] + [Cost_Per_Eff_V_FeV - Total_VIU_Credits]

            fig_wf = go.Figure(go.Waterfall(
                name="VIU Waterfall", orientation="v", measure=measures, x=wf_labels, y=wf_values_display,
                text=wf_text, textposition="auto", connector=dict(line=dict(color="#BDBDBD", width=1.5, dash="dot")),
                increasing=dict(marker=dict(color=C_DELTA)), decreasing=dict(marker=dict(color=C_FEV)),
                totals=dict(marker=dict(color=C_FEV)),
                hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
            ))
            fig_wf.add_hline(y=Cost_Per_Eff_V_NV, line_dash="dash", line_color=C_NV, line_width=1.5, annotation_text=f"Nitrovan Cost per Eff. V ₹{Cost_Per_Eff_V_NV:,.0f}", annotation_position="right")
            fig_wf.update_layout(**_layout("VIU Waterfall: Factoring FeV Operational Advantages (₹)", "₹ / MT Eff. V", 500))
            fig_wf.update_layout(showlegend=False, xaxis_tickangle=-25)
            st.plotly_chart(fig_wf, use_container_width=True)

            st.markdown("""
            <div class="info-box">
            <b>How to read this waterfall:</b> Starts with the material baseline Cost per Effective Vanadium of FeV80. The operational advantages of FeV (Throughput, Yield, Cleanliness) act as deductions (credits), lowering its "True Effective Cost". The final bar is then directly compared to the horizontal Nitrovan Cost line.
            </div>
            """, unsafe_allow_html=True)

            # ── SECTION 6: COST COMPARISON ──
            st.markdown('<div class="section-header">Cost Comparison & Sensitivity Analysis</div>', unsafe_allow_html=True)

            col_a, col_b = st.columns(2)

            with col_a:
                fig_stack = go.Figure()
                categories = ["FeV80 (Eq. to NV)", "Nitrovan"]

                fig_stack.add_trace(go.Bar(
                    name="Effective Cost", x=categories, 
                    y=[Cost_Per_Eff_V_FeV - Total_VIU_Credits, Cost_Per_Eff_V_NV],
                    marker_color=[C_FEV, C_NV], 
                    text=[f"₹{Cost_Per_Eff_V_FeV - Total_VIU_Credits:,.0f}", f"₹{Cost_Per_Eff_V_NV:,.0f}"], 
                    textposition="auto",
                    hovertemplate="%{x}<br>₹%{y:,.0f}<extra></extra>"
                ))
                fig_stack.update_layout(**_layout("Effective Cost Components (Cost/MT Eff. V - VIU Credits)", "₹", 380))
                st.plotly_chart(fig_stack, use_container_width=True)

            with col_b:
                nv_prices  = np.linspace(P_NV_Price * 0.7, P_NV_Price * 1.3, 80)
                net_adv_nvs   = [Cost_Per_Eff_V_FeV - (p / Eff_V_NV_VIU) - Total_VIU_Credits for p in nv_prices]

                fig_sens = go.Figure()
                fig_sens.add_trace(go.Scatter(
                    x=nv_prices, y=net_adv_nvs, mode="lines", name="Net VIU Advantage",
                    line=dict(color=C_DELTA, width=3), fill="tozeroy", fillcolor="rgba(76,175,80,0.1)",
                    hovertemplate="NV Price: ₹%{x:,.0f}<br>Net Savings: ₹%{y:,.0f}<extra></extra>",
                ))
                fig_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
                fig_sens.add_vline(x=P_NV_Price, line_dash="dot", line_color=C_NV, line_width=2, annotation_text=f"Current ₹{P_NV_Price:,}", annotation_position="top right")
                fig_sens.add_vline(x=nv_breakeven_price, line_dash="dot", line_color=C_NEG, line_width=2, annotation_text=f"Break-even ₹{nv_breakeven_price:,.0f}", annotation_position="top left")
                fig_sens.update_layout(**_layout("Nitrovan Price Sensitivity – Net Savings (₹)", "Net Savings (₹)", 380))
                st.plotly_chart(fig_sens, use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_c, col_d = st.columns(2)

            with col_c:
                fev_prices   = np.linspace(P_FeV_Price * 0.7, P_FeV_Price * 1.3, 80)
                net_adv_fevs = [(p / Eff_V_FeV_VIU) - Cost_Per_Eff_V_NV - Total_VIU_Credits for p in fev_prices]
        
                fig_lc_sens = go.Figure()
                fig_lc_sens.add_trace(go.Scatter(
                    x=fev_prices, y=net_adv_fevs, mode="lines", name="Net VIU (varying FeV price)",
                    line=dict(color=C_FEV, width=3), fill="tozeroy", fillcolor="rgba(33,150,243,0.1)",
                    hovertemplate="FeV80: ₹%{x:,.0f}<br>Net Savings: ₹%{y:,.0f}<extra></extra>",
                ))
                fig_lc_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
                fig_lc_sens.add_vline(x=P_FeV_Price, line_dash="dot", line_color=C_FEV, line_width=2, annotation_text=f"Current ₹{P_FeV_Price:,}", annotation_position="top right")
                fig_lc_sens.add_vline(x=fev_breakeven_price, line_dash="dot", line_color=C_NEG, line_width=2, annotation_text=f"Break-even ₹{fev_breakeven_price:,.0f}", annotation_position="top left")
                fig_lc_sens.update_layout(**_layout("FeV80 Price Sensitivity – Net Savings (₹)", "Net Savings (₹)", 380))
                st.plotly_chart(fig_lc_sens, use_container_width=True)

            with col_d:
                tornado_names  = all_benefit_names
                tornado_base   = all_benefit_values
                tornado_low    = [v * 0.80 for v in tornado_base]
                tornado_high   = [v * 1.20 for v in tornado_base]

                fig_tornado = go.Figure()
                fig_tornado.add_trace(go.Bar(
                    y=tornado_names[::-1], x=[h - b for h, b in zip(tornado_high[::-1], tornado_base[::-1])],
                    orientation="h", name="+20%", marker_color=C_DELTA, base=[b for b in tornado_base[::-1]],
                ))
                fig_tornado.add_trace(go.Bar(
                    y=tornado_names[::-1], x=[l - b for l, b in zip(tornado_low[::-1], tornado_base[::-1])],
                    orientation="h", name="−20%", marker_color="#EF9A9A", base=[b for b in tornado_base[::-1]],
                ))
                fig_tornado.update_layout(barmode="overlay", **_layout("Sensitivity Tornado (±20% Realization)", "Advantage Change (₹)", 380))
                st.plotly_chart(fig_tornado, use_container_width=True)

            # ── SECTION 7: ENTERPRISE SAVINGS ──
            st.markdown('<div class="section-header">Enterprise Savings Calculator</div>', unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)
            with s1: st.markdown(kpi("Substituted Volume", f"{NV_Consumption_FY * Substitution_Pct:,.0f} MT", f"at {Substitution_Pct*100:.0f}% substitution", ""), unsafe_allow_html=True)
            with s2: st.markdown(kpi("Savings / MT Alloy", f"₹{abs(Net_Savings):,.0f}", "Magnitude of net advantage", "kpi-card-green" if Net_Savings > 0 else "kpi-card-amber"), unsafe_allow_html=True)
            with s3: st.markdown(kpi("Annual Savings FY26", f"₹{abs(Annual_Savings_Cr):.2f} Cr", "at stated volume", "kpi-card-green" if Net_Savings > 0 else "kpi-card-amber"), unsafe_allow_html=True)
            with s4: st.markdown(kpi("Monthly Savings", f"₹{abs(Annual_Savings_Cr * 1e7 / 12 / 1e5):.1f} L", "per month average", "kpi-card-purple"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            col_lft, col_rgt = st.columns([2, 1])

            with col_lft:
                vol_range = np.arange(100, NV_Consumption_FY * 2.5, 200)
                savings_cr_curve = (abs(Net_Savings) * vol_range * Substitution_Pct) / 1e7

                fig_sav = go.Figure()
                fig_sav.add_trace(go.Scatter(
                    x=vol_range, y=savings_cr_curve, mode="lines", name="Annual Savings (₹ Cr)",
                    line=dict(color=C_DELTA if Net_Savings > 0 else C_NEG, width=3),
                    fill="tozeroy", fillcolor="rgba(76,175,80,0.12)" if Net_Savings > 0 else "rgba(244,67,54,0.12)",
                    hovertemplate="Consumption: %{x:,.0f} MT<br>Savings: ₹%{y:.2f} Cr<extra></extra>",
                ))
                fig_sav.add_vline(x=NV_Consumption_FY, line_dash="dash", line_color="#1A237E", line_width=2, annotation_text=f"Total: {NV_Consumption_FY:,} MT", annotation_position="top right")
                fig_sav.add_hline(y=0, line_dash="solid", line_color="#333", line_width=1.5)
                fig_sav.update_layout(**_layout(f"Enterprise Savings vs Total Consumption Volume (at {Substitution_Pct*100:.0f}% Sub)", "Savings (₹ Crore)", 400))
                st.plotly_chart(fig_sav, use_container_width=True)

            with col_rgt:
                st.markdown("#### Per-Benefit Annual Savings (₹ Cr)")
                benefits_annual = {n: (v * NV_Consumption_FY * Substitution_Pct) / 1e7 for n, v in zip(all_benefit_names, all_benefit_values)}
                df_bens = pd.DataFrame({"Benefit": list(benefits_annual.keys()), "₹ Crore / Year": [round(v, 3) for v in benefits_annual.values()]}).sort_values("₹ Crore / Year", ascending=False).set_index("Benefit")
                st.dataframe(df_bens.style.map(lambda val: "color:#1B5E20;font-weight:600" if val > 0 else "color:#B71C1C;font-weight:600", subset=["₹ Crore / Year"]), use_container_width=True, height=350)

            # Break-even calculator
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### Break-Even Price Analysis")
            be1, be2, be3 = st.columns(3)

            with be1: st.markdown(kpi("Nitrovan Break-Even", f"₹{nv_breakeven_price:,.0f}", f"Current NV: ₹{P_NV_Price:,} | {'BELOW' if P_NV_Price < nv_breakeven_price else 'ABOVE'} break-even", "kpi-card-green" if P_NV_Price <= nv_breakeven_price else "kpi-card-amber"), unsafe_allow_html=True)
            with be2: st.markdown(kpi("FeV80 Break-Even", f"₹{fev_breakeven_price:,.0f}", f"Current FeV: ₹{P_FeV_Price:,} | {'BELOW' if P_FeV_Price < fev_breakeven_price else 'ABOVE'} break-even", "kpi-card-amber"), unsafe_allow_html=True)
            with be3: st.markdown(kpi("Min. Credits Needed", f"₹{Cost_Gap if Cost_Gap > 0 else 0:,.0f}", f"Current VIU credits: ₹{Total_VIU_Credits:,.0f}", "kpi-card-green" if Total_VIU_Credits >= Cost_Gap else "kpi-card-red"), unsafe_allow_html=True)

            # ── SECTION 8: RECOMMENDATION ──
            st.markdown('<div class="section-header">Final Recommendation</div>', unsafe_allow_html=True)

            if Net_Savings > 0:
                st.markdown(f"""
                <div style="background:#E8F5E9; border-left:6px solid #4CAF50; padding:24px 32px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
                    <h2 style="color:#1B5E20; margin-top:0; font-size:28px;">🏆 Nitrovan Preferred</h2>
                    <p style="font-size:16px; color:#2E7D32; line-height:1.6; margin-bottom:0;">
                        <b>Projected Annual Savings: ₹{Annual_Savings_Cr:.2f} Crore</b><br>
                        By shifting {Substitution_Pct*100:.0f}% of your {NV_Consumption_FY:,} MT baseline consumption to Nitrovan, 
                        you realize a net advantage of <b>₹{Net_Savings:,.0f}</b>. 
                        The material pricing gap (₹{Cost_Gap:,.0f}) is strong enough to overcome FeV80's operational credits (₹{Total_VIU_Credits:,.0f}), making Nitrovan the superior choice.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background:#FFF3E0; border-left:6px solid #FF9800; padding:24px 32px; border-radius:12px; box-shadow:0 4px 12px rgba(0,0,0,0.05);">
                    <h2 style="color:#E65100; margin-top:0; font-size:28px;">🏆 FeV80 Preferred</h2>
                    <p style="font-size:16px; color:#EF6C00; line-height:1.6; margin-bottom:0;">
                        <b>FeV80 Cost Efficiency: ₹{abs(Net_Savings):,.0f}</b><br>
                        At current input parameters, FeV80 remains the more cost-effective option, yielding a projected <b>₹{Annual_Savings_Cr:.2f} Crore</b> in savings vs switching. 
                        FeV80's operational superiority (₹{Total_VIU_Credits:,.0f}) fully offsets its material premium (₹{Cost_Gap:,.0f}).
                    </p>
                </div>
                """, unsafe_allow_html=True)


        # ══════════════════════════════════════════════════════════════════════════════
        # TAB 2: SUBSTITUTION SOLVER
        # ══════════════════════════════════════════════════════════════════════════════


    # TAB 2: SUBSTITUTION SOLVER
    # ══════════════════════════════════════════════════════════════════════════════
    with tab2:
        if comparison_selection == "LC FeMn vs Mn Briquette":
            st.markdown('<div class="section-header">🧠 Optimal Alloy Substitution Solver</div>', unsafe_allow_html=True)
            st.markdown("Calculates the mathematically cheapest blend of LC FeMn and Mn Briquette that perfectly satisfies strict metallurgical limits.")

            grades_data = {
                "Commodity (IS2062)":   {"c_lim": 0.150, "rec_lim": 0.04,  "inc_lim": 0.10, "h2_lim": 1.00, "emm_max": 0.0},
                "TMT/Rebar (Fe500D)":   {"c_lim": 0.200, "rec_lim": 0.04,  "inc_lim": 0.10, "h2_lim": 1.00, "emm_max": 0.0},
                "HSLA/API (API X70)":   {"c_lim": 0.080, "rec_lim": 0.03,  "inc_lim": 0.06, "h2_lim": 0.60, "emm_max": 1.0},
                "Automotive (DP600)":   {"c_lim": 0.050, "rec_lim": 0.02,  "inc_lim": 0.04, "h2_lim": 0.40, "emm_max": 1.0},
                "Electrical (CRGO)":    {"c_lim": 0.020, "rec_lim": 0.015, "inc_lim": 0.02, "h2_lim": 0.20, "emm_max": 1.0},
                "IF Steel (Deep Draw)": {"c_lim": 0.010, "rec_lim": 0.01,  "inc_lim": 0.03, "h2_lim": 0.10, "emm_max": 1.0},
            }

            def update_lc_grade():
                limits = grades_data[st.session_state.lc_grade]
                st.session_state.lc_max_c = limits["c_lim"]
                st.session_state.lc_max_rec = limits["rec_lim"]
                st.session_state.lc_max_inc = limits["inc_lim"]
                st.session_state.lc_max_h2 = limits["h2_lim"]
                st.session_state.lc_max_emm = limits["emm_max"] * 100.0

            gc1, gc2 = st.columns([1.5, 2.5])
            sel_grade = gc1.selectbox("Select Target Steel Grade", list(grades_data.keys()), index=0, key="lc_grade", on_change=update_lc_grade)
            limits = grades_data[sel_grade]

            st.markdown("#### Metallurgical Constraints")
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            max_c   = sc1.slider("Max Carbon Limit",    0.001, 0.250, limits["c_lim"], step=0.005, format="%.3f", key="lc_max_c")
            max_rec = sc2.slider("Max Recovery Var",    0.005, 0.050, limits["rec_lim"], step=0.005, format="%.3f", key="lc_max_rec")
            max_inc = sc3.slider("Max Inclusion Index", 0.01,  0.15,  limits["inc_lim"], step=0.01, key="lc_max_inc")
            max_h2  = sc4.slider("Max Hydrogen Risk",   0.05,  1.20,  limits["h2_lim"], step=0.05, key="lc_max_h2")
            max_emm = sc5.slider("Max EMM Share (%)",   0.0,   100.0, limits["emm_max"] * 100, step=5.0, key="lc_max_emm") / 100.0

            c_cost = [Cost_Per_Mn_LC, Cost_Per_Mn_EMM]
            A_eq = [[1, 1]]
            b_eq = [1]
            A_ub = [
                [0.005, 0.0001], # Carbon Input Index 
                [0.03,  0.015],  # Recovery Variability
                [0.08,  0.02],   # Cleanliness Index
                [0.05,  0.8],    # Hydrogen Risk Index
                [0,     1],      # Max EMM (y <= max_emm)
            ]
            b_ub = [max_c, max_rec, max_inc, max_h2, max_emm]

            res = linprog(c_cost, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1), (0, 1)])

            st.markdown("#### Optimization Result")
            if res.success:
                mix = res.x
                blended_cost = res.fun
                costlier_commodity_cost = max(Cost_Per_Mn_LC, Cost_Per_Mn_EMM)
                savings = costlier_commodity_cost - blended_cost

                rc1, rc2 = st.columns(2)
                rc1.success(f"##### Final Effective Cost: \n ### **₹{blended_cost:,.0f}** per MT Eff. Mn")
                if savings > 10: rc2.info(f"##### Projected Savings vs Single Commodity: \n ### **₹{savings:,.0f}** per MT Eff. Mn")
                else: rc2.info(f"##### Projected Savings vs Single Commodity: \n ### **₹0** (100% Single Alloy is best)")

                fig_pie = go.Figure(data=[go.Pie(
                    labels=["LC FeMn Share", "Mn Briquette Share"], values=[round(m, 4) for m in mix], 
                    hole=0.4, marker_colors=[C_LCFEMN, C_EMM], textinfo="label+percent"
                )])
                fig_pie.update_layout(title=f"Optimal Procurement Ratio for {sel_grade}", height=380, template="plotly_white")
                st.plotly_chart(fig_pie, use_container_width=True)

                st.markdown("---")
                st.markdown("#### 📊 Deep Dive & Insights")
            
                col_insight1, col_insight2 = st.columns(2)
                with col_insight1:
                    fig_cost = go.Figure()
                    fig_cost.add_trace(go.Bar(
                        x=["100% LC FeMn", "Optimal Blend", "100% Mn Briquette"],
                        y=[Cost_Per_Mn_LC, blended_cost, Cost_Per_Mn_EMM],
                        marker_color=[C_LCFEMN, "#9C27B0", C_EMM],
                        text=[f"₹{Cost_Per_Mn_LC:,.0f}", f"₹{blended_cost:,.0f}", f"₹{Cost_Per_Mn_EMM:,.0f}"],
                        textposition="auto", hovertemplate="%{x}<br>₹%{y:,.0f}/MT<extra></extra>"
                    ))
                    fig_cost.update_layout(**_layout("Effective Cost Comparison (₹/MT Active Mn)", "Cost (₹)", 380))
                    st.plotly_chart(fig_cost, use_container_width=True)
                
                with col_insight2:
                    actual_c   = mix[0] * 0.005 + mix[1] * 0.0001
                    actual_rec = mix[0] * 0.03  + mix[1] * 0.015
                    actual_inc = mix[0] * 0.08  + mix[1] * 0.02
                    actual_h2  = mix[0] * 0.05  + mix[1] * 0.8
                    actual_emm = mix[1]
                
                    utils = [
                        (actual_emm / max_emm) * 100 if max_emm else 0,
                        (actual_h2 / max_h2) * 100 if max_h2 else 0,
                        (actual_inc / max_inc) * 100 if max_inc else 0,
                        (actual_rec / max_rec) * 100 if max_rec else 0,
                        (actual_c / max_c) * 100 if max_c else 0
                    ]
                    labels = ["Max EMM Share", "Hydrogen Risk", "Cleanliness", "Recovery Var", "Carbon Limit"]
                
                    fig_util = go.Figure()
                    fig_util.add_trace(go.Bar(
                        y=labels, x=utils, orientation='h', marker_color="#26A69A",
                        text=[f"{u:.1f}%" for u in utils], textposition="inside"
                    ))
                    fig_util.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Limit (100%)")
                    fig_util.update_layout(**_layout("Constraint Utilization (% of Max Limit Used)", "% Used", 380))
                    fig_util.update_xaxes(range=[0, max(110, max(utils)*1.1)])
                    st.plotly_chart(fig_util, use_container_width=True)
                
                st.markdown("#### Metallurgical Profile of the Optimal Blend")
                df_profile = pd.DataFrame({
                    "Parameter": ["Carbon Input Index", "Recovery Variability", "Cleanliness Index", "Hydrogen Risk Index", "EMM Share"],
                    "Blend Actual": [actual_c, actual_rec, actual_inc, actual_h2, actual_emm],
                    "Maximum Allowed": [max_c, max_rec, max_inc, max_h2, max_emm],
                })
                df_profile["Status"] = np.where(df_profile["Blend Actual"] >= df_profile["Maximum Allowed"] - 1e-6, "🛑 Binding Constraint", "✅ Safe")
            
                def format_val(val, is_pct): return f"{val*100:.2f}%" if is_pct else f"{val:.4f}"
                df_profile["Blend Actual"] = df_profile.apply(lambda row: format_val(row["Blend Actual"], row["Parameter"] == "EMM Share"), axis=1)
                df_profile["Maximum Allowed"] = df_profile.apply(lambda row: format_val(row["Maximum Allowed"], row["Parameter"] == "EMM Share"), axis=1)

                def color_status(val): return "color: #D32F2F; font-weight: bold" if "Binding" in val else "color: #388E3C"
                st.dataframe(df_profile.style.map(color_status, subset=["Status"]), use_container_width=True)

            else:
                st.error("⚠️ **Constraint Violation:** The chosen metallurgical limits are too strict to be met using these alloys. Please relax the constraints.")

        elif comparison_selection == "MC FeMn vs Mn Briquette":
            st.markdown('<div class="section-header">⚙️ Optimal Alloy Substitution Solver</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="info-box">
                <b>Linear Programming Engine:</b> Calculates the mathematically cheapest blend of <b>MC FeMn (70% Mn)</b> and <b>Mn Briquette (99% Mn)</b> that perfectly satisfies strict metallurgical limits for the selected steel grade. Uses inputs dynamically from the master sidebar.
            </div>
            """, unsafe_allow_html=True)
        
            grades_data = {
                "Commodity (IS2062)":   {"c_lim": 0.010, "rec_lim": 0.060, "inc_lim": 0.12, "reblow_lim": 0.05, "briq_max": 0.0},
                "TMT/Rebar (Fe500D)":   {"c_lim": 0.008, "rec_lim": 0.050, "inc_lim": 0.10, "reblow_lim": 0.04, "briq_max": 0.0},
                "HSLA/API (API X70)":   {"c_lim": 0.005, "rec_lim": 0.030, "inc_lim": 0.05, "reblow_lim": 0.02, "briq_max": 1.0},
                "Automotive (DP600)":   {"c_lim": 0.003, "rec_lim": 0.020, "inc_lim": 0.03, "reblow_lim": 0.02, "briq_max": 1.0},
                "Electrical (CRGO)":    {"c_lim": 0.0015,"rec_lim": 0.015, "inc_lim": 0.02, "reblow_lim": 0.01, "briq_max": 1.0},
                "IF Steel (Deep Draw)": {"c_lim": 0.001, "rec_lim": 0.015, "inc_lim": 0.02, "reblow_lim": 0.01, "briq_max": 1.0},
            }
        
            def update_mc_grade():
                limits = grades_data[st.session_state.mc_grade]
                st.session_state.mc_max_c = limits["c_lim"]
                st.session_state.mc_max_rec = limits["rec_lim"]
                st.session_state.mc_max_inc = limits["inc_lim"]
                st.session_state.mc_max_reb = limits["reblow_lim"]
                st.session_state.mc_max_briq = limits["briq_max"] * 100.0

            with st.container():
                st.markdown('<div class="solver-kpi-box" style="margin-bottom: 30px;">', unsafe_allow_html=True)
                sel_grade = st.selectbox("Select Target Steel Grade", list(grades_data.keys()), index=0, key="mc_grade", on_change=update_mc_grade)
                limits = grades_data[sel_grade]
            
                st.markdown("<h4 style='color:#333; margin-top:20px; font-size:16px; font-weight:700; border-bottom:1px solid #eee; padding-bottom:10px;'>Metallurgical Constraints (Active Limit Modifiers)</h4>", unsafe_allow_html=True)
            
                c1_s, c2_s, c3_s, c4_s, c5_s = st.columns(5)
                max_c      = c1_s.slider("Max Carbon Limit",    0.001, 0.250, limits["c_lim"], step=0.001, format="%.3f", key="mc_max_c")
                max_rec    = c2_s.slider("Max Recovery Var",    0.005, 0.100, limits["rec_lim"], step=0.005, format="%.3f", key="mc_max_rec")
                max_inc    = c3_s.slider("Max Cleanliness",     0.01,  0.20,  limits["inc_lim"], step=0.01, key="mc_max_inc")
                max_reblow = c4_s.slider("Max Reblow Risk",     0.01,  0.10,  limits["reblow_lim"], step=0.01, key="mc_max_reb")
                max_briq   = c5_s.slider("Max Briq Share (%)",  0.0,   100.0, limits["briq_max"] * 100, step=5.0, key="mc_max_briq") / 100.0
                st.markdown('</div>', unsafe_allow_html=True)
        
            cost_mc = (P_MCFeMn_Price / (P_MCFeMn_Mn * P_MCFeMn_Rec)) - (P_MCFeMn_Fe * P_Scrap_Price)
            cost_briq = P_Briq_Price / (P_Briq_Mn * P_Briq_Rec)
        
            c_cost = [cost_mc, cost_briq]
            A_eq = [[1, 1]]
            b_eq = [1]
            A_ub = [
                [P_MCFeMn_C, P_Briq_C],               
                [MCFeMn_Rec_Var, Briq_Rec_Var],       
                [0.09, 0.03],                         
                [Retreatment_MCFeMn, Retreatment_Briq], 
                [0, 1],                               
            ]
            b_ub = [max_c, max_rec, max_inc, max_reblow, max_briq]
        
            res = linprog(c_cost, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1), (0, 1)])
        
            st.markdown("<h4 style='color:#333; margin-top:10px; margin-bottom:15px; font-size:18px; font-weight:700;'>Optimization Result</h4>", unsafe_allow_html=True)
        
            if res.success:
                mix = res.x
                opt_x = mix[0]  
                opt_y = mix[1]  
                blended_cost = res.fun
            
                max_single_cost = max(cost_mc, cost_briq)
                savings = max_single_cost - blended_cost
        
                col_res1, col_res2 = st.columns(2)
                col_res1.markdown(f"""
                <div style="background:#F0FDF4; border:1px solid #BBF7D0; padding:16px; border-radius:12px;">
                    <div style="font-size:14px; font-weight:700; color:#15803D; margin-bottom:4px;">Final Effective Cost</div>
                    <div style="font-size:24px; font-weight:800; color:#166534;">₹{blended_cost:,.0f} <span style="font-size:14px; font-weight:400;">per MT Eff. Mn</span></div>
                </div>
                """, unsafe_allow_html=True)
            
                sav_display = f"₹{savings:,.0f}" if savings > 10 else "₹0"
                col_res2.markdown(f"""
                <div style="background:#EFF6FF; border:1px solid #BFDBFE; padding:16px; border-radius:12px;">
                    <div style="font-size:14px; font-weight:700; color:#1D4ED8; margin-bottom:4px;">Projected Savings vs Single Alloy</div>
                    <div style="font-size:24px; font-weight:800; color:#1E3A8A;">{sav_display} <span style="font-size:14px; font-weight:400;">per MT Eff. Mn</span></div>
                </div>
                """, unsafe_allow_html=True)
            
                st.markdown("<br>", unsafe_allow_html=True)
                chart_col1, chart_col2, chart_col3 = st.columns(3)
            
                with chart_col1:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=["MC FeMn Share", "Mn Briquette Share"], values=[opt_x, opt_y], 
                        hole=0.4, marker_colors=[C_MCFEMN, C_BRIQ], textinfo="label+percent"
                    )])
                    fig_pie.update_layout(**_layout_solver(f"Optimal Ratio for {sel_grade}"))
                    st.plotly_chart(fig_pie, use_container_width=True)
        
                with chart_col2:
                    fig_cost = go.Figure()
                    fig_cost.add_trace(go.Bar(
                        x=["100% MC FeMn", "Optimal Blend", "100% Mn Briq"],
                        y=[cost_mc, blended_cost, cost_briq], marker_color=[C_MCFEMN, "#9C27B0", C_BRIQ],
                        text=[f"₹{cost_mc:,.0f}", f"₹{blended_cost:,.0f}", f"₹{cost_briq:,.0f}"], textposition="auto",
                    ))
                    fig_cost.update_layout(**_layout_solver("Effective Cost Comparison", "Cost (₹)"))
                    st.plotly_chart(fig_cost, use_container_width=True)
        
                with chart_col3:
                    actual_c      = opt_x * P_MCFeMn_C + opt_y * P_Briq_C
                    actual_rec    = opt_x * MCFeMn_Rec_Var + opt_y * Briq_Rec_Var
                    actual_inc    = opt_x * 0.09 + opt_y * 0.03
                    actual_reblow = opt_x * Retreatment_MCFeMn + opt_y * Retreatment_Briq
                    actual_briq   = opt_y
                
                    utils = [
                        (actual_briq / max_briq) * 100 if max_briq else 0,
                        (actual_reblow / max_reblow) * 100 if max_reblow else 0,
                        (actual_inc / max_inc) * 100 if max_inc else 0,
                        (actual_rec / max_rec) * 100 if max_rec else 0,
                        (actual_c / max_c) * 100 if max_c else 0
                    ]
                    labels = ["Max Briq Share", "Reblow Risk", "Cleanliness", "Recovery Var", "Carbon Limit"]
                
                    fig_util = go.Figure()
                    fig_util.add_trace(go.Bar(
                        y=labels, x=utils, orientation='h', marker_color="#26A69A",
                        text=[f"{u:.1f}%" for u in utils], textposition="inside"
                    ))
                    fig_util.add_vline(x=100, line_dash="dash", line_color="red")
                    fig_util.update_layout(**_layout_solver("Constraint Utilization (%)", ""))
                    fig_util.update_xaxes(range=[0, max(110, max(utils)*1.1)])
                    st.plotly_chart(fig_util, use_container_width=True)
        
                st.markdown("<h4 style='color:#333; margin-top:20px; font-size:16px; font-weight:700; border-bottom:1px solid #eee; padding-bottom:10px;'>Metallurgical Profile of the Optimal Blend</h4>", unsafe_allow_html=True)
                df_profile = pd.DataFrame({
                    "Parameter": ["Carbon Input Index", "Recovery Variability", "Cleanliness Index", "Reblow Risk Index", "Briquette Share"],
                    "Blend Actual": [actual_c, actual_rec, actual_inc, actual_reblow, actual_briq],
                    "Maximum Allowed": [max_c, max_rec, max_inc, max_reblow, max_briq],
                })
            
                df_profile["Status"] = np.where(df_profile["Blend Actual"] >= df_profile["Maximum Allowed"] - 1e-5, "🛑 Binding Constraint", "✅ Safe")
            
                def format_val(val, is_pct): return f"{val*100:.2f}%" if is_pct else f"{val:.4f}"
                df_profile["Blend Actual"] = df_profile.apply(lambda row: format_val(row["Blend Actual"], row["Parameter"] == "Briquette Share"), axis=1)
                df_profile["Maximum Allowed"] = df_profile.apply(lambda row: format_val(row["Maximum Allowed"], row["Parameter"] == "Briquette Share"), axis=1)
        
                def color_status(val): return "color: #D32F2F; font-weight: bold" if "Binding" in val else "color: #388E3C; font-weight: 500"
                st.dataframe(df_profile.style.map(color_status, subset=["Status"]), use_container_width=True)
        
            else:
                st.markdown("""
                <div style="background:#FEF2F2; border-left:4px solid #EF4444; padding:16px; border-radius:8px;">
                    <div style="display:flex; align-items:center;">
                        <div style="font-size:24px; margin-right:12px;">⚠️</div>
                        <div>
                            <p style="font-size:14px; color:#B91C1C; font-weight:bold; margin:0;">Constraint Violation</p>
                            <p style="font-size:14px; color:#DC2626; margin:4px 0 0 0;">The chosen metallurgical limits are too strict to be met using these alloys simultaneously. Please relax the constraints.</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        elif comparison_selection == "FeSi vs Si Metal":
            # Bridging parameters: Map VIU Dashboard Sidebar parameters into Solver equivalents
            P_Slag_Handling = P_Slag_Handling_Cost
            P_Cycle_Saved = Time_Saved_SiMetal
            Ca_Wire_FeSi = CaWire_FeSi
            Ca_Wire_SiMetal = CaWire_SiMetal
            P_TempRise_FeSi = P_Temp_Rise_FeSi
            P_TempRise_Si = P_Temp_Rise_SiMetal
            Retreatment_Si = Retreatment_SiMetal
            Si_Consumption_FY = SiMetal_Consumption_FY

            # ══════════════════════════════════════════════════════════════════════════════
            # CORE COST CALCULATIONS FOR SOLVER OBJECTIVE FUNCTION
            # ══════════════════════════════════════════════════════════════════════════════
            # Objective: Minimize Effective Raw Material Cost per MT Active Si
            Cost_Per_Si_FeSi_Solver    = P_FeSi_Price / (P_FeSi_Si * P_FeSi_Rec)
            Cost_Per_Si_SiMetal_Solver = P_SiMetal_Price / (P_SiMetal_Si * P_SiMetal_Rec)

            # Deduct Iron Credit inherently present in FeSi70 (Using Raw Iron Value directly)
            Iron_Credit_Per_Si_FeSi_Solver = (P_FeSi_Fe * P_Scrap_Price)
            Net_Cost_FeSi_Solver = Cost_Per_Si_FeSi_Solver - Iron_Credit_Per_Si_FeSi_Solver
            Net_Cost_SiMetal_Solver = Cost_Per_Si_SiMetal_Solver


            # ══════════════════════════════════════════════════════════════════════════════
            # SUBSTITUTION SOLVER
            # ══════════════════════════════════════════════════════════════════════════════
            st.markdown("""
            <div style="background: linear-gradient(135deg,#1A237E 0%,#1565C0 60%,#0277BD 100%);
                        padding:22px 28px 18px 28px; border-radius:14px; margin-bottom:20px;
                        box-shadow:0 4px 24px rgba(26,35,126,0.25);">
              <h1 style="color:#FFFFFF;margin:0;font-size:26px;font-weight:800;letter-spacing:0.02em;">
                🧠 Optimal Alloy Substitution Solver
              </h1>
              <p style="color:#90CAF9;margin:6px 0 0 0;font-size:13px;">
                Calculates the mathematically cheapest blend of FeSi70 and 98% Si Metal that perfectly satisfies strict metallurgical limits.
              </p>
            </div>
            """, unsafe_allow_html=True)

            # Exact Steel Grade Constraints from Excel SOLVER Sheet
            grades_data = {
                "Commodity Structural (IS2062)":   {"chill_lim": -1500, "rec_lim": 5.0, "inc_lim": 1.50, "ret_lim": 25.0, "simetal_max": 0.0},
                "TMT / Rebar (Fe500D)":            {"chill_lim": -1500, "rec_lim": 5.0, "inc_lim": 1.50, "ret_lim": 25.0, "simetal_max": 0.0},
                "HSLA (API X60 / X70)":            {"chill_lim": -1300, "rec_lim": 3.0, "inc_lim": 0.40, "ret_lim": 10.0, "simetal_max": 0.40},
                "Automotive (DP / TRIP / Bearing)":{"chill_lim": -1200, "rec_lim": 2.0, "inc_lim": 0.15, "ret_lim": 5.0,  "simetal_max": 0.60},
                "IF Steel (Deep Draw)":            {"chill_lim": -1100, "rec_lim": 1.5, "inc_lim": 0.08, "ret_lim": 2.5,  "simetal_max": 1.00},
                "Electrical Steel (CRNGO / CRGO)": {"chill_lim": -1050, "rec_lim": 1.0, "inc_lim": 0.03, "ret_lim": 1.0,  "simetal_max": 1.00},
            }

            def update_fesi_grade():
                limits = grades_data[st.session_state.fesi_grade]
                st.session_state.fesi_min_chill = limits["chill_lim"]
                st.session_state.fesi_max_rec = limits["rec_lim"]
                st.session_state.fesi_max_inc = limits["inc_lim"]
                st.session_state.fesi_max_ret = limits["ret_lim"]
                st.session_state.fesi_max_simetal = limits["simetal_max"] * 100.0

            # Base Alloy Property Coefficients
            fesi_chill = -1500
            si_chill   = -1000
            fesi_rec   = 5.0
            si_rec     = 1.0
            fesi_inc   = 1.50
            si_inc     = 0.03
            fesi_ret   = 25.0
            si_ret     = 1.0

            gc1, gc2 = st.columns([1.5, 2.5])
            sel_grade = gc1.selectbox("Select Target Steel Grade", list(grades_data.keys()), index=0, key="fesi_grade", on_change=update_fesi_grade)
            limits = grades_data[sel_grade]

            st.markdown("#### Metallurgical Constraints")
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            min_chill   = sc1.slider("Min Thermal Limit (kJ/kg)", -2000, 0, limits["chill_lim"], step=50, key="fesi_min_chill")
            max_rec     = sc2.slider("Max Recovery Var (%)",      0.5, 10.0, limits["rec_lim"], step=0.5, key="fesi_max_rec")
            max_inc     = sc3.slider("Max Inclusion (Al wt%)",    0.01, 2.00, limits["inc_lim"], step=0.01, key="fesi_max_inc")
            max_ret     = sc4.slider("Max Re-treatment Risk",     0.5, 30.0, limits["ret_lim"], step=0.5, key="fesi_max_ret")
            max_simetal = sc5.slider("Max Si Metal Share (%)",    0.0, 100.0, limits["simetal_max"] * 100, step=5.0, key="fesi_max_simetal") / 100.0

            # ── Linear Programming Engine ──
            # Objective array: [FeSi, Si Metal] effective cost per MT of Active Si
            c_cost = [Net_Cost_FeSi_Solver, Net_Cost_SiMetal_Solver]

            # Equality constraints (Sum of allocations = 100% of Active Si target)
            A_eq = [[1, 1]]
            b_eq = [1]

            # Inequality constraints (A_ub * [x, y]^T <= b_ub)
            A_ub = [
                [-fesi_chill, -si_chill],  # Thermal Drop: fesi_chill*x + si_chill*y >= min_chill -> -fesi_chill*x - si_chill*y <= -min_chill
                [fesi_rec, si_rec],        # Recovery Variability
                [fesi_inc, si_inc],        # Cleanliness Index
                [fesi_ret, si_ret],        # LF Re-treatment Risk
                [0, 1],                    # Max Si Metal Share (y <= max_simetal)
            ]
            b_ub = [-min_chill, max_rec, max_inc, max_ret, max_simetal]

            # Execute Scipy Optimizer
            res = linprog(c_cost, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1), (0, 1)])

            st.markdown("#### Optimization Result")
            if res.success:
                mix = res.x
            
                # User defined calculations for display:
                fesi_share = mix[0]
                simetal_share = mix[1]
                gross_op_credits_per_mt_si = Gross_Op_Benefits * Alloy_Per_MT_Si_SiMetal
            
                blended_cost = (Cost_Per_Si_FeSi_Solver - Iron_Credit_Per_Si_FeSi_Solver) * fesi_share + \
                               (Cost_Per_Si_SiMetal_Solver) * simetal_share - \
                               (simetal_share * gross_op_credits_per_mt_si)
            
                max_single_cost = max(Cost_Per_Si_FeSi_Solver - Iron_Credit_Per_Si_FeSi_Solver, Cost_Per_Si_SiMetal_Solver)
                savings = max_single_cost - blended_cost

                rc1, rc2 = st.columns(2)
                rc1.success(f"##### Final Effective Cost: \n ### **₹{blended_cost:,.0f}** per MT Eff. Si")
            
                if savings > 10:
                    rc2.info(f"##### Projected Savings vs Single Commodity: \n ### **₹{savings:,.0f}** per MT Eff. Si")
                else:
                    rc2.info(f"##### Projected Savings vs Single Commodity: \n ### **₹0** (100% Single Alloy is best)")

                # Pie Chart Result
                fig_pie = go.Figure(data=[go.Pie(
                    labels=["FeSi70 Share", "Si Metal Share"], 
                    values=[round(m, 4) for m in mix], 
                    hole=0.4, 
                    marker_colors=[C_FESI_SOLVER, C_SIMETAL_SOLVER],
                    textinfo="label+percent"
                )])
                fig_pie.update_layout(title=f"Optimal Procurement Ratio for {sel_grade}", height=380, template="plotly_white")
                st.plotly_chart(fig_pie, use_container_width=True)

                # ── DEEP DIVE & INSIGHTS ──
                st.markdown("---")
                st.markdown("#### 📊 Deep Dive & Insights")
            
                col_insight1, col_insight2 = st.columns(2)
            
                with col_insight1:
                    # Cost Comparison Chart
                    fig_cost = go.Figure()
                    fig_cost.add_trace(go.Bar(
                        x=["100% FeSi70", "Optimal Blend", "100% Si Metal"],
                        y=[Cost_Per_Si_FeSi_Solver, blended_cost, Cost_Per_Si_SiMetal_Solver],
                        marker_color=[C_FESI_SOLVER, "#9C27B0", C_SIMETAL_SOLVER],
                        text=[f"₹{Cost_Per_Si_FeSi_Solver:,.0f}", f"₹{blended_cost:,.0f}", f"₹{Cost_Per_Si_SiMetal_Solver:,.0f}"],
                        textposition="auto",
                        hovertemplate="%{x}<br>₹%{y:,.0f}/MT<extra></extra>"
                    ))
                    fig_cost.update_layout(**_layout_solver_fesi("Effective Cost Comparison (₹/MT Active Si)", "Cost (₹)", 380))
                    st.plotly_chart(fig_cost, use_container_width=True)
                
                with col_insight2:
                    # Constraint Utilization Chart
                    actual_chill   = mix[0] * fesi_chill + mix[1] * si_chill
                    actual_rec     = mix[0] * fesi_rec   + mix[1] * si_rec
                    actual_inc     = mix[0] * fesi_inc   + mix[1] * si_inc
                    actual_ret     = mix[0] * fesi_ret   + mix[1] * si_ret
                    actual_simetal = mix[1]
                
                    utils = [
                        (actual_simetal / max_simetal) * 100 if max_simetal > 0 else 0,
                        (actual_ret / max_ret) * 100 if max_ret else 0,
                        (actual_inc / max_inc) * 100 if max_inc else 0,
                        (actual_rec / max_rec) * 100 if max_rec else 0,
                        (actual_chill / min_chill) * 100 if min_chill else 0  # Both values are negative, so division yields positive %
                    ]
                    labels = ["Max Si Metal Share", "Re-treatment Risk", "Cleanliness", "Recovery Var", "Thermal Limit"]
                
                    fig_util = go.Figure()
                    fig_util.add_trace(go.Bar(
                        y=labels,
                        x=utils,
                        orientation='h',
                        marker_color="#26A69A",
                        text=[f"{u:.1f}%" for u in utils],
                        textposition="inside"
                    ))
                    fig_util.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Limit (100%)")
                    fig_util.update_layout(**_layout_solver_fesi("Constraint Utilization (% of Limit Used)", "% Used", 380))
                    fig_util.update_xaxes(range=[0, max(110, max(utils)*1.1)])
                    st.plotly_chart(fig_util, use_container_width=True)
                
                # Table of exact values
                st.markdown("#### Metallurgical Profile of the Optimal Blend")
                df_profile = pd.DataFrame({
                    "Parameter": ["Thermal Limit (kJ/kg)", "Recovery Variability (%)", "Cleanliness Index (Al wt%)", "Re-treatment Risk", "Si Metal Share"],
                    "Blend Actual": [actual_chill, actual_rec, actual_inc, actual_ret, actual_simetal],
                    "Limit Threshold": [min_chill, max_rec, max_inc, max_ret, max_simetal],
                })
            
                # Check for binding constraints using standard linear programming tolerance
                df_profile["Status"] = [
                    "🛑 Binding Constraint" if abs(actual_chill - min_chill) < 1e-6 else "✅ Safe",
                    "🛑 Binding Constraint" if actual_rec >= max_rec - 1e-6 else "✅ Safe",
                    "🛑 Binding Constraint" if actual_inc >= max_inc - 1e-6 else "✅ Safe",
                    "🛑 Binding Constraint" if actual_ret >= max_ret - 1e-6 else "✅ Safe",
                    "🛑 Binding Constraint" if max_simetal > 0 and actual_simetal >= max_simetal - 1e-6 else "✅ Safe",
                ]
            
                # Format values (percentages for Si Metal Share, decimals for the rest)
                def format_val(val, is_pct, is_int):
                    if is_pct: return f"{val*100:.2f}%"
                    if is_int: return f"{val:.0f}"
                    return f"{val:.4f}"
                
                df_profile["Blend Actual"] = df_profile.apply(lambda row: format_val(row["Blend Actual"], row["Parameter"] == "Si Metal Share", "Thermal" in row["Parameter"]), axis=1)
                df_profile["Limit Threshold"] = df_profile.apply(lambda row: format_val(row["Limit Threshold"], row["Parameter"] == "Si Metal Share", "Thermal" in row["Parameter"]), axis=1)

                # Style the dataframe status column
                def color_status(val):
                    return "color: #D32F2F; font-weight: bold" if "Binding" in val else "color: #388E3C; font-weight: 500"
                
                st.dataframe(df_profile.style.map(color_status, subset=["Status"]), use_container_width=True)

            else:
                st.error("⚠️ **Constraint Violation:** The chosen metallurgical limits are too strict to be met using these alloys. Please relax the constraints.")


    # ══════════════════════════════════════════════════════════════════════════════

        elif comparison_selection == "Primary Al Ingot vs Secondary Al Ingot vs Al Notch Bar":
            # ── MAPPING: VIU Sidebar inputs mapped directly to Solver internal variables ──
            # Financial 
            P_Price          = P_Pri_Price
            S_Price          = P_Sec_Price
            N_Price          = P_Notch_Price
            P_Power_Cost     = P_Power_Cost
            P_Electrode_Cost = P_Electrode_Cost
            P_Steel_Value    = P_Steel_Value
            P_Slag_Cost      = P_Slag_Cost
            P_Margin         = P_Margin_Steel
            P_Retreat_Cost   = P_LF_Retreatment
        
            # Technical
            P_Purity         = P_Pri_Purity
            S_Purity         = P_Sec_Purity
            N_Purity         = P_Notch_Purity
            P_Heat_Size      = P_Heat_Size
            P_Cycle_Time     = P_Cycle_Time
            P_LF_Eff         = P_LF_Efficiency
            P_Spec_Heat      = 0.75             # Default constant required by solver engine structurally
            P_Imp_Sens_Heat  = P_SpHeat
            Target_Al_Pct    = Active_Al
            Al_Deox          = 0.68             # Default constant required by solver engine structurally
            Net_Al_Target    = 1.08             # Default constant required by solver engine structurally
            P_Rec            = P_Pri_Rec
            S_Rec            = P_Sec_Rec
            N_Rec            = P_Notch_Rec
        
            # Operational
            P_Overdose       = P_Pri_Overdose
            S_Overdose       = P_Sec_Overdose
            N_Overdose       = P_Notch_Overdose
            P_Reject         = P_Pri_Reject
            S_Reject         = P_Sec_Reject
            N_Reject         = P_Notch_Reject
            P_Reblow         = P_Pri_Retreat
            S_Reblow         = P_Sec_Retreat
            N_Reblow         = P_Notch_Retreat
            P_MetYield       = P_Pri_Yield
            S_MetYield       = P_Sec_Yield
            N_MetYield       = P_Notch_Yield
            S_Extra_Time     = Extra_Time_Sec
            N_Extra_Time     = Extra_Time_Notch
        
            # Thermodynamic (Implicit / Explicit)
            Stoich_Conv      = 1.889            # Default constant
            Conv_Fact        = 3.6              # Default constant
            Elec_Cons        = P_Elec_Wear
            Dross_S          = Dross_Pri_Sec
            Dross_N          = Dross_Pri_Notch
            Dross_SN         = 20.0             # Default constant
            Slag_Diff_S      = Slag_Pri_Sec
            Slag_Diff_N      = Slag_Pri_Notch
            Slag_Diff_SN     = 0.0              # Default constant
        
            # Realization
            R_Rec            = R_Stability
            R_Slag           = R_Slag
            R_Clean          = R_Cleanliness
            R_Thru           = R_Throughput
            R_Yield          = R_Yield
            Annual_Consumption = Al_Consumption_FY
    
            # ══════════════════════════════════════════════════════════════════════════════
            # BACKGROUND BREAKDOWN CALCULATIONS (Mapped Exactly to Excel Formulas)
            # ══════════════════════════════════════════════════════════════════════════════
    
            # Derived Input H1/I1 and H4/I4 Logic
            Req_S_T = (Target_Al_Pct / 100) * 1000 / S_Purity / S_Rec
            Supp_S  = 1000 / Req_S_T
    
            Req_N_T = (Target_Al_Pct / 100) * 1000 / N_Purity / N_Rec
            Supp_N  = 1000 / Req_N_T
    
            # --- Secondary Al Ingot Penalties (Primary over Secondary) ---
            kWh_Sav_S   = Dross_S * P_Imp_Sens_Heat / Conv_Fact / P_LF_Eff
            Pen_Power_S = kWh_Sav_S * P_Power_Cost
            Pen_Elec_S  = kWh_Sav_S * Elec_Cons * P_Electrode_Cost
            Pen_Thru_S  = S_Extra_Time * Supp_S / P_Heat_Size * P_Margin * R_Thru
            Pen_Rec_S   = (S_Overdose - P_Overdose) * S_Price / 1000 * Supp_S * R_Rec
            Pen_Slag_S  = Slag_Diff_S * P_Slag_Cost / 1000 * R_Slag
            Pen_Clean_S = (S_Reject - P_Reject) * P_Steel_Value * Supp_S * R_Clean
            Pen_Yield_S = (P_MetYield - S_MetYield) * P_Steel_Value * Supp_S * R_Yield
            Pen_Retrt_S = (S_Reblow - P_Reblow) * P_Retreat_Cost * Supp_S / P_Heat_Size
    
            Total_Pen_S = Pen_Power_S + Pen_Elec_S + Pen_Thru_S + Pen_Rec_S + Pen_Slag_S + Pen_Clean_S + Pen_Yield_S + Pen_Retrt_S
    
            # --- Al Notch Bar Penalties (Primary over Notch Bar) ---
            kWh_Sav_N   = Dross_N * P_Imp_Sens_Heat / Conv_Fact / P_LF_Eff
            Pen_Power_N = kWh_Sav_N * P_Power_Cost
            Pen_Elec_N  = kWh_Sav_N * Elec_Cons * P_Electrode_Cost
            Pen_Thru_N  = N_Extra_Time * Supp_N / P_Heat_Size * P_Margin * R_Thru
            Pen_Rec_N   = (N_Overdose - P_Overdose) * N_Price / 1000 * Supp_N * R_Rec
            Pen_Slag_N  = Slag_Diff_N * P_Slag_Cost / 1000 * R_Slag
            Pen_Clean_N = (N_Reject - P_Reject) * P_Steel_Value * Supp_N * R_Clean
            Pen_Yield_N = (P_MetYield - N_MetYield) * P_Steel_Value * Supp_N * R_Yield
            Pen_Retrt_N = (N_Reblow - P_Reblow) * P_Retreat_Cost * Supp_N / P_Heat_Size
    
            Total_Pen_N = Pen_Power_N + Pen_Elec_N + Pen_Thru_N + Pen_Rec_N + Pen_Slag_N + Pen_Clean_N + Pen_Yield_N + Pen_Retrt_N
    
            # --- VIU Summary Integration ---
            Eff_P = P_Purity * P_Rec
            Eff_S = S_Purity * S_Rec
            Eff_N = N_Purity * N_Rec
    
            Base_Cost_P = P_Price / Eff_P
            Base_Cost_S = S_Price / Eff_S
            Base_Cost_N = N_Price / Eff_N
    
            Converted_Pen_S = Total_Pen_S / Eff_S
            Converted_Pen_N = Total_Pen_N / Eff_N
    
            Net_Eff_Cost_P = Base_Cost_P
            Net_Eff_Cost_S = Base_Cost_S + Converted_Pen_S
            Net_Eff_Cost_N = Base_Cost_N + Converted_Pen_N
    
            st.markdown('<div class="section-header-solver">🧠 Optimal Aluminium Substitution Solver</div>', unsafe_allow_html=True)
            st.markdown("Calculates the mathematically cheapest blend of **Primary Al Ingot**, **Secondary Al Ingot**, and **Al Notch Bar** that perfectly satisfies strict metallurgical limits via Linear Programming.")
    
            # Dictionary containing grade-specific metallurgical constraints (with relaxed Effective Al limits)
            grades_data = {
                "Commodity Structural (IS2062/E250)": {"al_lim": 0.10, "therm_lim": 5.5, "rec_lim": 3.50, "inc_lim": 0.060, "ret_lim": 2.50, "sn_max": 1.00},
                "TMT / Rebar (Fe500D)":               {"al_lim": 0.20, "therm_lim": 4.0, "rec_lim": 3.20, "inc_lim": 0.050, "ret_lim": 2.30, "sn_max": 0.80},
                "HSLA / API (API X70)":               {"al_lim": 0.25, "therm_lim": 2.5, "rec_lim": 2.80, "inc_lim": 0.040, "ret_lim": 2.00, "sn_max": 0.25},
                "Automotive (DP600)":                 {"al_lim": 0.35, "therm_lim": 2.0, "rec_lim": 2.50, "inc_lim": 0.035, "ret_lim": 1.80, "sn_max": 0.15},
                "IF Steel (Deep Draw IF)":            {"al_lim": 0.45, "therm_lim": 1.5, "rec_lim": 2.15, "inc_lim": 0.032, "ret_lim": 1.55, "sn_max": 0.10},
                "Electrical Steel (CRGO / CRNO)":     {"al_lim": 0.48, "therm_lim": 1.1, "rec_lim": 2.05, "inc_lim": 0.031, "ret_lim": 1.50, "sn_max": 0.00},
            }
    
            gc1, gc2 = st.columns([1.5, 2.5])
            sel_grade = gc1.selectbox("Select Target Steel Grade", list(grades_data.keys()), index=0)
            limits = grades_data[sel_grade]
    
            st.markdown("#### Metallurgical Constraints")
            sc1, sc2, sc3, sc4, sc5, sc6 = st.columns(6)
            max_al   = sc1.slider("Min Effective Al (kg/T)", 0.05, 0.70, limits["al_lim"], step=0.01)
            max_thm  = sc2.slider("Max Thermal Burden",      1.0, 6.0, limits["therm_lim"], step=0.1)
            max_rec  = sc3.slider("Max Recovery Var",        1.5, 4.0, limits["rec_lim"], step=0.05)
            max_inc  = sc4.slider("Max Inclusion Idx",       0.0001, 0.0010, limits["inc_lim"], step=0.0001, format="%.4f")
            max_ret  = sc5.slider("Max Re-treatment Risk",   1.0, 3.0, limits["ret_lim"], step=0.1)
            max_sn   = sc6.slider("Max Sec/Notch Share",     0.0, 1.0, limits["sn_max"], step=0.05)
    
            # ── LINEAR PROGRAMMING ENGINE ──
            # Array Indices: [0] Primary, [1] Secondary, [2] Notch Bar
            # Objective: Minimize Final Effective Cost per MT Alloy
            c_cost = [P_Price, S_Price + Total_Pen_S, N_Price + Total_Pen_N]
    
            # Sum of all fractional alloy additions must equal 100%
            A_eq = [[1, 1, 1]]
            b_eq = [1]
    
            # Inequality constraints (A_ub * [P, S, N]^T <= b_ub)
            A_ub = [
                [-Eff_P, -Eff_S, -Eff_N],           # Effective Al Delivered (Negative for >= constraint)
                [1.0,    3.0,    5.0],              # Thermal Burden Index
                [2.0,    3.0,    3.5],              # Recovery Variability (StDev %)
                [P_Reject, S_Reject, N_Reject],     # Inclusion Severity Index
                [1.5,    2.0,    2.5],              # LF Re-treatment Risk
                [0.0,    1.0,    1.0],              # Max Secondary / Notch Share combined
            ]
    
            b_ub = [
                -max_al,
                max_thm,
                max_rec,
                max_inc,
                max_ret,
                max_sn
            ]
    
            # Execute Scipy Optimizer
            res = linprog(c_cost, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1), (0, 1), (0, 1)])
    
            st.markdown("#### Optimization Result")
            if res.success:
                mix = res.x
            
                # Calculate Effective Al of the resultant blend
                blend_eff_al = (mix[0] * Eff_P) + (mix[1] * Eff_S) + (mix[2] * Eff_N)
            
                # Calculate the True Final Effective Cost per MT of Effective Al: (B4+B9)/B3 equivalence
                cost_eff_blend = res.fun / blend_eff_al
            
                savings_vs_primary = Net_Eff_Cost_P - cost_eff_blend
    
                rc1, rc2 = st.columns(2)
                rc1.success(f"##### Final Effective Cost: \n ### **₹{cost_eff_blend:,.0f}** per MT Eff. Al")
            
                if savings_vs_primary > 10:
                    rc2.info(f"##### Projected Savings vs 100% Primary: \n ### **₹{savings_vs_primary:,.0f}** per MT Eff. Al")
                else:
                    rc2.info(f"##### Projected Savings vs 100% Primary: \n ### **₹0** (100% Primary is optimal)")
    
                # ── Pie Chart Result ──
                fig_pie = go.Figure(data=[go.Pie(
                    labels=["Primary Al Ingot", "Secondary Al Ingot", "Al Notch Bar"], 
                    values=[round(m, 4) for m in mix], 
                    hole=0.45, 
                    marker_colors=[C_SOLVER_PRIM, C_SOLVER_SEC, C_SOLVER_NOTCH],
                    textinfo="label+percent"
                )])
                fig_pie.update_layout(title=f"Optimal Procurement Ratio for {sel_grade}", height=380, template="plotly_white")
                st.plotly_chart(fig_pie, use_container_width=True)
    
                # ── Deep Dive & Insights ──
                st.markdown("---")
                st.markdown("#### 📊 Deep Dive & Insights")
            
                col_insight1, col_insight2 = st.columns(2)
            
                with col_insight1:
                    # Cost Comparison Chart
                    fig_cost = go.Figure()
                    fig_cost.add_trace(go.Bar(
                        x=["100% Primary", "Optimal Blend", "100% Secondary", "100% Notch Bar"],
                        y=[Net_Eff_Cost_P, cost_eff_blend, Net_Eff_Cost_S, Net_Eff_Cost_N],
                        marker_color=[C_SOLVER_PRIM, "#1A237E", C_SOLVER_SEC, C_SOLVER_NOTCH],
                        text=[f"₹{Net_Eff_Cost_P:,.0f}", f"₹{cost_eff_blend:,.0f}", f"₹{Net_Eff_Cost_S:,.0f}", f"₹{Net_Eff_Cost_N:,.0f}"],
                        textposition="auto",
                        hovertemplate="%{x}<br>₹%{y:,.0f}/MT Eff. Al<extra></extra>"
                    ))
                    fig_cost.update_layout(**_layout_solver("Net Effective Cost Comparison (₹/MT Effective Al)", "Cost (₹)", 380))
                    st.plotly_chart(fig_cost, use_container_width=True)
                
                with col_insight2:
                    # Constraint Utilization Chart
                    actual_therm = mix[0] * 1.0 + mix[1] * 3.0 + mix[2] * 5.0
                    actual_rec   = mix[0] * 2.0 + mix[1] * 3.0 + mix[2] * 3.5
                    actual_inc   = mix[0] * P_Reject + mix[1] * S_Reject + mix[2] * N_Reject
                    actual_ret   = mix[0] * 1.5 + mix[1] * 2.0 + mix[2] * 2.5
                    actual_sn    = mix[1] + mix[2]
                
                    utils = [
                        (actual_sn / max_sn) * 100 if max_sn else (100 if actual_sn > 0 else 0),
                        (actual_ret / max_ret) * 100 if max_ret else 0,
                        (actual_inc / max_inc) * 100 if max_inc else 0,
                        (actual_rec / max_rec) * 100 if max_rec else 0,
                        (actual_therm / max_thm) * 100 if max_thm else 0
                    ]
                    labels = ["Max Sec/Notch Share", "LF Re-treatment", "Inclusion Severity", "Recovery Var", "Thermal Burden"]
                
                    fig_util = go.Figure()
                    fig_util.add_trace(go.Bar(
                        y=labels, x=utils, orientation='h',
                        marker_color="#26A69A",
                        text=[f"{u:.1f}%" for u in utils],
                        textposition="inside"
                    ))
                    fig_util.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Limit (100%)")
                    fig_util.update_layout(**_layout_solver("Constraint Utilization (% of Max Limit Used)", "% Used", 380))
                    fig_util.update_xaxes(range=[0, max(110, max(utils)*1.1)])
                    st.plotly_chart(fig_util, use_container_width=True)
                
                # Table of exact values
                st.markdown("#### Metallurgical Profile of the Optimal Blend")
                df_profile = pd.DataFrame({
                    "Parameter": [
                        "Effective Al Delivered (kg/T)", 
                        "Thermal Burden Index", 
                        "Recovery Variability (StDev %)", 
                        "Inclusion Severity Index", 
                        "LF Re-treatment Risk (%)", 
                        "Secondary & Notch Share"
                    ],
                    "Blend Actual": [blend_eff_al, actual_therm, actual_rec, actual_inc, actual_ret, actual_sn],
                    "Maximum Allowed": [max_al, max_thm, max_rec, max_inc, max_ret, max_sn],
                })
            
                # Status calculation
                df_profile["Status"] = np.where(
                    df_profile["Parameter"] == "Effective Al Delivered (kg/T)",
                    np.where(df_profile["Blend Actual"] <= df_profile["Maximum Allowed"] + 1e-6, "🛑 Binding Constraint", "✅ Safe (Surplus)"),
                    np.where(df_profile["Blend Actual"] >= df_profile["Maximum Allowed"] - 1e-6, "🛑 Binding Constraint", "✅ Safe")
                )
            
                # Format values for display
                def format_val(val, param):
                    if param == "Secondary & Notch Share": return f"{val*100:.2f}%"
                    if param == "Inclusion Severity Index": return f"{val:.5f}"
                    return f"{val:.4f}"
                
                df_profile["Blend Actual"] = df_profile.apply(lambda row: format_val(row["Blend Actual"], row["Parameter"]), axis=1)
                df_profile["Maximum Allowed"] = df_profile.apply(lambda row: format_val(row["Maximum Allowed"], row["Parameter"]), axis=1)
    
                # Style the dataframe status column
                def color_status(val):
                    return "color: #D32F2F; font-weight: bold" if "Binding" in val else "color: #388E3C"
                
                st.dataframe(df_profile.style.map(color_status, subset=["Status"]), use_container_width=True)
    
            else:
                st.error("⚠️ **Constraint Violation:** The chosen metallurgical limits are too strict to be met using any combination of these three alloys. Please relax the constraints (e.g. lower the Min Effective Al requirement or raise the Max Sec/Notch Share).")
    
            st.divider()
            st.markdown("""
            <div style="text-align:center; color:#90A4AE; font-size:12px; padding:8px 0;">
              Aluminium Substitution Solver &nbsp;|&nbsp; All formulas sourced strictly from the Excel workbook 
              (SOLVER sheet & constraint equations) &nbsp;|&nbsp; 
              Three-Material optimization Engine.
            </div>
            """, unsafe_allow_html=True)


        elif comparison_selection == "FeV80 vs Nitrovan":
            st.markdown("""
            <div style="background: linear-gradient(135deg,#263238 0%,#455A64 60%,#607D8B 100%);
                        padding:22px 28px 18px 28px; border-radius:14px; margin-bottom:20px;
                        box-shadow:0 4px 24px rgba(38,50,56,0.25);">
              <h1 style="color:#FFFFFF;margin:0;font-size:26px;font-weight:800;letter-spacing:0.02em;">
                🧠 Alloy Substitution Solver — FeV vs NV Metal
              </h1>
              <p style="color:#CFD8DC;margin:6px 0 0 0;font-size:13px;">
                Calculates the mathematically cheapest blend of Ferrovanadium (80% V) and Nitrovan (16% N) 
                that perfectly satisfies strict metallurgical limits for target steel grades.
              </p>
            </div>
            """, unsafe_allow_html=True)

            # Grade Specific Definitions mapping exactly to Excel 'Solver' Sheet + IF Steel requirement
            grades_data = {
                "Commodity Structural (IS2062)":   {"v_lim": 0.030, "esu_lim": 0.60, "chill_lim": 15.00, "rec_lim": 5.00, "retrt_lim": 2.50, "inc_lim": 0.015, "h2_lim": 160, "nv_max": 1.0},
                "TMT / Rebar (Fe500D)":            {"v_lim": 0.045, "esu_lim": 0.75, "chill_lim": 15.00, "rec_lim": 5.00, "retrt_lim": 2.50, "inc_lim": 0.015, "h2_lim": 160, "nv_max": 1.0},
                "HSLA (API X60 / X70)":            {"v_lim": 0.055, "esu_lim": 0.80, "chill_lim": 10.00, "rec_lim": 2.45, "retrt_lim": 1.45, "inc_lim": 0.013, "h2_lim": 50,  "nv_max": 0.30},
                "Automotive (DP / TRIP / Bearing)":{"v_lim": 0.060, "esu_lim": 0.40, "chill_lim": 8.00,  "rec_lim": 2.30, "retrt_lim": 1.30, "inc_lim": 0.014, "h2_lim": 35,  "nv_max": 0.20},
                "IF Steel (Deep Draw)":            {"v_lim": 0.040, "esu_lim": 0.40, "chill_lim": 5.00,  "rec_lim": 2.00, "retrt_lim": 1.00, "inc_lim": 0.015, "h2_lim": 15,  "nv_max": 0.00},
            }

            gc1, gc2 = st.columns([1.5, 2.5])
            sel_grade = gc1.selectbox("Select Target Steel Grade", list(grades_data.keys()), index=0)
            limits = grades_data[sel_grade]

            st.markdown("#### Metallurgical Constraints (Mapped from Excel)")
            r1c1, r1c2, r1c3, r1c4 = st.columns(4)
            min_v     = r1c1.slider("Min Effective V (%)", 0.0, 0.150, limits["v_lim"], step=0.005, format="%.3f")
            min_esu   = r1c2.slider("ESU Target", 0.0, 1.5, limits["esu_lim"], step=0.05)
            max_chill = r1c3.slider("Max Chill Drop (°C)", 0.0, 30.0, float(limits["chill_lim"]), step=1.0)
            max_rec   = r1c4.slider("Max Rec. Var (%)", 0.0, 10.0, float(limits["rec_lim"]), step=0.1)

            r2c1, r2c2, r2c3, r2c4 = st.columns(4)
            max_retrt = r2c1.slider("Max Re-treat Risk (%)", 0.0, 10.0, float(limits["retrt_lim"]), step=0.1)
            max_inc   = r2c2.slider("Max Inclusion (%)", 0.001, 0.050, limits["inc_lim"], step=0.001, format="%.3f")
            max_n     = r2c3.slider("Max N Limit (ppm)", 0, 250, int(limits["h2_lim"]), step=5)
            max_nv    = r2c4.slider("Max NV Share (%)", 0.0, 100.0, limits["nv_max"] * 100, step=5.0) / 100.0

            # ══════════════════════════════════════════════════════════════════════════════
            # LINEAR PROGRAMMING ENGINE & VIU CALCULATION
            # ══════════════════════════════════════════════════════════════════════════════
            # Calculate effective parameters derived from sidebar
            Eff_V_FeV = P_FeV_V * P_FeV_Rec
            Eff_V_NV  = P_NV_V * P_NV_Rec
            ESU_FeV   = Eff_V_FeV * P_FeV_ESU_Fac
            ESU_NV    = Eff_V_NV * P_NV_ESU_Fac
            N_ppm_FeV = 1.0  # Derived equivalently to Excel's 0.001% mass assumption
            N_ppm_NV  = P_NV_N_pct * 1000  # 16% = 160 ppm N per kg alloy added

            # Define Objective: Minimize Base Alloy Purchase Cost per MT of Steel
            c_cost = [P_FeV_Price / 1000, P_NV_Price / 1000]

            A_ub, b_ub = [], []

            # Constraints Matrix
            A_ub.append([-Eff_V_FeV, -Eff_V_NV])      # 1. Min Effective V (kg per MT steel)
            b_ub.append(-min_v * 10)

            A_ub.append([-ESU_FeV, -ESU_NV])          # 2. ESU Target
            b_ub.append(-min_esu)

            A_ub.append([P_FeV_Chill_Solver, P_NV_Chill_Solver])    # 3. Chill Limit
            b_ub.append(max_chill)

            A_ub.append([P_FeV_Rec_Var, P_NV_Rec_Var])# 4. Recovery Variability Limit
            b_ub.append(max_rec)

            A_ub.append([P_FeV_Retrt, P_NV_Retrt])    # 5. Re-treatment Risk Limit
            b_ub.append(max_retrt)

            A_ub.append([P_FeV_Inc, P_NV_Inc])        # 6. Inclusion Limit
            b_ub.append(max_inc)

            A_ub.append([N_ppm_FeV, N_ppm_NV])        # 7. Nitrogen Limit (ppm)
            b_ub.append(max_n)

            if max_nv < 1.0:                          # 8. Max NV Share Constraint: y <= max_nv * (x + y)
                A_ub.append([-max_nv, 1 - max_nv])
                b_ub.append(0)

            # Execute Scipy Optimizer
            res = linprog(c_cost, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)])


            # --- VIU Credits (Operational Penalties for Nitrovan per MT) ---
            MT_Steel_per_NV = 389.6 

            benefit_power = 1.029 * P_Power_Tariff * MT_Steel_per_NV * R_Power
            benefit_electrode = 1.0288 * MT_Steel_per_NV * 0.01 * P_Electrode_Cost * R_Electrode
            benefit_throughput = (2.5 / 190.0) * 850.0 * MT_Steel_per_NV * R_Throughput
            benefit_stability = abs(P_NV_Rec_Var - P_FeV_Rec_Var) / 100.0 * P_FeV_Price * R_Stability * 1.4615
            benefit_reblow = abs(P_NV_Retrt - P_FeV_Retrt) / 100.0 * (15000.0 / 190.0) * MT_Steel_per_NV * R_Reblow * 0.9375
            benefit_clean = abs(P_FeV_Inc - P_NV_Inc) / 100.0 * P_Steel_Value * MT_Steel_per_NV * R_Cleanliness * 1.4
            benefit_yield = abs(P_NV_Yield - P_FeV_Yield) / 100.0 * P_Steel_Value * MT_Steel_per_NV * R_Yield

            # Total operational penalty applied to Nitrovan to account for its side effects vs FeV
            viu_credits = benefit_power + benefit_electrode + benefit_throughput + benefit_stability + benefit_reblow + benefit_clean + benefit_yield

            # Raw Material Cost per MT ESU (₹/MT ESU)
            Cost_ESU_FeV = P_FeV_Price / ESU_FeV
            Cost_ESU_NV  = P_NV_Price / ESU_NV

            # ══════════════════════════════════════════════════════════════════════════════
            # OUTPUTS & VISUALIZATIONS
            # ══════════════════════════════════════════════════════════════════════════════
            st.markdown("---")

            if res.success:
                x_fev, y_nv = res.x
                total_kg = x_fev + y_nv
        
                fev_share = x_fev / total_kg if total_kg > 0 else 0
                nv_share = y_nv / total_kg if total_kg > 0 else 0
        
                # NEW FINAL EFFECTIVE COST FORMULA
                # Cost_FeV_ESU * %FeV + Cost_NV_ESU * %NV + (%NV * VIU Credits)
                final_effective_cost = (Cost_ESU_FeV * fev_share) + (Cost_ESU_NV * nv_share) + (nv_share * viu_credits)

                # Cost Baseline: 100% FeV & 100% NV Effective Cost representations
                cost_fev_only = Cost_ESU_FeV
                cost_nv_only  = Cost_ESU_NV + viu_credits

                # Projected Savings calculated against the maximum ESU cost benchmark 
                proj_savings = max(Cost_ESU_FeV, Cost_ESU_NV) - final_effective_cost

                rc1, rc2 = st.columns(2)
                with rc1:
                    st.markdown(f"""
                    <div class="kpi-card">
                        <div class="kpi-label" style="color: #78909C; font-size: 12px;">Final Effective Cost</div>
                        <div class="kpi-value" style="color: #263238; font-size: 26px;">₹{final_effective_cost:,.0f}</div>
                        <div class="kpi-sub" style="color: #90A4AE; font-size: 12px;">True Operational Cost per MT of Target ESU <i>(Includes VIU: ₹{viu_credits:,.0f}/MT)</i></div>
                    </div>
                    """, unsafe_allow_html=True)
            
                with rc2:
                    if proj_savings > 1:
                        st.markdown(f"""
                        <div class="kpi-card kpi-card-green">
                            <div class="kpi-label" style="color: #78909C; font-size: 12px;">Projected Savings vs 100% FeV80</div>
                            <div class="kpi-value" style="color: #263238; font-size: 26px;">₹{proj_savings:,.2f}</div>
                            <div class="kpi-sub" style="color: #90A4AE; font-size: 12px;">Savings per MT of Steel ESU Delivered</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="kpi-card">
                            <div class="kpi-label" style="color: #78909C; font-size: 12px;">Projected Savings vs 100% FeV80</div>
                            <div class="kpi-value" style="color: #263238; font-size: 26px;">₹0.00</div>
                            <div class="kpi-sub" style="color: #90A4AE; font-size: 12px;">100% Single Alloy is best for this grade</div>
                        </div>
                        """, unsafe_allow_html=True)

                st.markdown("#### 📊 Optimal Blend & Cost Insights")
        
                col_pie, col_cost = st.columns([1, 1])
        
                with col_pie:
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=["FeV80 Share", "Nitrovan Share"], 
                        values=[fev_share, nv_share], 
                        hole=0.45, 
                        marker_colors=[C_FEV_SOLVER, C_NV_SOLVER],
                        textinfo="label+percent"
                    )])
                    fig_pie.update_layout(title="Optimal Procurement Ratio (Mass %)", height=380, template="plotly_white")
                    st.plotly_chart(fig_pie, use_container_width=True)
            
                with col_cost:
                    fig_cost = go.Figure()
                    fig_cost.add_trace(go.Bar(
                        x=["100% FeV80", "Optimal Blend", "100% Nitrovan"],
                        y=[cost_fev_only, final_effective_cost, cost_nv_only],
                        marker_color=[C_FEV_SOLVER, "#4CAF50", C_NV_SOLVER],
                        text=[f"₹{cost_fev_only:,.0f}", f"₹{final_effective_cost:,.0f}", f"₹{cost_nv_only:,.0f}"],
                        textposition="auto",
                        hovertemplate="%{x}<br>₹%{y:,.2f}/MT ESU<extra></extra>"
                    ))
                    fig_cost.update_layout(**_layout("True Effective Cost per MT ESU (₹)", "Cost (₹)", 380))
                    st.plotly_chart(fig_cost, use_container_width=True)

                # Calculate actual usage matrix
                actual_v = Eff_V_FeV * x_fev + Eff_V_NV * y_nv
                actual_esu = ESU_FeV * x_fev + ESU_NV * y_nv
                actual_chill = P_FeV_Chill_Solver * x_fev + P_NV_Chill_Solver * y_nv
                actual_rec = P_FeV_Rec_Var * x_fev + P_NV_Rec_Var * y_nv
                actual_retrt = P_FeV_Retrt * x_fev + P_NV_Retrt * y_nv
                actual_inc = P_FeV_Inc * x_fev + P_NV_Inc * y_nv
                actual_n = N_ppm_FeV * x_fev + N_ppm_NV * y_nv

                utils = [
                    (nv_share / max_nv) * 100 if max_nv else 0,
                    (actual_n / max_n) * 100 if max_n else 0,
                    (actual_inc / max_inc) * 100 if max_inc else 0,
                    (actual_retrt / max_retrt) * 100 if max_retrt else 0,
                    (actual_rec / max_rec) * 100 if max_rec else 0,
                    (actual_chill / max_chill) * 100 if max_chill else 0
                ]
                labels = ["Max NV Share", "Nitrogen Limit", "Cleanliness Limit", "Re-treatment Risk", "Recovery Var Limit", "Max Chill Drop"]
        
                st.markdown("#### Constraint Utilization & Metallurgical Profile")
                col_util, col_prof = st.columns([1, 1.2])
        
                with col_util:
                    fig_util = go.Figure()
                    fig_util.add_trace(go.Bar(
                        y=labels,
                        x=utils,
                        orientation='h',
                        marker_color="#26A69A",
                        text=[f"{u:.1f}%" for u in utils],
                        textposition="inside"
                    ))
                    fig_util.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Limit (100%)")
                    fig_util.update_layout(**_layout("Constraint Utilization (% of Max Limit Used)", "% Used", 350))
                    fig_util.update_xaxes(range=[0, max(110, max(utils)*1.1)])
                    st.plotly_chart(fig_util, use_container_width=True)

                with col_prof:
                    df_profile = pd.DataFrame({
                        "Parameter": [
                            "Min Effective V (%)", "ESU Target", "Max Chill Drop (°C)", 
                            "Recovery Variability", "Re-treatment Risk", "Cleanliness Index", 
                            "Nitrogen (ppm)", "Nitrovan Share (%)"
                        ],
                        "Blend Actual": [
                            actual_v / 10, actual_esu, actual_chill, actual_rec, 
                            actual_retrt, actual_inc, actual_n, nv_share * 100
                        ],
                        "Constraint Limit": [
                            min_v, min_esu, max_chill, max_rec, 
                            max_retrt, max_inc, max_n, max_nv * 100
                        ],
                        "Type": ["Min", "Min", "Max", "Max", "Max", "Max", "Max", "Max"]
                    })
            
                    # Check binding logic based on min/max formulation
                    tol = 1e-5
                    def check_binding(row):
                        if row["Type"] == "Min" and row["Blend Actual"] <= row["Constraint Limit"] + tol:
                            return "🛑 Binding Constraint"
                        elif row["Type"] == "Max" and row["Blend Actual"] >= row["Constraint Limit"] - tol:
                            return "🛑 Binding Constraint"
                        return "✅ Safe"
                
                    df_profile["Status"] = df_profile.apply(check_binding, axis=1)
            
                    # Formatting specific columns
                    df_profile["Blend Actual"] = df_profile["Blend Actual"].apply(lambda x: f"{x:.4f}" if x < 1 else f"{x:.2f}")
                    df_profile["Constraint Limit"] = df_profile["Constraint Limit"].apply(lambda x: f"{x:.4f}" if x < 1 else f"{x:.2f}")
            
                    def color_status(val):
                        return "color: #D32F2F; font-weight: bold" if "Binding" in val else "color: #388E3C"
                
                    st.dataframe(df_profile.drop(columns=["Type"]).style.map(color_status, subset=["Status"]), use_container_width=True, height=350)

            else:
                st.error("⚠️ **Constraint Violation:** The chosen metallurgical limits are too strict to be met simultaneously using these alloys. Please review the constraints for feasibility.")


            # ══════════════════════════════════════════════════════════════════════════════
            # FOOTER
            # ══════════════════════════════════════════════════════════════════════════════
            st.divider()
            st.markdown("""
            <div style="text-align:center; color:#90A4AE; font-size:12px; padding:8px 0;">
              Alloy Substitution Solver – FeV80 vs Nitrovan Metal &nbsp;|&nbsp; 
              All mathematical logic extracted and replicated precisely from Excel constraints. &nbsp;|&nbsp; 
              Optimal blend calculations based on minimizing active raw material cost under grade specifications.
            </div>
            """, unsafe_allow_html=True)

# COMMON FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
if comparison_selection in ["LC FeMn vs Mn Briquette", "MC FeMn vs Mn Briquette"]:
    st.markdown("""
    <div style="text-align:center; color:#90A4AE; font-size:12px; padding:8px 0;">
      <b>Integrated Analytics Suite</b> &nbsp;|&nbsp; 
      VIU Dashboard & Substitution Solver Matrix &nbsp;|&nbsp; 
      All operational logic uniquely synced across mathematical models.
    </div>
    """, unsafe_allow_html=True)
elif comparison_selection == "FeSi vs Si Metal":
    st.markdown("""
    <div style="text-align:center; color:#90A4AE; font-size:12px; padding:8px 0;">
      VIU Dashboard – FeSi70 vs Si Metal &nbsp;|&nbsp; All formulas sourced from Excel workbook 
      (INPUT_PARAMETER → BREAKDOWN_CALC → VIU_SUMMARY) &nbsp;|&nbsp; 
      Operational benefits calculated per MT of High-Purity Si Metal at stated realization factors.
    </div>
    """, unsafe_allow_html=True)
elif comparison_selection == "Primary Al Ingot vs Secondary Al Ingot vs Al Notch Bar":
    st.markdown("""
    <div style="text-align:center; color:#90A4AE; font-size:12px; padding:8px 0;">
      Aluminium VIU Dashboard &nbsp;|&nbsp; Primary vs Secondary vs Notch Bar &nbsp;|&nbsp; 
      All metallurgical formulas sourced from standardized Aluminium VIU Workbook.
    </div>
    """, unsafe_allow_html=True)
elif comparison_selection == "FeV80 vs Nitrovan":
    st.markdown("""
    <div style="text-align:center; color:#90A4AE; font-size:12px; padding:8px 0;">
      FeV80 vs Nitrovan VIU Dashboard &nbsp;|&nbsp; All metallurgical formulas and solver logic preserved unmodified.
    </div>
    """, unsafe_allow_html=True)
