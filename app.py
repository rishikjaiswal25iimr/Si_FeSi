"""
VIU DASHBOARD – Unified Alloy Comparison
================================================
Value-in-Use comparison of Standard Ferromanganese Alloys (LC/MC)
against Electrolytic Manganese Metal / Mn Briquette (99.7% / 99.0%).

All formulas sourced exclusively from the Excel logic:
  • INPUT_PARAMETER sheet  → adjustable parameters and dynamic defaults
  • BREAKDOWN_CALC sheet   → physical mass-balance benefit calculations
  • VIU_SUMMARY sheet      → synthesis & enterprise savings
  • SOLVER sheet           → Linear programming grade-wise constraints
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.optimize import linprog

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VIU Dashboard & Solver",
    page_icon="⚗️",
    layout="wide",
)

# Colour palette
C_ALLOY1   = "#2196F3"   # blue  – LC/MC FeMn
C_ALLOY2   = "#4CAF50"   # green – Mn Briquette / EMM
C_DELTA    = "#FF9800"   # amber – delta / benefit
C_NEG      = "#F44336"   # red   – penalties / negative
C_GRID     = "#EEEEEE"
C_BG       = "#FAFAFA"
C_TEXT     = "#333333"

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: #F0F4F8; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A237E 0%, #283593 40%, #1565C0 100%);
}
[data-testid="stSidebar"] * { color: #E8EAF6 !important; }
[data-testid="stSidebar"] .stSlider > div > div > div { background: #5C6BC0 !important; }
[data-testid="stSidebar"] hr { border-color: #3949AB; }
[data-testid="stSidebar"] .stNumberInput input { background: #283593; border-color: #5C6BC0; color: #fff !important; }
[data-testid="stSidebar"] .stSelectbox select { background: #283593; color: #fff; }
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label { color: #fff !important; }

.kpi-card {
    background: #FFFFFF; border-radius: 12px; padding: 18px 22px 14px 22px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); border-left: 5px solid #2196F3;
    margin-bottom: 8px;
}
.kpi-card-green  { border-left-color: #4CAF50; }
.kpi-card-amber  { border-left-color: #FF9800; }
.kpi-card-red    { border-left-color: #F44336; }
.kpi-card-purple { border-left-color: #9C27B0; }
.kpi-label { font-size: 12px; font-weight: 600; color: #78909C; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
.kpi-value { font-size: 26px; font-weight: 700; color: #1A237E; line-height: 1.15; }
.kpi-sub   { font-size: 12px; color: #90A4AE; margin-top: 3px; }

.section-header {
    font-size: 20px; font-weight: 800; color: #1A237E; text-transform: uppercase; 
    letter-spacing: 0.05em; border-bottom: 3px solid #2196F3;
    padding-bottom: 8px; margin-bottom: 24px; margin-top: 32px;
}
.info-box { background: #E3F2FD; border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #1565C0; border-left: 4px solid #2196F3; margin-bottom: 10px; }
.warn-box { background: #FFF3E0; border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #E65100; border-left: 4px solid #FF9800; margin-bottom: 10px; }
.success-box { background: #E8F5E9; border-radius: 8px; padding: 12px 16px; font-size: 13px; color: #1B5E20; border-left: 4px solid #4CAF50; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

def _layout(title: str, y_title: str = "", height: int = 420) -> dict:
    return dict(
        template="plotly_white", paper_bgcolor="white", plot_bgcolor=C_BG,
        font=dict(family="Inter, sans-serif", size=12, color=C_TEXT),
        title=dict(text=title, font=dict(size=15, color="#1A237E"), x=0.01),
        legend=dict(bgcolor="rgba(255,255,255,0.85)", bordercolor="#DDD", borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False, title=y_title),
        hovermode="x unified", height=height, margin=dict(l=60, r=30, t=55, b=45),
    )

def kpi(label: str, value: str, sub: str = "", colour: str = "") -> str:
    cls = f"kpi-card {colour}"
    return f"""<div class="{cls}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>"""

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – DYNAMIC INPUT PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚗️ VIU Dashboard & Substitution Solver")
    st.divider()

    st.markdown("### A. Comparison Selection")
    comparison_selection = st.radio(
        "Select Analysis:",
        ["Not selected", "LC FeMn vs Mn Briquette", "MC FeMn vs Mn Briquette"],
        index=0
    )
    
    if comparison_selection != "Not selected":
        # Dynamic Defaults based on selection
        is_mc = (comparison_selection == "MC FeMn vs Mn Briquette")
        
        alloy1_name = "MC FeMn" if is_mc else "LC FeMn"
        alloy2_name = "Mn Briquette" if is_mc else "Mn Briquette (EMM)"
        
        # Base Excel Defaults
        p1_price_def = 130000 if is_mc else 145000
        p2_price_def = 175000 if is_mc else 240000
        p1_mn_def = 70.0 if is_mc else 80.0
        p2_mn_def = 99.0 if is_mc else 99.7
        p1_rec_def = 85.0 if is_mc else 90.0
        p2_rec_def = 95.0 if is_mc else 97.0
        p1_fe_def = 20.0 if is_mc else 15.0
        p1_c_def = 1.50 if is_mc else 0.50
        p2_c_def = 0.10 if is_mc else 0.01
        p1_chill_def = 2.500 if is_mc else 2.057
        p2_chill_def = 1.000
        p1_overdose_def = 5.0 if is_mc else 2.0
        p2_overdose_def = 1.5 if is_mc else 0.5
        p1_recvar_def = 5.0 if is_mc else 3.0
        p2_recvar_def = 1.5 if is_mc else 1.5
        reject1_def = 0.0002 if is_mc else 0.0005
        reject2_def = 0.0000 if is_mc else 0.00035
        retreat1_def = 4.0 if is_mc else 3.0
        retreat2_def = 2.0 if is_mc else 2.5
        c_corr_freq_def = 0.10

        st.divider()
        st.markdown("### B. Financial Parameters")
        P_Alloy1_Price       = st.number_input(f"{alloy1_name} Price (₹/MT)",    value=p1_price_def, step=1000, min_value=50000, max_value=400000)
        P_Alloy2_Price       = st.number_input(f"{alloy2_name} Price (₹/MT)",    value=p2_price_def, step=1000, min_value=50000, max_value=600000)
        P_Power_Tariff       = st.number_input("Power Tariff (₹/kWh)",         value=6.5,   step=0.1, min_value=1.0,   max_value=20.0, format="%.2f")
        P_Electrode_Cost     = st.number_input("Electrode Cost (₹/kg)",        value=240,   step=10,  min_value=50,    max_value=800)
        P_Steel_Value        = st.number_input("Steel Value (₹/MT)",           value=60000, step=1000, min_value=20000, max_value=200000)
        P_Margin_Steel       = st.number_input("Throughput Margin (₹/MT)",     value=2800,  step=100, min_value=500,   max_value=10000)
        P_LF_Retreatment_Cost= st.number_input("LF Re-treatment Cost (₹/heat)",value=15000, step=500, min_value=2000,  max_value=50000)
        P_RH_Minute_Cost     = st.number_input("RH Cost per Min (₹/min)",      value=2500,  step=100, min_value=500,   max_value=10000)
        P_Ladle_Reline_Cost  = st.number_input("Ladle Reline Cost (₹)",        value=1500000,step=50000,min_value=200000,max_value=5000000)
        P_Scrap_Price        = st.number_input("Scrap / Fe Credit (₹/MT)",     value=35000, step=500, min_value=5000,  max_value=80000)

        st.divider()
        st.markdown("### C. Technical Parameters")
        P_Alloy1_Mn  = st.slider(f"{alloy1_name} Mn Content (%)", 50.0, 95.0, p1_mn_def, 0.5) / 100
        P_Alloy2_Mn  = st.slider(f"{alloy2_name} Mn Content (%)", 90.0, 100.0, p2_mn_def, 0.1) / 100
        P_Alloy1_Rec = st.slider(f"{alloy1_name} Recovery (%)",   60.0, 99.0, p1_rec_def, 0.5) / 100
        P_Alloy2_Rec = st.slider(f"{alloy2_name} Recovery (%)",   80.0, 99.9, p2_rec_def, 0.5) / 100
        P_Alloy1_Fe  = st.slider(f"{alloy1_name} Fe Content (%)", 5.0,  35.0, p1_fe_def, 0.5) / 100
        P_Alloy1_C   = st.slider(f"{alloy1_name} Carbon (%)",     0.1,  3.0,  p1_c_def,  0.1) / 100
        P_SpHeat_Steel  = st.slider("Steel Specific Heat (MJ/T/°C)", 0.5, 1.0, 0.75, 0.01)
        P_Chill_Alloy1  = st.slider(f"{alloy1_name} Chill Factor (°C/kg/t)", 1.0, 4.0, p1_chill_def, 0.001)
        P_Chill_Alloy2  = st.slider(f"{alloy2_name} Chill Factor (°C/kg/t)", 0.5, 2.5, p2_chill_def, 0.05)
        H2_Degas_Rate   = st.slider("H₂ Degas Rate (ppm/min)",      0.02, 0.10, 0.045, 0.005)

        st.divider()
        st.markdown("### D. Operational Parameters")
        P_Heat_Size  = st.slider("Heat Size (MT)",            100,  350,  190,  5)
        P_Cycle_Time = st.slider("LF Cycle Time (min)",        30,   90,   53,  1)
        P_Ladle_Life = st.slider("Ladle Life (heats)",         50,  200,  100,  5)
        P_Alloy_Target = st.number_input("Alloy Addition Rate (kg/T)", value=5.0, step=0.1, min_value=1.0, max_value=20.0, format="%.1f")
        P_LF_Efficiency = st.slider("LF Efficiency (%)",       25.0, 80.0, 45.0, 1.0) / 100
        P_Arc_Duty      = st.slider("Arc Duty Cycle (%)",      30.0, 90.0, 60.0, 1.0) / 100
        P_Reheat_Rate   = st.slider("Reheat Rate (°C/min)",     2.0,  6.0,  3.5,  0.1)
        P_Graphite_Factor = st.slider("Electrode Wear (kg/kWh)", 0.005, 0.020, 0.010, 0.001)
        Alloy1_Overdose    = st.slider(f"{alloy1_name} Overdose Buffer (%)",  0.5,  6.0,  p1_overdose_def,  0.1) / 100
        Alloy2_Overdose    = st.slider(f"{alloy2_name} Overdose Buffer (%)",  0.1,  3.0,  p2_overdose_def,  0.1) / 100
        Reject_Alloy1      = st.number_input(f"{alloy1_name} Rejection Rate", value=reject1_def, format="%.5f", step=0.0001)
        Reject_Alloy2      = st.number_input(f"{alloy2_name} Rejection Rate", value=reject2_def, format="%.5f", step=0.0001)
        Retreatment_Alloy1 = st.slider(f"{alloy1_name} Re-treatment Rate (%)",1.0,  10.0, retreat1_def, 0.1) / 100
        Retreatment_Alloy2 = st.slider(f"{alloy2_name} Re-treatment Rate (%)",0.5,  6.0,  retreat2_def, 0.1) / 100
        C_Corr_Freq_Alloy1 = st.slider("Carbon Correction Frequency", 0.02, 0.30, c_corr_freq_def, 0.01)
        RH_Corr_Time       = st.slider("RH Carbon Corr. Time (min)",   2,   15,    5,    1)
        H2_Bath_Pickup     = st.slider("H₂ Bath Pickup (ppm)",         0.5,  3.0,  1.5,  0.1)
        Refractory_Wear_Drop = st.slider("Refractory Wear Reduction (%)", 0.5, 8.0, 2.0, 0.5) / 100

        st.divider()
        st.markdown("### E. Realization Factors")
        R_Power       = st.slider("Power Realization",       0.50, 1.00, 0.90, 0.01)
        R_Electrode   = st.slider("Electrode Realization",   0.50, 1.00, 0.90, 0.01)
        R_Throughput  = st.slider("Throughput Realization",  0.10, 0.80, 0.40, 0.01)
        R_Stability   = st.slider("Stability Realization",   0.20, 1.00, 0.50, 0.01)
        R_Reblow      = st.slider("Reblow Realization",      0.30, 1.00, 0.75, 0.01)
        R_Cleanliness = st.slider("Cleanliness Realization", 0.10, 0.70, 0.30, 0.01)
        R_Yield       = st.slider("Yield Realization",       0.05, 0.50, 0.20, 0.01)

        st.divider()
        st.markdown("### F. Enterprise Savings")
        EMM_Consumption_FY = st.number_input("Consumption (MT)", value=8300, step=100, min_value=100, max_value=100000)
        Substitution_Pct   = st.slider("% Substitution", 0.0, 1.0, 0.50, 0.05)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT GUARD
# ══════════════════════════════════════════════════════════════════════════════
if comparison_selection == "Not selected":
    st.info("Please select substitution combination to run the VIU analysis.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# CORE CALCULATIONS (GENERALIZED EXCEL REPLICATION - COLUMN F LOGIC)
# ══════════════════════════════════════════════════════════════════════════════

# Calculate true mass balance for substitutions (Excel Power Calc Engine basis)
Alloy_Base = P_Alloy_Target
Active_Mn = Alloy_Base * P_Alloy1_Mn * P_Alloy1_Rec
Alloy_Alt = Active_Mn / (P_Alloy2_Mn * P_Alloy2_Rec)

Steel_Per_MT_Alt = 1000.0 / Alloy_Alt
kWh_MJ = 3.6

# --- Power Saving (Rigorous Mass Balance Basis per MT Alternative) ---
Temp_Drop_Base = P_Chill_Alloy1 * Alloy_Base
Temp_Drop_Alt = P_Chill_Alloy2 * Alloy_Alt
Delta_Temp_Rigorous = Temp_Drop_Base - Temp_Drop_Alt

Eff_LF_Effective = P_LF_Efficiency * P_Arc_Duty
Energy_Saved_per_T_steel = (Delta_Temp_Rigorous * P_SpHeat_Steel) / (kWh_MJ * Eff_LF_Effective)
Power_kWh_Saved_Per_MT = Energy_Saved_per_T_steel * Steel_Per_MT_Alt
Benefit_Power = Power_kWh_Saved_Per_MT * P_Power_Tariff * R_Power

# --- Electrode Saving ---
Benefit_Electrode = Power_kWh_Saved_Per_MT * P_Graphite_Factor * P_Electrode_Cost * R_Electrode

# --- Throughput Gain ---
Delta_Chill_Simple = P_Chill_Alloy1 - P_Chill_Alloy2
Thermal_Gain_Total = Delta_Chill_Simple * P_Alloy_Target 
Time_Saved_Min = Thermal_Gain_Total / P_Reheat_Rate
Benefit_Throughput = (Time_Saved_Min / P_Cycle_Time) * P_Heat_Size * P_Margin_Steel * R_Throughput * (1000.0 / (P_Alloy_Target * P_Heat_Size))

# --- Recovery Stability ---
Benefit_Stability = (Alloy1_Overdose - Alloy2_Overdose) * P_Alloy1_Price * R_Stability

# --- Re-treatment Reduction ---
Benefit_Retreatment = (Retreatment_Alloy1 - Retreatment_Alloy2) * P_LF_Retreatment_Cost * (1000.0 / (P_Alloy_Target * P_Heat_Size)) * R_Reblow

# --- Cleanliness ---
Benefit_Cleanliness = (Reject_Alloy1 - Reject_Alloy2) * P_Steel_Value * (1000.0 / P_Alloy_Target) * R_Cleanliness

# --- Yield Improvement ---
P_Yield_Factor = 2.5e-05
Benefit_Yield = P_Yield_Factor * P_Steel_Value * (1000.0 / P_Alloy_Target) * R_Yield

# --- Carbon Correction Avoidance ---
Benefit_Carbon = C_Corr_Freq_Alloy1 * RH_Corr_Time * P_RH_Minute_Cost * (1000.0 / (P_Alloy_Target * P_Heat_Size))

# --- Hydrogen Penalty ---
Benefit_Hydrogen = -(H2_Bath_Pickup / H2_Degas_Rate) * P_RH_Minute_Cost * (1000.0 / (P_Alloy_Target * P_Heat_Size))

# --- Refractory Life ---
Benefit_Refractory = (P_Ladle_Reline_Cost / P_Ladle_Life) * Refractory_Wear_Drop * (1000.0 / (P_Alloy_Target * P_Heat_Size))

# --- Gross Operational Credits ---
Gross_Op_Benefits = (
    Benefit_Power + Benefit_Electrode + Benefit_Throughput +
    Benefit_Stability + Benefit_Retreatment + Benefit_Cleanliness +
    Benefit_Yield + Benefit_Carbon + Benefit_Hydrogen + Benefit_Refractory
)


# ══ VIU SUMMARY EXACT LOGIC ═══════════════════════════════════════════════════
Alloy_Per_MT_Mn_Base = 1.0 / (P_Alloy1_Mn * P_Alloy1_Rec)
Alloy_Per_MT_Mn_Alt  = 1.0 / (P_Alloy2_Mn * P_Alloy2_Rec)

Cost_Per_Mn_Base = Alloy_Per_MT_Mn_Base * P_Alloy1_Price
Cost_Per_Mn_Alt  = Alloy_Per_MT_Mn_Alt  * P_Alloy2_Price
Iron_Credit_Base = P_Alloy1_Fe * P_Scrap_Price

Cost_Per_Mn_Delta = Cost_Per_Mn_Alt - Cost_Per_Mn_Base

# Deduct lost Iron Credit as a penalty to get Total Net Credits
Total_Op_Credits = Gross_Op_Benefits - Iron_Credit_Base
Net_VIU_Advantage = Cost_Per_Mn_Delta - Total_Op_Credits
Savings_Per_MT = Total_Op_Credits - Cost_Per_Mn_Delta

Annual_Savings_Rs = EMM_Consumption_FY * Substitution_Pct * abs(Savings_Per_MT)
Annual_Savings_Cr = Annual_Savings_Rs / 1e7


# ══════════════════════════════════════════════════════════════════════════════
# TABS SETUP
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["VIU Dashboard", "Substitution Solver"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: VIU DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown(f"""
    <div style="background: linear-gradient(135deg,#1A237E 0%,#1565C0 60%,#0277BD 100%);
                padding:22px 28px 18px 28px; border-radius:14px; margin-bottom:20px;
                box-shadow:0 4px 24px rgba(26,35,126,0.25);">
      <h1 style="color:#FFFFFF;margin:0;font-size:26px;font-weight:800;letter-spacing:0.02em;">
        ⚗️ VIU Dashboard — {alloy1_name} vs {alloy2_name}
      </h1>
      <p style="color:#90CAF9;margin:6px 0 0 0;font-size:13px;">
        Value-In-Use Economic Analysis &nbsp;|&nbsp; {alloy1_name} ({P_Alloy1_Mn*100:.0f}% Mn) 
        vs {alloy2_name} ({P_Alloy2_Mn*100:.1f}% Mn)
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.markdown(kpi(f"{alloy1_name} Price", f"₹{P_Alloy1_Price:,.0f}", "per MT alloy", ""), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi(f"{alloy2_name} Price", f"₹{P_Alloy2_Price:,.0f}", "per MT alloy", "kpi-card-green"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Mn Cost Gap", f"₹{Cost_Per_Mn_Delta:,.0f}", "per MT Active Mn", "kpi-card-amber"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Total VIU Credits", f"₹{Total_Op_Credits:,.0f}", "net benefit / MT alloy", "kpi-card-green"), unsafe_allow_html=True)
    with c5:
        col = "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-red"
        st.markdown(kpi("Net Savings / MT", f"₹{Savings_Per_MT:+,.0f}", "Briquette advantage", col), unsafe_allow_html=True)
    with c6:
        col_yr = "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-red"
        st.markdown(kpi("Annual Savings", f"₹{abs(Annual_Savings_Cr):.2f} Cr", f"@ {Substitution_Pct*100:.0f}% Substitution", col_yr), unsafe_allow_html=True)

    st.markdown('<div class="section-header">VIU Economic Synthesis</div>', unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1])
    
    with col_l:
        st.markdown("#### Cost per Active Manganese (₹/MT Mn)")
        km1, km2 = st.columns(2)
        with km1:
            st.markdown(kpi(f"{alloy1_name} Cost/Mn", f"₹{Cost_Per_Mn_Base:,.0f}", f"@ {P_Alloy1_Mn*100:.1f}% Mn × {P_Alloy1_Rec*100:.0f}% rec", ""), unsafe_allow_html=True)
        with km2:
            st.markdown(kpi(f"{alloy2_name} Cost/Mn", f"₹{Cost_Per_Mn_Alt:,.0f}", f"@ {P_Alloy2_Mn*100:.1f}% Mn × {P_Alloy2_Rec*100:.0f}% rec", "kpi-card-green"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### VIU Components")
        data_summary = {
            "Component": [
                "Cost per MT Active Mn",
                "Direct Cost Delta (Premium)",
                "Gross Operational Credits",
                "Lost Iron Credit Penalty",
                "Total Net Credits",
                "Net VIU Advantage (Credits − Delta)",
            ],
            f"{alloy1_name} (₹/MT)": [f"₹{Cost_Per_Mn_Base:,.0f}", "—", "—", "—", "—", "—"],
            f"{alloy2_name} (₹/MT)": [
                f"₹{Cost_Per_Mn_Alt:,.0f}", f"₹{Cost_Per_Mn_Delta:,.0f}",
                f"₹{Gross_Op_Benefits:,.0f}", f"-₹{Iron_Credit_Base:,.0f}",
                f"₹{Total_Op_Credits:,.0f}", f"₹{Savings_Per_MT:+,.0f}",
            ],
        }
        st.dataframe(pd.DataFrame(data_summary).set_index("Component"), use_container_width=True)

        if Savings_Per_MT > 0:
            st.markdown(f"""<div class="success-box">✅ <b>{alloy2_name} offers a net advantage of ₹{Savings_Per_MT:,.0f}/MT alloy.</b><br>
            Operational credits exceed the price premium, making it economically superior.</div>""", unsafe_allow_html=True)
        elif Savings_Per_MT < -2000:
            st.markdown(f"""<div class="warn-box">⚠️ <b>{alloy1_name} is currently more cost-effective by ₹{abs(Savings_Per_MT):,.0f}/MT.</b><br>
            At current prices, the base alloy price advantage outweighs operational credits.</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="info-box">ℹ️ <b>Near economic parity.</b> Net VIU: ₹{Savings_Per_MT:+,.0f}/MT alloy.
            Consider plant-specific factors and grade-specific requirements.</div>""", unsafe_allow_html=True)

    with col_r:
        benefit_names = ["Power Saving", "Electrode Saving", "Throughput Gain", "Recovery Stability", "Re-treatment Reduction",
                         "Cleanliness Benefit", "Yield Improvement", "Carbon Avoidance", "Refractory Benefit"]
        benefit_values = [Benefit_Power, Benefit_Electrode, Benefit_Throughput, Benefit_Stability, Benefit_Retreatment, 
                          Benefit_Cleanliness, Benefit_Yield, Benefit_Carbon, Benefit_Refractory]
        pos_names  = [n for n, v in zip(benefit_names, benefit_values) if v > 0]
        pos_values = [v for v in benefit_values if v > 0]
        colours_donut = ["#2196F3", "#1565C0", "#42A5F5", "#4CAF50", "#66BB6A", "#81C784", "#FF9800", "#FFA726", "#FFC107"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=pos_names, values=pos_values, hole=0.52,
            marker=dict(colors=colours_donut[:len(pos_names)], line=dict(color="#fff", width=2)),
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}/MT<extra></extra>",
        )])
        fig_donut.add_annotation(text=f"<b>₹{Gross_Op_Benefits:,.0f}</b><br><span style='font-size:10px'>Gross Credits</span>", x=0.5, y=0.5, font_size=14, showarrow=False)
        fig_donut.update_layout(title="Gross Operational Credit Composition (₹/MT Alloy)", template="plotly_white", height=420, margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig_donut, use_container_width=True)

        k1, k2 = st.columns(2)
        with k1: st.markdown(kpi(f"MT Alloy/MT Active Mn ({alloy1_name})", f"{Alloy_Per_MT_Mn_Base:.3f}", "", ""), unsafe_allow_html=True)
        with k2: st.markdown(kpi(f"MT Alloy/MT Active Mn ({alloy2_name})", f"{Alloy_Per_MT_Mn_Alt:.3f}", "", "kpi-card-green"), unsafe_allow_html=True)

    # ── SECTION 4: BENEFIT BREAKDOWN ──
    st.markdown('<div class="section-header">Detailed Benefit Breakdown</div>', unsafe_allow_html=True)
    all_benefit_names = benefit_names.copy()
    all_benefit_names.insert(-1, "Hydrogen Penalty")
    all_benefit_values = benefit_values.copy()
    all_benefit_values.insert(-1, Benefit_Hydrogen)
    
    all_benefit_basis = [
        f"ΔT={Delta_Temp_Rigorous:.3f}°C/t steel, {P_LF_Efficiency*100:.0f}% LF eff, {R_Power*100:.0f}% real.",
        f"P_kWh_saved={Power_kWh_Saved_Per_MT:.1f} kWh/MT, {P_Graphite_Factor*1000:.0f}g/kWh, {R_Electrode*100:.0f}% real.",
        f"Time saved={Time_Saved_Min:.2f} min/heat, {R_Throughput*100:.0f}% real.",
        f"Overdose Δ={(Alloy1_Overdose-Alloy2_Overdose)*100:.1f}%, {R_Stability*100:.0f}% real.",
        f"Miss Δ={(Retreatment_Alloy1-Retreatment_Alloy2)*100:.1f}%, {R_Reblow*100:.0f}% real.",
        f"Reject Δ={(Reject_Alloy1-Reject_Alloy2)*100:.4f}%, {R_Cleanliness*100:.0f}% real.",
        f"Yield factor={P_Yield_Factor*1e6:.1f}ppm, {R_Yield*100:.0f}% real.",
        f"C-corr freq={C_Corr_Freq_Alloy1*100:.0f}%, {RH_Corr_Time}min, ₹{P_RH_Minute_Cost}/min.",
        f"H₂ pickup={H2_Bath_Pickup:.3f}ppm, degas={H2_Degas_Rate:.3f}ppm/min.",
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
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_table:
        df_breakdown = pd.DataFrame({"Benefit Component": all_benefit_names, "₹/MT Alloy": [f"₹{v:+,.0f}" for v in all_benefit_values], "Basis & Assumptions": all_benefit_basis}).set_index("Benefit Component")
        def color_values(val):
            num = float(val.replace("₹", "").replace(",", "").replace("+", ""))
            return "color: #1B5E20; font-weight: 600" if num > 0 else ("color: #B71C1C; font-weight: 600" if num < 0 else "")
        st.dataframe(df_breakdown.style.map(color_values, subset=["₹/MT Alloy"]), use_container_width=True, height=460)

    # ── SECTION 5: WATERFALL ANALYSIS ──
    st.markdown('<div class="section-header">VIU Waterfall Analysis</div>', unsafe_allow_html=True)
    wf_labels = [f"{alloy1_name} Active Mn Cost", "Power Saving", "Electrode Saving", "Throughput Gain", "Recovery Stability",
                 "Re-treatment Reduction", "Cleanliness", "Yield", "Carbon Avoidance", "Hydrogen Penalty", "Refractory Life",
                 "Lost Iron Credit", f"{alloy2_name} Active Mn Cost"]
    wf_values = [Cost_Per_Mn_Base, Benefit_Power, Benefit_Electrode, Benefit_Throughput, Benefit_Stability, Benefit_Retreatment,
                 Benefit_Cleanliness, Benefit_Yield, Benefit_Carbon, Benefit_Hydrogen, Benefit_Refractory, -Iron_Credit_Base, 0]
    
    measures = ["absolute"] + ["relative"] * (len(wf_labels) - 2) + ["total"]
    wf_text = [f"₹{abs(v):,.0f}" for v in wf_values[:-1]] + [f"₹{Cost_Per_Mn_Alt:,.0f}"]
    wf_values_display = wf_values[:-1] + [Cost_Per_Mn_Alt]

    fig_wf = go.Figure(go.Waterfall(
        name="VIU Waterfall", orientation="v", measure=measures, x=wf_labels, y=wf_values_display, text=wf_text, textposition="outside",
        connector=dict(line=dict(color="#BDBDBD", width=1.5, dash="dot")),
        increasing=dict(marker=dict(color=C_DELTA)), decreasing=dict(marker=dict(color=C_NEG)),
        totals=dict(marker=dict(color="#4CAF50" if Cost_Per_Mn_Alt <= Cost_Per_Mn_Base + Total_Op_Credits else C_NEG)),
    ))
    fig_wf.add_hline(y=Cost_Per_Mn_Alt, line_dash="dash", line_color="#4CAF50", line_width=1.5, annotation_text=f"Market Cost/MT Mn ₹{Cost_Per_Mn_Alt:,.0f}", annotation_position="right")
    fig_wf.update_layout(**_layout("VIU Waterfall: Active Mn Cost & Operational Adjustments (₹/MT)", "₹/MT", 520))
    fig_wf.update_layout(showlegend=False, xaxis_tickangle=-30)
    st.plotly_chart(fig_wf, use_container_width=True)

    # ── SECTION 6: COST COMPARISON ──
    st.markdown('<div class="section-header">Cost Comparison & Sensitivity Analysis</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        fig_stack = go.Figure()
        cats = [alloy1_name, alloy2_name]
        fig_stack.add_trace(go.Bar(name="Cost per MT Active Mn", x=cats, y=[Cost_Per_Mn_Base, Cost_Per_Mn_Alt], marker_color=[C_ALLOY1, C_ALLOY2], text=[f"₹{Cost_Per_Mn_Base:,.0f}", f"₹{Cost_Per_Mn_Alt:,.0f}"], textposition="inside"))
        fig_stack.add_trace(go.Bar(name="Gross Operational Credits (deduct)", x=cats, y=[0, -Gross_Op_Benefits], marker_color=["rgba(0,0,0,0)", "#FFC107"], text=["", f"-₹{Gross_Op_Benefits:,.0f}"], textposition="inside"))
        fig_stack.add_trace(go.Bar(name="Lost Iron Credit Penalty (add)", x=cats, y=[0, Iron_Credit_Base], marker_color=["rgba(0,0,0,0)", "#FF7043"], text=["", f"+₹{Iron_Credit_Base:,.0f}"], textposition="inside"))
        fig_stack.update_layout(barmode="relative", **_layout("Effective Cost Components (₹/MT Active Mn)", "₹/MT", 420))
        st.plotly_chart(fig_stack, use_container_width=True)

    with col_b:
        emm_prices = np.linspace(P_Alloy1_Price * 0.8, P_Alloy1_Price * 2.5, 80)
        cost_mn_alts = (1.0 / (P_Alloy2_Mn * P_Alloy2_Rec)) * emm_prices
        net_viuss = Total_Op_Credits - (cost_mn_alts - Cost_Per_Mn_Base)
        breakeven = (Cost_Per_Mn_Base + Total_Op_Credits) * (P_Alloy2_Mn * P_Alloy2_Rec)

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=emm_prices, y=net_viuss, mode="lines", name="Net Advantage", line=dict(color=C_DELTA, width=3), fill="tozeroy", fillcolor="rgba(76,175,80,0.1)"))
        fig_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
        fig_sens.add_vline(x=P_Alloy2_Price, line_dash="dot", line_color=C_ALLOY2, line_width=2, annotation_text=f"Current ₹{P_Alloy2_Price:,}", annotation_position="top right")
        fig_sens.add_vline(x=breakeven, line_dash="dot", line_color=C_NEG, line_width=2, annotation_text=f"Break-even ₹{breakeven:,.0f}", annotation_position="top left")
        fig_sens.update_layout(**_layout(f"{alloy2_name} Price Sensitivity", "Net Advantage (₹/MT)", 420))
        st.plotly_chart(fig_sens, use_container_width=True)

    # ── SECTION 7: ENTERPRISE SAVINGS ──
    st.markdown('<div class="section-header">Enterprise Savings Calculator</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1: st.markdown(kpi("Substituted Volume", f"{EMM_Consumption_FY * Substitution_Pct:,.0f} MT", f"at {Substitution_Pct*100:.0f}% substitution", ""), unsafe_allow_html=True)
    with s2: st.markdown(kpi("Savings / MT Alloy", f"₹{abs(Savings_Per_MT):,.0f}", "Magnitude of net advantage", "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-amber"), unsafe_allow_html=True)
    with s3: st.markdown(kpi("Annual Savings", f"₹{abs(Annual_Savings_Cr):.2f} Cr", "at stated volume", "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-amber"), unsafe_allow_html=True)
    with s4: st.markdown(kpi("Monthly Savings", f"₹{abs(Annual_Savings_Cr * 1e7 / 12 / 1e5):.1f} L", "per month average", "kpi-card-purple"), unsafe_allow_html=True)

    # ── SECTION 8: RECOMMENDATION ──
    st.markdown('<div class="section-header">Final Recommendation</div>', unsafe_allow_html=True)
    if Savings_Per_MT > 0:
        st.markdown(f"""<div class="success-box" style="padding:24px 32px; border-radius:12px;"><h2 style="color:#1B5E20; margin-top:0;">🏆 {alloy2_name} Preferred</h2>
        <p style="font-size:16px;"><b>Projected Annual Savings: ₹{Annual_Savings_Cr:.2f} Crore</b><br>
        By shifting {Substitution_Pct*100:.0f}% of consumption to {alloy2_name}, you realize a net advantage of <b>₹{Savings_Per_MT:,.0f}/MT alloy</b>. 
        Operational credits offset the cost premium.</p></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="warn-box" style="padding:24px 32px; border-radius:12px;"><h2 style="color:#E65100; margin-top:0;">🏆 {alloy1_name} Preferred</h2>
        <p style="font-size:16px;"><b>Cost Efficiency: ₹{abs(Savings_Per_MT):,.0f}/MT alloy</b><br>
        {alloy1_name} remains more cost-effective, saving <b>₹{Annual_Savings_Cr:.2f} Crore</b> vs switching. 
        Operational credits do not fully offset the Active Mn cost premium.</p></div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: SUBSTITUTION SOLVER
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🧠 Optimal Alloy Substitution Solver</div>', unsafe_allow_html=True)
    st.markdown(f"Calculates the mathematically cheapest blend of **{alloy1_name}** and **{alloy2_name}** that satisfies strict metallurgical limits.")

    if is_mc:
        grades_data = {
            "Commodity (IS2062)":   {"c_lim": 0.020, "rec_lim": 0.06,  "inc_lim": 0.12, "metric4_lim": 0.05, "metric5_lim": 0.40, "briq_max": 0.0},
            "TMT/Rebar (Fe500D)":   {"c_lim": 0.018, "rec_lim": 0.05,  "inc_lim": 0.10, "metric4_lim": 0.04, "metric5_lim": 0.30, "briq_max": 0.0},
            "HSLA/API (API X70)":   {"c_lim": 0.010, "rec_lim": 0.03,  "inc_lim": 0.06, "metric4_lim": 0.03, "metric5_lim": 0.20, "briq_max": 1.0},
            "Automotive (DP600)":   {"c_lim": 0.005, "rec_lim": 0.02,  "inc_lim": 0.04, "metric4_lim": 0.02, "metric5_lim": 0.15, "briq_max": 1.0},
            "Electrical (CRGO)":    {"c_lim": 0.002, "rec_lim": 0.015, "inc_lim": 0.02, "metric4_lim": 0.01, "metric5_lim": 0.10, "briq_max": 1.0},
            "IF Steel (Deep Draw)": {"c_lim": 0.001, "rec_lim": 0.01,  "inc_lim": 0.02, "metric4_lim": 0.01, "metric5_lim": 0.10, "briq_max": 1.0},
        }
        metric4_name, metric5_name = "Reblow Risk Allowed", "Max RH Correction Need"
        # Constants from Excel Solver matrix for MC FeMn
        a_carbon, a_recvar, a_clean, a_metric4, a_metric5 = [0.015, 0.001], [0.05, 0.015], [0.09, 0.03], [0.04, 0.02], [0.3, 0.1]
    else:
        grades_data = {
            "Commodity (IS2062)":   {"c_lim": 0.150, "rec_lim": 0.04,  "inc_lim": 0.10, "metric4_lim": 1.00, "metric5_lim": 1.50, "briq_max": 0.0},
            "TMT/Rebar (Fe500D)":   {"c_lim": 0.200, "rec_lim": 0.04,  "inc_lim": 0.10, "metric4_lim": 1.00, "metric5_lim": 1.50, "briq_max": 0.0},
            "HSLA/API (API X70)":   {"c_lim": 0.080, "rec_lim": 0.03,  "inc_lim": 0.06, "metric4_lim": 0.60, "metric5_lim": 1.00, "briq_max": 1.0},
            "Automotive (DP600)":   {"c_lim": 0.050, "rec_lim": 0.02,  "inc_lim": 0.04, "metric4_lim": 0.40, "metric5_lim": 0.80, "briq_max": 1.0},
            "Electrical (CRGO)":    {"c_lim": 0.020, "rec_lim": 0.015, "inc_lim": 0.02, "metric4_lim": 0.20, "metric5_lim": 0.60, "briq_max": 1.0},
            "IF Steel (Deep Draw)": {"c_lim": 0.010, "rec_lim": 0.01,  "inc_lim": 0.03, "metric4_lim": 0.10, "metric5_lim": 0.50, "briq_max": 1.0},
        }
        metric4_name, metric5_name = "Max Hydrogen Risk", "Refractory Wear Limit"
        # Constants from Excel Solver matrix for LC FeMn
        a_carbon, a_recvar, a_clean, a_metric4, a_metric5 = [0.005, 0.0001], [0.03, 0.015], [0.08, 0.02], [0.05, 0.8], [1.5, 0.5]

    gc1, gc2 = st.columns([1.5, 2.5])
    sel_grade = gc1.selectbox("Select Target Steel Grade", list(grades_data.keys()), index=0)
    limits = grades_data[sel_grade]

    st.markdown("#### Metallurgical Constraints")
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    max_c       = sc1.slider("Max Carbon Limit",   0.001, 0.250, limits["c_lim"], step=0.005, format="%.3f")
    max_rec     = sc2.slider("Max Recovery Var",   0.005, 0.080, limits["rec_lim"], step=0.005, format="%.3f")
    max_inc     = sc3.slider("Max Inclusion Index",0.01,  0.15,  limits["inc_lim"], step=0.01)
    max_metric4 = sc4.slider(metric4_name,         0.01,  1.20,  limits["metric4_lim"], step=0.05)
    max_briq    = sc5.slider(f"Max {alloy2_name} (%)",0.0, 100.0, limits["briq_max"] * 100, step=5.0) / 100.0

    # Linear Programming Engine
    c_cost = [Cost_Per_Mn_Base, Cost_Per_Mn_Alt]
    A_eq = [[1, 1]]
    b_eq = [1]

    A_ub = [
        a_carbon,
        a_recvar,
        a_clean,
        a_metric4,
        [0, 1], # Max Alternative Alloy
    ]
    b_ub = [max_c, max_rec, max_inc, max_metric4, max_briq]

    res = linprog(c_cost, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1), (0, 1)])

    st.markdown("#### Optimization Result")
    if res.success:
        mix = res.x
        blended_cost = res.fun
        costlier_commodity_cost = max(Cost_Per_Mn_Base, Cost_Per_Mn_Alt)
        savings = costlier_commodity_cost - blended_cost

        rc1, rc2 = st.columns(2)
        rc1.success(f"##### Final Effective Cost: \n ### **₹{blended_cost:,.0f}** per MT Eff. Mn")
        if savings > 10:
            rc2.info(f"##### Projected Savings vs Single Commodity: \n ### **₹{savings:,.0f}** per MT Eff. Mn")
        else:
            rc2.info(f"##### Projected Savings vs Single Commodity: \n ### **₹0** (100% Single Alloy is best)")

        fig_pie = go.Figure(data=[go.Pie(
            labels=[f"{alloy1_name} Share", f"{alloy2_name} Share"], 
            values=[round(m, 4) for m in mix], hole=0.4, 
            marker_colors=[C_ALLOY1, C_ALLOY2], textinfo="label+percent"
        )])
        fig_pie.update_layout(title=f"Optimal Procurement Ratio for {sel_grade}", height=380, template="plotly_white")
        st.plotly_chart(fig_pie, use_container_width=True)

        # --- INSIGHTS ---
        st.markdown("---")
        st.markdown("#### 📊 Deep Dive & Insights")
        col_insight1, col_insight2 = st.columns(2)
        
        with col_insight1:
            fig_cost = go.Figure()
            fig_cost.add_trace(go.Bar(
                x=[f"100% {alloy1_name}", "Optimal Blend", f"100% {alloy2_name}"],
                y=[Cost_Per_Mn_Base, blended_cost, Cost_Per_Mn_Alt],
                marker_color=[C_ALLOY1, "#9C27B0", C_ALLOY2],
                text=[f"₹{Cost_Per_Mn_Base:,.0f}", f"₹{blended_cost:,.0f}", f"₹{Cost_Per_Mn_Alt:,.0f}"],
                textposition="auto", hovertemplate="%{x}<br>₹%{y:,.0f}/MT<extra></extra>"
            ))
            fig_cost.update_layout(**_layout("Effective Cost Comparison (₹/MT Active Mn)", "Cost (₹)", 380))
            st.plotly_chart(fig_cost, use_container_width=True)
            
        with col_insight2:
            actual_c       = mix[0] * a_carbon[0]  + mix[1] * a_carbon[1]
            actual_rec     = mix[0] * a_recvar[0]  + mix[1] * a_recvar[1]
            actual_inc     = mix[0] * a_clean[0]   + mix[1] * a_clean[1]
            actual_metric4 = mix[0] * a_metric4[0] + mix[1] * a_metric4[1]
            actual_briq    = mix[1]
            
            utils = [
                (actual_briq / max_briq) * 100 if max_briq else 0,
                (actual_metric4 / max_metric4) * 100 if max_metric4 else 0,
                (actual_inc / max_inc) * 100 if max_inc else 0,
                (actual_rec / max_rec) * 100 if max_rec else 0,
                (actual_c / max_c) * 100 if max_c else 0
            ]
            util_labels = [f"Max {alloy2_name} Share", metric4_name, "Cleanliness", "Recovery Var", "Carbon Limit"]
            
            fig_util = go.Figure()
            fig_util.add_trace(go.Bar(y=util_labels, x=utils, orientation='h', marker_color="#26A69A", text=[f"{u:.1f}%" for u in utils], textposition="inside"))
            fig_util.add_vline(x=100, line_dash="dash", line_color="red", annotation_text="Limit (100%)")
            fig_util.update_layout(**_layout("Constraint Utilization (% of Max Limit Used)", "% Used", 380))
            fig_util.update_xaxes(range=[0, max(110, max(utils)*1.1)])
            st.plotly_chart(fig_util, use_container_width=True)
            
        st.markdown("#### Metallurgical Profile of the Optimal Blend")
        df_profile = pd.DataFrame({
            "Parameter": ["Carbon Input Index", "Recovery Variability", "Cleanliness Index", metric4_name, f"Max {alloy2_name} Share"],
            "Blend Actual": [actual_c, actual_rec, actual_inc, actual_metric4, actual_briq],
            "Maximum Allowed": [max_c, max_rec, max_inc, max_metric4, max_briq],
        })
        
        df_profile["Status"] = np.where(df_profile["Blend Actual"] >= df_profile["Maximum Allowed"] - 1e-6, "🛑 Binding Constraint", "✅ Safe")
        
        def format_val(val, is_pct): return f"{val*100:.2f}%" if is_pct else f"{val:.4f}"
        df_profile["Blend Actual"] = df_profile.apply(lambda row: format_val(row["Blend Actual"], "Share" in row["Parameter"]), axis=1)
        df_profile["Maximum Allowed"] = df_profile.apply(lambda row: format_val(row["Maximum Allowed"], "Share" in row["Parameter"]), axis=1)

        def color_status(val): return "color: #D32F2F; font-weight: bold" if "Binding" in val else "color: #388E3C"
        st.dataframe(df_profile.style.map(color_status, subset=["Status"]), use_container_width=True)

    else:
        st.error("⚠️ **Constraint Violation:** The chosen metallurgical limits are too strict to be met using these alloys. Please relax the constraints.")


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="text-align:center; color:#90A4AE; font-size:12px; padding:8px 0;">
  VIU Dashboard & Solver – Unified Comparison Matrix &nbsp;|&nbsp; All formulas sourced from Excel workbook logic &nbsp;|&nbsp; 
  Operational benefits per MT alloy at stated realization factors.
</div>
""", unsafe_allow_html=True)