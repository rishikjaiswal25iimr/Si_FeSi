"""
VIU DASHBOARD – LC FeMn vs Mn Briquette (EMM)
================================================
Value-in-Use comparison of Low-Carbon Ferromanganese (80% Mn)
against Electrolytic Manganese Metal / Mn Briquette (99.7% Mn).

All formulas sourced exclusively from the Excel file:
  • INPUT_PARAMETER sheet  → every adjustable parameter
  • BREAKDOWN_CALC sheet   → all benefit calculations
  • VIU_SUMMARY sheet      → synthesis & enterprise savings

Architecture & UX inspired by the Manganese Intelligence dashboard.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG & THEME
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="VIU Dashboard – LC FeMn vs Mn Briquette",
    page_icon="⚗️",
    layout="wide",
)

# Colour palette
C_LCFEMN   = "#2196F3"   # blue  – LC FeMn
C_EMM      = "#4CAF50"   # green – Mn Briquette / EMM
C_DELTA    = "#FF9800"   # amber – delta / benefit
C_NEG      = "#F44336"   # red   – penalties / negative
C_GRID     = "#EEEEEE"
C_BG       = "#FAFAFA"
C_TEXT     = "#333333"
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
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Plotly layout template
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

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARD HELPER
# ══════════════════════════════════════════════════════════════════════════════
def kpi(label: str, value: str, sub: str = "", colour: str = "") -> str:
    cls = f"kpi-card {colour}"
    return f"""
    <div class="{cls}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>"""

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR – ALL INPUT PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚗️ VIU Dashboard")
    st.divider()

    st.markdown("### A. Comparison Selection")
    comparison_selection = st.radio(
        "Select Analysis:",
        ["Not selected", "LC FeMn vs Mn Briquette"],
        index=0
    )
    
    st.divider()
    st.markdown("### B. Financial Parameters")
    P_LCFeMn_Price       = st.number_input("LC FeMn Price (₹/MT)",        value=145000, step=1000, min_value=50000, max_value=400000)
    P_EMM_Price          = st.number_input("Mn Briquette Price (₹/MT)", value=240000, step=1000, min_value=50000, max_value=600000)
    P_Power_Tariff       = st.number_input("Power Tariff (₹/kWh)",         value=6.5,   step=0.1, min_value=1.0,   max_value=20.0, format="%.2f")
    P_Electrode_Cost     = st.number_input("Electrode Cost (₹/kg)",        value=240,   step=10,  min_value=50,    max_value=800)
    P_Steel_Value        = st.number_input("Steel Value (₹/MT)",           value=60000, step=1000, min_value=20000, max_value=200000)
    P_Margin_Steel       = st.number_input("Throughput Margin (₹/MT)",     value=2800,  step=100, min_value=500,   max_value=10000)
    P_LF_Retreatment_Cost= st.number_input("LF Re-treatment Cost (₹/heat)",value=15000, step=500, min_value=2000,  max_value=50000)
    P_RH_Minute_Cost     = st.number_input("RH Cost per Minute (₹/min)",   value=2500,  step=100, min_value=500,   max_value=10000)
    P_Ladle_Reline_Cost  = st.number_input("Ladle Reline Cost (₹)",        value=1500000,step=50000,min_value=200000,max_value=5000000)
    P_Scrap_Price        = st.number_input("Scrap / Fe Credit (₹/MT)",     value=35000, step=500, min_value=5000,  max_value=80000)

    st.divider()
    st.markdown("### C. Technical Parameters")
    P_LCFeMn_Mn  = st.slider("LC FeMn Mn Content (%)",   60.0, 95.0, 80.0, 0.5) / 100
    P_EMM_Mn     = st.slider("EMM Mn Content (%)",        95.0, 100.0, 99.7, 0.1) / 100
    P_LCFeMn_Rec = st.slider("LC FeMn Recovery (%)",      70.0, 99.0, 90.0, 0.5) / 100
    P_EMM_Mn_Rec = st.slider("EMM Recovery (%)",          80.0, 99.9, 97.0, 0.5) / 100
    P_LCFeMn_Fe  = st.slider("LC FeMn Fe Content (%)",    5.0,  35.0, 15.0, 0.5) / 100
    P_LCFeMn_C   = st.slider("LC FeMn Carbon (%)",        0.1,  2.0,  0.5,  0.1) / 100
    P_SpHeat_Steel  = st.slider("Steel Specific Heat (MJ/T/°C)", 0.5, 1.0, 0.75, 0.01)
    P_Chill_LCFeMn  = st.slider("LC FeMn Chill Factor (°C/kg/t)", 1.0, 4.0, 2.057, 0.001)
    P_Chill_EMM     = st.slider("EMM Chill Factor (°C/kg/t)",     0.5, 2.5, 1.0,  0.05)
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
    LCFeMn_Overdose    = st.slider("LC FeMn Overdose Buffer (%)",  0.5,  5.0,  2.0,  0.1) / 100
    EMM_Overdose       = st.slider("EMM Overdose Buffer (%)",       0.1,  2.0,  0.5,  0.1) / 100
    LCFeMn_Rec_Var     = st.slider("LC FeMn Recovery Std-Dev (%)", 0.5,  6.0,  3.0,  0.1) / 100
    EMM_Rec_Var        = st.slider("EMM Recovery Std-Dev (%)",      0.5,  3.0,  1.5,  0.1) / 100
    Reject_LCFeMn      = st.number_input("LC FeMn Rejection Rate", value=0.0005, format="%.5f", step=0.0001)
    Reject_EMM         = st.number_input("EMM Rejection Rate",      value=0.00035, format="%.5f", step=0.0001)
    Retreatment_LCFeMn = st.slider("LC FeMn Re-treatment Rate (%)",1.0,  8.0,  3.0,  0.1) / 100
    Retreatment_EMM    = st.slider("EMM Re-treatment Rate (%)",     0.5,  5.0,  2.5,  0.1) / 100
    C_Corr_Freq_LCFeMn = st.slider("Carbon Correction Frequency", 0.02, 0.30, 0.10, 0.01)
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
    st.info("Please select LC FeMn vs Mn Briquette to run the VIU analysis.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# CORE CALCULATIONS (EXACT EXCEL REPLICATION)
# ══════════════════════════════════════════════════════════════════════════════

Steel_Per_MT_Alloy = 1000.0 / P_Alloy_Target
kWh_MJ = 3.6

# --- Power Saving ---
Delta_Chill = P_Chill_LCFeMn - P_Chill_EMM
# Excel dynamically scales LF_Efficiency down by an arc duty multiplier (0.45 * 0.60 = 0.27 effective)
Eff_LF_Effective = P_LF_Efficiency * P_Arc_Duty
Power_kWh_Saved_Per_MT = (1000.0 * Delta_Chill * P_SpHeat_Steel) / (kWh_MJ * Eff_LF_Effective)
Benefit_Power = Power_kWh_Saved_Per_MT * P_Power_Tariff * R_Power

# --- Electrode Saving ---
Benefit_Electrode = Power_kWh_Saved_Per_MT * P_Graphite_Factor * P_Electrode_Cost * R_Electrode

# --- Throughput Gain ---
Thermal_Gain_Total = Delta_Chill * P_Alloy_Target 
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
# Excel directly correlates H2 penalty to the H2 Bath Pickup mapping limit (1.5 ppm default)
Benefit_Hydrogen = -(H2_Bath_Pickup / H2_Degas_Rate) * P_RH_Minute_Cost * (1000.0 / (P_Alloy_Target * P_Heat_Size))

# --- Refractory Life ---
Benefit_Refractory = (P_Ladle_Reline_Cost / P_Ladle_Life) * Refractory_Wear_Drop * (1000.0 / (P_Alloy_Target * P_Heat_Size))

# --- Total Operational Credits ---
Total_Op_Credits = (
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

# Excel's Final Synthesis Logic: Net Value in Use Advantage = Direct Cost Delta - Total VIU Credits
# (Positive Net VIU implies EMM is at a cost penalty)
Net_VIU_Advantage = Cost_Per_Mn_Delta - Total_Op_Credits

# Savings Per MT applies inversely (if penalty is positive, savings for EMM is negative)
Savings_Per_MT = Total_Op_Credits - Cost_Per_Mn_Delta

# --- New Enterprise Savings Formula ---
Annual_Savings_Rs = EMM_Consumption_FY * Substitution_Pct * abs(Savings_Per_MT)
Annual_Savings_Cr = Annual_Savings_Rs / 1e7


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DASHBOARD HEADER
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: TOP KPI CARDS
# ══════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    st.markdown(kpi("LC FeMn Price", f"₹{P_LCFeMn_Price:,.0f}", "per MT alloy", ""), unsafe_allow_html=True)
with c2:
    st.markdown(kpi("Mn Briquette Price", f"₹{P_EMM_Price:,.0f}", "per MT alloy", "kpi-card-green"), unsafe_allow_html=True)
with c3:
    st.markdown(kpi("Mn Cost Gap", f"₹{Cost_Per_Mn_Delta:,.0f}", "per MT Active Mn", "kpi-card-amber"), unsafe_allow_html=True)
with c4:
    st.markdown(kpi("Total VIU Credits", f"₹{Total_Op_Credits:,.0f}", "operational benefit / MT alloy", "kpi-card-green"), unsafe_allow_html=True)
with c5:
    col = "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-red"
    lbl = "Net Savings / MT Alloy"
    st.markdown(kpi(lbl, f"₹{Savings_Per_MT:+,.0f}", "EMM advantage (positive = better)", col), unsafe_allow_html=True)
with c6:
    col_yr = "kpi-card-green" if Savings_Per_MT > 0 else "kpi-card-red"
    st.markdown(kpi("Annual Savings FY26", f"₹{abs(Annual_Savings_Cr):.2f} Cr", f"@ {Substitution_Pct*100:.0f}% Substitution", col_yr), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: VIU SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
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
            "Iron Credit (Not in Net Sum)",
            "Total Operational Credits",
            "Net VIU Advantage (Credits − Delta)",
        ],
        "LC FeMn (₹/MT)": [
            f"₹{Cost_Per_Mn_LC:,.0f}", "—",
            f"₹{Iron_Credit_LC:,.0f}", "—", "—",
        ],
        "EMM (₹/MT)": [
            f"₹{Cost_Per_Mn_EMM:,.0f}", f"₹{Cost_Per_Mn_Delta:,.0f}",
            "—", f"₹{Total_Op_Credits:,.0f}", f"₹{Savings_Per_MT:+,.0f}",
        ],
    }
    df_sum = pd.DataFrame(data_summary).set_index("Component")
    st.dataframe(df_sum, use_container_width=True)

    # Verdict
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
    # --- VIU Donut: Credit composition ---
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
    # Only positive credits for the donut
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
        text=f"<b>₹{Total_Op_Credits:,.0f}</b><br><span style='font-size:10px'>Total Credits</span>",
        x=0.5, y=0.5, font_size=14, showarrow=False,
    )
    fig_donut.update_layout(
        title="Operational Credit Composition (₹/MT Alloy)",
        template="plotly_white", height=420,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(font=dict(size=11)),
    )
    st.plotly_chart(fig_donut, use_container_width=True)

    # Summary KPIs
    k1, k2 = st.columns(2)
    with k1:
        st.markdown(kpi("Alloy/MT Active Mn (LC)", f"{Alloy_Per_MT_Mn_LC:.3f} MT", "LC FeMn required", ""), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi("Alloy/MT Active Mn (EMM)", f"{Alloy_Per_MT_Mn_EMM:.3f} MT", "Mn Briquette required", "kpi-card-green"), unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: BENEFIT BREAKDOWN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Detailed Benefit Breakdown</div>', unsafe_allow_html=True)

all_benefit_names = [
    "Power Saving",
    "Electrode Saving",
    "Throughput Gain",
    "Recovery Stability",
    "Re-treatment Reduction",
    "Cleanliness Benefit",
    "Yield Improvement",
    "Carbon Avoidance",
    "Hydrogen Penalty",
    "Refractory Benefit",
]
all_benefit_values = [
    Benefit_Power, Benefit_Electrode, Benefit_Throughput,
    Benefit_Stability, Benefit_Retreatment, Benefit_Cleanliness,
    Benefit_Yield, Benefit_Carbon, Benefit_Hydrogen, Benefit_Refractory,
]
all_benefit_basis = [
    f"ΔChill={Delta_Chill:.3f}°C/kg/t, {Eff_LF_Effective*100:.0f}% eff (Arc*LF), {R_Power*100:.0f}% real.",
    f"P_kWh_saved={Power_kWh_Saved_Per_MT:.1f} kWh, {P_Graphite_Factor*1000:.0f}g/kWh, {R_Electrode*100:.0f}% real.",
    f"Time saved={Time_Saved_Min:.2f} min/heat, {R_Throughput*100:.0f}% real.",
    f"Overdose Δ={(LCFeMn_Overdose-EMM_Overdose)*100:.1f}%, {R_Stability*100:.0f}% real.",
    f"Miss Δ={(Retreatment_LCFeMn-Retreatment_EMM)*100:.1f}%, {R_Reblow*100:.0f}% real.",
    f"Reject Δ={(Reject_LCFeMn-Reject_EMM)*100:.4f}%, {R_Cleanliness*100:.0f}% real.",
    f"Yield factor={P_Yield_Factor*1e6:.1f}ppm, {R_Yield*100:.0f}% real.",
    f"C-corr freq={C_Corr_Freq_LCFeMn*100:.0f}%, {RH_Corr_Time}min, ₹{P_RH_Minute_Cost}/min.",
    f"H₂ pickup={H2_Bath_Pickup:.3f}ppm, degas={H2_Degas_Rate:.3f}ppm/min.",
    f"Wear drop={Refractory_Wear_Drop*100:.1f}%, ladle cost=₹{P_Ladle_Reline_Cost:,}.",
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
        **_layout("Benefit Contribution per MT Alloy (₹/MT)", "₹/MT Alloy", 460)
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
        df_breakdown.style.applymap(color_values, subset=["₹/MT Alloy"]),
        use_container_width=True, height=460,
    )

st.markdown("<br>", unsafe_allow_html=True)

# Heatmap of benefits by realization factor sensitivity
st.markdown("#### Benefit Sensitivity Heatmap (₹/MT at varying Realization Factors)")
real_range = np.arange(0.1, 1.05, 0.1)
heat_matrix = []
# only realizable benefits (excluding hydrogen penalty which is fixed physics)
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
    z=heat_matrix,
    x=[f"{r*100:.0f}%" for r in real_range],
    y=heat_names,
    colorscale="Blues",
    text=np.round(heat_matrix, 0).astype(int),
    texttemplate="₹%{text}",
    textfont=dict(size=10),
    hovertemplate="<b>%{y}</b><br>Realization: %{x}<br>₹%{z:,.0f}/MT<extra></extra>",
))
fig_heat.update_layout(
    **_layout("VIU Benefit Heatmap — Realization Factor Sensitivity", "", 380)
)
fig_heat.update_layout(xaxis_title="Realization Factor", yaxis_title="")
st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: WATERFALL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">VIU Waterfall Analysis</div>', unsafe_allow_html=True)

# Build waterfall mirroring Excel's Active Mn Synthesis approach
wf_labels = [
    "LC FeMn Active Mn Cost",
    "Power Saving",
    "Electrode Saving",
    "Throughput Gain",
    "Recovery Stability",
    "Re-treatment Reduction",
    "Cleanliness",
    "Yield",
    "Carbon Avoidance",
    "Hydrogen Penalty",
    "Refractory Life",
    "EMM Active Mn Cost",
]
# Start from Active Mn Cost, apply credits (as offsets mapping towards EMM equivalent cost)
wf_values = [
    Cost_Per_Mn_LC,           # base
    Benefit_Power,            # positive shift towards closing the EMM premium gap
    Benefit_Electrode,
    Benefit_Throughput,
    Benefit_Stability,
    Benefit_Retreatment,
    Benefit_Cleanliness,
    Benefit_Yield,
    Benefit_Carbon,
    Benefit_Hydrogen,         # negative benefit shifts downwards (increases gap)
    Benefit_Refractory,
    0,                        # total placeholder
]

# Compute running cumulative for waterfall
measures = ["absolute"] + ["relative"] * (len(wf_labels) - 2) + ["total"]
wf_text = [f"₹{abs(v):,.0f}" for v in wf_values[:-1]] + [f"₹{Cost_Per_Mn_EMM:,.0f}"]

# EMM total bar
wf_values_display = wf_values[:-1] + [Cost_Per_Mn_EMM]

wf_colors = ["#1A237E"]  # base
for v in wf_values[1:-1]:
    wf_colors.append(C_DELTA if v > 0 else C_NEG)
wf_colors.append("#4CAF50")  # EMM total

fig_wf = go.Figure(go.Waterfall(
    name="VIU Waterfall",
    orientation="v",
    measure=measures,
    x=wf_labels,
    y=wf_values_display,
    text=wf_text,
    textposition="outside",
    connector=dict(line=dict(color="#BDBDBD", width=1.5, dash="dot")),
    increasing=dict(marker=dict(color=C_DELTA)),
    decreasing=dict(marker=dict(color=C_NEG)),
    totals=dict(marker=dict(color="#4CAF50" if Cost_Per_Mn_EMM <= Cost_Per_Mn_LC + Total_Op_Credits else C_NEG)),
    hovertemplate="<b>%{x}</b><br>₹%{y:,.0f}<extra></extra>",
))
fig_wf.add_hline(
    y=Cost_Per_Mn_EMM, line_dash="dash", line_color="#4CAF50", line_width=1.5,
    annotation_text=f"EMM Cost/MT Mn ₹{Cost_Per_Mn_EMM:,.0f}", annotation_position="right",
)
fig_wf.update_layout(
    **_layout("VIU Waterfall: Active Mn Cost & Operational Adjustments (₹/MT)", "₹/MT", 520)
)
fig_wf.update_layout(showlegend=False, xaxis_tickangle=-30)
st.plotly_chart(fig_wf, use_container_width=True)

# Explanation
st.markdown("""
<div class="info-box">
<b>How to read this waterfall:</b> Visualizes the synthesis algorithm from the Excel model. 
Starting from the Base Cost per MT Active Mn of LC FeMn, we add the operational 
advantage benefits (Power, Electrode, Throughput, etc.) as credits mapping up towards the EMM Active Mn market price. 
The Hydrogen Penalty pushes the threshold back down. 
The final bar is the market cost per MT of Active Mn for EMM. If the total height of LC FeMn + Benefits 
exceeds the EMM bar, EMM is more cost effective. <i>Note: Iron credit is excluded from this summation per Excel's final logic</i>.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: COST COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Cost Comparison & Sensitivity Analysis</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2)

with col_a:
    # Stacked bar: price components
    fig_stack = go.Figure()
    categories = ["LC FeMn", "Mn Briquette (EMM)"]

    fig_stack.add_trace(go.Bar(
        name="Cost per MT Active Mn", x=categories,
        y=[Cost_Per_Mn_LC, Cost_Per_Mn_EMM],
        marker_color=[C_LCFEMN, C_EMM],
        text=[f"₹{Cost_Per_Mn_LC:,.0f}", f"₹{Cost_Per_Mn_EMM:,.0f}"],
        textposition="inside",
    ))
    fig_stack.add_trace(go.Bar(
        name="Operational Credits (deduct)", x=categories,
        y=[0, -Total_Op_Credits],
        marker_color=["rgba(0,0,0,0)", "#FFC107"],
        text=["", f"-₹{Total_Op_Credits:,.0f}"],
        textposition="inside",
    ))
    fig_stack.update_layout(
        barmode="relative",
        **_layout("Effective Cost Components (₹/MT Active Mn)", "₹/MT", 420),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

with col_b:
    # EMM Price sensitivity on Net VIU
    emm_prices  = np.linspace(P_LCFeMn_Price * 0.8, P_LCFeMn_Price * 2.5, 80)
    # Re-running the math for varying EMM prices
    cost_mn_emms = (1.0 / (P_EMM_Mn * P_EMM_Mn_Rec)) * emm_prices
    net_viuss   = Total_Op_Credits - (cost_mn_emms - Cost_Per_Mn_LC)
    breakeven   = (Cost_Per_Mn_LC + Total_Op_Credits) * (P_EMM_Mn * P_EMM_Mn_Rec)

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=emm_prices, y=net_viuss,
        mode="lines", name="Net VIU Advantage",
        line=dict(color=C_DELTA, width=3),
        fill="tozeroy",
        fillcolor="rgba(76,175,80,0.1)",
        hovertemplate="EMM Price: ₹%{x:,.0f}<br>Net Advantage: ₹%{y:,.0f}/MT<extra></extra>",
    ))
    fig_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
    fig_sens.add_vline(x=P_EMM_Price, line_dash="dot", line_color=C_EMM, line_width=2,
                       annotation_text=f"Current ₹{P_EMM_Price:,}", annotation_position="top right")
    fig_sens.add_vline(x=breakeven, line_dash="dot", line_color=C_NEG, line_width=2,
                       annotation_text=f"Break-even ₹{breakeven:,.0f}", annotation_position="top left")
    fig_sens.update_layout(
        **_layout("EMM Price Sensitivity – Net VIU Advantage (₹/MT)", "Net Advantage (₹/MT)", 420)
    )
    st.plotly_chart(fig_sens, use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)

# --- LC FeMn price sensitivity ---
col_c, col_d = st.columns(2)

with col_c:
    lc_prices   = np.linspace(P_EMM_Price * 0.3, P_EMM_Price * 1.2, 80)
    cost_mn_lcs = (1.0 / (P_LCFeMn_Mn * P_LCFeMn_Rec)) * lc_prices
    net_lc_sens = Total_Op_Credits - (Cost_Per_Mn_EMM - cost_mn_lcs)
    
    fig_lc_sens = go.Figure()
    fig_lc_sens.add_trace(go.Scatter(
        x=lc_prices, y=net_lc_sens,
        mode="lines", name="Net VIU (varying LC FeMn price)",
        line=dict(color=C_LCFEMN, width=3),
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.1)",
        hovertemplate="LC FeMn: ₹%{x:,.0f}<br>Net Advantage: ₹%{y:,.0f}/MT<extra></extra>",
    ))
    fig_lc_sens.add_hline(y=0, line_dash="dash", line_color="#333", line_width=1.5)
    fig_lc_sens.add_vline(x=P_LCFeMn_Price, line_dash="dot", line_color=C_LCFEMN, line_width=2,
                          annotation_text=f"Current ₹{P_LCFeMn_Price:,}", annotation_position="top right")
    fig_lc_sens.update_layout(
        **_layout("LC FeMn Price Sensitivity – Net VIU Advantage (₹/MT)", "Net Advantage (₹/MT)", 380)
    )
    st.plotly_chart(fig_lc_sens, use_container_width=True)

with col_d:
    # Tornado chart: individual benefit sensitivity (±20%)
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
        orientation="h", name="+20%", marker_color=C_DELTA,
        base=[b for b in tornado_base[::-1]],
    ))
    fig_tornado.add_trace(go.Bar(
        y=tornado_names[::-1], x=[l - b for l, b in zip(tornado_low[::-1], tornado_base[::-1])],
        orientation="h", name="−20%", marker_color="#EF9A9A",
        base=[b for b in tornado_base[::-1]],
    ))
    fig_tornado.update_layout(
        barmode="overlay",
        **_layout("Sensitivity Tornado (±20% Realization)", "₹/MT Alloy", 380),
    )
    st.plotly_chart(fig_tornado, use_container_width=True)

# --- Per-MT-Mn cost comparison table ---
st.markdown("#### Side-by-Side Cost per Active Manganese Summary")
df_cmp = pd.DataFrame({
    "Metric": [
        "Market Price (₹/MT alloy)",
        "Active Mn Content (%)",
        "Mn Recovery (%)",
        "Effective Mn Efficiency (%)",
        "Alloy Needed per MT Active Mn (MT)",
        "Raw Cost per MT Active Mn (₹)",
        "Fe Credit (₹/MT alloy)",
        "Total Operational Credits (₹/MT alloy)",
        "Net Adjusted Cost per MT Active Mn (₹)",
    ],
    "LC FeMn": [
        f"₹{P_LCFeMn_Price:,}",
        f"{P_LCFeMn_Mn*100:.1f}%",
        f"{P_LCFeMn_Rec*100:.1f}%",
        f"{P_LCFeMn_Mn*P_LCFeMn_Rec*100:.1f}%",
        f"{Alloy_Per_MT_Mn_LC:.3f} MT",
        f"₹{Cost_Per_Mn_LC:,.0f}",
        f"₹{Iron_Credit_LC:,}",
        "—",
        f"₹{Cost_Per_Mn_LC - (Iron_Credit_LC * Alloy_Per_MT_Mn_LC):,.0f}",
    ],
    "Mn Briquette (EMM)": [
        f"₹{P_EMM_Price:,}",
        f"{P_EMM_Mn*100:.1f}%",
        f"{P_EMM_Mn_Rec*100:.1f}%",
        f"{P_EMM_Mn*P_EMM_Mn_Rec*100:.1f}%",
        f"{Alloy_Per_MT_Mn_EMM:.3f} MT",
        f"₹{Cost_Per_Mn_EMM:,.0f}",
        "—",
        f"₹{Total_Op_Credits:,.0f}",
        f"₹{Cost_Per_Mn_EMM - Total_Op_Credits:,.0f}",
    ],
}).set_index("Metric")
st.dataframe(df_cmp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: ENTERPRISE SAVINGS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Enterprise Savings Calculator</div>', unsafe_allow_html=True)

# Top KPI row
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
    # Savings vs consumption volume chart
    vol_range = np.arange(1000, EMM_Consumption_FY * 2.5, 500)
    # Savings dynamically driven by substitution percentage parameter
    savings_cr = (abs(Savings_Per_MT) * vol_range * Substitution_Pct) / 1e7

    fig_sav = go.Figure()
    fig_sav.add_trace(go.Scatter(
        x=vol_range, y=savings_cr,
        mode="lines", name="Annual Savings (₹ Cr)",
        line=dict(color=C_DELTA if Savings_Per_MT > 0 else C_NEG, width=3),
        fill="tozeroy",
        fillcolor="rgba(76,175,80,0.12)" if Savings_Per_MT > 0 else "rgba(244,67,54,0.12)",
        hovertemplate="Consumption: %{x:,.0f} MT<br>Savings: ₹%{y:.2f} Cr<extra></extra>",
    ))
    fig_sav.add_vline(
        x=EMM_Consumption_FY, line_dash="dash", line_color="#1A237E", line_width=2,
        annotation_text=f"Total: {EMM_Consumption_FY:,} MT → ₹{Annual_Savings_Cr:.2f} Cr (@ {Substitution_Pct*100:.0f}% Sub)",
        annotation_position="top right",
    )
    fig_sav.add_hline(y=0, line_dash="solid", line_color="#333", line_width=1.5)
    fig_sav.update_layout(
        **_layout(f"Enterprise Savings vs Total Consumption Volume (at {Substitution_Pct*100:.0f}% Sub)", "Savings (₹ Crore)", 400)
    )
    st.plotly_chart(fig_sav, use_container_width=True)

    # 3-year projection (annual compounding with 5% price escalation)
    st.markdown(f"#### 3-Year Savings Projection (5% annual price escalation)")
    years = ["FY 2026", "FY 2027", "FY 2028"]
    escalation = [1.0, 1.05, 1.1025]
    proj_savings = [Annual_Savings_Cr * e for e in escalation]
    cumulative_cr = np.cumsum(proj_savings)

    fig_3yr = go.Figure()
    fig_3yr.add_trace(go.Bar(
        x=years, y=proj_savings,
        name="Annual Savings (₹ Cr)",
        marker_color=[C_DELTA if s > 0 else C_NEG for s in proj_savings],
        text=[f"₹{v:.2f} Cr" for v in proj_savings],
        textposition="outside",
    ))
    fig_3yr.add_trace(go.Scatter(
        x=years, y=cumulative_cr,
        mode="lines+markers+text", name="Cumulative (₹ Cr)",
        line=dict(color="#9C27B0", width=2.5, dash="dash"),
        marker=dict(size=9, color="#9C27B0"),
        text=[f"₹{v:.2f} Cr" for v in cumulative_cr],
        textposition="top center",
    ))
    fig_3yr.update_layout(
        **_layout("3-Year Enterprise Savings Projection (₹ Crore)", "₹ Crore", 380)
    )
    st.plotly_chart(fig_3yr, use_container_width=True)

with col_rgt:
    # Savings breakdown by benefit
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

    st.dataframe(
        df_bens.style.applymap(style_ben, subset=["₹ Crore / Year"]),
        use_container_width=True, height=350,
    )

    st.markdown("#### Savings Components Sunburst")
    pos_bens  = [(n, (v * EMM_Consumption_FY * Substitution_Pct) / 1e7) for n, v in zip(all_benefit_names, all_benefit_values) if v > 0]
    sun_labels = ["Total VIU Credits"] + [p[0] for p in pos_bens]
    sun_parents = [""] + ["Total VIU Credits"] * len(pos_bens)
    sun_values = [sum(p[1] for p in pos_bens)] + [p[1] for p in pos_bens]

    fig_sun = go.Figure(go.Sunburst(
        labels=sun_labels, parents=sun_parents, values=sun_values,
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>₹%{value:.3f} Cr<extra></extra>",
        marker=dict(colors=["#1A237E"] + colours_donut[:len(pos_bens)]),
    ))
    fig_sun.update_layout(
        title="Savings Sunburst (₹ Cr)",
        template="plotly_white", height=380,
        margin=dict(l=5, r=5, t=40, b=5),
    )
    st.plotly_chart(fig_sun, use_container_width=True)

# Break-even calculator
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("#### Break-Even Price Analysis")
be1, be2, be3 = st.columns(3)
# To find break-even, Cost_Per_Mn_Delta must equal Total_Op_Credits
# Cost_Per_Mn_Delta = (EMM_Price / eff_emm) - (LC_Price / eff_lc)
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

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="text-align:center; color:#90A4AE; font-size:12px; padding:8px 0;">
  VIU Dashboard – LC FeMn vs Mn Briquette &nbsp;|&nbsp; All formulas sourced from Excel workbook 
  (INPUT_PARAMETER → BREAKDOWN_CALC → VIU_SUMMARY) &nbsp;|&nbsp; 
  Operational benefits per MT alloy at stated realization factors.
</div>
""", unsafe_allow_html=True)
