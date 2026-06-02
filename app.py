"""
VIU DASHBOARD – Manganese Alloy Substitution
================================================
Value-in-Use comparison of LC/MC Ferromanganese against Mn Briquette / EMM.

All formulas sourced exclusively from the Excel file:
  • INPUT_PARAMETER sheet  → adjustable parameters
  • BREAKDOWN_CALC sheet   → all benefit calculations
  • VIU_SUMMARY sheet      → synthesis & enterprise savings
  • SOLVER sheet           → Linear programming constraints

Architecture & UX inspired by the Manganese Intelligence dashboard.
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
    page_title="VIU Dashboard & Solver",
    page_icon="⚗️",
    layout="wide",
)

# Colour palette
C_BASE     = "#2196F3"   # blue  – Base Alloy (LC/MC FeMn)
C_NEW      = "#4CAF50"   # green – Mn Briquette / EMM
C_DELTA    = "#FF9800"   # amber – delta / benefit
C_NEG      = "#F44336"   # red   – penalties / negative
C_GRID     = "#E0E0E0"

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.title("⚗️ Configuration")
mode = st.sidebar.radio(
    "A. Select Comparison Mode", 
    ["LC FeMn vs Mn Briquette", "MC FeMn vs Mn Briquette"]
)

st.sidebar.divider()

# Load parameters dynamically based on the selected mode
if mode == "LC FeMn vs Mn Briquette":
    base_name = "LC FeMn"
    new_name = "Mn Briquette"
    
    with st.sidebar.expander("B. Financial Parameters", expanded=True):
        p_base_price = st.number_input(f"{base_name} Price (₹/MT)", value=105000, step=1000)
        p_new_price = st.number_input(f"{new_name} Price (₹/MT)", value=160000, step=1000)
        p_power_tariff = st.number_input("Power Tariff (₹/kWh)", value=6.5, step=0.1)
        p_electrode_cost = st.number_input("Electrode Cost (₹/kg)", value=240, step=10)
        p_steel_value = st.number_input("Finished Steel Value (₹/MT)", value=60000, step=1000)
        p_margin = st.number_input("Variable Margin (₹/MT)", value=2800, step=100)
        p_lf_retreat = st.number_input("LF Retreatment Cost (₹/heat)", value=15000, step=500)
        p_scrap = st.number_input("Scrap / Iron Credit (₹/MT)", value=35000, step=500)

    with st.sidebar.expander("C. Thermodynamic & Technical", expanded=False):
        p_base_mn = st.slider(f"{base_name} Mn Fraction", 0.75, 0.85, 0.80)
        p_new_mn = st.slider(f"{new_name} Mn Fraction", 0.95, 0.999, 0.997)
        p_base_rec = st.slider(f"{base_name} Recovery", 0.80, 0.98, 0.95)
        p_new_rec = st.slider(f"{new_name} Recovery", 0.80, 0.99, 0.95)
        p_base_fe = st.slider(f"{base_name} Fe Fraction", 0.0, 0.20, 0.15)
        p_base_c = st.slider(f"{base_name} C Fraction", 0.0, 0.010, 0.005, format="%.3f")
        p_new_c = st.slider(f"{new_name} C Fraction", 0.0, 0.005, 0.0005, format="%.4f")

    with st.sidebar.expander("D. Operational Realization", expanded=False):
        r_power = st.slider("Power Saving Realization", 0.0, 1.0, 1.0)
        r_elec = st.slider("Electrode Realization", 0.0, 1.0, 1.0)
        r_tp = st.slider("Throughput Gain Realization", 0.0, 1.0, 0.25)
        r_yield = st.slider("Yield Gain Realization", 0.0, 1.0, 0.25)
        r_qual = st.slider("Quality Realization", 0.0, 1.0, 1.0)
        r_chem = st.slider("Chemistry Realization", 0.0, 1.0, 0.50)
        r_rh = st.slider("RH Corr Realization", 0.0, 1.0, 1.0)

    with st.sidebar.expander("E. Process Variables", expanded=False):
        p_heat = st.number_input("Heat Size (MT)", value=190)
        t_delta = st.number_input("Thermal Advantage (°C)", value=4.0)
        p_lf_eff = st.slider("LF Efficiency", 0.5, 1.0, 0.78)
        p_spec_heat = st.number_input("Specific Heat", value=0.75)
        cool_rate = st.number_input("Cooling Rate (°C/min)", value=3.5)
        cast_tat = st.number_input("Caster TAT (min)", value=53)
        
        # Internal implicit solver/VIU variables for LC baseline
        base_divisor = 1.05 
        yield_gain = 0.0003
        var_base, var_new = 0.04, 0.01
        overdose_base, overdose_new = 0.04, 0.01
        reject_base, reject_new = 0.0001, 0.0
        retreat_base, retreat_new = 0.03, 0.01
        carbon_freq = 0.05
        rh_cost = 2500
        
        inc_base, inc_new = 0.08, 0.02
        reblow_base, reblow_new = 0.03, 0.01

else:
    base_name = "MC FeMn"
    new_name = "Mn Briquette"
    
    with st.sidebar.expander("B. Financial Parameters", expanded=True):
        p_base_price = st.number_input(f"{base_name} Price (₹/MT)", value=130000, step=1000)
        p_new_price = st.number_input(f"{new_name} Price (₹/MT)", value=175000, step=1000)
        p_power_tariff = st.number_input("Power Tariff (₹/kWh)", value=6.5, step=0.1)
        p_electrode_cost = st.number_input("Electrode Cost (₹/kg)", value=240, step=10)
        p_steel_value = st.number_input("Finished Steel Value (₹/MT)", value=60000, step=1000)
        p_margin = st.number_input("Variable Margin (₹/MT)", value=2800, step=100)
        p_lf_retreat = st.number_input("LF Retreatment Cost (₹/heat)", value=15000, step=500)
        p_scrap = st.number_input("Scrap / Iron Credit (₹/MT)", value=35000, step=500)

    with st.sidebar.expander("C. Thermodynamic & Technical", expanded=False):
        p_base_mn = st.slider(f"{base_name} Mn Fraction", 0.60, 0.85, 0.70)
        p_new_mn = st.slider(f"{new_name} Mn Fraction", 0.90, 0.999, 0.99)
        p_base_rec = st.slider(f"{base_name} Recovery", 0.70, 0.95, 0.85)
        p_new_rec = st.slider(f"{new_name} Recovery", 0.80, 0.99, 0.95)
        p_base_fe = st.slider(f"{base_name} Fe Fraction", 0.0, 0.30, 0.20)
        p_base_c = st.slider(f"{base_name} C Fraction", 0.0, 0.050, 0.015, format="%.3f")
        p_new_c = st.slider(f"{new_name} C Fraction", 0.0, 0.010, 0.001, format="%.3f")

    with st.sidebar.expander("D. Operational Realization", expanded=False):
        r_power = st.slider("Power Saving Realization", 0.0, 1.0, 1.0)
        r_elec = st.slider("Electrode Realization", 0.0, 1.0, 1.0)
        r_tp = st.slider("Throughput Gain Realization", 0.0, 1.0, 0.25)
        r_yield = st.slider("Yield Gain Realization", 0.0, 1.0, 0.25)
        r_qual = st.slider("Quality Realization", 0.0, 1.0, 1.0)
        r_chem = st.slider("Chemistry Realization", 0.0, 1.0, 0.50)
        r_rh = st.slider("RH Corr Realization", 0.0, 1.0, 1.0)

    with st.sidebar.expander("E. Process Variables", expanded=False):
        p_heat = st.number_input("Heat Size (MT)", value=190)
        t_delta = st.number_input("Thermal Advantage (°C)", value=4.0)
        p_lf_eff = st.slider("LF Efficiency", 0.5, 1.0, 0.78)
        p_spec_heat = st.number_input("Specific Heat", value=0.75)
        cool_rate = st.number_input("Cooling Rate (°C/min)", value=3.5)
        cast_tat = st.number_input("Caster TAT (min)", value=53)
        
        # Internal variables matching the MC FeMn Excel breakdown logic perfectly
        base_divisor = 0.808 # MT of Active Mn added per heat baseline in Excel
        yield_gain = 0.0003
        var_base, var_new = 0.05, 0.015
        overdose_base, overdose_new = 0.05, 0.015
        reject_base, reject_new = 0.0002, 0.0
        retreat_base, retreat_new = 0.04, 0.02
        carbon_freq = 0.10
        rh_cost = 2500
        
        inc_base, inc_new = 0.09, 0.03
        reblow_base, reblow_new = 0.04, 0.02


# ══════════════════════════════════════════════════════════════════════════════
# CORE VIU CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════
# Note: Excel formulations normalize around the Active Mn added metric (~0.808 MT).
# 1. Energy Savings
energy_kwh = (p_heat * p_spec_heat * t_delta) / (3.6 * p_lf_eff)
b_power = (energy_kwh * p_power_tariff * r_power) / base_divisor

# 2. Electrode Savings
b_elec = (energy_kwh * 0.0028 * p_electrode_cost * r_elec) / base_divisor

# 3. Throughput Gains
time_saved = t_delta / cool_rate
frac_heat = time_saved / cast_tat
b_tp = (frac_heat * p_heat * p_margin * r_tp) / base_divisor

# 4. Recovery Stability
b_rec = ((overdose_base - overdose_new) * p_new_price * r_chem) / base_divisor

# 5. Retreatment Reductions
b_retreat = ((retreat_base - retreat_new) * p_lf_retreat * r_chem) / base_divisor

# 6. Cleanliness Improvements
b_clean = ((reject_base - reject_new) * p_steel_value * p_heat * r_qual) / base_divisor

# 7. Yield Enhancements
b_yield_val = (yield_gain * p_steel_value * p_heat * r_yield) / base_divisor

# 8. Carbon Avoidance (RH Correction)
b_carb = (carbon_freq * rh_cost * r_rh) / base_divisor

# Total Operational VIU
op_viu = b_power + b_elec + b_tp + b_rec + b_retreat + b_clean + b_yield_val + b_carb

# Unit Costs per MT Active Mn
base_cost_mn = p_base_price / (p_base_mn * p_base_rec)
new_cost_mn = p_new_price / (p_new_mn * p_new_rec)
raw_delta = base_cost_mn - new_cost_mn # Positive -> Briquette is cheaper per Active MT Mn

# Iron Credit Penalty (since Briquette lacks Fe)
b_iron = p_base_fe * p_scrap

# Final Enterprise Net Value
net_value = raw_delta + op_viu - b_iron

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DASHBOARD TABS
# ══════════════════════════════════════════════════════════════════════════════
st.title(f"🏭 Value-in-Use: {base_name} vs {new_name}")

tab1, tab2 = st.tabs(["📊 VIU Dashboard", "🧮 Substitution Solver"])

with tab1:
    st.markdown("### Top KPIs (₹ per MT Active Mn)")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    kpi1.metric(
        label="Raw Cost Difference",
        value=f"₹ {raw_delta:,.0f}",
        delta="Cheaper Base Alloy" if raw_delta < 0 else f"Cheaper {new_name}",
        delta_color="inverse" if raw_delta < 0 else "normal"
    )
    
    kpi2.metric(
        label="Operational VIU (Benefits)",
        value=f"₹ {op_viu:,.0f}",
        delta="Process Savings",
        delta_color="normal"
    )
    
    kpi3.metric(
        label="Iron Credit Penalty",
        value=f"₹ {-b_iron:,.0f}",
        delta="Lost Fe Value",
        delta_color="inverse"
    )
    
    kpi4.metric(
        label="Net Enterprise Value",
        value=f"₹ {net_value:,.0f}",
        delta=f"Advantage {new_name}" if net_value > 0 else f"Advantage {base_name}",
        delta_color="normal" if net_value > 0 else "inverse"
    )

    st.divider()

    col_chart, col_table = st.columns([1.2, 1])

    with col_chart:
        st.markdown(f"#### VIU Economic Synthesis: {new_name} Advantage")
        
        # Waterfall Chart
        fig = go.Figure(go.Waterfall(
            name="20", orientation="v",
            measure=["relative", "relative", "relative", "relative", "relative", 
                     "relative", "relative", "relative", "relative", "relative", "total"],
            x=["Raw Cost Δ", "Power", "Electrode", "Throughput", "Recovery", 
               "Retreatment", "Cleanliness", "Yield", "C-Avoidance", "Iron Penalty", "Net Value"],
            textposition="outside",
            text=[f"{v/1000:.1f}k" for v in [raw_delta, b_power, b_elec, b_tp, b_rec, b_retreat, 
                                             b_clean, b_yield_val, b_carb, -b_iron, net_value]],
            y=[raw_delta, b_power, b_elec, b_tp, b_rec, b_retreat, b_clean, b_yield_val, b_carb, -b_iron, net_value],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": C_NEG}},
            increasing={"marker": {"color": C_NEW}},
            totals={"marker": {"color": C_BASE}}
        ))
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=40),
            waterfallgap=0.3,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(title="Value (₹ / MT Active Mn)", gridcolor=C_GRID),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("#### Final Recommendation Breakdown")
        breakdown_data = {
            "Benefit Component": [
                "Raw Purchase Difference", "Power Saving", "Electrode Saving", 
                "Throughput Gain", "Recovery Stability", "Retreatment Reduction", 
                "Cleanliness Inclusion", "Yield Gain", "Carbon Avoidance", "Iron Credit Penalty"
            ],
            "Net Benefit (₹/MT)": [
                raw_delta, b_power, b_elec, b_tp, b_rec, b_retreat, 
                b_clean, b_yield_val, b_carb, -b_iron
            ]
        }
        df_bd = pd.DataFrame(breakdown_data)
        
        # Format the table
        st.dataframe(
            df_bd.style.format({"Net Benefit (₹/MT)": "₹ {:,.0f}"})
            .applymap(lambda x: f"color: {C_NEG}; font-weight:bold" if x < 0 else f"color: {C_NEW}", subset=["Net Benefit (₹/MT)"]),
            hide_index=True,
            use_container_width=True,
            height=400
        )

with tab2:
    st.markdown(f"### Optimal Alloy Substitution Solver")
    
    # Define constraints per mode mapping
    if mode == "LC FeMn vs Mn Briquette":
        grade_data = pd.DataFrame({
            "Grade": ["Structural (IS2062)", "Micro-alloyed (API)", "AHSS (DP780)", "Silicon Steel (CRGO)", "IF/ULC (EDD)"],
            "Mn Target (%)": [0.80, 1.20, 1.60, 0.40, 0.20],
            "C Limit (%)": [0.015, 0.010, 0.005, 0.003, 0.002],
            "Rec Var Limit (%)": [5.0, 3.5, 2.5, 2.0, 1.5],
            "Inc Limit": [0.10, 0.06, 0.04, 0.02, 0.01],
            "Reblow Limit (%)": [4.0, 2.5, 1.5, 1.0, 0.5],
            "Max Briq Share": [0.5, 1.0, 1.0, 1.0, 1.0]
        })
    else:
        grade_data = pd.DataFrame({
            "Grade": ["Commodity (IS2062)", "TMT/Rebar (Fe500D)", "HSLA/API (API X70)", "AHSS/DP (DP780)", "Electrical (CRGO)", "IF/ULC (EDD)"],
            "Mn Target (%)": [0.65, 0.70, 1.20, 1.80, 0.30, 0.15],
            "C Limit (%)": [0.010, 0.008, 0.005, 0.004, 0.002, 0.001],
            "Rec Var Limit (%)": [6.0, 5.0, 3.0, 2.5, 2.0, 1.5],
            "Inc Limit": [0.12, 0.10, 0.05, 0.04, 0.02, 0.01],
            "Reblow Limit (%)": [5.0, 4.0, 2.0, 1.0, 1.0, 0.5],
            "Max Briq Share": [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        })

    col_sel, col_empty = st.columns([1, 2])
    with col_sel:
        selected_grade = st.selectbox("Select Target Steel Grade", grade_data["Grade"].tolist())
    
    g_row = grade_data[grade_data["Grade"] == selected_grade].iloc[0]
    
    # Fetch limits
    t_mn = g_row["Mn Target (%)"]
    t_c = g_row["C Limit (%)"]
    t_var = g_row["Rec Var Limit (%)"] / 100
    t_inc = g_row["Inc Limit"]
    t_reb = g_row["Reblow Limit (%)"] / 100
    t_max_briq = g_row["Max Briq Share"]
    
    # Linear Programming Setup (variables: x = kg Base per T steel, y = kg New per T steel)
    # Objective: Minimize cost in ₹/MT steel
    c_obj = [p_base_price/1000, p_new_price/1000]
    
    # 1. Mn Target Equality (Mn_Fraction * Recovery * Alloy_Kg = Mn_Target_Kg)
    A_eq = [[p_base_mn * p_base_rec, p_new_mn * p_new_rec]]
    b_eq = [t_mn * 10]
    
    # 2. Inequality Constraints A_ub * [x, y]^T <= b_ub
    A_ub = []
    b_ub = []
    
    # Carbon limit
    A_ub.append([p_base_c, p_new_c])
    b_ub.append(t_c * 10)
    
    # Index properties weighted by absolute mass (x+y)
    # Rec Var Limit: (var_base)*x + (var_new)*y <= t_var * (x + y)  => (var_base - t_var)*x + (var_new - t_var)*y <= 0
    A_ub.append([var_base - t_var, var_new - t_var])
    b_ub.append(0)
    
    # Cleanliness Limit
    A_ub.append([inc_base - t_inc, inc_new - t_inc])
    b_ub.append(0)
    
    # Reblow Limit
    A_ub.append([reblow_base - t_reb, reblow_new - t_reb])
    b_ub.append(0)
    
    # Max Briq Share Limit: y <= Max * (x + y) => -Max * x + (1 - Max) * y <= 0
    A_ub.append([-t_max_briq, 1 - t_max_briq])
    b_ub.append(0)
    
    # Solve LP
    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None), (0, None)], method='highs')
    
    st.divider()

    if res.success:
        x_opt, y_opt = res.x
        total_alloy = x_opt + y_opt
        cost_opt = res.fun
        
        c1, c2 = st.columns([1, 1.2])
        
        with c1:
            st.markdown("#### Optimal Recipe per MT Steel")
            
            # Pie Chart
            fig_pie = px.pie(
                values=[x_opt, y_opt], 
                names=[base_name, new_name],
                color=[base_name, new_name],
                color_discrete_map={base_name: C_BASE, new_name: C_NEW},
                hole=0.4
            )
            fig_pie.update_traces(textinfo='percent+label')
            fig_pie.update_layout(
                margin=dict(t=20, b=20, l=20, r=20),
                annotations=[dict(text=f'<b>₹ {cost_opt:,.0f}</b>', x=0.5, y=0.5, font_size=20, showarrow=False)]
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            st.info(f"**Blend Cost**: ₹ {cost_opt:,.2f} per MT Steel\n\n**Addition**: {x_opt:.2f} kg {base_name} + {y_opt:.2f} kg {new_name}")

        with c2:
            st.markdown("#### Metallurgical Profile of Optimal Blend")
            
            # Evaluate constraints metrics
            actual_c = (p_base_c * x_opt + p_new_c * y_opt) / 10
            actual_var = (var_base * x_opt + var_new * y_opt) / total_alloy if total_alloy > 0 else 0
            actual_inc = (inc_base * x_opt + inc_new * y_opt) / total_alloy if total_alloy > 0 else 0
            actual_reb = (reblow_base * x_opt + reblow_new * y_opt) / total_alloy if total_alloy > 0 else 0
            actual_share = y_opt / total_alloy if total_alloy > 0 else 0
            
            profile_data = [
                {"Parameter": "Carbon Contribution (%)", "Target Limit": t_c, "Actual": actual_c},
                {"Parameter": "Recovery Variance (%)", "Target Limit": t_var*100, "Actual": actual_var*100},
                {"Parameter": "Cleanliness Index", "Target Limit": t_inc, "Actual": actual_inc},
                {"Parameter": "Reblow Risk Index (%)", "Target Limit": t_reb*100, "Actual": actual_reb*100},
                {"Parameter": f"Max {new_name} Share (%)", "Target Limit": t_max_briq*100, "Actual": actual_share*100}
            ]
            
            df_prof = pd.DataFrame(profile_data)
            df_prof["Status"] = df_prof.apply(
                lambda row: "🟢 Passed" if row["Actual"] <= row["Target Limit"] + 1e-6 else "🔴 Binding / Failed", axis=1
            )
            
            # Format and Style Dataframe
            st.dataframe(
                df_prof.style.format({
                    "Target Limit": "{:.3f}",
                    "Actual": "{:.3f}"
                }).applymap(
                    lambda x: "color: #388E3C; font-weight:bold" if "Passed" in x else "color: #D32F2F; font-weight:bold", 
                    subset=["Status"]
                ),
                use_container_width=True,
                hide_index=True
            )
            
            st.caption("Solver strictly guarantees all limits are met while minimizing costs. Binding constraints represent process bottlenecks driving up alloy mix costs.")
    else:
        st.error(f"⚠️ **Infeasible Constraints:** The metallurgical limits for **{selected_grade}** are too strict to be met with the current parameters of {base_name} and {new_name}. Relax the Carbon, Cleanliness, or Share constraints.")

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("""
<div style="text-align:center; color:#90A4AE; font-size:12px; padding:8px 0;">
  VIU Dashboard & Solver – Dynamic Substitution Modelling &nbsp;|&nbsp; All formulas sourced strictly from the uploaded inputs.
</div>
""", unsafe_allow_html=True)