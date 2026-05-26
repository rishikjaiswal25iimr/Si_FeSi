"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      SILICON PROCUREMENT INTELLIGENCE & PROCESS ECONOMICS PLATFORM          ║
║      app_silicon.py  |  v3.3  (2026 Industrial Framework Sync)             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Tabs  : Price Forecast | Market Comparison | Regime & Drivers              ║
║          VIU & TCO Optimizer | Substitution Solver                          ║
║  Products : Silicon Metal (Si) | FeSi Alloy (FeSi 70%)                     ║
║                                                                              ║
║  AUDIT FIXES v3.3:                                                           ║
║  • CACHE FIX: Reduced data caching TTL so pipeline updates show instantly   ║
║  • DRIVER SYNC: Added full mapping for all new/old econometric drivers      ║
║  • PALETTE EXPANSION: Added more colors to support up to 24 chart traces    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import json
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.optimize import linprog

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Silicon Intelligence Platform",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design palette ─────────────────────────────────────────────────────────────
C_SI_IDX    = "#1565C0"                    # Si Index — deep blue
C_FESI_IDX  = "#6A1B9A"                    # FeSi Index — deep purple
C_MARKET    = "#E65100"                    # Market actual — burnt orange
C_FUTURE    = "#2E7D32"                    # Future forecast — forest green
C_CI        = "rgba(46, 125, 50, 0.10)"
C_REG_HIGH  = "rgba(211, 47, 47, 0.12)"
C_REG_MED   = "rgba(255, 152, 0, 0.10)"
C_GRID      = "#EEEEEE"
C_TEXT      = "#212121"
C_ACCENT    = "#0288D1"
C_VIU_GREEN = "#43A047"
C_VIU_GRAY  = "#9E9E9E"
C_ERR_POS   = "#1565C0"
C_ERR_NEG   = "#E65100"

HORIZON_OPTIONS = {
    "4 weeks (1 month)":   4,
    "12 weeks (3 months)": 12,
    "26 weeks (6 months)": 26,
    "52 weeks (1 year)":   52,
    "104 weeks (2 years)": 104,
    "156 weeks (3 years)": 156,
}


# ═══════════════════════════════════════════════════════════════════════════════
# LAYOUT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _layout(title: str, y_title: str = "Price", height: int = 460) -> dict:
    return dict(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="#FAFAFA",
        font=dict(family="Inter, sans-serif", size=12, color=C_TEXT),
        title=dict(text=title, font=dict(size=15, color="#111"), x=0.01),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#DDD",
                    borderwidth=1, font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=C_GRID, zeroline=False, title=y_title),
        hovermode="x unified",
        height=height,
        margin=dict(l=60, r=20, t=55, b=45),
    )


def _regime_shapes(dates: pd.DatetimeIndex, probs: np.ndarray,
                   high: float = 0.65, med: float = 0.45) -> list:
    shapes = []
    in_high, in_med = False, False
    t0_h, t0_m = None, None

    for d, p in zip(dates, probs):
        if p >= high:
            if in_med:
                shapes.append(dict(type="rect", xref="x", yref="paper",
                                   x0=str(t0_m), x1=str(d), y0=0, y1=1,
                                   fillcolor=C_REG_MED, line_width=0, layer="below"))
                in_med = False
            if not in_high:
                in_high, t0_h = True, d
        elif p >= med:
            if in_high:
                shapes.append(dict(type="rect", xref="x", yref="paper",
                                   x0=str(t0_h), x1=str(d), y0=0, y1=1,
                                   fillcolor=C_REG_HIGH, line_width=0, layer="below"))
                in_high = False
            if not in_med:
                in_med, t0_m = True, d
        else:
            if in_high:
                shapes.append(dict(type="rect", xref="x", yref="paper",
                                   x0=str(t0_h), x1=str(d), y0=0, y1=1,
                                   fillcolor=C_REG_HIGH, line_width=0, layer="below"))
                in_high = False
            if in_med:
                shapes.append(dict(type="rect", xref="x", yref="paper",
                                   x0=str(t0_m), x1=str(d), y0=0, y1=1,
                                   fillcolor=C_REG_MED, line_width=0, layer="below"))
                in_med = False

    if in_high:
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=str(t0_h), x1=str(dates[-1]), y0=0, y1=1,
                           fillcolor=C_REG_HIGH, line_width=0, layer="below"))
    if in_med:
        shapes.append(dict(type="rect", xref="x", yref="paper",
                           x0=str(t0_m), x1=str(dates[-1]), y0=0, y1=1,
                           fillcolor=C_REG_MED, line_width=0, layer="below"))
    return shapes


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=5)  # Cache heavily reduced to allow immediate market price detection
def load_alloy_data(alloy_code: str, horizon_weeks: int):
    output_dir = f"./outputs_{alloy_code}"
    hist   = pd.read_csv(f"{output_dir}/historical_predictions.csv",
                         index_col=0, parse_dates=True)
    future = pd.read_csv(f"{output_dir}/future_forecast.csv",
                         index_col=0, parse_dates=True).iloc[:horizon_weeks]
    fi     = (pd.read_csv(f"{output_dir}/feature_importance.csv")
                .sort_values("importance", ascending=False).head(25))
    with open(f"{output_dir}/model_metadata.json") as f:
        meta = json.load(f)
    return hist, future, fi, meta


def get_real_price_series(alloy_code: str) -> pd.Series | None:
    try:
        df = pd.read_csv(f"./outputs_{alloy_code}/historical_predictions.csv",
                         index_col=0, parse_dates=True)
        return df["real_price"]
    except Exception:
        return None


def get_future_price_series(alloy_code: str) -> pd.Series | None:
    try:
        df = pd.read_csv(f"./outputs_{alloy_code}/future_forecast.csv",
                         index_col=0, parse_dates=True)
        return df["real_price"]
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTIVE INSIGHT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def generate_executive_insights(alloy_code, hist, future, meta, fi) -> dict:
    last_price  = float(hist["real_price"].iloc[-1])
    regime_now  = float(hist["regime_probability"].iloc[-1])
    nxt_price   = float(future["real_price"].iloc[0])
    end_price   = float(future["real_price"].iloc[-1])
    price_trend = (end_price - last_price) / (last_price + 1e-9) * 100

    known_drivers = {
        "al_price", "solar_etf", "semiconductor_etf", "china_etf", 
        "bdry_freight", "vix", "thermal_coal_futures", "crude_oil", 
        "usd_inr", "silica_quartz_index", "cny_inr", 
        "hydrology_rainfall_index", "petcoke_charcoal_index", 
        "electrode_consumables_index", "gfex_silicon_futures", 
        "electricity_power_index", "steel_etf", "iron_ore", 
        "india_steel", "zce_fesi_futures", "carbon_emissions_futures", 
        "fx_effect", "shaanxi_semicoke", "magnesium_demand"
    }

    top_driver = fi["feature"].iloc[0]
    for feat in fi["feature"]:
        base = feat.split("_lag")[0].split("_rm")[0].split("_rz")[0].split("_ret")[0]
        if base in known_drivers:
            top_driver = base
            break

    if regime_now > 0.65:
        regime_alert = "🔴 HIGH STRESS: Market in high-volatility regime. Procurement risk elevated."
        regime_color = "error"
    elif regime_now > 0.45:
        regime_alert = "🟡 ELEVATED RISK: Transition regime. Monitor closely."
        regime_color = "warning"
    else:
        regime_alert = "🟢 STABLE REGIME: Low volatility. Favourable procurement window."
        regime_color = "success"

    wk4_idx = min(3, len(future) - 1)
    wk4_chg = (float(future["real_price"].iloc[wk4_idx]) - last_price) / last_price * 100
    if wk4_chg > 3:
        momentum = f"📈 UPWARD: +{wk4_chg:.1f}% forecast over next 4 weeks. Consider forward procurement."
    elif wk4_chg < -3:
        momentum = f"📉 DOWNWARD: {wk4_chg:.1f}% forecast over next 4 weeks. Defer non-urgent orders."
    else:
        momentum = f"➡️ SIDEWAYS: {wk4_chg:+.1f}% over next 4 weeks. Neutral procurement stance."

    if price_trend > 10:
        outlook = f"⚠️ RISING TREND: +{price_trend:.1f}% over forecast horizon. Lock in forward contracts."
    elif price_trend < -10:
        outlook = f"💡 FALLING TREND: {price_trend:.1f}% over forecast horizon. Spot buying preferred."
    else:
        outlook = f"📊 RANGE-BOUND: {price_trend:+.1f}% over forecast horizon. Blend spot and term procurement."

    driver_label = top_driver.replace("_", " ").title()
    driver_comment = f"🔧 DOMINANT DRIVER: '{driver_label}' has highest predictive weight. Monitor weekly."

    return dict(
        regime_alert=regime_alert, regime_color=regime_color,
        momentum=momentum, outlook=outlook, driver_comment=driver_comment,
        last_price=last_price, regime_prob=regime_now, price_trend=price_trend,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# VIU PROCESS ECONOMICS — Silicon System
# ═══════════════════════════════════════════════════════════════════════════════

HEAT_SIZE         = 190.0     # MT per heat (typical RH/LF)
STEEL_ALLOY_RATIO = 200.0     # 1 MT alloy supports ~200 T steel (≈5 kg/T)
POWER_RATE        = 6.5       # ₹/kVAh industrial tariff
ELECTRODE_RATE    = 240.0     # ₹/kg
THROUGHPUT_MARGIN = 2800.0    # ₹/MT variable margin
REBLOW_COST       = 15000.0   # ₹/heat
STEEL_VALUE       = 60000.0   # ₹/MT reference steel value


def compute_si_operational_benefits(
    power_kwh: float,       # kWh power saving vs FeSi
    recovery_pct: float,    # % recovery improvement vs FeSi
    slag_red_kg: float,     # Kg slag reduction per T steel
    inclusion_rej: float,   # % rejection reduction
    reblow_pct: float,      # % reblow risk reduction
    yield_gain_pct: float,  # % yield gain
    throughput_min: float,  # minutes saved per heat
    si_price_mt: float,     # Si Metal price ₹/MT
) -> dict:
    power_benefit      = power_kwh * POWER_RATE
    electrode_benefit  = power_kwh * 0.00286 * ELECTRODE_RATE
    recovery_benefit   = (recovery_pct / 100) * si_price_mt * 0.50
    scale              = STEEL_ALLOY_RATIO / HEAT_SIZE
    slag_benefit       = slag_red_kg * 600 * HEAT_SIZE * 0.05 * scale
    inclusion_benefit  = (inclusion_rej / 100) * STEEL_VALUE * STEEL_ALLOY_RATIO * 0.40
    reblow_benefit     = (reblow_pct / 100) * REBLOW_COST * scale
    yield_benefit      = (yield_gain_pct / 100) * STEEL_VALUE * STEEL_ALLOY_RATIO * 0.50
    throughput_benefit = (throughput_min / 53.0) * HEAT_SIZE * 0.25 * THROUGHPUT_MARGIN * scale

    return {
        "Power Saving":          round(power_benefit, 2),
        "Electrode Saving":      round(electrode_benefit, 2),
        "Recovery Improvement":  round(recovery_benefit, 2),
        "Slag Reduction":        round(slag_benefit, 2),
        "Inclusion Cleanliness": round(inclusion_benefit, 2),
        "Reblow Reduction":      round(reblow_benefit, 2),
        "Yield Improvement":     round(yield_benefit, 2),
        "Throughput/RH Saving":  round(throughput_benefit, 2),
    }


def calc_tco(price_mt: float, si_pct: float, fe_pct: float,
             recovery: float, fe_credit_rate: float,
             viu_benefits_total: float) -> dict:
    effective_si = (si_pct / 100) * (recovery / 100)
    fe_credit    = (fe_pct / 100) * fe_credit_rate
    adj_price    = price_mt - fe_credit
    base_viu     = adj_price / (effective_si + 1e-9)
    viu_per_eff  = viu_benefits_total / (effective_si + 1e-9)
    tco          = base_viu - viu_per_eff

    return dict(
        effective_si=effective_si,
        fe_credit=fe_credit,
        adj_price=adj_price,
        base_viu=base_viu,
        viu_benefits_per_eff=viu_per_eff,
        tco=tco,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STEEL GRADE DEFINITIONS & DRIVER MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

STEEL_GRADES = {
    "Commodity Structural (IS2062 / E250)": {
        "eff_si_min": 0.72, "chill_limit": 11.0, "rec_var": 6.0,
        "reblow_max": 5.0,  "inclusion_max": 0.050, "power_min": 40.0,
        "si_share_max": 0.40,
    },
    "TMT / Rebar (Fe500D)": {
        "eff_si_min": 0.75, "chill_limit": 10.5, "rec_var": 5.0,
        "reblow_max": 4.5,  "inclusion_max": 0.048, "power_min": 50.0,
        "si_share_max": 0.50,
    },
    "HSLA (API X60 / X70)": {
        "eff_si_min": 0.80, "chill_limit": 10.0, "rec_var": 3.0,
        "reblow_max": 4.0,  "inclusion_max": 0.042, "power_min": 60.0,
        "si_share_max": 0.70,
    },
    "Automotive (DP600 / DP780)": {
        "eff_si_min": 0.88, "chill_limit": 9.5,  "rec_var": 2.0,
        "reblow_max": 3.0,  "inclusion_max": 0.038, "power_min": 65.0,
        "si_share_max": 0.90,
    },
    "IF Steel (Deep Draw)": {
        "eff_si_min": 0.92, "chill_limit": 9.0,  "rec_var": 1.5,
        "reblow_max": 2.5,  "inclusion_max": 0.035, "power_min": 70.0,
        "si_share_max": 1.00,
    },
    "Electrical Steel (CRGO / CRNO)": {
        "eff_si_min": 0.95, "chill_limit": 8.5,  "rec_var": 1.0,
        "reblow_max": 2.0,  "inclusion_max": 0.033, "power_min": 80.0,
        "si_share_max": 1.00,
    },
}

DRIVER_LABELS = {
    "al_price": "Aluminium Price (LME)",
    "solar_etf": "Solar PV Demand (TAN)",
    "semiconductor_etf": "Semiconductor ETF (SOXX)",
    "china_etf": "China Industrial ETF (FXI)",
    "bdry_freight": "Baltic Dry Freight",
    "vix": "VIX (Risk Regime)",
    "thermal_coal_futures": "Thermal Coal Futures",
    "crude_oil": "WTI Crude Oil",
    "usd_inr": "USD/INR FX",
    "silica_quartz_index": "High-Purity Quartz Spot",
    "cny_inr": "CNY/INR FX",
    "hydrology_rainfall_index": "Hydrology / Reservoir Level",
    "petcoke_charcoal_index": "Reductant Index (Petcoke/Charcoal)",
    "electrode_consumables_index": "Graphite Electrode Prices",
    "gfex_silicon_futures": "GFEX Silicon Futures",
    "electricity_power_index": "Synthetic Electricity Power Index",
    "steel_etf": "Steel ETF (SLX)",
    "iron_ore": "Iron Ore Futures",
    "india_steel": "India Steel (JSW)",
    "zce_fesi_futures": "ZCE FeSi Futures",
    "carbon_emissions_futures": "EU ETS Carbon Spot (CBAM)",
    "fx_effect": "USD/CNY FX Effect",
    "shaanxi_semicoke": "Shaanxi Semi-Coke Spot",
    "magnesium_demand": "Magnesium Demand (Pidgeon)"
}

DRIVER_PALETTE = [
    "#1565C0", "#6A1B9A", "#2E7D32", "#E65100", "#C62828",
    "#00838F", "#F9A825", "#4A148C", "#1B5E20", "#BF360C",
    "#0277BD", "#D84315", "#558B2F", "#6D4C41", "#546E7A",
    "#F06292", "#BA68C8", "#4DD0E1", "#9575CD", "#7986CB",
    "#81C784", "#FF8A65", "#A1887F", "#90A4AE", "#E57373"
]


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## ⚗️ Silicon Intelligence")
    st.markdown("*Procurement & Process Economics Platform*")
    st.divider()

    st.markdown("### 🎯 Product Selection")
    product_choice = st.radio(
        "Select Silicon Product:",
        ["Silicon Metal (98% Si)", "FeSi Alloy (70% Si)"],
        index=0,
    )
    alloy_code = "Si" if "Metal" in product_choice else "FeSi"

    st.divider()
    st.markdown("### 📐 Display Options")
    display_window = st.slider("History to show (weeks)", 52, 520, 260, step=26)
    horizon_label  = st.selectbox("Forecast horizon", list(HORIZON_OPTIONS.keys()), index=2)
    price_mode     = st.radio("Price display", ["₹/Kg (Real Price)", "Index Value"], index=0)
    show_regime    = st.checkbox("Show regime shading", value=True)
    show_market    = st.checkbox("Show Indian market price", value=True)
    show_ci        = st.checkbox("Show 95% confidence band", value=True)

    st.divider()
    st.markdown("### ℹ️ Platform Info")
    st.markdown("""
    **Anchors (08-Jan-2024)**
    - Si Metal: ₹187.50 / Kg
    - FeSi: ₹107.75 / Kg

    **Sources**
    - yfinance (online drivers)
    - Indian market prices (provided)
    - VIU from process Excel model

    **Engine**
    - LightGBM / GradientBoosting
    - Driver-based proxy construction
    - Regime-aware future forecasting
    """)

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOAD
# ═══════════════════════════════════════════════════════════════════════════════

horizon_weeks = HORIZON_OPTIONS[horizon_label]

try:
    hist, future, fi, meta = load_alloy_data(alloy_code, horizon_weeks)
except FileNotFoundError:
    st.error(
        f"⛔ Pipeline outputs for **{alloy_code}** not found.  \n"
        f"Run the pipeline first:  \n"
        f"```bash\npython pipeline_silicon.py --alloy {alloy_code}\n```"
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER KPIs
# ═══════════════════════════════════════════════════════════════════════════════

product_label = "Silicon Metal" if alloy_code == "Si" else "FeSi Alloy"
color = C_SI_IDX if alloy_code == "Si" else C_FESI_IDX

st.markdown(
    f"<h1 style='color:{color}; margin-bottom:4px;'>⚗️ {product_label} Intelligence Platform</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#666; margin-top:0; font-size:14px;'>"
    "Procurement Intelligence · Process Economics · Substitution Solver</p>",
    unsafe_allow_html=True,
)

insights = generate_executive_insights(alloy_code, hist, future, meta, fi)

if insights["regime_color"] == "error":
    st.error(insights["regime_alert"])
elif insights["regime_color"] == "warning":
    st.warning(insights["regime_alert"])
else:
    st.success(insights["regime_alert"])

# KPI row
last_real = float(hist["real_price"].iloc[-1])
last_idx  = float(hist["actual"].iloc[-1])
nxt_real  = float(future["real_price"].iloc[0])
end_real  = float(future["real_price"].iloc[-1])
pct_1wk   = (nxt_real - last_real) / (last_real + 1e-9) * 100
pct_end   = (end_real - last_real) / (last_real + 1e-9) * 100
mape_hist = float(np.mean(np.abs(
    (hist["actual"] - hist["hybrid_prediction"]) / (hist["actual"] + 1e-9)
)) * 100)

overlap_rows = hist.dropna(subset=["market_price"]) if "market_price" in hist.columns else pd.DataFrame()
if len(overlap_rows) > 5:
    mkt_rmse = float(np.sqrt(np.mean(
        (overlap_rows["real_price"] - overlap_rows["market_price"]) ** 2
    )))
    mkt_mape = float(np.mean(np.abs(
        (overlap_rows["real_price"] - overlap_rows["market_price"]) /
        (overlap_rows["market_price"] + 1e-9)
    )) * 100)
else:
    mkt_rmse, mkt_mape = None, None

k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
k1.metric("📍 Last Index",         f"{last_idx:.2f}")
k2.metric("💰 Current Price",      f"₹{last_real:.2f}/Kg")
k3.metric("📅 Next-Week Forecast", f"₹{nxt_real:.2f}/Kg",
          delta=f"{pct_1wk:+.1f}%",
          delta_color="inverse" if alloy_code == "Si" else "normal")
k4.metric("🎯 Horizon-End Price",  f"₹{end_real:.2f}/Kg",
          delta=f"{pct_end:+.1f}% ({len(future)} wks)",
          delta_color="off")
k5.metric("📊 In-Sample MAPE",     f"{mape_hist:.1f}%")
k6.metric("🏪 Mkt MAPE",
          f"{mkt_mape:.1f}%" if mkt_mape is not None else "N/A",
          help="MAPE vs Indian market prices where available")
k7.metric("⚖️ Scale Factor",       f"{meta.get('scaling_factor', 0):.5f}",
          help=f"Anchored: {meta.get('anchor_date')} = ₹{meta.get('anchor_price_kg', 0):.2f}/Kg")

st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB LAYOUT
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Price Forecast",
    "📊 Market Comparison",
    "🔀 Regime & Drivers",
    "⚖️ VIU & TCO Optimizer",
    "🧬 Substitution Solver",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PRICE Forecast
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    h = hist[hist.index >= hist.index[-1] - pd.DateOffset(weeks=display_window)].copy()

    use_real  = price_mode.startswith("₹")
    hist_vals = h["real_price"]   if use_real else h["actual"]
    fut_col   = "real_price"      if use_real else "predicted_index"
    y_label   = "Price (₹/Kg)"   if use_real else "Price Index"
    hover_fmt = "₹%{y:.2f}/Kg"   if use_real else "Index: %{y:.3f}"

    N = len(h)
    M = len(future)
    full_x = list(h.index) + list(future.index)

    fig1 = go.Figure()

    if show_regime and "regime_probability" in h.columns:
        for s in _regime_shapes(h.index, h["regime_probability"].values):
            fig1.add_shape(**s)

    fig1.add_trace(go.Scatter(
        x=full_x,
        y=list(hist_vals) + [None] * M,
        name=f"{product_label} Index",
        line=dict(color=color, width=2.8),
        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>{product_label}: {hover_fmt}<extra></extra>",
    ))

    if show_market and "market_price" in h.columns:
        mkt_vals = h["market_price"]
        if mkt_vals.notna().any():
            fig1.add_trace(go.Scatter(
                x=full_x,
                y=list(mkt_vals) + [None] * M,
                name="Indian Market Price",
                line=dict(color=C_MARKET, width=2.2, dash="dot"),
                hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Market: {hover_fmt}<extra></extra>",
            ))

    conn_val = float(hist_vals.iloc[-1])
    fp       = future[fut_col]

    if show_ci and len(fp) > 1:
        idx_arr   = np.arange(1, M + 1)
        base_sigma = float(np.std(fp.values)) if np.std(fp.values) > 0 else conn_val * 0.03
        sigma     = base_sigma * 0.015 * idx_arr
        ci_up     = list(np.array(fp.values) + 1.96 * sigma)
        ci_dn     = list(np.array(fp.values) - 1.96 * sigma)
        ci_x      = [h.index[-1]] + list(future.index)
        
        fig1.add_trace(go.Scatter(
            x=ci_x + ci_x[::-1],
            y=[conn_val] + ci_up + ([conn_val] + ci_dn)[::-1],
            fill="toself",
            fillcolor=C_CI,
            line=dict(width=0),
            name="95% CI Band",
            showlegend=True,
            hoverinfo="skip",
        ))

    fut_y = [None] * (N - 1) + [conn_val] + list(fp.values)
    fig1.add_trace(go.Scatter(
        x=full_x,
        y=fut_y,
        name="Future Forecast",
        line=dict(color=C_FUTURE, width=2.8, dash="dash"),
        hovertemplate=f"<b>%{{x|%d %b %Y}}</b><br>Forecast: {hover_fmt}<extra></extra>",
    ))

    # AUDIT FIX: Replaced parameter-based annotation to avoid Timestamp crash
    x_forecast_start = h.index[-1]
    fig1.add_vline(x=x_forecast_start, line_width=1.5, line_dash="dot", line_color="#999")
    fig1.add_annotation(
        x=x_forecast_start, y=1, yref="paper",
        text="Forecast →", showarrow=False,
        xanchor="left", yanchor="top", font=dict(size=11), xshift=5
    )

    fig1.update_layout(**_layout(
        f"{product_label} — Price Trajectory & {horizon_label} Forecast",
        y_label, 490,
    ))
    st.plotly_chart(fig1, use_container_width=True)

    col_ins1, col_ins2 = st.columns(2)
    col_ins1.info(f"**4-Week Outlook:** {insights['momentum']}")
    col_ins2.info(f"**Horizon Outlook:** {insights['outlook']}")
    st.caption(insights["driver_comment"])

    with st.expander("📊 Show Dual-Axis: Index vs Real Price", expanded=False):
        fig_dual = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dual.add_trace(
            go.Scatter(x=h.index, y=h["actual"],
                       name="Index Value", line=dict(color=color, width=2.5)),
            secondary_y=False,
        )
        fig_dual.add_trace(
            go.Scatter(x=h.index, y=h["real_price"],
                       name="Real Price (₹/Kg)", line=dict(color=C_MARKET, width=2, dash="dash")),
            secondary_y=True,
        )
        fig_dual.update_yaxes(title_text="Price Index",  secondary_y=False,
                               showgrid=True, gridcolor=C_GRID)
        fig_dual.update_yaxes(title_text="₹/Kg",         secondary_y=True,
                               showgrid=False)
        fig_dual.update_xaxes(showgrid=True, gridcolor=C_GRID)
        fig_dual.update_layout(
            title=f"{product_label}: Index vs Real Price (Dual Axis)",
            template="plotly_white", height=380,
            legend=dict(orientation="h", y=1.06),
            hovermode="x unified",
        )
        st.plotly_chart(fig_dual, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MARKET COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 Model vs Indian Market Price — Validation Dashboard")
    st.caption(
        "Compares model-generated calibrated index price against actual Indian market prices. "
        "Overlap period only — shows where real data exists."
    )

    if "market_price" not in hist.columns or hist["market_price"].isna().all():
        st.warning(
            "⚠️ No market price data found in pipeline outputs. "
            "Ensure `market_prices_Si_FeSi.xlsx` is in the working directory and re-run pipeline."
        )
    else:
        mkt_overlap = hist.dropna(subset=["market_price"]).copy()
        mkt_overlap = mkt_overlap[
            mkt_overlap.index >= hist.index[-1] - pd.DateOffset(weeks=display_window)
        ]

        if len(mkt_overlap) < 3:
            st.warning(
                f"Only {len(mkt_overlap)} overlapping data points found in the selected display window. "
                "Increase history window on the sidebar to view more data."
            )
        else:
            rmse = float(np.sqrt(np.mean(
                (mkt_overlap["real_price"] - mkt_overlap["market_price"]) ** 2)))
            mae  = float(np.mean(np.abs(
                mkt_overlap["real_price"] - mkt_overlap["market_price"])))
            mape = float(np.mean(np.abs(
                (mkt_overlap["real_price"] - mkt_overlap["market_price"]) /
                (mkt_overlap["market_price"] + 1e-9)
            )) * 100)
            corr = float(np.corrcoef(
                mkt_overlap["real_price"], mkt_overlap["market_price"])[0, 1])
            bias = float((mkt_overlap["real_price"] - mkt_overlap["market_price"]).mean())

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("RMSE (₹/Kg)",  f"₹{rmse:.3f}")
            m2.metric("MAE (₹/Kg)",   f"₹{mae:.3f}")
            m3.metric("MAPE",         f"{mape:.2f}%")
            m4.metric("Correlation",  f"{corr:.3f}")
            m5.metric("Mean Bias",    f"₹{bias:+.3f}/Kg",
                      help="Positive = model overestimates market")

            st.divider()

            fig_mkt = go.Figure()
            fig_mkt.add_trace(go.Scatter(
                x=mkt_overlap.index, y=mkt_overlap["market_price"],
                name="Indian Market Price (Actual)",
                line=dict(color=C_MARKET, width=2.8),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Market: ₹%{y:.2f}/Kg<extra></extra>",
            ))
            fig_mkt.add_trace(go.Scatter(
                x=mkt_overlap.index, y=mkt_overlap["real_price"],
                name="Model Index Price",
                line=dict(color=color, width=2.5, dash="dash"),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Model: ₹%{y:.2f}/Kg<extra></extra>",
            ))
            fig_mkt.add_trace(go.Scatter(
                x=list(mkt_overlap.index) + list(mkt_overlap.index[::-1]),
                y=list(mkt_overlap["market_price"]) + list(mkt_overlap["real_price"][::-1]),
                fill="toself", fillcolor="rgba(100,100,200,0.07)",
                line=dict(width=0), name="Model Error Band", hoverinfo="skip",
            ))
            fig_mkt.add_trace(go.Scatter(
                x=list(future.index), y=list(future["real_price"]),
                name="Future Forecast (Model)",
                line=dict(color=C_FUTURE, width=2.2, dash="dot"),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Forecast: ₹%{y:.2f}/Kg<extra></extra>",
            ))

            # AUDIT FIX: Replaced parameter-based annotation
            x_mkt_forecast = mkt_overlap.index[-1]
            fig_mkt.add_vline(x=x_mkt_forecast, line_width=1.5, line_dash="dot", line_color="#999")
            fig_mkt.add_annotation(
                x=x_mkt_forecast, y=1, yref="paper",
                text="Forecast Start →", showarrow=False,
                xanchor="left", yanchor="top", font=dict(size=11), xshift=5
            )

            fig_mkt.update_layout(**_layout(
                f"{product_label}: Indian Market Price vs Model Index Price",
                "Price (₹/Kg)", 490,
            ))
            st.plotly_chart(fig_mkt, use_container_width=True)

            error_s = mkt_overlap["real_price"] - mkt_overlap["market_price"]
            fig_err = go.Figure()
            fig_err.add_trace(go.Bar(
                x=mkt_overlap.index, y=error_s,
                name="Error (Model − Market)",
                marker=dict(
                    color=np.where(error_s >= 0, C_ERR_POS, C_ERR_NEG),
                    opacity=0.75,
                ),
                hovertemplate="<b>%{x|%d %b %Y}</b><br>Error: ₹%{y:.3f}/Kg<extra></extra>",
            ))
            fig_err.add_hline(y=0, line_color="#333", line_width=1)
            
            # AUDIT FIX: Safe hline annotation rendering
            fig_err.add_hline(y=bias, line_dash="dot", line_color="#888")
            fig_err.add_annotation(
                x=1, xref="paper", y=bias,
                text=f"Mean Bias: ₹{bias:+.3f}", showarrow=False,
                xanchor="right", yanchor="bottom", yshift=2
            )

            fig_err.update_layout(**_layout("Weekly Model Error (₹/Kg)", "Error (₹/Kg)", 300))
            st.plotly_chart(fig_err, use_container_width=True)

            col_s1, col_s2 = st.columns(2)

            fig_scat = go.Figure()
            fig_scat.add_trace(go.Scatter(
                x=mkt_overlap["market_price"], y=mkt_overlap["real_price"],
                mode="markers",
                marker=dict(color=color, size=7, opacity=0.65),
                hovertemplate="Market: ₹%{x:.2f}<br>Model: ₹%{y:.2f}<extra></extra>",
                name="Observations",
            ))
            mn = mkt_overlap["market_price"].min()
            mx = mkt_overlap["market_price"].max()
            fig_scat.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx], mode="lines", name="Perfect Fit",
                line=dict(color="#aaa", dash="dash", width=1.5),
            ))
            fig_scat.update_layout(**_layout(
                "Calibration Scatter: Model vs Market", "Model Price (₹/Kg)", 380))
            fig_scat.update_xaxes(title="Market Price (₹/Kg)")
            col_s1.plotly_chart(fig_scat, use_container_width=True)

            fig_hist_err = go.Figure(go.Histogram(
                x=error_s, nbinsx=20,
                marker_color=color, opacity=0.75, name="Error Dist",
            ))
            fig_hist_err.add_vline(x=0, line_color="#333", line_width=1.5)
            
            # AUDIT FIX: Safe vertical line annotation rendering
            fig_hist_err.add_vline(x=bias, line_dash="dot", line_color=C_MARKET)
            fig_hist_err.add_annotation(
                x=bias, y=1, yref="paper",
                text=f"Bias={bias:+.3f}", showarrow=False,
                xanchor="right", yanchor="top", xshift=-5
            )

            fig_hist_err.update_layout(**_layout("Error Distribution", "Frequency", 380))
            col_s2.plotly_chart(fig_hist_err, use_container_width=True)

            with st.expander("📋 View Weekly Comparison Data", expanded=False):
                tbl = mkt_overlap[["real_price", "market_price"]].copy()
                tbl.columns = ["Model Price (₹/Kg)", "Market Price (₹/Kg)"]
                tbl["Error (₹/Kg)"] = (tbl["Model Price (₹/Kg)"] -
                                        tbl["Market Price (₹/Kg)"]).round(3)
                tbl["Error %"] = (tbl["Error (₹/Kg)"] /
                                   (tbl["Market Price (₹/Kg)"] + 1e-9) * 100).round(2)
                st.dataframe(
                    tbl.sort_index(ascending=False).style.format({
                        "Model Price (₹/Kg)":  "₹{:.3f}",
                        "Market Price (₹/Kg)": "₹{:.3f}",
                        "Error (₹/Kg)":        "₹{:+.3f}",
                        "Error %":             "{:+.2f}%",
                    }),
                    use_container_width=True,
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — REGIME & DRIVERS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🌍 Macro Driver Analysis & Market Regime")

    h_drv = hist[hist.index >= hist.index[-1] - pd.DateOffset(weeks=display_window)].copy()

    if "regime_probability" in h_drv.columns:
        fig_reg = go.Figure()
        reg_vals = h_drv["regime_probability"].values
        fig_reg.add_trace(go.Scatter(
            x=h_drv.index, y=reg_vals,
            name="P(High Volatility Regime)",
            fill="tozeroy",
            fillcolor="rgba(211, 47, 47, 0.15)",
            line=dict(color="#D32F2F", width=2),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Regime Prob: %{y:.2%}<extra></extra>",
        ))
        
        # AUDIT FIX: Safe hline annotation rendering
        fig_reg.add_hline(y=0.65, line_dash="dot", line_color="#D32F2F")
        fig_reg.add_annotation(
            x=1, xref="paper", y=0.65,
            text="High Stress (0.65)", showarrow=False,
            xanchor="right", yanchor="bottom", yshift=2
        )
        
        fig_reg.add_hline(y=0.45, line_dash="dot", line_color="#FF9800")
        fig_reg.add_annotation(
            x=1, xref="paper", y=0.45,
            text="Elevated Risk (0.45)", showarrow=False,
            xanchor="right", yanchor="bottom", yshift=2
        )

        fig_reg.update_layout(**_layout(
            f"{product_label}: Market Stress Regime Probability", "Probability", 280))
        st.plotly_chart(fig_reg, use_container_width=True)

    st.divider()

    si_drivers   = [
        "al_price", "solar_etf", "semiconductor_etf", "china_etf", 
        "bdry_freight", "vix", "thermal_coal_futures", "crude_oil", 
        "usd_inr", "silica_quartz_index", "cny_inr", 
        "hydrology_rainfall_index", "petcoke_charcoal_index", 
        "electrode_consumables_index", "gfex_silicon_futures", 
        "electricity_power_index"
    ]
    fesi_drivers = [
        "steel_etf", "iron_ore", "thermal_coal_futures", "bdry_freight", 
        "al_price", "india_steel", "vix", "usd_inr", "silica_quartz_index", 
        "cny_inr", "hydrology_rainfall_index", "shaanxi_semicoke", 
        "magnesium_demand", "zce_fesi_futures", "carbon_emissions_futures", 
        "electricity_power_index", "fx_effect"
    ]
                    
    driver_list = si_drivers if alloy_code == "Si" else fesi_drivers
    avail_drv   = [d for d in driver_list if d in h_drv.columns]

    if avail_drv:
        st.markdown("#### 📈 Relative Driver Movement (Indexed to 100)")
        st.caption("All drivers normalised to 100 at start of history for relative comparison.")
        fig_drv = go.Figure()
        for i, d_col in enumerate(avail_drv):
            s = h_drv[d_col].dropna()
            if s.empty or s.iloc[0] == 0:
                continue
            norm = (s / s.iloc[0]) * 100
            fig_drv.add_trace(go.Scatter(
                x=s.index, y=norm,
                name=DRIVER_LABELS.get(d_col, d_col.replace("_", " ").title()),
                line=dict(color=DRIVER_PALETTE[i % len(DRIVER_PALETTE)], width=1.8),
                hovertemplate=(
                    f"<b>%{{x|%d %b %Y}}</b><br>"
                    f"{DRIVER_LABELS.get(d_col, d_col)}: %{{y:.1f}}<extra></extra>"
                ),
            ))
        fig_drv.update_layout(**_layout(
            "Relative Driver Movement (Indexed = 100 at start)", "Relative Index", 430))
        st.plotly_chart(fig_drv, use_container_width=True)

    st.divider()

    fc1, fc2 = st.columns([3, 2])

    fig_bar = go.Figure(go.Bar(
        x=fi["importance"].iloc[:15][::-1],
        y=fi["feature"].iloc[:15][::-1],
        orientation="h",
        marker=dict(
            color=fi["importance"].iloc[:15][::-1],
            colorscale="Blues", showscale=False, opacity=0.88,
        ),
        hovertemplate="%{y}: %{x:.1f}<extra></extra>",
    ))
    fig_bar.update_layout(**_layout(
        f"Top 15 Feature Importances — {product_label}", "Importance Score", 430))
    fc1.plotly_chart(fig_bar, use_container_width=True)

    top7 = fi.head(7).copy()
    rest_imp = fi.iloc[7:]["importance"].sum()
    if rest_imp > 0:
        top7 = pd.concat([
            top7,
            pd.DataFrame([{"feature": "Other Variables", "importance": rest_imp}])
        ], ignore_index=True)
    fig_pie = go.Figure(go.Pie(
        labels=[DRIVER_LABELS.get(f, f.replace("_", " ").title()) for f in top7["feature"]],
        values=(top7["importance"] / top7["importance"].sum() * 100),
        hole=0.48,
        marker=dict(colors=DRIVER_PALETTE[:len(top7)]),
        textinfo="label+percent", textfont=dict(size=10),
    ))
    fig_pie.update_layout(
        title=dict(text=f"{product_label}: Driver Dependence %",
                   font=dict(size=14), x=0.5),
        template="plotly_white", height=430, showlegend=False,
    )
    fc2.plotly_chart(fig_pie, use_container_width=True)

    with st.expander("⚡ View Historical Shock Event Periods", expanded=False):
        
        # AUDIT FIX: Decoupled SHOCK_EVENTS dictionary to prevent ModuleNotFoundError
        SHOCKS = {
            "covid_disruption":        ("2020-03-01", "2020-08-31"),
            "ukraine_energy_crisis":   ("2022-02-24", "2022-12-31"),
            "logistics_spike":         ("2021-01-01", "2021-12-31"),
            "china_power_curbs_2021":  ("2021-08-01", "2021-12-31"),
            "china_export_restrict":   ("2023-07-01", "2024-03-31"),
            "energy_crisis_europe":    ("2022-06-01", "2023-03-31"),
            "steel_downturn_2015":     ("2015-06-01", "2016-06-30"),
            "steel_downturn_2019":     ("2019-01-01", "2019-12-31"),
            "china_steel_curbs":       ("2021-05-01", "2021-12-31"),
            "india_infra_push":        ("2023-01-01", "2024-12-31"),
        }

        shock_df = pd.DataFrame([
            {"Event": k.replace("_", " ").title(), "Start": v[0], "End": v[1]}
            for k, v in SHOCKS.items()
        ])
        st.dataframe(shock_df, use_container_width=True)

        fig_shk = go.Figure()
        fig_shk.add_trace(go.Scatter(
            x=h_drv.index, y=h_drv["actual"],
            name=f"{product_label} Index", line=dict(color=color, width=2),
        ))
        
        for k, (s_d, e_d) in SHOCKS.items():
            # AUDIT FIX: Replaced parameter-based annotation to avoid str/datetime math crashes
            fig_shk.add_vrect(
                x0=s_d, x1=e_d,
                fillcolor="rgba(100,100,200,0.08)",
                layer="below", line_width=0,
            )
            fig_shk.add_annotation(
                x=s_d, y=1, yref="paper",
                text=k.replace("_", " ").title()[:18], showarrow=False,
                xanchor="left", yanchor="top", font=dict(size=8), xshift=2
            )

        fig_shk.update_layout(**_layout(
            f"{product_label}: Shock Event Periods vs Index", "Index Value", 380))
        st.plotly_chart(fig_shk, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — VIU & TCO OPTIMIZER
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### ⚖️ Value-In-Use (VIU) & Total Cost of Ownership Optimizer")
    st.caption(
        "Computes the true economic cost of Silicon Metal vs FeSi per unit of "
        "Effective Silicon delivered to steel melt."
    )

    si_real_last   = get_real_price_series("Si")
    fesi_real_last = get_real_price_series("FeSi")

    p_si_kg   = float(si_real_last.iloc[-1])   if si_real_last   is not None else 187.5
    p_fesi_kg = float(fesi_real_last.iloc[-1]) if fesi_real_last is not None else 107.75
    p_si_mt   = p_si_kg   * 1000
    p_fesi_mt = p_fesi_kg * 1000

    st.markdown("#### 1. Current Market Prices")
    pa, pb, pc, pd_ = st.columns(4)
    si_price_inp   = pa.number_input("Si Metal Price (₹/MT)",          value=int(p_si_mt),   step=1000)
    fesi_price_inp = pb.number_input("FeSi Price (₹/MT)",              value=int(p_fesi_mt), step=1000)
    fe_credit_rate = pc.number_input("Fe Scrap Credit (₹/MT)", value=38000, step=1000,
                                     help="Market price of iron scrap for Fe credit")
    annual_si_mt   = pd_.number_input("Annual Si Consumption (MT Eff. Si)", value=8000, step=500)

    st.divider()
    st.markdown("#### 2. Alloy Specifications & Process Parameters")

    with st.expander("⚙️ Si Metal Specifications", expanded=True):
        s1, s2, s3, s4 = st.columns(4)
        si_si_pct  = s1.slider("Si Metal — Si Content (%)",      96.0, 99.5, 98.0, step=0.1)
        si_fe_pct  = s2.slider("Si Metal — Fe Content (%)",       0.0,  2.0,  0.5, step=0.1)
        si_rec     = s3.slider("Si Metal — Recovery (%)",        85.0, 99.0, 93.0, step=0.5)
        si_add_kg  = s4.number_input("Si Metal — Addition Rate (Kg/T steel)", value=2.5, step=0.1)

        sv1, sv2, sv3, sv4 = st.columns(4)
        si_power_kwh  = sv1.slider("Power Saving vs FeSi (kWh/MT)",         50.0, 120.0, 80.0, step=5.0)
        si_rec_imp    = sv2.slider("Recovery Improvement vs FeSi (%)",       0.5,   3.0,  1.5, step=0.1)
        si_slag_red   = sv3.slider("Slag Reduction vs FeSi (Kg/T steel)",    3.0,  15.0,  8.0, step=0.5)
        si_incl       = sv4.slider("Inclusion Reject Reduction (%)",      0.005, 0.030, 0.015, step=0.001, format="%.3f")

        sv5, sv6, sv7 = st.columns(3)
        si_reblow     = sv5.slider("Reblow Risk Reduction (%)",  0.5, 3.0, 1.0, step=0.1)
        si_yield      = sv6.slider("Yield Gain (%)",          0.005, 0.040, 0.020, step=0.001, format="%.3f")
        si_throughput = sv7.slider("Throughput Time Saved (min/heat)", 0.2, 2.0, 0.5, step=0.05)

    with st.expander("⚙️ FeSi Specifications", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        fe_si_pct  = f1.slider("FeSi — Si Content (%)",   65.0, 75.0, 70.0, step=0.5)
        fe_fe_pct  = f2.slider("FeSi — Fe Content (%)",   22.0, 32.0, 25.0, step=0.5)
        fe_rec     = f3.slider("FeSi — Recovery (%)",     78.0, 95.0, 87.0, step=0.5)
        fe_add_kg  = f4.number_input("FeSi — Addition Rate (Kg/T steel)", value=4.0, step=0.1)

    st.divider()

    si_benefits_dict = compute_si_operational_benefits(
        power_kwh=si_power_kwh,
        recovery_pct=si_rec_imp,
        slag_red_kg=si_slag_red,
        inclusion_rej=si_incl * 100,
        reblow_pct=si_reblow,
        yield_gain_pct=si_yield * 100,
        throughput_min=si_throughput,
        si_price_mt=float(si_price_inp),
    )
    si_total_viu = sum(si_benefits_dict.values())

    si_tco   = calc_tco(si_price_inp,   si_si_pct, si_fe_pct, si_rec,
                         fe_credit_rate, si_total_viu)
    fesi_tco = calc_tco(fesi_price_inp, fe_si_pct, fe_fe_pct, fe_rec,
                         fe_credit_rate, 0.0)

    best     = "Si Metal" if si_tco["tco"] < fesi_tco["tco"] else "FeSi"
    tco_diff = abs(si_tco["tco"] - fesi_tco["tco"])

    if best == "Si Metal":
        st.success(
            f"### 🏆 Procurement Recommendation: **Silicon Metal** \n"
            f"True cost advantage: **₹{tco_diff:,.0f} / MT Effective Si** over FeSi."
        )
    else:
        st.info(
            f"### 💡 Procurement Recommendation: **FeSi Alloy** \n"
            f"FeSi delivers lower TCO by **₹{tco_diff:,.0f} / MT Effective Si** at current prices."
        )

    st.markdown("#### 3. TCO Comparison Summary")
    k1c, k2c = st.columns(2)

    with k1c:
        st.markdown("**Silicon Metal**")
        sa, sb, sc_ = st.columns(3)
        sa.metric("Effective Si",  f"{si_tco['effective_si']*100:.1f}%")
        sb.metric("Fe Credit",     f"₹{si_tco['fe_credit']:,.0f}/MT")
        sc_.metric("Adj. Price",   f"₹{si_tco['adj_price']:,.0f}/MT")
        sd, se, sf = st.columns(3)
        sd.metric("Base VIU",      f"₹{si_tco['base_viu']:,.0f}/MT Eff. Si")
        se.metric("VIU Benefits",  f"₹{si_tco['viu_benefits_per_eff']:,.0f}/MT Eff. Si",
                  help="Process economics advantage of Si Metal over FeSi")
        sf.metric("**Final TCO**", f"₹{si_tco['tco']:,.0f}/MT Eff. Si")

    with k2c:
        st.markdown("**FeSi Alloy**")
        fa_, fb, fc_ = st.columns(3)
        fa_.metric("Effective Si", f"{fesi_tco['effective_si']*100:.1f}%")
        fb.metric("Fe Credit",     f"₹{fesi_tco['fe_credit']:,.0f}/MT")
        fc_.metric("Adj. Price",   f"₹{fesi_tco['adj_price']:,.0f}/MT")
        fd, fe, ff = st.columns(3)
        fd.metric("Base VIU",      f"₹{fesi_tco['base_viu']:,.0f}/MT Eff. Si")
        fe.metric("VIU Benefits",  "₹0 / MT Eff. Si",
                  help="FeSi is baseline — Si VIU benefits measured relative to FeSi")
        ff.metric("**Final TCO**", f"₹{fesi_tco['tco']:,.0f}/MT Eff. Si")

    st.divider()

    st.markdown("#### 4. TCO Breakdown — Visual Comparison")
    fc_b1, fc_b2 = st.columns(2)

    fig_tco_bar = go.Figure()
    alloys    = ["Si Metal", "FeSi Alloy"]
    base_vius = [si_tco["base_viu"], fesi_tco["base_viu"]]
    viu_bens  = [-si_tco["viu_benefits_per_eff"], 0.0]

    fig_tco_bar.add_trace(go.Bar(
        name="Base VIU (Adj Price / Eff Si)", x=alloys, y=base_vius,
        marker_color=[C_SI_IDX, C_FESI_IDX],
        text=[f"₹{v:,.0f}" for v in base_vius], textposition="inside",
    ))
    fig_tco_bar.add_trace(go.Bar(
        name="VIU Process Benefits (Si Metal advantage)", x=alloys, y=viu_bens,
        marker_color=[C_VIU_GREEN, C_VIU_GRAY],
        text=[f"₹{v:,.0f}" if v != 0 else "" for v in viu_bens], textposition="inside",
    ))
    fig_tco_bar.update_layout(
        barmode="stack",
        title="TCO Breakdown (₹ per MT Effective Silicon)",
        template="plotly_white", height=400,
        yaxis_title="₹ / MT Eff. Si",
        legend=dict(orientation="h", y=1.06),
    )
    fc_b1.plotly_chart(fig_tco_bar, use_container_width=True)

    bens   = list(si_benefits_dict.items())
    labels = [b[0] for b in bens] + ["Total VIU Benefit"]
    values = [b[1] for b in bens] + [si_total_viu]
    colors = [C_VIU_GREEN] * len(bens) + [C_SI_IDX]
    fig_wf = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors, opacity=0.85,
        text=[f"₹{v:,.0f}" for v in values], textposition="outside",
        hovertemplate="%{x}: ₹%{y:,.0f}<extra></extra>",
    ))
    fig_wf.update_layout(**_layout(
        "Si Metal VIU Benefit Components (₹/MT of Effective Si)", "₹/MT", 400))
    fc_b2.plotly_chart(fig_wf, use_container_width=True)

    st.divider()

    st.markdown("#### 5. Enterprise Savings Calculator")
    savings_per_mt = abs(si_tco["tco"] - fesi_tco["tco"])
    annual_savings = savings_per_mt * annual_si_mt
    savings_cr     = annual_savings / 1e7

    if best == "Si Metal":
        st.success(
            f"### 💰 Annual Procurement Savings: **₹{savings_cr:.2f} Crore** \n"
            f"By switching **{annual_si_mt:,} MT/yr** of Effective Si procurement to Si Metal."
        )
    else:
        st.info(
            f"### 💰 Savings by Staying on FeSi: **₹{savings_cr:.2f} Crore/yr** \n"
            f"FeSi currently offers lower TCO at current relative prices."
        )

    st.divider()

    st.markdown("#### 6. Future TCO Projections — 3-Year Outlook")
    st.caption("Projects TCO over the forecast horizon using ML-generated future prices with current VIU parameters.")

    si_fut   = get_future_price_series("Si")
    fesi_fut = get_future_price_series("FeSi")

    if si_fut is not None and fesi_fut is not None:
        fut_aligned = pd.DataFrame({
            "si_price":   si_fut,
            "fesi_price": fesi_fut,
        }).dropna()

        si_tco_fut = (
            (fut_aligned["si_price"] * 1000 - (si_fe_pct / 100) * fe_credit_rate)
            / (si_tco["effective_si"] + 1e-9)
            - si_tco["viu_benefits_per_eff"]
        )
        fesi_tco_fut = (
            (fut_aligned["fesi_price"] * 1000 - (fe_fe_pct / 100) * fe_credit_rate)
            / (fesi_tco["effective_si"] + 1e-9)
        )

        fig_fut_tco = go.Figure()
        fig_fut_tco.add_trace(go.Scatter(
            x=fut_aligned.index, y=si_tco_fut,
            name="Si Metal TCO",
            line=dict(color=C_SI_IDX, width=2.5),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Si Metal TCO: ₹%{y:,.0f}/MT Eff. Si<extra></extra>",
        ))
        fig_fut_tco.add_trace(go.Scatter(
            x=fut_aligned.index, y=fesi_tco_fut,
            name="FeSi TCO",
            line=dict(color=C_FESI_IDX, width=2.5, dash="dash"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>FeSi TCO: ₹%{y:,.0f}/MT Eff. Si<extra></extra>",
        ))
        crossover = si_tco_fut - fesi_tco_fut
        fig_fut_tco.add_trace(go.Scatter(
            x=list(fut_aligned.index) + list(fut_aligned.index[::-1]),
            y=list(np.where(crossover < 0, si_tco_fut, fesi_tco_fut)) +
              list(np.where(crossover < 0, fesi_tco_fut, si_tco_fut)[::-1]),
            fill="toself", fillcolor="rgba(46,125,50,0.07)",
            line=dict(width=0), name="Si Metal Advantage Zone", hoverinfo="skip",
        ))
        fig_fut_tco.update_layout(**_layout(
            "Projected TCO — Si Metal vs FeSi (Next 3 Years)",
            "TCO (₹/MT Effective Si)", 470,
        ))
        st.plotly_chart(fig_fut_tco, use_container_width=True)

        si_cheaper_pct = (crossover < 0).mean() * 100
        if si_cheaper_pct > 60:
            st.success(f"📊 **TCO Outlook**: Si Metal cheaper than FeSi in **{si_cheaper_pct:.0f}%** of forecast. Consider forward contracts.")
        elif si_cheaper_pct > 35:
            st.warning(f"📊 **TCO Outlook**: Mixed signals — Si Metal cheaper in {si_cheaper_pct:.0f}% of forecast. Blend procurement.")
        else:
            st.info(f"📊 **TCO Outlook**: FeSi maintains TCO advantage in {100-si_cheaper_pct:.0f}% of forecast period.")
    else:
        st.info("📂 Run both Si and FeSi pipelines to generate the comparative future TCO chart.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — SUBSTITUTION SOLVER
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 🧬 Silicon Alloy Substitution Solver")
    st.caption(
        "LP-based optimization to find the cheapest blend of Si Metal and FeSi "
        "that satisfies metallurgical constraints for the target steel grade."
    )

    sc_left, sc_right = st.columns([1, 2])

    with sc_left:
        st.markdown("#### Grade Selection")
        sel_grade    = st.selectbox("Target Steel Grade", list(STEEL_GRADES.keys()))
        grade_params = STEEL_GRADES[sel_grade]

        st.markdown("#### Metallurgical Constraints")
        st.caption("Pre-filled from Excel Solver sheet. Adjust as needed.")

        min_eff_si    = st.slider("Min Effective Si (%)", 0.55, 1.00,
                                  float(grade_params["eff_si_min"]), step=0.01)
        max_chill     = st.slider("Max Chill Factor (°C/MT)", 8.0, 14.0,
                                  float(grade_params["chill_limit"]), step=0.5)
        max_rec_var   = st.slider("Max Recovery Variability (±%)", 1.0, 8.0,
                                  float(grade_params["rec_var"]), step=0.5)
        max_reblow    = st.slider("Max Reblow Risk (%)", 1.0, 6.0,
                                  float(grade_params["reblow_max"]), step=0.5)
        max_inclusion = st.slider("Max Inclusion Index", 0.030, 0.080,
                                  float(grade_params["inclusion_max"]),
                                  step=0.001, format="%.3f")
        min_power_ben = st.slider("Min Power Benefit (kWh/MT)", 0.0, 90.0,
                                  float(grade_params["power_min"]), step=5.0)
        max_si_share  = st.slider("Max Si Metal Share", 0.0, 1.0,
                                  float(grade_params["si_share_max"]), step=0.05)

    with sc_right:
        si_eff   = (si_si_pct / 100) * (si_rec / 100)
        fesi_eff = (fe_si_pct / 100) * (fe_rec / 100)

        si_tco_solver   = si_tco["tco"]
        fesi_tco_solver = fesi_tco["tco"]

        chill_si      = 9.0;  chill_fesi   = 7.0
        rec_var_si    = 2.0;  rec_var_fesi = 5.0
        reblow_si     = 1.5;  reblow_fesi  = 3.5
        incl_si       = 0.040; incl_fesi   = 0.060
        power_si      = float(si_power_kwh)
        power_fesi    = 0.0

        c_obj = [fesi_tco_solver, si_tco_solver]
        A_eq  = [[1, 1]]
        b_eq  = [1]

        A_ub = [
            [-fesi_eff,    -si_eff],
            [chill_fesi,   chill_si],
            [rec_var_fesi, rec_var_si],
            [reblow_fesi,  reblow_si],
            [incl_fesi,    incl_si],
            [-power_fesi,  -power_si],
        ]
        b_ub = [
            -min_eff_si,
            max_chill,
            max_rec_var,
            max_reblow,
            max_inclusion,
            -min_power_ben,
        ]

        bounds = [(0, max(0.0, 1.0 - max_si_share)), (0, max_si_share)]

        res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub,
                      bounds=bounds, method="highs")

        st.markdown("#### Optimization Result")
        if res.success:
            x_fesi, y_si = float(res.x[0]), float(res.x[1])
            blended_tco  = float(res.fun)

            def single_feasible(eff, chill, rec, rb, incl, pwr):
                return (eff >= min_eff_si and chill <= max_chill and
                        rec <= max_rec_var and rb <= max_reblow and
                        incl <= max_inclusion and pwr >= min_power_ben)

            pure_si_ok   = single_feasible(si_eff,   chill_si,   rec_var_si,
                                            reblow_si,  incl_si,   power_si)
            pure_fesi_ok = single_feasible(fesi_eff,  chill_fesi, rec_var_fesi,
                                            reblow_fesi, incl_fesi, power_fesi)

            single_costs = []
            if pure_si_ok:   single_costs.append(si_tco_solver)
            if pure_fesi_ok: single_costs.append(fesi_tco_solver)
            baseline_single = min(single_costs) if single_costs else None
            savings_vs_single = (baseline_single - blended_tco) if baseline_single else 0.0

            rk1, rk2, rk3 = st.columns(3)
            rk1.success(
                f"**Optimal Blended TCO** \n"
                f"### ₹{blended_tco:,.0f} / MT Eff. Si"
            )
            rk2.metric("FeSi Share",     f"{x_fesi*100:.1f}%")
            rk3.metric("Si Metal Share", f"{y_si*100:.1f}%")

            if savings_vs_single and savings_vs_single > 100:
                st.info(f"💰 **Blending Saves:** ₹{savings_vs_single:,.0f}/MT Eff. Si vs best single alloy.")
            elif single_costs:
                st.info("ℹ️ **100% Single Alloy is Optimal** at this constraint level.")

            fig_blend = go.Figure(go.Pie(
                labels=["FeSi Alloy", "Si Metal"],
                values=[x_fesi, y_si],
                hole=0.45,
                marker=dict(colors=[C_FESI_IDX, C_SI_IDX]),
                textinfo="label+percent",
                texttemplate="%{label}<br>%{percent:.1%}",
                hovertemplate="%{label}: %{percent:.1%}<br>Share: %{value:.3f}<extra></extra>",
            ))
            fig_blend.update_layout(
                title=f"Optimal Blend for {sel_grade}",
                template="plotly_white", height=380, showlegend=False,
            )
            st.plotly_chart(fig_blend, use_container_width=True)

            blend_eff   = fesi_eff   * x_fesi + si_eff   * y_si
            blend_chill = chill_fesi * x_fesi + chill_si * y_si
            blend_rec   = rec_var_fesi * x_fesi + rec_var_si  * y_si
            blend_rb    = reblow_fesi  * x_fesi + reblow_si   * y_si
            blend_incl  = incl_fesi    * x_fesi + incl_si     * y_si
            blend_pwr   = power_fesi   * x_fesi + power_si    * y_si

            cons_df = pd.DataFrame({
                "Constraint": [
                    "Effective Si (%)", "Chill Factor (°C/MT)",
                    "Recovery Variability (%)", "Reblow Risk (%)",
                    "Inclusion Index", "Power Benefit (kWh/MT)",
                ],
                "Limit": [
                    f"≥ {min_eff_si:.2f}", f"≤ {max_chill:.1f}",
                    f"≤ {max_rec_var:.1f}", f"≤ {max_reblow:.1f}",
                    f"≤ {max_inclusion:.3f}", f"≥ {min_power_ben:.0f}",
                ],
                "Blend Value": [
                    f"{blend_eff:.3f}", f"{blend_chill:.2f}",
                    f"{blend_rec:.2f}", f"{blend_rb:.2f}",
                    f"{blend_incl:.4f}", f"{blend_pwr:.1f}",
                ],
                "Status": [
                    "✅" if blend_eff   >= min_eff_si    else "❌",
                    "✅" if blend_chill <= max_chill     else "❌",
                    "✅" if blend_rec   <= max_rec_var   else "❌",
                    "✅" if blend_rb    <= max_reblow    else "❌",
                    "✅" if blend_incl  <= max_inclusion else "❌",
                    "✅" if blend_pwr   >= min_power_ben else "❌",
                ],
            })
            st.dataframe(cons_df, use_container_width=True, hide_index=True)

            st.markdown("#### Process Stage Suitability")
            stage_df = pd.DataFrame({
                "Stage": ["HMDS", "BOF", "LF (Ladle Furnace)", "RH (Vacuum Degasser)", "Tundish"],
                "Si Metal": ["⚠️ Limited", "⚠️ Limited", "✅ Good", "✅ Best", "✅ Minor Trim"],
                "FeSi":     ["⚠️ Limited", "✅ Good",     "✅ Good", "✅ Good", "✅ Minor Trim"],
                "Reason": [
                    "Si oxidation losses in hot metal are high",
                    "High oxidation in BOF favours FeSi",
                    "Controlled atmosphere — both suitable",
                    "Best precision chemistry + highest Si recovery",
                    "Late-stage trimming only — both suitable",
                ],
            })
            st.dataframe(stage_df, use_container_width=True, hide_index=True)

            st.markdown("#### Annual Procurement Savings")
            vol_a, vol_b, vol_c = st.columns(3)
            annual_vol_blend = vol_a.number_input(
                "Annual Volume (MT Eff. Si)", value=int(annual_si_mt), step=500, key="solver_vol")
            blend_savings_cr = (savings_vs_single * annual_vol_blend / 1e7) if savings_vs_single > 100 else 0
            vol_b.metric("Savings per MT Eff. Si",
                         f"₹{savings_vs_single:,.0f}" if savings_vs_single > 100 else "₹0")
            vol_c.metric("Projected Annual Savings", f"₹{blend_savings_cr:.2f} Crore")

        else:
            st.error(
                "⚠️ **Constraint Infeasible**: The metallurgical limits cannot be met "
                "by any FeSi + Si Metal blend at the specified grade.  \n"
                "Try relaxing: Effective Si minimum, Inclusion Index, or Power Benefit floor."
            )
            st.info(
                "**Diagnostic hints:**\n"
                f"- Si Metal eff Si: {si_eff:.3f} | FeSi: {fesi_eff:.3f} | Required: {min_eff_si:.3f}\n"
                f"- Si Metal power benefit: {power_si:.0f} kWh | FeSi: {power_fesi:.0f} | Min: {min_power_ben:.0f}\n"
                f"- Max Si share ({max_si_share:.2f}) may be too restrictive for high Si grades."
            )


# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown(
    "<div style='text-align:center; color:#999; font-size:11px;'>"
    "Silicon Procurement Intelligence & Process Economics Platform | "
    "Anchors: Si Metal ₹187.50/Kg · FeSi ₹107.75/Kg (08-Jan-2024) | "
    "Model: LightGBM Hybrid Engine | Indian Market Data Integrated"
    "</div>",
    unsafe_allow_html=True,
)