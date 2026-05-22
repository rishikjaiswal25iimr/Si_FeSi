"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         SILICON INTELLIGENCE PLATFORM — FORECASTING PIPELINE                ║
║         pipeline_silicon.py  |  v2.1  (Production-Audited)                 ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Engines  : Silicon Metal Index  |  FeSi Alloy Index                        ║
║  Method   : LightGBM hybrid + Driver-based Target Proxy + Anchor Calibration║
║  Output   : historical_predictions.csv, future_forecast.csv,                ║
║             feature_importance.csv, model_metadata.json                     ║
║  Anchors  : Si Metal  08-Jan-2024 = ₹187.50/Kg                             ║
║             FeSi      08-Jan-2024 = ₹107.75/Kg                             ║
║                                                                              ║
║  AUDIT FIXES v2.1 (Manual Drivers Update):                                   ║
║  • REMOVED: hrc_proxy entirely (no online fetch, no synthetic fallback)      ║
║  • ADDED: Local Excel loader for cny_inr.xlsx (Date, Price)                  ║
║  • ADDED: Local Excel loader for silica_quartz_index.xlsx (Date, Index)      ║
║  • UPDATED: Target Proxy weights to incorporate Silica Quartz index          ║
║  • IMPROVED: Future forecast stability for manual drivers (carry-forward)    ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import argparse
import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── optional heavy deps ────────────────────────────────────────────────────────
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    print("[WARN] yfinance not installed — all drivers will use synthetic fallbacks.")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

HAS_SKL = False
HAS_RIDGE = False

try:
    from sklearn.ensemble import GradientBoostingRegressor
    HAS_SKL = True
    if not HAS_LGB:
        print("[INFO] LightGBM not found — using sklearn GradientBoosting fallback.")
except ImportError:
    pass

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    HAS_RIDGE = True
except ImportError:
    pass

if not HAS_LGB and not HAS_SKL and not HAS_RIDGE:
    print("[WARN] No ML library found. Install lightgbm or scikit-learn.")


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

START_DATE      = "2016-01-01"
END_DATE        = "2026-04-25"
DEFAULT_HORIZON = 156          # 3 years of weekly forecasts

LAG_PERIODS  = [1, 2, 4, 6, 8, 13, 26]
ROLLING_WINS = [4, 8, 13, 26]

# ── Anchor calibration points ────────────────────────────────────────────────
ANCHORS = {
    "Si":   {"date": pd.Timestamp("2024-01-08"), "price_per_kg": 187.50},
    "FeSi": {"date": pd.Timestamp("2024-01-08"), "price_per_kg": 107.75},
}

# ── Shock / Event dummy framework ─────────────────────────────────────────────
SHOCK_EVENTS = {
    # Common shocks
    "covid_disruption":        ("2020-03-01", "2020-08-31"),
    "ukraine_energy_crisis":   ("2022-02-24", "2022-12-31"),
    "logistics_spike":         ("2021-01-01", "2021-12-31"),

    # Silicon-specific shocks
    "china_power_curbs_2021":  ("2021-08-01", "2021-12-31"),
    "china_export_restrict":   ("2023-07-01", "2024-03-31"),
    "energy_crisis_europe":    ("2022-06-01", "2023-03-31"),

    # FeSi / Steel-specific shocks
    "steel_downturn_2015":     ("2015-06-01", "2016-06-30"),
    "steel_downturn_2019":     ("2019-01-01", "2019-12-31"),
    "china_steel_curbs":       ("2021-05-01", "2021-12-31"),
    "india_infra_push":        ("2023-01-01", "2024-12-31"),
}
SHOCK_COLS = list(SHOCK_EVENTS.keys())

# ── Driver column lists ────────────────────────────────────────────────────────
SI_PRICE_COLS   = ["Target_Proxy", "al_price", "solar_etf", "semiconductor_etf",
                   "coal_energy", "crude_oil", "cny_inr", "silica_quartz_index", 
                   "usd_inr", "china_etf", "bdry_freight", "steel_etf", "vix"]

FESI_PRICE_COLS = ["Target_Proxy", "steel_etf", "iron_ore", "coal_energy",
                   "silica_quartz_index", "usd_inr", "cny_inr", "bdry_freight",
                   "al_price", "india_steel", "vix"]


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET PRICE LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_market_prices(alloy_type: str, idx: pd.DatetimeIndex) -> pd.Series:
    """
    Load real Indian market prices from market_prices_Si_FeSi.xlsx.
    """
    market_file = None
    search_paths = [
        "market_prices_Si_FeSi.xlsx",
        "../market_prices_Si_FeSi.xlsx",
        "/mnt/user-data/uploads/market_prices_Si_FeSi.xlsx",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_prices_Si_FeSi.xlsx"),
    ]
    for p in search_paths:
        if os.path.exists(p):
            market_file = p
            break

    if market_file is None:
        print("  [Market] market_prices_Si_FeSi.xlsx not found — skipping market overlay.")
        return pd.Series(np.nan, index=idx, name="market_price")

    try:
        raw = pd.read_excel(market_file)
        raw.columns = [c.strip() for c in raw.columns]

        col_map = {"Si": "Si Price", "FeSi": "India FeSi Price"}
        price_col = col_map.get(alloy_type)
        if price_col not in raw.columns:
            print(f"  [Market] Column '{price_col}' not found. Columns: {list(raw.columns)}")
            return pd.Series(np.nan, index=idx, name="market_price")

        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", dayfirst=True)
        raw[price_col] = pd.to_numeric(
            raw[price_col].astype(str).str.replace(",", "").str.strip().str.replace("#N/A", ""),
            errors="coerce"
        )
        raw = raw.dropna(subset=["Date", price_col]).copy()
        raw = raw.sort_values("Date").reset_index(drop=True)

        raw["Date_norm"] = raw["Date"] + pd.to_timedelta(
            (4 - raw["Date"].dt.weekday) % 7, unit="D"
        )
        raw["Date_norm"] = raw["Date_norm"].dt.normalize()

        idx_norm = pd.to_datetime(idx).normalize()
        idx_norm = idx_norm + pd.to_timedelta((4 - idx_norm.weekday) % 7, unit="D")

        lookup_df = pd.DataFrame({"Date_norm": idx_norm, "orig_idx": range(len(idx))})
        lookup_df = lookup_df.sort_values("Date_norm").reset_index(drop=True)

        market_clean = raw[["Date_norm", price_col]].rename(
            columns={price_col: "market_price"}
        ).sort_values("Date_norm")

        merged = pd.merge_asof(
            lookup_df,
            market_clean,
            on="Date_norm",
            direction="nearest",
            tolerance=pd.Timedelta("7 days"),
        )

        merged = merged.sort_values("orig_idx").reset_index(drop=True)
        result = pd.Series(merged["market_price"].values, index=idx, name="market_price")

        non_null = result.notna().sum()
        print(f"  [Market] Loaded {non_null}/{len(idx)} non-null market prices for {alloy_type}.")
        return result

    except Exception as e:
        print(f"  [Market] Failed to load market prices: {e}")
        return pd.Series(np.nan, index=idx, name="market_price")


# ═══════════════════════════════════════════════════════════════════════════════
# MANUAL DRIVER EXCEL LOADER
# ═══════════════════════════════════════════════════════════════════════════════

def load_manual_driver(filename: str, value_col: str, idx: pd.DatetimeIndex, fallback_val: float) -> pd.Series:
    """
    Safely load custom manual drivers from local Excel files.
    """
    search_paths = [
        filename,
        f"../{filename}",
        f"/mnt/user-data/uploads/{filename}",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename),
    ]
    file_path = None
    for p in search_paths:
        if os.path.exists(p):
            file_path = p
            break

    if file_path is None:
        print(f"  [Manual Driver] {filename} not found — using fallback synthetic generator.")
        return _synthetic_driver(idx, fallback_val, 0.05, 0.0, 104, 0.02, abs(hash(filename)) % 100)

    try:
        raw = pd.read_excel(file_path)
        raw.columns = [c.strip() for c in raw.columns]

        if "Date" not in raw.columns or value_col not in raw.columns:
            print(f"  [Manual Driver] Missing 'Date' or '{value_col}' columns in {filename}.")
            return pd.Series(fallback_val, index=idx, name=value_col)

        raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", dayfirst=False)
        raw[value_col] = pd.to_numeric(
            raw[value_col].astype(str).str.replace(",", "").str.strip(),
            errors="coerce"
        )
        raw = raw.dropna(subset=["Date", value_col]).copy()
        raw = raw.sort_values("Date").reset_index(drop=True)

        raw["Date_norm"] = raw["Date"] + pd.to_timedelta(
            (4 - raw["Date"].dt.weekday) % 7, unit="D"
        )
        raw["Date_norm"] = raw["Date_norm"].dt.normalize()

        idx_norm = pd.to_datetime(idx).normalize()
        idx_norm = idx_norm + pd.to_timedelta((4 - idx_norm.weekday) % 7, unit="D")

        lookup_df = pd.DataFrame({"Date_norm": idx_norm, "orig_idx": range(len(idx))})
        lookup_df = lookup_df.sort_values("Date_norm").reset_index(drop=True)

        market_clean = raw[["Date_norm", value_col]].sort_values("Date_norm")

        # Direction backward ensures we carry forward the last available price up to that week
        merged = pd.merge_asof(
            lookup_df,
            market_clean,
            on="Date_norm",
            direction="backward",
        )

        merged = merged.sort_values("orig_idx").reset_index(drop=True)
        result = pd.Series(merged[value_col].values, index=idx, name=value_col)

        # Fill gaps
        result = result.ffill().bfill()

        non_null = result.notna().sum()
        print(f"  [Manual Driver] Loaded {non_null}/{len(idx)} valid rows from {filename}.")

        if result.isna().all():
             return pd.Series(fallback_val, index=idx, name=value_col)

        return result

    except Exception as e:
        print(f"  [Manual Driver] Failed to load {filename}: {e}")
        return pd.Series(fallback_val, index=idx, name=value_col)


# ═══════════════════════════════════════════════════════════════════════════════
# yfinance FETCHER with fallback
# ═══════════════════════════════════════════════════════════════════════════════

def _fetch_yf(ticker: str, start: str, end: str) -> pd.Series | None:
    if not HAS_YF:
        return None
    try:
        df = yf.download(ticker, start=start, end=end,
                         interval="1wk", progress=False, auto_adjust=True)
        if df is None or df.empty:
            return None
        s = df["Close"].squeeze()
        if isinstance(s, pd.DataFrame):
            s = s.iloc[:, 0]
        s.index = pd.to_datetime(s.index).tz_localize(None)
        s = s.dropna()
        if len(s) < 10:
            return None
        return s
    except Exception:
        return None


def _synthetic_driver(idx: pd.DatetimeIndex, base: float,
                      annual_vol: float = 0.18,
                      trend_pct: float = 0.0,
                      cycle_period_wks: int = 104,
                      cycle_amp: float = 0.08,
                      seed: int = 0) -> pd.Series:
    """
    Realistic synthetic time-series: GBM + sinusoidal cycle + regime jumps.
    """
    np.random.seed(seed)
    n     = len(idx)
    dt    = 1 / 52
    sigma = annual_vol * np.sqrt(dt)
    mu    = (trend_pct / 100) * dt

    shocks = np.random.normal(mu, sigma, n)
    jumps  = np.where(np.random.uniform(size=n) < 0.015,
                      np.random.choice([-1, 1], n) * sigma * 3.5, 0)
    log_rets = shocks + jumps

    cycle = cycle_amp * np.sin(2 * np.pi * np.arange(n) / cycle_period_wks)
    log_rets += np.gradient(cycle)

    prices = base * np.exp(np.cumsum(log_rets))
    return pd.Series(prices, index=idx, name="synthetic")


SYNTHETIC_PARAMS = {
    "al_price":          (2300,  0.22, 1.5,  104, 0.10, 1),
    "solar_etf":         (18,    0.30, 8.0,   78, 0.15, 2),
    "semiconductor_etf": (180,   0.28, 6.0,   52, 0.12, 3),
    "coal_energy":       (120,   0.35, 0.5,   78, 0.20, 4),
    "crude_oil":         (75,    0.32,-1.0,   52, 0.18, 5),
    "usd_inr":           (83,    0.07, 1.2,  156, 0.04, 7),
    "china_etf":         (35,    0.25,-1.0,  104, 0.12, 8),
    "bdry_freight":      (12,    0.55,-2.0,   52, 0.25, 9),
    "steel_etf":         (55,    0.28, 0.5,  104, 0.14, 10),
    "vix":               (18,    0.55, 0.0,   26, 0.30, 11),
    "iron_ore":          (110,   0.30,-2.0,   78, 0.18, 12),
    "india_steel":       (800,   0.20, 4.0,  104, 0.10, 14),
}


# ═══════════════════════════════════════════════════════════════════════════════
# DRIVER BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

def _get_driver(label: str, ticker: str, idx: pd.DatetimeIndex,
                fallback_base: float = 100.0) -> pd.Series:
    """Fetch driver from yfinance; fall back to synthetic GBM if unavailable."""
    s = _fetch_yf(ticker, START_DATE, END_DATE)
    if s is not None and not s.dropna().empty:
        out = s.resample("W-FRI").last().ffill().reindex(idx).ffill().bfill()
        if not out.dropna().empty:
            return out

    params = SYNTHETIC_PARAMS.get(label)
    if params:
        base, vol, trend, cyc, amp, seed = params
        return _synthetic_driver(idx, base, vol, trend, cyc, amp, seed)
    return _synthetic_driver(idx, fallback_base, 0.20, 0.0, 104, 0.10,
                             abs(hash(label)) % 100)


def build_si_drivers(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Silicon Metal driver set — energy & custom manual drivers."""
    print("  [Si Drivers] Fetching Silicon Metal drivers...")
    df = pd.DataFrame(index=idx)

    driver_spec = {
        "al_price":          "ALI=F",
        "solar_etf":         "ICLN",
        "semiconductor_etf": "SOXX",
        "coal_energy":       "MTF=F",
        "crude_oil":         "CL=F",
        "usd_inr":           "INR=X",
        "china_etf":         "FXI",
        "bdry_freight":      "BDRY",
        "steel_etf":         "SLX",
        "vix":               "^VIX",
    }
    fallbacks = {"al_price": 2300, "solar_etf": 18, "semiconductor_etf": 180,
                 "coal_energy": 120, "crude_oil": 75, "usd_inr": 83, 
                 "china_etf": 35, "bdry_freight": 12, "steel_etf": 55, "vix": 18}

    # Fetch online drivers
    for label, ticker in driver_spec.items():
        df[label] = _get_driver(label, ticker, idx, fallbacks.get(label, 100))
        src = "online" if HAS_YF else "synthetic"
        print(f"    {label}: {df[label].notna().sum()} valid rows [{src}]")

    # Load Manual Drivers
    df["cny_inr"] = load_manual_driver("cny_inr.xlsx", "Price", idx, fallback_val=11.5)
    df["silica_quartz_index"] = load_manual_driver("silica_quartz_index.xlsx", "Index", idx, fallback_val=100.0)

    return df


def build_fesi_drivers(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """FeSi Alloy driver set — steel cycle & custom manual drivers."""
    print("  [FeSi Drivers] Fetching FeSi Alloy drivers...")
    df = pd.DataFrame(index=idx)

    driver_spec = {
        "steel_etf":    "SLX",
        "iron_ore":     "TIO=F",
        "coal_energy":  "MTF=F",
        "usd_inr":      "INR=X",
        "bdry_freight": "BDRY",
        "al_price":     "ALI=F",
        "india_steel":  "JSWSTEEL.NS",
        "vix":          "^VIX",
    }
    fallbacks = {"steel_etf": 55, "iron_ore": 110, "coal_energy": 120,
                 "usd_inr": 83, "bdry_freight": 12, "al_price": 2300, 
                 "india_steel": 800, "vix": 18}

    # Fetch online drivers
    for label, ticker in driver_spec.items():
        df[label] = _get_driver(label, ticker, idx, fallbacks.get(label, 100))
        src = "online" if HAS_YF else "synthetic"
        print(f"    {label}: {df[label].notna().sum()} valid rows [{src}]")

    # Load Manual Drivers
    df["cny_inr"] = load_manual_driver("cny_inr.xlsx", "Price", idx, fallback_val=11.5)
    df["silica_quartz_index"] = load_manual_driver("silica_quartz_index.xlsx", "Index", idx, fallback_val=100.0)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SHOCK / EVENT DUMMY INJECTION
# ═══════════════════════════════════════════════════════════════════════════════

def apply_shock_dummies(df: pd.DataFrame, alloy_type: str) -> pd.DataFrame:
    for col, (start, end) in SHOCK_EVENTS.items():
        df[col] = ((df.index >= start) & (df.index <= end)).astype(float)
    return df

def _shock_weight(col: str, alloy_type: str) -> float:
    si_amplified   = {"china_power_curbs_2021", "china_export_restrict", "energy_crisis_europe"}
    fesi_amplified = {"steel_downturn_2015", "steel_downturn_2019",
                      "china_steel_curbs", "india_infra_push"}
    if alloy_type == "Si"   and col in si_amplified:   return 1.5
    if alloy_type == "FeSi" and col in fesi_amplified: return 1.5
    return 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TARGET PROXY CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════════

def build_target_proxy(df: pd.DataFrame, alloy_type: str,
                       df_n: pd.DataFrame) -> pd.Series:
    """
    Driver-based target proxy. Weights rebalanced to incorporate Silica Quartz.
    """
    _s = lambda key: df_n.get(key, pd.Series(100.0, index=df.index))

    if alloy_type == "Si":
        # Adjusted coal from 0.35 -> 0.25 to insert silica_quartz_index (0.10)
        proxy = (
            0.25 * _s("coal_energy")         +
            0.20 * _s("al_price")            +
            0.15 * _s("solar_etf")           +
            0.12 * _s("cny_inr")             +
            0.10 * _s("silica_quartz_index") +
            0.08 * _s("crude_oil")           +
            0.05 * _s("semiconductor_etf")   +
            0.05 * _s("china_etf")
        )
    else:  
        # FeSi: Replaced defunct hrc_proxy with silica_quartz_index (0.10)
        proxy = (
            0.40 * _s("steel_etf")           +
            0.20 * _s("iron_ore")            +
            0.20 * _s("coal_energy")         +
            0.10 * _s("silica_quartz_index") +
            0.10 * _s("usd_inr")
        )

    for col in SHOCK_COLS:
        if col in df.columns:
            w = _shock_weight(col, alloy_type)
            proxy = proxy + df[col] * w * 0.5

    return proxy


# ═══════════════════════════════════════════════════════════════════════════════
# RAW DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_raw_dataset(alloy_type: str) -> pd.DataFrame:
    idx = pd.date_range(start=START_DATE, end=END_DATE, freq="W-FRI")
    print(f"  [Dataset] Building {alloy_type} dataset: {len(idx)} weekly periods")

    if alloy_type == "Si":
        raw_drivers = build_si_drivers(idx)
    else:
        raw_drivers = build_fesi_drivers(idx)

    raw_drivers = raw_drivers.ffill().bfill().fillna(100.0)

    base_vals = raw_drivers.iloc[0].replace(0, 1e-9)
    df_n = raw_drivers / base_vals * 100

    proxy = build_target_proxy(raw_drivers, alloy_type, df_n)

    df = raw_drivers.copy()
    df["Target_Proxy"] = proxy.values

    df = apply_shock_dummies(df, alloy_type)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(df: pd.DataFrame, alloy_type: str) -> pd.DataFrame:
    price_cols = [c for c in (SI_PRICE_COLS if alloy_type == "Si" else FESI_PRICE_COLS)
                  if c in df.columns]

    new = {}
    for col in price_cols:
        s = df[col]
        new[f"{col}_ret"] = np.log(s / s.shift(1) + 1e-9)

        for lag in LAG_PERIODS:
            new[f"{col}_lag{lag}"] = s.shift(lag)

        for w in ROLLING_WINS:
            rm = s.rolling(w).mean()
            rs = s.rolling(w).std().replace(0, 1e-9)
            new[f"{col}_rm{w}"]   = rm
            new[f"{col}_rz{w}"]   = (s - rm) / rs
            new[f"{col}_rmax{w}"] = s.rolling(w).max()
            new[f"{col}_rmin{w}"] = s.rolling(w).min()

    if alloy_type == "Si" and "al_price" in df.columns and "coal_energy" in df.columns:
        new["al_coal_ratio"] = df["al_price"] / (df["coal_energy"] + 1e-9)
    if alloy_type == "FeSi" and "steel_etf" in df.columns and "iron_ore" in df.columns:
        new["steel_ore_ratio"] = df["steel_etf"] / (df["iron_ore"] + 1e-9)

    new_df = pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1)
    new_df["month"]        = new_df.index.month
    new_df["quarter"]      = new_df.index.quarter
    new_df["week_of_year"] = new_df.index.isocalendar().week.astype(int)

    for sc in SHOCK_COLS:
        if sc in df.columns:
            new_df[sc] = df[sc]

    return new_df.ffill().bfill().dropna(subset=["Target_Proxy"])


# ═══════════════════════════════════════════════════════════════════════════════
# REGIME PROBABILITY
# ═══════════════════════════════════════════════════════════════════════════════

def compute_regime(df: pd.DataFrame) -> np.ndarray:
    returns   = pd.Series(df["Target_Proxy"].values).pct_change()
    roll_vol  = returns.rolling(8, min_periods=4).std()
    long_mean = roll_vol.rolling(52, min_periods=26).mean()
    long_std  = roll_vol.rolling(52, min_periods=26).std().replace(0, 1e-9)
    z = ((roll_vol - long_mean) / long_std).fillna(0.0)
    regime = (1 / (1 + np.exp(-z.values))).clip(0.05, 0.95)
    return regime


# ═══════════════════════════════════════════════════════════════════════════════
# ANCHOR CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

class PriceCalibrator:
    def __init__(self, alloy_type: str):
        anchor            = ANCHORS[alloy_type]
        self.anchor_date  = anchor["date"]
        self.anchor_price = anchor["price_per_kg"]
        self.scaling_factor = 1.0

    def fit(self, index_series: pd.Series) -> "PriceCalibrator":
        pos = index_series.index.get_indexer([self.anchor_date], method="nearest")[0]
        idx_val = float(index_series.iloc[pos])
        if abs(idx_val) < 1e-9:
            idx_val = 1.0
        self.scaling_factor = self.anchor_price / idx_val
        print(f"  [Calibration] Anchor {self.anchor_date.date()} | "
              f"Target ₹{self.anchor_price}/Kg | Index@anchor={idx_val:.4f} | "
              f"Scale={self.scaling_factor:.6f}")
        return self

    def transform(self, index_series: pd.Series) -> pd.Series:
        return (index_series * self.scaling_factor).rename("real_price")


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_model(X: np.ndarray, y: np.ndarray, feat_cols: list):
    if HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=1200,
            learning_rate=0.025,
            num_leaves=63,
            min_child_samples=10,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1,
        )
        model.fit(X, y)
        preds = model.predict(X)
        importances = dict(zip(feat_cols, model.feature_importances_))
        return model, preds, importances

    if HAS_SKL:
        model = GradientBoostingRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.85,
            random_state=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        importances = dict(zip(feat_cols, model.feature_importances_))
        return model, preds, importances

    if HAS_RIDGE:
        scaler = StandardScaler()
        Xs     = scaler.fit_transform(X)
        model  = Ridge(alpha=1.0)
        model.fit(Xs, y)
        preds = model.predict(Xs)
        imp = np.abs(model.coef_)
        imp = imp / (imp.sum() + 1e-9) * 1000
        importances = dict(zip(feat_cols, imp))
        model._scaler = scaler
        return model, preds, importances

    raise RuntimeError("No ML library available. Install lightgbm or scikit-learn.")


def model_predict(model, X_vec: np.ndarray) -> float:
    if hasattr(model, "_scaler"):
        X_vec = model._scaler.transform(X_vec)
    return float(model.predict(X_vec)[0])


# ═══════════════════════════════════════════════════════════════════════════════
# FUTURE FORECAST — Regime-aware, Volatility-preserving
# ═══════════════════════════════════════════════════════════════════════════════

def generate_future_forecast(
    alloy_type: str,
    df: pd.DataFrame,
    model,
    feat_cols: list,
    calibrator: PriceCalibrator,
    regime_probs: np.ndarray,
    horizon: int,
) -> pd.DataFrame:
    price_cols = [c for c in (SI_PRICE_COLS if alloy_type == "Si" else FESI_PRICE_COLS)
                  if c in df.columns and c != "Target_Proxy"]

    if alloy_type == "Si":
        VOL_MULT     = 1.0
        CYCLE_AMP    = 0.0
        SHOCK_PROB   = 0.04
        SHOCK_MAG    = 2.5
        ML_BLEND     = 0.35
    else:
        VOL_MULT     = 0.70
        CYCLE_AMP    = 0.40
        SHOCK_PROB   = 0.0
        SHOCK_MAG    = 0.0
        ML_BLEND     = 0.40

    last_date        = df.index[-1]
    reversion_target = float(df["Target_Proxy"].tail(104).mean())
    hist_vol         = float(np.std(
        np.log(df["Target_Proxy"] / df["Target_Proxy"].shift(1) + 1e-9).dropna().tail(52)
    ))
    hist_vol = max(hist_vol, 0.005)

    log_rets = np.log(df["Target_Proxy"] / df["Target_Proxy"].shift(1) + 1e-9).dropna().values

    vols   = {c: max(float(np.log(df[c] / df[c].shift(1) + 1e-9).std()), 0.005)
              for c in price_cols if c in df.columns}
    drifts = {c: float(np.log(df[c] / df[c].shift(1) + 1e-9).tail(52).mean())
              for c in price_cols if c in df.columns}

    regime_p  = float(regime_probs[-1])
    current_p = float(df["Target_Proxy"].iloc[-1])

    np.random.seed(42)
    block_size = 8
    boot_blocks = [
        log_rets[i:i+block_size]
        for i in range(0, len(log_rets) - block_size, block_size // 2)
    ]
    if not boot_blocks:
        boot_blocks = [log_rets]
    boot_sequence = np.concatenate(
        [boot_blocks[i % len(boot_blocks)] for i in range((horizon // block_size) + 2)]
    )[:horizon]

    buffer_cols   = price_cols + SHOCK_COLS + ["Target_Proxy"]
    buffer_vals   = {c: df[c].values.copy() if c in df.columns else np.full(len(df), 100.0)
                     for c in buffer_cols}
    buffer_index  = list(df.index)

    rows = []

    for step in range(horizon):
        next_date = last_date + pd.Timedelta(weeks=step + 1)
        next_date = next_date + pd.Timedelta(days=(4 - next_date.weekday()) % 7)

        row = {}
        decay = 0.96 ** step

        for c in price_cols:
            last_val  = float(buffer_vals[c][-1])

            # GUARDRAIL: Extrapolating manual drivers causes systemic instability.
            # We enforce a strict carry-forward (flat line extension) per instructions.
            if c in ["cny_inr", "silica_quartz_index"]:
                row[c] = last_val
            else:
                shock     = np.random.normal(0, vols.get(c, 0.02))
                drift_adj = drifts.get(c, 0.0) * decay
                row[c]    = float(np.clip(last_val * np.exp(drift_adj + shock),
                                          last_val * 0.5, last_val * 2.0))

        row["month"]        = next_date.month
        row["quarter"]      = next_date.quarter
        row["week_of_year"] = next_date.isocalendar()[1]

        for col in price_cols:
            if col not in buffer_vals: continue
            s_arr = buffer_vals[col]
            row[f"{col}_ret"] = float(
                np.log((s_arr[-1] + 1e-9) / (s_arr[-2] + 1e-9))
            ) if len(s_arr) > 1 else 0.0
            for lag in LAG_PERIODS:
                idx_back = max(len(s_arr) - lag, 0)
                row[f"{col}_lag{lag}"] = float(s_arr[idx_back])
            for w in ROLLING_WINS:
                tail   = s_arr[-w:] if len(s_arr) >= w else s_arr
                rm     = float(np.mean(tail))
                rs     = float(np.std(tail)) + 1e-9
                row[f"{col}_rm{w}"]   = rm
                row[f"{col}_rz{w}"]   = (float(s_arr[-1]) - rm) / rs
                row[f"{col}_rmax{w}"] = float(np.max(tail))
                row[f"{col}_rmin{w}"] = float(np.min(tail))

        if alloy_type == "Si":
            row["al_coal_ratio"] = row.get("al_price", 100) / (row.get("coal_energy", 100) + 1e-9)
        else:
            row["steel_ore_ratio"] = row.get("steel_etf", 55) / (row.get("iron_ore", 110) + 1e-9)

        for sc in SHOCK_COLS: row[sc] = 0.0
        if alloy_type == "FeSi": row["india_infra_push"] = 1.0

        vec = np.array([row.get(f, 0.0) for f in feat_cols], dtype=np.float32).reshape(1, -1)
        ml_pred = model_predict(model, vec)

        replay_ret   = float(boot_sequence[step])
        regime_noise = np.random.normal(0, hist_vol * VOL_MULT * (1 + regime_p * 0.8))

        if alloy_type == "Si":
            if SHOCK_PROB > 0 and np.random.uniform() < SHOCK_PROB:
                regime_noise += np.random.choice([-1, 1]) * hist_vol * SHOCK_MAG
            p = current_p * np.exp(replay_ret * 0.6 + regime_noise * 0.4)
        else:
            cycle_pos = np.sin(2 * np.pi * step / 104) * hist_vol * CYCLE_AMP
            p = current_p * np.exp(replay_ret * 0.7 + regime_noise * 0.3 + cycle_pos)

        deviation = (reversion_target - p) / (abs(reversion_target) + 1e-9)
        mean_rev_strength = 0.05 + 0.03 * (step / horizon)
        p = p + mean_rev_strength * deviation * abs(p)

        p = (1 - ML_BLEND) * p + ML_BLEND * ml_pred

        anchor_p = float(df["Target_Proxy"].iloc[-1])
        p = float(np.clip(p, anchor_p * 0.25, anchor_p * 4.0))

        current_p = p * 0.70 + float(df["Target_Proxy"].iloc[-1]) * 0.30
        regime_p = float(np.clip(regime_p * 0.94 + 0.20 * 0.06 + np.random.normal(0, 0.03), 0.05, 0.95))

        out_row = {
            "date":               next_date,
            "predicted_index":    round(p, 5),
            "real_price":         round(p * calibrator.scaling_factor, 4),
            "regime_probability": round(regime_p, 4),
        }
        for dc in price_cols:
            if dc in row: out_row[dc] = round(row[dc], 4)
        rows.append(out_row)

        row["Target_Proxy"] = p
        for c in buffer_cols:
            if c in row:
                buffer_vals[c] = np.append(buffer_vals[c], float(row[c]))
            elif c in SHOCK_COLS:
                buffer_vals[c] = np.append(buffer_vals[c], 0.0)
            else:
                buffer_vals[c] = np.append(buffer_vals[c], buffer_vals[c][-1])
        buffer_index.append(next_date)

    fut_df = pd.DataFrame(rows).set_index("date")
    fut_df.index.name = "date"
    return fut_df


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(alloy_type: str, horizon: int = DEFAULT_HORIZON):
    print(f"\n{'='*70}")
    print(f"  SILICON INTELLIGENCE PIPELINE — {alloy_type}")
    print(f"{'='*70}")

    output_dir = f"./outputs_{alloy_type}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df_raw = build_raw_dataset(alloy_type)
    print(f"  [Dataset] Raw rows: {len(df_raw)}")

    df = engineer_features(df_raw, alloy_type)
    print(f"  [Features] Engineered rows: {len(df)} | columns: {len(df.columns)}")

    if df.empty:
        print("  [FATAL] Empty dataset after feature engineering. Aborting.")
        return

    feat_cols = [c for c in df.columns if c != "Target_Proxy"]
    X = df[feat_cols].values.astype(np.float32)
    y = df["Target_Proxy"].values.astype(np.float32)

    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.nan_to_num(y, nan=float(np.nanmean(y)))

    print(f"  [Model] Training on {X.shape[0]} samples × {X.shape[1]} features...")
    model, preds, importances = train_model(X, y, feat_cols)
    print(f"  [Model] Training complete.")

    regime_probs = compute_regime(df)

    pred_series = pd.Series(preds, index=df.index)
    calibrator  = PriceCalibrator(alloy_type).fit(pred_series)
    hist_real   = calibrator.transform(pred_series)

    price_cols = [c for c in (SI_PRICE_COLS if alloy_type == "Si" else FESI_PRICE_COLS)
                  if c in df.columns and c != "Target_Proxy"]
    hist_dict = {
        "actual":             df["Target_Proxy"],
        "hybrid_prediction":  preds,
        "real_price":         hist_real,
        "regime_probability": regime_probs,
    }
    for col in price_cols:
        hist_dict[col] = df[col]
    hist_df = pd.DataFrame(hist_dict, index=df.index)
    hist_df.index.name = "date"

    market_s = load_market_prices(alloy_type, df.index)
    hist_df["market_price"] = market_s.values

    non_null = hist_df["market_price"].notna().sum()
    print(f"  [Market Validation] Non-null market_price rows: {non_null}")
    if non_null == 0:
        print("  [WARN] No market prices merged — check file path / column names.")

    overlap = hist_df.dropna(subset=["market_price"])
    if len(overlap) > 5:
        rmse = float(np.sqrt(np.mean((overlap["real_price"] - overlap["market_price"]) ** 2)))
        mae  = float(np.mean(np.abs(overlap["real_price"] - overlap["market_price"])))
        mape = float(np.mean(np.abs(
            (overlap["real_price"] - overlap["market_price"]) /
            (overlap["market_price"] + 1e-9)
        )) * 100)
        print(f"  [Accuracy] Market overlap: {len(overlap)} rows | "
              f"RMSE={rmse:.3f} | MAE={mae:.3f} | MAPE={mape:.2f}%")

    print(f"  [Forecast] Generating {horizon}-week future forecast...")
    fut_df = generate_future_forecast(
        alloy_type, df, model, feat_cols, calibrator, regime_probs, horizon
    )

    hist_df.to_csv(f"{output_dir}/historical_predictions.csv")
    fut_df.to_csv(f"{output_dir}/future_forecast.csv")

    fi_df = pd.DataFrame(
        sorted(importances.items(), key=lambda x: x[1], reverse=True),
        columns=["feature", "importance"]
    )
    fi_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)

    meta = {
        "alloy":           alloy_type,
        "scaling_factor":  calibrator.scaling_factor,
        "anchor_date":     str(calibrator.anchor_date.date()),
        "anchor_price_kg": calibrator.anchor_price,
        "horizon_weeks":   horizon,
        "train_rows":      int(len(df)),
        "feature_count":   int(len(feat_cols)),
        "market_overlap":  int(non_null),
    }
    with open(f"{output_dir}/model_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ {alloy_type} pipeline complete — outputs saved to '{output_dir}/'")
    print(f"   historical_predictions.csv : {len(hist_df)} rows")
    print(f"   future_forecast.csv        : {len(fut_df)} rows")
    print(f"   feature_importance.csv     : {len(fi_df)} features")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silicon Intelligence Forecasting Pipeline")
    parser.add_argument("--alloy", type=str, required=True,
                        choices=["Si", "FeSi", "all"],
                        help="Alloy to run: Si | FeSi | all")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help=f"Forecast horizon in weeks (default: {DEFAULT_HORIZON})")
    args = parser.parse_args()

    if args.alloy == "all":
        run_pipeline("Si",   args.horizon)
        run_pipeline("FeSi", args.horizon)
    else:
        run_pipeline(args.alloy, args.horizon)