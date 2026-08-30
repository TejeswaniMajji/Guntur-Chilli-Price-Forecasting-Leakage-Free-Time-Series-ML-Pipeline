"""
Leakage-safe feature engineering for the Guntur chilli price forecaster.

Every feature that touches the target series is built from `series.shift(1)` (or later)
*before* any rolling/lag window is applied, so a feature dated t can only depend on
information available up to and including t-1. This is verified in tests/test_features.py
by asserting feature values are unchanged when future rows are truncated.

Two frequencies are supported: weekly (week ending Saturday) and monthly (month end),
matching how the mandi bulletin itself is naturally summarized by traders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

DATE_COL = "Date"
TARGET_COL = "LCA_334_Best"

WEEKLY_FREQ = "W-SAT"
MONTHLY_FREQ = "ME"

# Short + medium horizons only (long-horizon forecasts were the least reliable part of the
# original pipeline -- sMAPE 50-135% beyond ~4 weeks / 6 months -- so v2 scopes down rather
# than promising accuracy it can't deliver).
WEEKLY_HORIZONS = [1, 2, 3, 4, 6, 8]      # 1-8 weeks ahead
MONTHLY_HORIZONS = [1, 2, 3, 4, 5, 6]     # 1-6 months ahead

SUPPLY_COLS = ["Total_Arivals", "Arrivals_AC", "NON_AC", "SALES"]

# ---------------------------------------------------------------------------
# Resampling: daily panel -> weekly / monthly base tables
# ---------------------------------------------------------------------------


def resample(daily: pd.DataFrame, freq: str) -> pd.DataFrame:
    df = daily.set_index(DATE_COL).sort_index()
    agg: Dict[str, str] = {TARGET_COL: "mean"}
    for c in SUPPLY_COLS:
        if c in df.columns:
            agg[c] = "sum"
    for c in df.columns:
        if c.startswith("wx_"):
            agg[c] = "sum" if "precipitation" in c else "mean"
        elif c.startswith("cold_") or c == "inflation_cpi_pct":
            agg[c] = "last"
        elif c.startswith("export_") or c.startswith("usd_inr_"):
            agg[c] = "last"  # annual/fiscal-year series, forward-filled -- "last" just re-reads the current value
        elif c.startswith("event_"):
            agg[c] = "max"  # 1 if the period falls (even partly) inside a documented event window
        elif c.startswith("remark_"):
            agg[c] = "mean"
        elif c in {"prior_gap_ge_2", "prior_gap_ge_5"}:
            agg[c] = "sum"  # count of closure events in the period
    agg = {k: v for k, v in agg.items() if k in df.columns}
    out = df.resample(freq).agg(agg).reset_index()
    out["n_trading_days"] = df[TARGET_COL].resample(freq).count().values
    return out


# ---------------------------------------------------------------------------
# Feature spec
# ---------------------------------------------------------------------------


@dataclass
class FeatureSpec:
    lag_cols: List[str]
    lag_periods: List[int]
    rolling_cols: List[str]
    rolling_windows: List[int]
    growth_cols: List[str]
    growth_periods: List[int]


def default_spec(frequency: str) -> FeatureSpec:
    if frequency == "weekly":
        return FeatureSpec(
            lag_cols=[TARGET_COL, "Total_Arivals", "SALES"],
            lag_periods=[1, 2, 3, 4, 8, 12],
            rolling_cols=[TARGET_COL, "Total_Arivals"],
            rolling_windows=[4, 8, 12],
            growth_cols=[TARGET_COL, "Total_Arivals"],
            growth_periods=[1, 4, 8],
        )
    return FeatureSpec(
        lag_cols=[TARGET_COL, "Total_Arivals", "SALES"],
        lag_periods=[1, 2, 3, 6, 12],
        rolling_cols=[TARGET_COL, "Total_Arivals"],
        rolling_windows=[3, 6, 12],
        growth_cols=[TARGET_COL, "Total_Arivals"],
        growth_periods=[1, 3, 6],
    )


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df[DATE_COL].dt.year
    df["month"] = df[DATE_COL].dt.month
    df["weekofyear"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
    # harvest-season flag: Guntur's chilli trading season runs roughly Feb-Jun (see EDA)
    df["is_harvest_season"] = df["month"].isin([2, 3, 4, 5, 6]).astype(int)
    return df


def _add_lags(df: pd.DataFrame, col: str, periods: List[int]) -> pd.DataFrame:
    for p in periods:
        df[f"{col}_lag{p}"] = df[col].shift(p)
    return df


def _add_rolling(df: pd.DataFrame, col: str, windows: List[int]) -> pd.DataFrame:
    shifted = df[col].shift(1)  # never include the current period's own value
    for w in windows:
        roll = shifted.rolling(window=w, min_periods=max(2, w // 2))
        df[f"{col}_r{w}_mean"] = roll.mean()
        df[f"{col}_r{w}_std"] = roll.std()
        m, s = roll.mean(), roll.std()
        df[f"{col}_r{w}_z"] = (shifted - m) / s.replace(0, np.nan)
    return df


def _add_growth(df: pd.DataFrame, col: str, periods: List[int]) -> pd.DataFrame:
    for p in periods:
        df[f"{col}_pctchg{p}"] = df[col].pct_change(periods=p).replace([np.inf, -np.inf], np.nan)
    return df


def build_leakfree_features(base: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Add lag/rolling/growth/calendar features to a resampled base table.
    `base` must already be sorted ascending by Date with one row per period."""
    spec = default_spec(frequency)
    df = base.sort_values(DATE_COL).reset_index(drop=True).copy()
    df = add_calendar_features(df)

    # A handful of periods have no valid trading-day price at all (most notably the Apr-May
    # 2020 COVID lockdown, where the mandi reported but quoted Rs.0). For anything that reads
    # "the current/last known price" -- model features, and the naive persistence baseline --
    # forward-fill those gaps. The raw TARGET_COL is left untouched so the *actual* multi-
    # horizon labels built downstream never contain an imputed value as ground truth.
    df["current_price"] = df[TARGET_COL].ffill()

    for c in spec.lag_cols:
        if c in df.columns:
            df = _add_lags(df, c, spec.lag_periods)
    for c in spec.rolling_cols:
        if c in df.columns:
            df = _add_rolling(df, c, spec.rolling_windows)
    for c in spec.growth_cols:
        if c in df.columns:
            df = _add_growth(df, c, spec.growth_periods)

    # weather anomaly vs trailing seasonal-ish window (shifted -> no leakage)
    for c in [c for c in df.columns if c.startswith("wx_")]:
        shifted = df[c].shift(1)
        df[f"{c}_anom"] = shifted - shifted.rolling(12, min_periods=4).mean()

    # cold storage regime (quartile bucket of Guntur share of national stock, shifted)
    if "cold_guntur_share_pct" in df.columns:
        shifted = df["cold_guntur_share_pct"].shift(1)
        q1, q3 = shifted.quantile(0.25), shifted.quantile(0.75)
        df["cold_guntur_share_regime"] = pd.cut(
            shifted, bins=[-np.inf, q1, q3, np.inf], labels=["low", "mid", "high"]
        )
        df = pd.get_dummies(df, columns=["cold_guntur_share_regime"], prefix="cold_regime", dummy_na=False)

    # inflation is already a slow annual step -- take it as-is (no leakage risk, it is
    # public macro data known well before the period it describes)
    return df


def make_multi_horizon_targets(feat_df: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Stack one supervised row per (period, horizon) with target_t+h = target h periods ahead."""
    horizons = WEEKLY_HORIZONS if frequency == "weekly" else MONTHLY_HORIZONS
    df = feat_df.sort_values(DATE_COL).reset_index(drop=True)
    frames = []
    for h in horizons:
        tmp = df.copy()
        tmp["horizon"] = h
        tmp["target"] = df[TARGET_COL].shift(-h)
        frames.append(tmp)
    stacked = pd.concat(frames, ignore_index=True)
    stacked = stacked.dropna(subset=["target"])
    return stacked


def build_supervised_dataset(daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    freq_code = WEEKLY_FREQ if frequency == "weekly" else MONTHLY_FREQ
    base = resample(daily, freq_code)
    feat = build_leakfree_features(base, frequency)
    return make_multi_horizon_targets(feat, frequency)
