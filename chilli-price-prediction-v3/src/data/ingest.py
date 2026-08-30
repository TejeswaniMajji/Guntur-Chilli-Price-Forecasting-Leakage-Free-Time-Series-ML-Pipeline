"""
Data ingestion & cleaning for the Guntur chilli price forecasting system.

Sources (all rooted at data/raw/):
    Chilli.csv                          Daily Guntur mandi bulletin (2012-2025): prices for ~11
                                         chilli varieties x 4 grades, arrivals, sales, free-text
                                         broker remarks.
    weather_data_Dec_31_2024.csv        Daily weather for 5 Andhra Pradesh districts (2019-2024),
                                         explicitly labelled by DISTRICT (preferred over the
                                         lat/lon-only weather_data_Andhra Pradesh.csv).
    Cold_Stock_pavan.csv                Monthly cold-storage stock (2012-2025) by location
                                         (Guntur Local / Non-Local / other states), in lakh bags.
    Inflation_India.xlsx                Annual CPI inflation, India (1960-2024).

Target: LCA_334_Best -- the "Best" grade of LCA-334 (the benchmark Guntur variety), daily
mandi price in Rs/quintal-equivalent units as recorded in the bulletin.

Design notes
------------
* Everything here is leakage-safe by construction: this module only *assembles* a clean daily
  panel. No lag/rolling/target features are created here -- that happens downstream in
  src/features/build_features.py, which is where leakage would actually be introduced if done
  carelessly.
* Trading-day gaps (the mandi doesn't report on Sundays, festivals, and occasional strikes) are
  detected directly from the gaps between consecutive dates in the bulletin itself, rather than
  from a separate holiday-calendar file (none exists / was ever populated in this project).
* The broker remarks column is free text ("Market steady", "Market up by Rs.2/Kg for 334...").
  We derive a small set of sentiment/activity flags from it rather than discarding it.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

DATE_COL = "Date"
TARGET_COL = "LCA_334_Best"
DEFAULT_ENCODING = "latin1"

RAW_DIR = Path("data/raw")
EXTERNAL_DIR = Path("data/external")
PROCESSED_DIR = Path("data/processed")

MONTH_MAP = {m: i for i, m in enumerate(
    ["JANUARY", "FEBRUARY", "MARCH", "APRIL", "MAY", "JUNE", "JULY",
     "AUGUST", "SEPTEMBER", "OCTOBER", "NOVEMBER", "DECEMBER"], start=1
)}
MONTH_ABBR_MAP = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1
)}

QTY_COLS = ["Arrivals_AC", "NON_AC", "New_Crop_Arrivals", "Carry_Farward", "Total_Arivals", "SALES"]

# ---------------------------------------------------------------------------
# 1. Chilli market bulletin (the core / target series)
# ---------------------------------------------------------------------------


def _flatten_multiheader(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten the bulletin's 2-row header (variety on row 1, grade on row 2) into
    single names like LCA_334_Best, 334_TALU_Medium, TEJA_D_Grade, etc."""
    new_cols, last_valid_top, counts = [], "", {}
    for top, sub in df.columns:
        top_s, sub_s = str(top).strip(), str(sub).strip()
        if "Unnamed:" not in top_s and top_s:
            last_valid_top = top_s
            name = f"{top_s}_{sub_s}" if ("Unnamed:" not in sub_s and sub_s) else top_s
        else:
            if "Unnamed:" not in sub_s and sub_s:
                base = last_valid_top if last_valid_top and "Unnamed:" not in last_valid_top else ""
                name = f"{base}_{sub_s}" if base else sub_s
            else:
                name = f"{top_s}_{sub_s}"
        name = (name.replace(" ", "_").replace("-", "_")
                    .replace("(", "").replace(")", "").replace(".", "").strip())
        if name in counts:
            counts[name] += 1
            name = f"{name}_{counts[name]}"
        else:
            counts[name] = 1
        new_cols.append(name)
    df.columns = new_cols
    df = df.rename(columns={c: c.replace("D_Grade_D_Grade", "D_Grade") for c in df.columns if "D_Grade_D_Grade" in c})
    df = df.rename(columns={c: c.replace("TALU_TALU", "TALU") for c in df.columns if "TALU_TALU" in c})
    df = df.rename(columns={c: "NON_AC" for c in df.columns if "NON_A/C" in c or c == "NON_AC"})
    # the free-text remarks column has no real "variety" header above it, so the generic
    # top-header carry-forward logic mislabels it (e.g. "Armour_talu_Remarks") -- restore it
    remark_like = [c for c in df.columns if c.endswith("_Remarks") or c == "Remarks"]
    if remark_like:
        df = df.rename(columns={remark_like[0]: "Remarks"})
    return df.dropna(axis=1, how="all")


def _compose_date(df: pd.DataFrame) -> pd.DataFrame:
    df["Month_Num"] = df["Month"].astype(str).str.upper().str.strip().map(MONTH_MAP)
    df["Day"] = pd.to_numeric(df["Day"], errors="coerce")
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df[DATE_COL] = pd.to_datetime(
        dict(year=df["Year"], month=df["Month_Num"], day=df["Day"]), errors="coerce"
    )
    return df.drop(columns=["Day", "Month", "Year", "Month_Num"])


def _parse_price_range(v) -> float:
    """'48-49' -> 48.5 ; '48' -> 48.0 ; NaN -> NaN."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s == "" or s.lower() in {"nan", "-"}:
        return np.nan
    if "-" in s:
        parts = s.split("-", 1)
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


REMARK_PATTERNS = {
    "remark_steady": r"steady",
    "remark_slow": r"slow",
    "remark_up": r"\bup\b",
    "remark_down": r"\bdown\b",
    "remark_new_crop": r"new crop",
    "remark_has_text": None,  # handled separately
}


def _extract_remark_features(remarks: pd.Series) -> pd.DataFrame:
    r = remarks.fillna("").astype(str).str.lower()
    out = pd.DataFrame(index=remarks.index)
    out["remark_has_text"] = (r.str.len() > 0).astype(int)
    for col, pattern in REMARK_PATTERNS.items():
        if pattern is None:
            continue
        out[col] = r.str.contains(pattern, regex=True, na=False).astype(int)
    # crude Rs/Kg move magnitude e.g. "up by Rs.2/Kg" / "up by Rs2/-"
    move = r.str.extract(r"(?:up|down)\D{0,12}?rs\.?\s*(\d+(?:\.\d+)?)", expand=False)
    magnitude = pd.to_numeric(move, errors="coerce").fillna(0)
    out["remark_move_signed"] = magnitude * np.where(out["remark_up"] == 1, 1, np.where(out["remark_down"] == 1, -1, 0))
    return out


def load_chilli_market(path: Path = RAW_DIR / "Chilli.csv") -> pd.DataFrame:
    """Load, flatten, clean, and date-index the daily Guntur bulletin.

    Returns one row per *unique* trading date (duplicates in the source are averaged),
    sorted ascending, with all price columns as floats (range strings midpoint-parsed)
    and quantity columns as floats with thousands separators removed.
    """
    df = pd.read_csv(path, encoding=DEFAULT_ENCODING, header=[0, 1])
    df = _flatten_multiheader(df)
    df = _compose_date(df)
    df = df.dropna(subset=[DATE_COL])

    remark_col = "Remarks" if "Remarks" in df.columns else None
    remark_feats = _extract_remark_features(df[remark_col]) if remark_col else None

    price_keywords = ["_Deluxe", "_Best", "_Med_Best", "_Medium", "_D_Grade", "_TALU"]
    price_cols = [c for c in df.columns if any(k in c for k in price_keywords)]
    for c in price_cols:
        df[c] = df[c].apply(_parse_price_range)
    # a quoted price of 0 is not economically meaningful for a graded commodity -- it shows
    # up as a ~6-week stretch in Apr-May 2020 (the COVID-19 lockdown, when the mandi was
    # non-functional but the bulletin kept issuing rows) and a handful of other isolated days.
    # Treat as missing rather than a real observation.
    df[price_cols] = df[price_cols].replace(0, np.nan)

    for c in QTY_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", "", regex=False), errors="coerce")

    # merge duplicate "_2" columns produced by repeated variety headers (e.g. Deluxe appears twice)
    for c in [c for c in df.columns if re.search(r"_2$", c)]:
        base = c[:-2]
        if base in df.columns:
            df[base] = df[[base, c]].mean(axis=1)
            df = df.drop(columns=[c])

    keep_cols = [DATE_COL] + price_cols + [c for c in QTY_COLS if c in df.columns]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    if remark_feats is not None:
        df = pd.concat([df, remark_feats], axis=1)

    # Guard against duplicate trading dates in the raw bulletin (rare data-entry errors):
    # average numeric fields, keep max of remark flags.
    n_before = len(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df.groupby(DATE_COL, as_index=False)[numeric_cols].mean()
    n_after = len(df)
    if n_after < n_before:
        df.attrs["n_duplicate_dates_merged"] = n_before - n_after

    df = df.sort_values(DATE_COL).reset_index(drop=True)
    df.attrs["target_col"] = TARGET_COL
    return df


def add_trading_gap_flags(df: pd.DataFrame, date_col: str = DATE_COL) -> pd.DataFrame:
    """Flag non-trading gaps directly from the bulletin's own date sequence, rather than
    from an external holiday calendar (none exists for this market). A 1-day gap is an
    ordinary weekday-to-weekday step; anything longer implies one or more closed days
    (typically Sunday, a festival holiday, or occasionally a market strike)."""
    df = df.sort_values(date_col).reset_index(drop=True)
    gap_days = df[date_col].diff().dt.days
    df["days_since_last_trading_day"] = gap_days.fillna(1)
    df["prior_gap_ge_2"] = (gap_days >= 2).astype("Int64").fillna(0)
    df["prior_gap_ge_5"] = (gap_days >= 5).astype("Int64").fillna(0)  # long closure (>4 non-trading days)
    df["dayofweek"] = df[date_col].dt.dayofweek
    return df


# ---------------------------------------------------------------------------
# 2. Weather (Guntur district)
# ---------------------------------------------------------------------------

def load_weather(path: Path = RAW_DIR / "weather_data_Dec_31_2024.csv", district: str = "Guntur") -> pd.DataFrame:
    w = pd.read_csv(path, encoding=DEFAULT_ENCODING)
    w = w[w["DISTRICT"].astype(str).str.strip().str.lower() == district.lower()].copy()
    w["date"] = pd.to_datetime(w["date"], errors="coerce")
    w = w.dropna(subset=["date"]).drop_duplicates(subset=["date"])
    w = w.rename(columns={"date": DATE_COL})
    keep = [DATE_COL, "temperatures_min", "temperatures_max", "precipitation_amount",
            "solar_amount", "wind_morningMax", "wind_dayMax", "wind_average",
            "relativeHumidity_max", "relativeHumidity_min"]
    w = w[[c for c in keep if c in w.columns]]
    w = w.rename(columns={c: f"wx_{c}" for c in w.columns if c != DATE_COL})
    w["wx_temp_range"] = w["wx_temperatures_max"] - w["wx_temperatures_min"]
    return w.sort_values(DATE_COL).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 3. Cold storage stock (monthly -> forward-filled daily)
# ---------------------------------------------------------------------------

def load_cold_stock(path: Path = RAW_DIR / "Cold_Stock_pavan.csv") -> pd.DataFrame:
    c = pd.read_csv(path, encoding=DEFAULT_ENCODING, low_memory=False)
    named = [col for col in c.columns if not str(col).startswith("Unnamed")]
    c = c[named].copy()
    c["month_num"] = c["Month"].astype(str).str.upper().str.strip().map(MONTH_ABBR_MAP)
    c["Year"] = pd.to_numeric(c["Year"], errors="coerce")
    c = c.dropna(subset=["month_num", "Year"])
    c["period"] = pd.to_datetime(dict(year=c["Year"].astype(int), month=c["month_num"].astype(int), day=1))

    is_guntur_local = c["Location"].astype(str).str.contains("Guntur", na=False)
    guntur = (c[is_guntur_local].groupby("period", as_index=False)["Bags_Lakh"].sum()
                .rename(columns={"Bags_Lakh": "cold_guntur_bags_lakh"}))
    national = (c.groupby("period", as_index=False)["Bags_Lakh"].sum()
                  .rename(columns={"Bags_Lakh": "cold_national_bags_lakh"}))
    merged = guntur.merge(national, on="period", how="outer").sort_values("period")
    merged["cold_guntur_share_pct"] = 100 * merged["cold_guntur_bags_lakh"] / merged["cold_national_bags_lakh"]
    merged = merged.rename(columns={"period": DATE_COL})
    return merged.reset_index(drop=True)


def expand_monthly_to_daily(monthly: pd.DataFrame, daily_index: pd.DatetimeIndex, date_col: str = DATE_COL) -> pd.DataFrame:
    """Forward-fill a monthly series onto a daily calendar so it can be safely merged with
    the daily price panel (no look-ahead: a month's stock figure only becomes 'known' from
    that month's start date onward)."""
    m = monthly.set_index(date_col).sort_index()
    full_daily = pd.DataFrame({date_col: pd.date_range(m.index.min(), daily_index.max(), freq="D")})
    full_daily = full_daily.merge(m.reset_index(), on=date_col, how="left").sort_values(date_col)
    full_daily = full_daily.ffill()
    return full_daily


# ---------------------------------------------------------------------------
# 4. Inflation (annual -> forward-filled daily)
# ---------------------------------------------------------------------------

def load_inflation(path: Path = RAW_DIR / "Inflation_India.xlsx") -> pd.DataFrame:
    x = pd.read_excel(path)
    x.columns = ["year", "inflation_cpi_pct"]
    x["period"] = pd.to_datetime(x["year"].astype(int).astype(str) + "-01-01")
    return x[["period", "inflation_cpi_pct"]].rename(columns={"period": DATE_COL}).sort_values(DATE_COL).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 5. International trade: exports & exchange rate (annual/fiscal-year -> forward-filled daily)
# ---------------------------------------------------------------------------
# LCA-334 is an export-oriented variety (alongside Teja, Sannam), and Guntur is India's largest
# chilli trading hub -- so export demand, and the exchange rate that determines how competitive
# Indian chilli is for foreign buyers, are plausible domestic-price drivers that the original
# version of this project didn't model at all. Granular (monthly, variety-specific) export data
# is not publicly available; this uses the best public annual series found, sourced in each
# CSV's `source` column and cited in full in reports/ANALYSIS_REPORT.md.

def load_export_stats(path: Path = EXTERNAL_DIR / "india_chilli_export_stats.csv") -> pd.DataFrame:
    e = pd.read_csv(path, parse_dates=["period_start"])
    e = e.sort_values("period_start").reset_index(drop=True)
    e["export_value_per_tonne_rs_lakh"] = e["export_value_rs_lakh"] / e["export_qty_tonnes"]
    e["export_qty_yoy_growth_pct"] = e["export_qty_tonnes"].pct_change() * 100
    e["export_value_yoy_growth_pct"] = e["export_value_rs_lakh"].pct_change() * 100
    return e.rename(columns={"period_start": DATE_COL})[
        [DATE_COL, "export_value_rs_lakh", "export_qty_tonnes", "export_value_per_tonne_rs_lakh",
         "export_qty_yoy_growth_pct", "export_value_yoy_growth_pct"]
    ]


def load_exchange_rate(path: Path = EXTERNAL_DIR / "usd_inr_annual_avg.csv") -> pd.DataFrame:
    x = pd.read_csv(path, parse_dates=["period_start"])
    x = x.sort_values("period_start").reset_index(drop=True)
    x["usd_inr_yoy_change_pct"] = x["usd_inr_avg"].pct_change() * 100
    return x.rename(columns={"period_start": DATE_COL})[[DATE_COL, "usd_inr_avg", "usd_inr_yoy_change_pct"]]


def add_trade_event_flags(df: pd.DataFrame, path: Path = EXTERNAL_DIR / "trade_events.csv", date_col: str = DATE_COL) -> pd.DataFrame:
    """One boolean column per documented trade/export event (see data/external/trade_events.csv
    for sources) -- e.g. `event_china_demand_high_price_regime`. These are hand-curated from
    published market reports, not derived from a granular time series, so they mark *whole
    documented periods* rather than trying to quantify magnitude."""
    events = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    df = df.copy()
    for row in events.itertuples():
        col = f"event_{row.event}"
        df[col] = ((df[date_col] >= row.start_date) & (df[date_col] <= row.end_date)).astype(int)
    return df


# ---------------------------------------------------------------------------
# 6. Full merge -> daily base panel
# ---------------------------------------------------------------------------

def build_daily_panel(save: bool = True) -> pd.DataFrame:
    market = load_chilli_market()
    market = add_trading_gap_flags(market)

    weather = load_weather()
    weather_daily = weather  # already daily

    cold = load_cold_stock()
    cold_daily = expand_monthly_to_daily(cold, market[DATE_COL])

    infl = load_inflation()
    infl_daily = expand_monthly_to_daily(infl, market[DATE_COL])

    export = load_export_stats()
    export_daily = expand_monthly_to_daily(export, market[DATE_COL])

    fx = load_exchange_rate()
    fx_daily = expand_monthly_to_daily(fx, market[DATE_COL])

    daily = market.merge(weather_daily, on=DATE_COL, how="left")
    daily = daily.merge(cold_daily, on=DATE_COL, how="left")
    daily = daily.merge(infl_daily, on=DATE_COL, how="left")
    daily = daily.merge(export_daily, on=DATE_COL, how="left")
    daily = daily.merge(fx_daily, on=DATE_COL, how="left")
    daily = add_trade_event_flags(daily)
    daily = daily.sort_values(DATE_COL).reset_index(drop=True)

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        daily.to_csv(PROCESSED_DIR / "chilli_daily_panel.csv", index=False)
    return daily


if __name__ == "__main__":
    d = build_daily_panel(save=True)
    print(f"Daily panel: {d.shape[0]} rows x {d.shape[1]} cols, {d[DATE_COL].min().date()} to {d[DATE_COL].max().date()}")
    print(f"Target ({TARGET_COL}) non-null: {d[TARGET_COL].notna().sum()} / {len(d)}")
