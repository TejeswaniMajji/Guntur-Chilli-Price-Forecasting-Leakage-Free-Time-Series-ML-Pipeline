"""
Exploratory analysis for the Guntur chilli (LCA-334 'Best' grade) price series.

Produces:
  reports/figures/*.png          -- charts referenced in the analysis report
  reports/metrics/eda_findings.json -- every number quoted in the written report,
                                        computed here so nothing in the report is hand-typed.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.ingest import build_daily_panel, TARGET_COL, DATE_COL
from src.features.build_features import resample, WEEKLY_FREQ, MONTHLY_FREQ
from src.utils.viz import apply_style, CATEGORICAL, SEQ_BLUE, INK_SECONDARY, INK_MUTED, GRID

apply_style()
FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = Path("reports/metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

findings = {}

daily = pd.read_csv("data/processed/chilli_daily_panel.csv", parse_dates=[DATE_COL])
weekly = resample(daily, WEEKLY_FREQ)
monthly = resample(daily, MONTHLY_FREQ)

# ---------------------------------------------------------------------------
# 1. Full price history
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(daily[DATE_COL], daily[TARGET_COL], color=CATEGORICAL[0], linewidth=0.9)
ax.set_title("Guntur LCA-334 'Best' grade price, 2012-2025")
ax.set_ylabel("Rs / quintal-equiv.")
fig.tight_layout()
fig.savefig(FIG_DIR / "01_price_history.png")
plt.close(fig)

findings["price_overview"] = {
    "n_daily_obs": int(daily[TARGET_COL].notna().sum()),
    "date_range": [str(daily[DATE_COL].min().date()), str(daily[DATE_COL].max().date())],
    "min": float(daily[TARGET_COL].min()),
    "max": float(daily[TARGET_COL].max()),
    "mean": float(daily[TARGET_COL].mean()),
    "median": float(daily[TARGET_COL].median()),
    "std": float(daily[TARGET_COL].std()),
    "cv_pct": float(100 * daily[TARGET_COL].std() / daily[TARGET_COL].mean()),
}

# ---------------------------------------------------------------------------
# 2. Trading calendar / seasonality of market activity
# ---------------------------------------------------------------------------
daily["month"] = daily[DATE_COL].dt.month
daily["year"] = daily[DATE_COL].dt.year
trading_days_by_month = daily.groupby("month")[TARGET_COL].count()

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(trading_days_by_month.index, trading_days_by_month.values, color=CATEGORICAL[0], width=0.65)
ax.set_title("Total trading-day records by calendar month (2012-2025)")
ax.set_xlabel("Month")
ax.set_ylabel("Trading-day records")
ax.set_xticks(range(1, 13))
fig.tight_layout()
fig.savefig(FIG_DIR / "02_trading_days_by_month.png")
plt.close(fig)

findings["trading_calendar"] = {
    "trading_days_by_month": trading_days_by_month.to_dict(),
    "peak_month": int(trading_days_by_month.idxmax()),
    "trough_month": int(trading_days_by_month.idxmin()),
}

# ---------------------------------------------------------------------------
# 3. Seasonality of price by month
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
month_groups = [daily.loc[daily["month"] == m, TARGET_COL].dropna().values for m in range(1, 13)]
bp = ax.boxplot(month_groups, positions=range(1, 13), widths=0.6, patch_artist=True,
                 medianprops=dict(color=CATEGORICAL[7], linewidth=1.6),
                 boxprops=dict(facecolor=SEQ_BLUE[1], edgecolor=INK_SECONDARY, linewidth=0.9),
                 whiskerprops=dict(color=INK_MUTED), capprops=dict(color=INK_MUTED),
                 flierprops=dict(markersize=2, markerfacecolor=INK_MUTED, markeredgecolor="none"))
ax.set_title("Price distribution by calendar month (seasonality)")
ax.set_xlabel("Month")
ax.set_ylabel("Rs / quintal-equiv.")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_price_seasonality.png")
plt.close(fig)

monthly_price_stats = daily.groupby("month")[TARGET_COL].agg(["mean", "median", "std"])
findings["price_seasonality"] = {
    "monthly_mean": monthly_price_stats["mean"].round(2).to_dict(),
    "cheapest_month": int(monthly_price_stats["mean"].idxmin()),
    "most_expensive_month": int(monthly_price_stats["mean"].idxmax()),
    "seasonal_swing_pct": float(100 * (monthly_price_stats["mean"].max() - monthly_price_stats["mean"].min()) / monthly_price_stats["mean"].mean()),
}

# ---------------------------------------------------------------------------
# 4. Year-over-year trend + volatility
# ---------------------------------------------------------------------------
yearly = daily.groupby("year")[TARGET_COL].mean().dropna()
yearly = yearly[(daily.groupby("year")[TARGET_COL].count() >= 20)]  # drop partial first/last years
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(yearly.index, yearly.values, marker="o", color=CATEGORICAL[0], linewidth=1.6, markersize=4)
ax.set_title("Average annual price, by year")
ax.set_ylabel("Rs / quintal-equiv.")
fig.tight_layout()
fig.savefig(FIG_DIR / "04_yearly_trend.png")
plt.close(fig)

weekly_ret = weekly[TARGET_COL].pct_change()
rolling_vol = weekly_ret.rolling(12).std() * 100
fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(weekly[DATE_COL], rolling_vol, color=CATEGORICAL[1], linewidth=1.0)
ax.set_title("Rolling 12-week volatility of weekly price changes")
ax.set_ylabel("Std. dev. of weekly % change")
fig.tight_layout()
fig.savefig(FIG_DIR / "05_rolling_volatility.png")
plt.close(fig)

findings["trend_and_volatility"] = {
    "yearly_mean_price": {int(k): round(float(v), 2) for k, v in yearly.items()},
    "weekly_return_std_pct": float(weekly_ret.std() * 100),
    "weekly_return_mean_pct": float(weekly_ret.mean() * 100),
    "max_weekly_drop_pct": float(weekly_ret.min() * 100),
    "max_weekly_jump_pct": float(weekly_ret.max() * 100),
}

# ---------------------------------------------------------------------------
# 5. Arrivals vs price relationship
# ---------------------------------------------------------------------------
sub = weekly.dropna(subset=[TARGET_COL, "Total_Arivals"])
sub = sub[sub["Total_Arivals"] > 0]
corr_arrivals = sub[TARGET_COL].corr(sub["Total_Arivals"])
corr_log_arrivals = sub[TARGET_COL].corr(np.log1p(sub["Total_Arivals"]))

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(sub["Total_Arivals"], sub[TARGET_COL], s=10, alpha=0.45, color=CATEGORICAL[0], edgecolors="none")
ax.set_title(f"Weekly arrivals vs. price  (r = {corr_arrivals:.2f})")
ax.set_xlabel("Total weekly arrivals")
ax.set_ylabel("Price (Rs / quintal-equiv.)")
fig.tight_layout()
fig.savefig(FIG_DIR / "06_arrivals_vs_price.png")
plt.close(fig)

findings["supply_relationship"] = {
    "corr_price_arrivals": float(corr_arrivals),
    "corr_price_log_arrivals": float(corr_log_arrivals),
}

# ---------------------------------------------------------------------------
# 6. Weather correlation (2019-2024 subset only)
# ---------------------------------------------------------------------------
wx_cols = [c for c in weekly.columns if c.startswith("wx_") and "anom" not in c]
wx_sub = weekly.dropna(subset=[TARGET_COL] + wx_cols)
wx_corr = wx_sub[[TARGET_COL] + wx_cols].corr()[TARGET_COL].drop(TARGET_COL).sort_values()

fig, ax = plt.subplots(figsize=(8, 5))
colors = [CATEGORICAL[7] if v < 0 else CATEGORICAL[0] for v in wx_corr.values]
ax.barh(wx_corr.index.str.replace("wx_", ""), wx_corr.values, color=colors)
ax.axvline(0, color=INK_MUTED, linewidth=0.8)
ax.set_title(f"Correlation of weekly weather with price (n={len(wx_sub)} weeks, 2019-2024)")
ax.set_xlabel("Pearson r")
fig.tight_layout()
fig.savefig(FIG_DIR / "07_weather_correlation.png")
plt.close(fig)

findings["weather_relationship"] = {
    "n_weeks_with_weather": int(len(wx_sub)),
    "correlations": {k.replace("wx_", ""): round(float(v), 3) for k, v in wx_corr.items()},
}

# ---------------------------------------------------------------------------
# 7. Cold storage relationship
# ---------------------------------------------------------------------------
cold_sub = weekly.dropna(subset=[TARGET_COL, "cold_guntur_share_pct"])
corr_cold = cold_sub[TARGET_COL].corr(cold_sub["cold_guntur_share_pct"])
corr_cold_bags = weekly.dropna(subset=[TARGET_COL, "cold_guntur_bags_lakh"])[TARGET_COL].corr(
    weekly.dropna(subset=[TARGET_COL, "cold_guntur_bags_lakh"])["cold_guntur_bags_lakh"]
)

fig, ax = plt.subplots(figsize=(7, 5.5))
ax.scatter(cold_sub["cold_guntur_share_pct"], cold_sub[TARGET_COL], s=10, alpha=0.45, color=CATEGORICAL[2], edgecolors="none")
ax.set_title(f"Guntur share of national cold-stock vs price  (r = {corr_cold:.2f})")
ax.set_xlabel("Guntur share of national cold storage (%)")
ax.set_ylabel("Price")
fig.tight_layout()
fig.savefig(FIG_DIR / "08_coldstock_vs_price.png")
plt.close(fig)

findings["cold_storage_relationship"] = {
    "corr_price_guntur_share_pct": float(corr_cold),
    "corr_price_guntur_bags": float(corr_cold_bags),
}

# ---------------------------------------------------------------------------
# 8. Broker-remark sentiment sanity check
# ---------------------------------------------------------------------------
d2 = daily.copy()
d2["fwd_ret_5d"] = d2[TARGET_COL].shift(-5) / d2[TARGET_COL] - 1
up_ret = d2.loc[d2["remark_up"] == 1, "fwd_ret_5d"].dropna()
down_ret = d2.loc[d2["remark_down"] == 1, "fwd_ret_5d"].dropna()
steady_ret = d2.loc[(d2["remark_up"] == 0) & (d2["remark_down"] == 0) & (d2["remark_steady"] == 1), "fwd_ret_5d"].dropna()

findings["remark_sentiment_validation"] = {
    "n_up_days": int(len(up_ret)), "mean_fwd_5d_return_after_up_pct": float(up_ret.mean() * 100),
    "n_down_days": int(len(down_ret)), "mean_fwd_5d_return_after_down_pct": float(down_ret.mean() * 100),
    "n_steady_days": int(len(steady_ret)), "mean_fwd_5d_return_after_steady_pct": float(steady_ret.mean() * 100),
}

fig, ax = plt.subplots(figsize=(8, 4.5))
cats = ["Remark: 'up'", "Remark: 'steady'", "Remark: 'down'"]
vals = [up_ret.mean() * 100, steady_ret.mean() * 100, down_ret.mean() * 100]
colors_b = [CATEGORICAL[0], INK_MUTED, CATEGORICAL[7]]
ax.bar(cats, vals, color=colors_b, width=0.55)
ax.axhline(0, color=INK_MUTED, linewidth=0.8)
ax.set_title("Forward 5-day return by same-day broker remark")
ax.set_ylabel("Mean forward 5-day return (%)")
fig.tight_layout()
fig.savefig(FIG_DIR / "09_remark_sentiment_check.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# 9. Missingness
# ---------------------------------------------------------------------------
key_cols = [TARGET_COL, "Total_Arivals", "wx_temperatures_max", "cold_guntur_bags_lakh", "inflation_cpi_pct"]
missing_pct = (daily[key_cols].isna().mean() * 100).round(2)
findings["missingness_pct"] = missing_pct.to_dict()

# ---------------------------------------------------------------------------
# 10. Autocorrelation of weekly price changes (manual ACF, no statsmodels available)
# ---------------------------------------------------------------------------
def acf(series: pd.Series, nlags: int) -> np.ndarray:
    s = series.dropna().values
    s = s - s.mean()
    n = len(s)
    denom = (s ** 2).sum()
    return np.array([1.0] + [np.dot(s[:-k], s[k:]) / denom for k in range(1, nlags + 1)])

lags = 20
acf_vals = acf(weekly_ret, lags)
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(acf_vals)), acf_vals, color=CATEGORICAL[0], width=0.5)
ci = 1.96 / np.sqrt(weekly_ret.dropna().shape[0])
ax.axhline(ci, color=INK_MUTED, linewidth=0.8, linestyle="--")
ax.axhline(-ci, color=INK_MUTED, linewidth=0.8, linestyle="--")
ax.set_title("Autocorrelation of weekly price % change (95% CI band)")
ax.set_xlabel("Lag (weeks)")
fig.tight_layout()
fig.savefig(FIG_DIR / "10_acf_weekly_returns.png")
plt.close(fig)

findings["autocorrelation"] = {
    "weekly_return_acf_lag1_8": [round(float(v), 3) for v in acf_vals[1:9]],
    "significant_lags_95pct": [int(i) for i, v in enumerate(acf_vals) if i > 0 and abs(v) > ci],
}

# ---------------------------------------------------------------------------
# 11. International trade & exchange-rate dynamics (export volume/value, USD/INR)
#
# LCA-334 is predominantly an export variety, so international demand and the rupee's
# exchange rate are plausible price drivers alongside the domestic supply/weather/storage
# factors above. The export and exchange-rate series are annual/fiscal-year resolution
# (see data/external/*.csv), so this section works at the yearly level rather than
# weekly/daily -- correlating each calendar year's mean domestic price against that year's
# export volume, export value, and USD/INR rate, plus a visual check of whether documented
# trade events (COVID export surge, China-demand era, export-price softening) line up with
# the price series.
# ---------------------------------------------------------------------------
trade_yearly = daily.groupby("year").agg(
    price_mean=(TARGET_COL, "mean"),
    export_qty_tonnes=("export_qty_tonnes", "mean"),
    export_value_rs_lakh=("export_value_rs_lakh", "mean"),
    usd_inr_avg=("usd_inr_avg", "mean"),
).dropna()
trade_yearly = trade_yearly[(daily.groupby("year")[TARGET_COL].count() >= 20)]

corr_export_qty = trade_yearly["price_mean"].corr(trade_yearly["export_qty_tonnes"])
corr_export_value = trade_yearly["price_mean"].corr(trade_yearly["export_value_rs_lakh"])
corr_usd_inr = trade_yearly["price_mean"].corr(trade_yearly["usd_inr_avg"])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axes[0]
ax2 = ax.twinx()
ax.plot(trade_yearly.index, trade_yearly["price_mean"], color=CATEGORICAL[0], marker="o", markersize=4, linewidth=1.6, label="Domestic price (Rs, LHS)")
ax2.plot(trade_yearly.index, trade_yearly["export_value_rs_lakh"], color=CATEGORICAL[2], marker="s", markersize=4, linewidth=1.6, linestyle="--", label="Export value (Rs lakh, RHS)")
ax.set_title(f"Domestic price vs India chilli export value, by year (r = {corr_export_value:.2f})")
ax.set_ylabel("Rs / quintal-equiv.", color=CATEGORICAL[0])
ax2.set_ylabel("Export value (Rs lakh)", color=CATEGORICAL[2])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8, frameon=False)

ax = axes[1]
ax2 = ax.twinx()
ax.plot(trade_yearly.index, trade_yearly["price_mean"], color=CATEGORICAL[0], marker="o", markersize=4, linewidth=1.6, label="Domestic price (Rs, LHS)")
ax2.plot(trade_yearly.index, trade_yearly["usd_inr_avg"], color=CATEGORICAL[7], marker="^", markersize=4, linewidth=1.6, linestyle="--", label="USD/INR (RHS)")
ax.set_title(f"Domestic price vs USD/INR exchange rate, by year (r = {corr_usd_inr:.2f})")
ax.set_ylabel("Rs / quintal-equiv.", color=CATEGORICAL[0])
ax2.set_ylabel("USD/INR", color=CATEGORICAL[7])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(FIG_DIR / "12_trade_dynamics.png")
plt.close(fig)

# China-demand / high-price-regime alignment check: is the FY2022-23-FY2023-24 price peak
# concurrent with the two highest export-value years on record?
top2_export_years = trade_yearly["export_value_rs_lakh"].sort_values(ascending=False).head(2).index.tolist()
peak_price_year = int(trade_yearly["price_mean"].idxmax())

findings["trade_dynamics"] = {
    "note": "Export/exchange-rate series are annual (export) or annual-average (USD/INR); correlated here "
            "against each calendar year's mean domestic price. See data/external/*.csv for sourcing and "
            "reports/ANALYSIS_REPORT.md Limitations for the fiscal-year-vs-calendar-year and known-from-date "
            "simplifications this implies.",
    "n_years": int(len(trade_yearly)),
    "corr_price_export_qty": float(corr_export_qty),
    "corr_price_export_value": float(corr_export_value),
    "corr_price_usd_inr": float(corr_usd_inr),
    "peak_price_year": peak_price_year,
    "top2_export_value_years": [int(y) for y in top2_export_years],
    "peak_price_year_in_top2_export_years": peak_price_year in top2_export_years,
    "yearly_series": {
        int(y): {
            "price_mean": round(float(r.price_mean), 2),
            "export_qty_tonnes": round(float(r.export_qty_tonnes), 1),
            "export_value_rs_lakh": round(float(r.export_value_rs_lakh), 1),
            "usd_inr_avg": round(float(r.usd_inr_avg), 2),
        }
        for y, r in trade_yearly.iterrows()
    },
}

with open(METRICS_DIR / "eda_findings.json", "w") as f:
    json.dump(findings, f, indent=2, default=str)

print("EDA complete.")
print(json.dumps(findings, indent=2, default=str)[:3000])
