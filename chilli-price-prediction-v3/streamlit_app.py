"""
Guntur Chilli (LCA-334 'Best') Price Forecast -- interactive dashboard.

Run locally with:  streamlit run streamlit_app.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.pipelines.scoring import run_scoring
from src.utils.viz import apply_style, CATEGORICAL, INK_MUTED

st.set_page_config(page_title="Chilli Price Forecast", layout="wide", page_icon="\U0001F336️")
apply_style()

METRICS_DIR = Path("reports/metrics")
FIG_DIR = Path("reports/figures")


@st.cache_data(ttl=3600)
def load_forecasts():
    path = Path("reports/forecasts/latest_forecasts.csv")
    if path.exists():
        return pd.read_csv(path, parse_dates=["forecast_date"])
    return run_scoring(save=True)


@st.cache_data
def load_walk_forward_summary():
    return pd.read_csv(METRICS_DIR / "walk_forward_summary.csv")


@st.cache_data
def load_model_selection():
    path = METRICS_DIR / "model_selection.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


@st.cache_data
def load_eda_findings():
    path = METRICS_DIR / "eda_findings.json"
    return json.loads(path.read_text()) if path.exists() else {}


st.title("\U0001F336️ Guntur Chilli Price Forecast")
st.caption("LCA-334 'Best' grade, Guntur mandi -- multi-horizon forecasts with walk-forward validated accuracy.")

tab_forecast, tab_accuracy, tab_explore, tab_about = st.tabs(
    ["Forecasts", "Model Accuracy", "Data Explorer", "About"]
)

# ---------------------------------------------------------------------------
with tab_forecast:
    forecasts = load_forecasts()
    current_price = forecasts["current_price"].iloc[0]
    st.metric("Latest known price", f"Rs {current_price:,.1f}")

    freq_choice = st.radio("Frequency", ["weekly", "monthly"], horizontal=True)
    fc = forecasts[forecasts["frequency"] == freq_choice].sort_values("horizon")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.axhline(current_price, color=INK_MUTED, linewidth=1, linestyle="--", label="Current price")
    ax.plot(fc["forecast_date"], fc["prediction"], marker="o", color=CATEGORICAL[0], label="Forecast")
    if fc["pi_80_low"].notna().any():
        ax.fill_between(fc["forecast_date"], fc["pi_80_low"], fc["pi_80_high"],
                         color=CATEGORICAL[0], alpha=0.15, label="80% interval")
    ax.set_ylabel("Rs / quintal-equiv.")
    ax.legend()
    st.pyplot(fig)

    st.dataframe(
        fc[["horizon", "forecast_date", "prediction", "pi_80_low", "pi_80_high", "model", "reliability"]]
        .rename(columns={"pi_80_low": "80% low", "pi_80_high": "80% high"})
        .style.format({"prediction": "{:.1f}", "80% low": "{:.1f}", "80% high": "{:.1f}"}),
        use_container_width=True, hide_index=True,
    )
    st.info(
        "**Reliability** is read from walk-forward validated MASE against a naive "
        "'price stays the same' baseline. 'At parity' or 'close to naive' means the model is "
        "roughly as good as assuming no change -- treat those numbers as informed estimates, "
        "not high-confidence predictions. See the Model Accuracy tab for the full picture."
    )

# ---------------------------------------------------------------------------
with tab_accuracy:
    st.subheader("Walk-forward validated accuracy, by horizon")
    st.caption(
        "Every model was compared against a naive persistence baseline using expanding-window "
        "walk-forward cross-validation (5 folds per horizon, never trained on future data). "
        "MASE below 1.0 means the model beats naive; above 1.0 means naive still wins."
    )
    summary = load_walk_forward_summary()
    selection = load_model_selection()
    freq_choice2 = st.radio("Frequency ", ["weekly", "monthly"], horizontal=True, key="acc_freq")

    has_pools = "feature_pool" in summary.columns
    if has_pools:
        pool_choice = st.radio(
            "Feature pool", ["Winning pool per horizon (what's actually shipped)", "domestic_only", "with_trade"],
            horizontal=True, key="acc_pool",
            help="LCA-334 is an export variety, so export volume/value, USD/INR, and sourced trade-event "
                 "features were tested as an additional candidate pool alongside the domestic-only features "
                 "(price/arrivals/weather/cold-storage/broker-remarks). Walk-forward CV decides which pool "
                 "wins per horizon -- see reports/ANALYSIS_REPORT.md §5.5.",
        )
        if pool_choice.startswith("Winning"):
            sel_freq = selection[selection["frequency"] == freq_choice2] if not selection.empty else pd.DataFrame()
            winning_pool_by_h = dict(zip(sel_freq["horizon"], sel_freq["feature_pool"])) if not sel_freq.empty else {}
            sub = summary[(summary["frequency"] == freq_choice2) & (summary["model"] != "naive")]
            sub = sub[sub.apply(lambda r: r["feature_pool"] == winning_pool_by_h.get(r["horizon"], "with_trade"), axis=1)]
        else:
            sub = summary[(summary["frequency"] == freq_choice2) & (summary["model"] != "naive")
                          & (summary["feature_pool"] == pool_choice)]
    else:
        sub = summary[(summary["frequency"] == freq_choice2) & (summary["model"] != "naive")]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, model in enumerate(sub["model"].unique()):
        m = sub[sub["model"] == model].sort_values("horizon")
        ax.plot(m["horizon"], m["mase_mean"], marker="o", label=model, color=CATEGORICAL[i])
    ax.axhline(1.0, color=INK_MUTED, linewidth=1.2, linestyle="--", label="naive baseline (MASE=1)")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MASE (lower is better)")
    ax.legend()
    st.pyplot(fig)

    table_cols = ["horizon", "model", "mae_mean", "mase_mean", "r2_mean"]
    if has_pools:
        table_cols.insert(2, "feature_pool")
    st.dataframe(
        sub[table_cols].sort_values(["horizon", "mase_mean"])
        .style.format({"mae_mean": "{:.2f}", "mase_mean": "{:.2f}", "r2_mean": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )

    if has_pools and not selection.empty:
        st.caption(
            "Does the export/trade angle actually help? Best-model MASE by feature pool, per horizon "
            "(lower wins; winner in **bold**) -- see reports/ANALYSIS_REPORT.md §5.5 for the full read."
        )
        ablation = (
            summary[(summary["frequency"] == freq_choice2) & (summary["model"] != "naive")]
            .groupby(["horizon", "feature_pool"])["mase_mean"].min().unstack("feature_pool")
        )
        if {"domestic_only", "with_trade"}.issubset(ablation.columns):
            def _bold_winner(row):
                if row["domestic_only"] <= row["with_trade"]:
                    return pd.Series({"domestic_only": f"**{row['domestic_only']:.2f}**",
                                       "with_trade": f"{row['with_trade']:.2f}"})
                return pd.Series({"domestic_only": f"{row['domestic_only']:.2f}",
                                   "with_trade": f"**{row['with_trade']:.2f}**"})
            st.dataframe(ablation.apply(_bold_winner, axis=1), use_container_width=True)

    st.subheader("Diagnostics for the two headline horizons")
    c1, c2 = st.columns(2)
    with c1:
        st.image(str(FIG_DIR / "11_weekly_h1_feature_importance.png"), caption="Weekly h=1 top features")
        st.image(str(FIG_DIR / "12_weekly_h1_actual_vs_predicted.png"))
    with c2:
        st.image(str(FIG_DIR / "14_monthly_h1_feature_importance.png"), caption="Monthly h=1 top features")
        st.image(str(FIG_DIR / "15_monthly_h1_actual_vs_predicted.png"))

# ---------------------------------------------------------------------------
with tab_explore:
    st.subheader("Exploratory analysis")
    findings = load_eda_findings()
    figs = sorted(FIG_DIR.glob("0*.png")) + sorted(FIG_DIR.glob("12_*.png"))
    captions = {
        "01": "13 years of daily price history", "02": "Trading-day activity by month",
        "03": "Price seasonality by month", "04": "Annual average price trend",
        "05": "Rolling 12-week volatility", "06": "Arrivals vs price",
        "07": "Weather correlation with price (2019-2024)", "08": "Cold storage vs price",
        "09": "Broker-remark sentiment validation", "10": "Autocorrelation of weekly returns",
        "12": "Price vs export value & USD/INR, by year (trade dynamics -- see report §3.6)",
    }
    cols = st.columns(2)
    for i, fig_path in enumerate(figs):
        key = fig_path.stem[:2]
        with cols[i % 2]:
            st.image(str(fig_path), caption=captions.get(key, fig_path.stem))

    if findings:
        with st.expander("Raw EDA findings (JSON)"):
            st.json(findings)

# ---------------------------------------------------------------------------
with tab_about:
    st.markdown(Path("docs/model_card.md").read_text() if Path("docs/model_card.md").exists() else "See docs/model_card.md")
