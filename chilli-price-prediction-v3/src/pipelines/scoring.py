"""Production scoring: load the latest data, rebuild features, and produce forecasts from the
already-trained per-horizon model artifacts. Shared by the Streamlit dashboard, the API, and
the scheduled refresh script so all three ever agree with each other."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd

from src.data.ingest import build_daily_panel, DATE_COL, TARGET_COL
from src.features.build_features import (
    resample, build_leakfree_features, WEEKLY_FREQ, MONTHLY_FREQ,
    WEEKLY_HORIZONS, MONTHLY_HORIZONS,
)

ARTIFACTS_DIR = Path("models/artifacts")
FORECAST_DIR = Path("reports/forecasts")
METRICS_DIR = Path("reports/metrics")


def _load_diagnostics() -> dict:
    path = METRICS_DIR / "model_diagnostics.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _load_model_selection() -> pd.DataFrame:
    path = METRICS_DIR / "model_selection.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def score_frequency(feat_df: pd.DataFrame, frequency: str, horizons: List[int], diagnostics: dict) -> pd.DataFrame:
    last_row = feat_df.iloc[-1]
    last_date = feat_df[DATE_COL].iloc[-1]
    current_price = last_row["current_price"]
    records = []
    for h in horizons:
        path = ARTIFACTS_DIR / f"{frequency}_h{h}.pkl"
        if not path.exists():
            continue
        bundle = joblib.load(path)
        feats = bundle["features"]
        row = last_row.reindex(feats)
        if row.isna().any():
            row = row.fillna(feat_df[feats].median())
        pred_raw = bundle["model"].predict(row.to_frame().T)[0]
        pred_level = pred_raw + current_price if bundle.get("predict_delta", True) else pred_raw

        key = f"{frequency}_h{h}"
        diag = diagnostics.get(key, {})
        offsets = diag.get("prediction_interval_offsets_80pct")
        if offsets:
            lower = pred_level + offsets["q10"]
            upper = pred_level + offsets["q90"]
        else:
            lower = upper = np.nan

        forecast_date = last_date + (pd.Timedelta(weeks=h) if frequency == "weekly" else pd.DateOffset(months=h))
        records.append({
            "frequency": frequency, "horizon": h, "forecast_date": forecast_date,
            "current_price": float(current_price), "prediction": float(pred_level),
            "pi_80_low": float(lower) if lower == lower else None,
            "pi_80_high": float(upper) if upper == upper else None,
            "model": bundle.get("model_name"),
        })
    return pd.DataFrame(records)


def run_scoring(save: bool = True) -> pd.DataFrame:
    daily = pd.read_csv("data/processed/chilli_daily_panel.csv", parse_dates=[DATE_COL]) \
        if Path("data/processed/chilli_daily_panel.csv").exists() else build_daily_panel(save=True)

    weekly_base = resample(daily, WEEKLY_FREQ)
    monthly_base = resample(daily, MONTHLY_FREQ)
    weekly_feat = build_leakfree_features(weekly_base, "weekly")
    monthly_feat = build_leakfree_features(monthly_base, "monthly")

    diagnostics = _load_diagnostics()
    weekly_fc = score_frequency(weekly_feat, "weekly", WEEKLY_HORIZONS, diagnostics)
    monthly_fc = score_frequency(monthly_feat, "monthly", MONTHLY_HORIZONS, diagnostics)
    forecasts = pd.concat([weekly_fc, monthly_fc], ignore_index=True)

    # attach a reliability tag from the validated walk-forward MASE, so nothing is presented
    # with more confidence than the evaluation actually earned
    sel = _load_model_selection()
    if not sel.empty:
        forecasts = forecasts.merge(sel[["frequency", "horizon", "mase_mean"]], on=["frequency", "horizon"], how="left")
        forecasts["reliability"] = np.select(
            [forecasts["mase_mean"] <= 1.05, forecasts["mase_mean"] <= 1.20],
            ["at parity with naive persistence", "close to naive persistence"],
            default="naive persistence still wins -- use with caution",
        )

    if save:
        FORECAST_DIR.mkdir(parents=True, exist_ok=True)
        forecasts.to_csv(FORECAST_DIR / "latest_forecasts.csv", index=False)
    return forecasts


if __name__ == "__main__":
    fc = run_scoring(save=True)
    print(fc.to_string(index=False))
