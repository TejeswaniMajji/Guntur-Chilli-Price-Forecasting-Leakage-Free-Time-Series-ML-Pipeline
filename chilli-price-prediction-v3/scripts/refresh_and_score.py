"""
The one script to run end to end: re-ingest raw data, rebuild features, retrain every
horizon's model with walk-forward-selected best algorithm, and re-score the latest forecasts.

    python scripts/refresh_and_score.py

Wire this to Windows Task Scheduler (see docs/architecture.md#scheduling) for an automated
refresh. Runs in well under a minute at this dataset's size.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.data.ingest import build_daily_panel, DATE_COL
from src.features.build_features import build_supervised_dataset, WEEKLY_HORIZONS, MONTHLY_HORIZONS
from src.models.train import run_full_evaluation, train_final_models
from src.pipelines.scoring import run_scoring


def main():
    t0 = time.time()

    print("[1/4] Ingesting raw data...")
    daily = build_daily_panel(save=True)
    print(f"      {len(daily)} daily rows, {daily[DATE_COL].min().date()} to {daily[DATE_COL].max().date()}")

    print("[2/4] Building leakage-safe multi-horizon feature sets...")
    weekly_sup = build_supervised_dataset(daily, "weekly")
    monthly_sup = build_supervised_dataset(daily, "monthly")
    weekly_sup.to_csv("data/processed/weekly_supervised.csv", index=False)
    monthly_sup.to_csv("data/processed/monthly_supervised.csv", index=False)
    print(f"      weekly: {weekly_sup.shape}, monthly: {monthly_sup.shape}")

    print("[3/4] Walk-forward evaluating naive/ridge/random_forest/hist_gbm per horizon, "
          "then training the best model for each on all available history...")
    summary = run_full_evaluation(weekly_sup, monthly_sup, WEEKLY_HORIZONS, MONTHLY_HORIZONS)
    model_summary = summary[summary.model != "naive"].copy()
    # idxmin() picks the single best (model, feature_pool) row per horizon -- the ablation
    # (domestic_only vs with_trade) is decided on exactly the same walk-forward footing as the
    # model choice itself, not assumed either way.
    best = model_summary.loc[model_summary.groupby(["frequency", "horizon"])["mase_mean"].idxmin()]
    best[["frequency", "horizon", "model", "feature_pool", "mae_mean", "mase_mean", "r2_mean"]] \
        .sort_values(["frequency", "horizon"]).to_csv("reports/metrics/model_selection.csv", index=False)

    selection = {(r.frequency, int(r.horizon)): (r.model, r.feature_pool) for r in best.itertuples()}
    for freq, sup, horizons in [("weekly", weekly_sup, WEEKLY_HORIZONS), ("monthly", monthly_sup, MONTHLY_HORIZONS)]:
        for h in horizons:
            model_name, feature_pool = selection[(freq, h)]
            train_final_models(sup, [h], freq, best_model_name=model_name, feature_pool=feature_pool,
                                top_k_features=15)

    print("[4/4] Scoring latest forecasts...")
    forecasts = run_scoring(save=True)
    print(forecasts[["frequency", "horizon", "forecast_date", "prediction", "reliability"]].to_string(index=False))

    print(f"\nDone in {time.time() - t0:.1f}s. Run scripts/run_diagnostics.py separately if you "
          f"also want refreshed prediction-interval / feature-importance figures (slower, and "
          f"not needed for a routine data refresh).")


if __name__ == "__main__":
    main()
