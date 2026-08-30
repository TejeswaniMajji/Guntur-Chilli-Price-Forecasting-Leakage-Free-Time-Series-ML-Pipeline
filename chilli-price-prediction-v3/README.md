# Guntur Chilli Price Forecasting

An end-to-end forecasting system for the LCA-334 "Best" grade chilli price at the Guntur
mandi (Andhra Pradesh, India) -- 13 years of daily trading data, walk-forward validated
multi-horizon models, and an interactive dashboard.

**Start here:** [`reports/ANALYSIS_REPORT.md`](reports/ANALYSIS_REPORT.md) is the full
write-up -- business framing, data-quality issues found and fixed, exploratory findings,
modeling methodology, and an honest account of what the models do and don't achieve.

## What's in this project

| Question | Answer |
|---|---|
| What are we predicting? | LCA-334 "Best" grade price, Guntur mandi, in Rs / quintal-equivalent |
| What horizons? | Weekly: 1, 2, 3, 4, 6, 8 weeks ahead. Monthly: 1-6 months ahead |
| What data feeds it? | Guntur mandi bulletin (price, arrivals, broker remarks), Guntur-district weather, cold-storage stock levels, national CPI inflation, and (since LCA-334 is an export variety) India chilli export volume/value, USD/INR exchange rate, and sourced trade-event windows |
| How good is it? | Walk-forward validated against a naive "price stays the same" baseline. Best results are near parity with naive, not a decisive win -- see the report for why that's the honest finding, not a shortcoming to hide |
| How is it served? | A Streamlit dashboard, a small FastAPI prediction service, and a script wired for scheduled local retraining |

## Project structure

```
data/                  raw/external/processed data (raw = original bulletins/exports; external = researched export/exchange-rate/trade-event data)
src/
  data/ingest.py        cleans & merges all raw + external sources into one daily panel
  features/build_features.py   leakage-safe lag/rolling/growth/calendar features + multi-horizon targets
  models/train.py        walk-forward CV, model comparison, final model training
  evaluation/metrics.py  MAE / RMSE / sMAPE / MASE / R2
  pipelines/scoring.py    shared production scoring logic (used by the dashboard, API, and scheduler)
  utils/viz.py            shared chart styling
scripts/
  run_eda.py              regenerates reports/figures/*.png + reports/metrics/eda_findings.json
  run_diagnostics.py       feature importance, actual-vs-predicted, prediction intervals
  refresh_and_score.py    the one script to re-run end to end (ingest -> features -> train -> score)
api/main.py             FastAPI prediction service
streamlit_app.py        the dashboard
docs/                  architecture, data dictionary, model card
reports/                ANALYSIS_REPORT.md, figures, metrics, forecasts
tests/                  leakage tests + pipeline smoke tests
```

## Running it locally

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 1. Rebuild data, features, models, and forecasts from scratch:
python scripts/refresh_and_score.py

# 2. Explore the dashboard:
streamlit run streamlit_app.py

# 3. Or serve predictions over HTTP:
uvicorn api.main:app --reload
```

### Keeping forecasts current

`scripts/refresh_and_score.py` re-ingests the raw data, rebuilds features, retrains all 12
horizon models, and re-scores -- runs in well under a minute on this dataset size. Wire it to
Windows Task Scheduler for an automated daily/weekly refresh (see
`docs/architecture.md#scheduling` for the exact steps); no cloud hosting is required since the
whole system runs from your local Python environment.

## Notes on the model stack

Model training uses scikit-learn's `HistGradientBoostingRegressor` (a LightGBM-style
histogram-based booster) alongside Ridge regression and Random Forest, rather than requiring
LightGBM/XGBoost as hard dependencies -- see `src/models/train.py` for a one-line hook to add
either if you have them installed and want to compare.

## License

MIT.
