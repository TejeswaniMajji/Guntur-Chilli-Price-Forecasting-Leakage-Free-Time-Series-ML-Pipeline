# Architecture

## Pipeline

```
data/raw/*      ──►  src/data/ingest.py       ──►  data/processed/chilli_daily_panel.csv
data/external/*      (clean, merge, gap/event        (112 columns -- domestic + trade/exchange)
                       flags, remark sentiment,
                       export/exchange-rate merge)

daily panel ──►  src/features/build_features.py  ──►  weekly_supervised.csv / monthly_supervised.csv
                  (resample, leak-safe lag/            (93-105 features x 12 horizons, stacked)
                   rolling/growth/calendar features,
                   multi-horizon target construction)

supervised  ──►  src/models/train.py          ──►  models/artifacts/{freq}_h{horizon}.pkl
  datasets         (walk-forward CV across            reports/metrics/walk_forward_summary.csv
                    naive/ridge/RF/hist_gbm             reports/metrics/model_selection.csv
                    x {domestic_only, with_trade}
                    feature pools, per-horizon
                    best (model, pool) pick,
                    final refit on all history)

models      ──►  src/pipelines/scoring.py      ──►  reports/forecasts/latest_forecasts.csv
                  (rebuild latest features,
                   load each horizon's model,
                   attach 80% prediction intervals)
                        │
                        ├──► streamlit_app.py        (interactive dashboard)
                        ├──► api/main.py              (FastAPI prediction service)
                        └──► scripts/refresh_and_score.py  (the one script that runs everything above)
```

Two supporting scripts sit outside the main data flow and only need to be re-run when you want
to regenerate the EDA/diagnostic artifacts, not every scoring pass:

- `scripts/run_eda.py` -- exploratory figures + `reports/metrics/eda_findings.json`.
- `scripts/run_diagnostics.py` -- feature importance, actual-vs-predicted, and the empirical
  prediction-interval offsets stored in `reports/metrics/model_diagnostics.json` (consumed by
  `scoring.py`).

## Design choices worth knowing about

- **Direct multi-horizon models, not recursive.** Each horizon gets its own model trained on
  its own shifted target, rather than one model whose own output is fed back in as next
  period's input. Direct models don't compound error the way recursive ones do; the cost is
  training more models, which is cheap at this data size.
- **Leakage safety is tested, not just asserted** -- see `tests/test_features.py`. Every
  target-derived feature uses `.shift(1)` or later before any window function is applied.
- **Models predict price *change* from the current price, not the raw level** -- see
  `reports/ANALYSIS_REPORT.md` §5.1 for the reasoning; this is the single most impactful
  modeling decision in the project.
- **Fold-internal feature selection** (top-15 by correlation with the training fold's target)
  keeps the feature:row ratio sane in the earliest walk-forward folds, without ever looking at
  the test fold to choose features.
- **Feature-pool ablation (`domestic_only` vs. `with_trade`) is walk-forward tested per horizon,
  not assumed.** `run_full_evaluation` in `src/models/train.py` runs every (model, feature pool)
  combination through the same walk-forward harness and keeps whichever wins on MASE for each
  horizon independently -- see `reports/ANALYSIS_REPORT.md` §5.1/§5.5 for why (adding the trade
  features directly to a single undifferentiated pool made results worse) and what the result
  actually was (trade features win only at weekly h=1 and monthly h=5/6).
- **No LightGBM/XGBoost hard dependency.** `HistGradientBoostingRegressor` from scikit-learn
  fills that role so the project has no compiled external ML dependency; `src/models/train.py`
  has a commented one-line hook to add LightGBM back in if you have it installed locally.

## Scheduling {#scheduling}

To keep forecasts current without any cloud hosting, wire `scripts/refresh_and_score.py` to
Windows Task Scheduler:

1. Open Task Scheduler → *Create Basic Task*.
2. Trigger: daily or weekly, whatever matches how often the mandi bulletin actually updates.
3. Action → *Start a program*:
   - Program/script: the full path to `venv\Scripts\python.exe` inside this project.
   - Arguments: `scripts\refresh_and_score.py`
   - Start in: the project's root folder.
4. The script re-ingests `data/raw/`, rebuilds features, retrains all 12 models, and writes a
   fresh `reports/forecasts/latest_forecasts.csv` -- the dashboard and API both read that file
   directly, so no restart is needed for them to pick up new numbers (the dashboard caches for
   1 hour; refresh the page or wait it out).

## Serving

- **Dashboard** (`streamlit_app.py`): `streamlit run streamlit_app.py` → http://localhost:8501
- **API** (`api/main.py`): `uvicorn api.main:app --reload` → http://localhost:8000/docs for the
  interactive OpenAPI page. Both read the same `reports/forecasts/latest_forecasts.csv` that
  `scoring.py` produces, so they can never disagree with each other.
