# Model Card

## Overview

Twelve models forecast the Guntur LCA-334 "Best" grade chilli price at fixed horizons: weekly
1, 2, 3, 4, 6, 8 weeks ahead, and monthly 1-6 months ahead. Each horizon has its own model,
chosen from {Ridge regression, Random Forest, scikit-learn HistGradientBoostingRegressor} and
its own **feature pool** (`domestic_only` vs. `with_trade` -- whether export/exchange-rate/
trade-event features from `data/external/` are candidates), based on which (model, feature
pool) combination won walk-forward cross-validation for that specific horizon (see
`reports/metrics/model_selection.csv`). LCA-334 is an export-oriented variety, so the trade
features were researched to test whether international demand/exchange-rate dynamics improve
forecasts; `reports/ANALYSIS_REPORT.md` §5.5 reports the honest, per-horizon result rather than
assuming they help.

## Intended use

Informing (not automating) decisions by farmers, cold-storage operators, and traders about when
to sell or store LCA-334-grade chilli from the Guntur market. Forecasts should be read as a
**mild adjustment to the current price**, weighted by the reliability tag shown for that
horizon -- not as an independent, high-confidence prediction. See
`reports/ANALYSIS_REPORT.md` for the full reasoning.

## Performance (walk-forward validated, 5 expanding-window folds per horizon)

| Horizon | Model | Feature pool | MAE (Rs) | MASE vs. naive |
|---|---|---|---|---|
| Weekly 1 wk | random_forest | with_trade | 3.97 | 1.06 |
| Weekly 2 wk | random_forest | domestic_only | 7.10 | 1.06 |
| Weekly 3 wk | ridge | domestic_only | 9.65 | 1.07 |
| Weekly 4 wk | ridge | domestic_only | 11.89 | 1.05 |
| Weekly 6 wk | ridge | domestic_only | 16.11 | 1.10 |
| Weekly 8 wk | ridge | domestic_only | 17.34 | 1.03 |
| Monthly 1 mo | random_forest | domestic_only | 10.08 | **0.97** |
| Monthly 2 mo | random_forest | domestic_only | 16.42 | 1.02 |
| Monthly 3 mo | random_forest | domestic_only | 21.12 | 1.05 |
| Monthly 4 mo | hist_gbm | domestic_only | 27.86 | 1.17 |
| Monthly 5 mo | hist_gbm | with_trade | 34.55 | 1.24 |
| Monthly 6 mo | hist_gbm | with_trade | 40.37 | 1.32 |

MASE = model MAE / naive-persistence MAE. 1.0 means no better than "assume the price doesn't
change"; only the 1-month horizon crosses below that line, and only slightly. `domestic_only`
wins 10 of 12 horizons; `with_trade` wins weekly 1-week-ahead and the two longest monthly
horizons (5, 6 months) -- see `reports/ANALYSIS_REPORT.md` §5.5 for the honest read on why, and
§5.3 for why this is reported as a finding rather than hidden.

## Training data

`data/processed/weekly_supervised.csv` and `monthly_supervised.csv`, built from the Guntur
mandi bulletin (2012-2025), Guntur-district weather (2019-2024 only), monthly cold-storage
stock (2012-2025), annual CPI inflation, and (new) India chilli export volume/value
(FY2011-12-2024-25), USD/INR annual average (2012-2025), and five sourced trade-event windows
(all forward-filled -- see `docs/data_dictionary.md`). 93 domestic candidate features per
horizon (105 including trade features), reduced to the top 15 by fold-internal correlation with
the training target before fitting (see `docs/architecture.md`).

## Limitations

- No horizon beats naive persistence by a wide margin; most are within 2-17% of it.
- Weather features only cover 2019-2024, a window that overlaps almost entirely with the
  2022-2023 price spike -- weather-price relationships found in EDA may partly reflect that
  shared trend rather than a clean causal effect. The same trend-confounding risk applies to the
  export-value and USD/INR correlations in the EDA (report §3.6).
- The 2022-2023 price spike (Guntur price roughly tripled) has no earlier comparable analogue
  in the training data; models were measurably worse in walk-forward folds covering that period
  (several show negative R²).
- Monthly 1-month-ahead is the only horizon at parity with naive, and its feature-importance
  profile is noisy (weather-dominated, small magnitudes) rather than clearly explainable --
  treat that result as worth monitoring, not as a proven, stable edge.
- Cold storage, inflation, export, and exchange-rate figures update monthly/annually and are
  forward-filled to daily granularity; they cannot react to anything that happens mid-period,
  and are treated as "known" from the start of the period they describe even though real-world
  publication lags mean that's optimistic (report §6).
- The trade-event flags are a hand-curated set of five sourced events, not an exhaustive
  catalog (report §6).

## Retraining

`scripts/refresh_and_score.py` re-ingests raw data, rebuilds features, and retrains all 12
models from scratch (runs in well under a minute at this data size). See
`docs/architecture.md#scheduling` for wiring it to Windows Task Scheduler.
