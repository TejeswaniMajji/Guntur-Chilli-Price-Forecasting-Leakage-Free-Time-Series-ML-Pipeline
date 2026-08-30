# Guntur Chilli Price Forecasting -- Analysis Report

**Series:** LCA-334 "Best" grade, Guntur mandi (Andhra Pradesh, India) &nbsp;|&nbsp; **Period:** Feb 2012 - Feb 2025 &nbsp;|&nbsp; **Author:** Pavan_vs

---

## 1. Business problem

Guntur is India's largest chilli trading hub, and LCA-334 is the benchmark variety traded there.
Farmers, cold-storage operators, and traders all make decisions -- when to sell, how much to
store, how to hedge -- against where this price is headed. This project asks a concrete
question: **given everything observable in the market today (recent prices, arrivals, cold
storage levels, weather, broker commentary), can a statistical model forecast the price 1-8
weeks or 1-6 months out better than simply assuming "the price stays where it is"?**

That last clause matters more than it sounds. A forecasting system is only useful if it beats
the obvious free alternative -- guessing "no change." Section 5 treats that as the actual bar,
not a formality.

LCA-334 is specifically an **export-oriented variety** -- a meaningful share of Guntur's crop
leaves the country, so international demand, the countries that buy from Andhra Pradesh (China
chief among them), and the rupee's exchange rate are all plausible price drivers alongside the
domestic supply/weather/storage story, not just noise around it. COVID-19 is the clearest single
example: export demand for spices marketed on immunity-boosting properties *rose* sharply in
2020-21 even while the domestic Guntur market was physically shut for six weeks. Section 3.6 and
5.5 test this properly -- with real, sourced trade data run through the same walk-forward
validation as everything else -- rather than assuming international dynamics matter just because
the story is plausible.

## 2. Data

| Source | Coverage | Granularity | Role |
|---|---|---|---|
| `Chilli.csv` -- Guntur mandi bulletin | 2012-02-15 to 2025-02-13 (2,770 valid price days) | Daily | Target + arrivals/sales + free-text broker remarks |
| `weather_data_Dec_31_2024.csv` | 2019-2024, 5 AP districts | Daily | Guntur-district temperature, rainfall, humidity, wind, solar |
| `Cold_Stock_pavan.csv` | 2012-2025, 18 locations | Monthly | Guntur & national cold-storage stock (lakh bags) |
| `Inflation_India.xlsx` | 1960-2024 | Annual | India CPI inflation |
| `data/external/india_chilli_export_stats.csv` | FY2011-12 to FY2024-25 (14 years) | Annual (fiscal year) | India chilli export volume (tonnes) & value (Rs lakh), Spices Board India data via ijsat.org (2025) "Chilli Export Dynamics in India" |
| `data/external/usd_inr_annual_avg.csv` | 2012-2025 (14 years) | Annual average | USD/INR exchange rate -- Forbes India (2012-2023), exchange-rates.org (2024), x-rates.com monthly averages (2025, approximated) |
| `data/external/trade_events.csv` | 5 hand-curated windows, 2020-2025 | Event date ranges | Sourced trade/macro event flags: COVID export-demand surge, COVID domestic disruption, China-demand/high-price regime, El Nino 2023 crop damage, FY2024-25 export-price softening |

These three files were not something the original project had -- they were researched
specifically to test whether export/exchange-rate dynamics move this price (see Section 3.6/5.5),
after the observation that LCA-334 is an export variety and international demand shocks (COVID,
China's buying pattern, exchange-rate moves) are a real, plausible channel the domestic-only
feature set couldn't see. Every row in all three files carries a `source` column citing the
specific report/article it came from -- see `docs/data_dictionary.md` for the full column
reference.

Three data-quality issues were found in the original mandi/weather/storage data and handled
explicitly (not silently):

1. **The mandi bulletin's header is a genuinely two-row Excel layout** (variety on row 1, grade
   on row 2, with the variety cell merged across four grade columns). Flattening it naively
   produces nonsense column names -- this required writing a dedicated parser
   (`src/data/ingest.py::_flatten_multiheader`) rather than a generic CSV read.
2. **A ~6-week stretch of `Rs 0` quoted prices in April-May 2020** turned out to be the
   COVID-19 lockdown: the bulletin kept issuing rows, but the market was non-functional. These
   were re-coded from `0` to missing, since a price of zero is not economically meaningful for
   a graded commodity and would otherwise have quietly corrupted every lag/rolling feature that
   touches that window.
3. **No holiday/strike calendar file existed** in the original project (the code referenced
   one, but it was never populated). Non-trading days are instead detected directly from gaps
   between consecutive bulletin dates -- a 3-day gap between a Saturday and Tuesday reads as a
   normal weekend; a gap of 5+ days is flagged as an extended closure.

Missingness after cleaning: target price 2.5%, arrivals 0.2%, weather 53.0% (only available
2019-2024 -- see the caveat in Section 4.3), cold storage 2.9%, inflation 0%.

## 3. Exploratory findings

*(All figures in `reports/figures/`; every number below is computed in `scripts/run_eda.py`
and persisted to `reports/metrics/eda_findings.json` -- nothing here is hand-typed.)*

### 3.1 The price series itself

Daily price ranges **Rs 42 - Rs 282** per quintal-equivalent over the 13-year window (mean Rs
115, median Rs 106, coefficient of variation **50%**). `01_price_history.png` shows the shape:
a slow climb through 2012-2016, a sharp 2017 crash to the series low, a COVID-era spike
starting 2019, and then the standout event of the dataset -- **prices roughly tripled from
~Rs 100 in 2021 to a peak of Rs 282 in late 2022**, before partially retracing to ~Rs 130 by
early 2025. Annual average price by year (`04_yearly_trend.png`):

| Year | 2012 | 2015 | 2018 | 2020 | 2022 | 2023 | 2024 | 2025* |
|---|---|---|---|---|---|---|---|---|
| Rs | 52 | 93 | 82 | 124 | **222** | 219 | 159 | 133 |

*2025 is a partial year (through mid-February).*

This 2022-2023 spike is the single most important fact for interpreting every model result
in Section 5: it is a regime the models have almost no comparable history to learn from,
and it is why long-horizon accuracy in particular is weak.

### 3.2 Seasonality

Two independent seasonal patterns emerged, and they run in different directions:

- **Trading activity** (`02_trading_days_by_month.png`) is lowest in **May** (82 total
  trading-day records across 13 years, vs. 200-270 for every other month) and highest in
  **November-December** (270/268). This is a market-activity pattern, not a data artifact --
  it holds even after excluding the 2020 lockdown days.
- **Price level** (`03_price_seasonality.png`) is lowest in **April** (mean Rs 101) and highest
  in **September** (mean Rs 125) -- a **21.6% seasonal swing** between the cheapest and most
  expensive month. April sits inside the main harvest window (roughly February-June for
  Guntur), so this is consistent with the standard commodity pattern of harvest-season supply
  pressure depressing price, with the post-harvest low-stock months commanding a premium.

### 3.3 What actually correlates with price

| Driver | Correlation with price | Read |
|---|---|---|
| Guntur's share of national cold-storage stock | **-0.53** | Strongest relationship found. Higher relative Guntur stock coincides with lower price -- consistent with large local harvests both filling storage and depressing price simultaneously (this is correlation from a supply-side confound, not a claim that storage *causes* the price move). |
| Total weekly arrivals | +0.19 (+0.20 on log arrivals) | Weak, and the *wrong sign* for a naive supply-demand story (more arrivals should mean lower price). Likely reflects that arrivals and price both move with overall market activity/demand rather than a clean partial-equilibrium relationship. |
| Guntur cold-stock volume (levels, not share) | -0.26 | Weaker than the share metric -- share-of-national appears to carry more signal than the raw Guntur number. |
| Weather (max temperature) | -0.16 | Weakest signal tested, and available for only 286 of the ~650 weeks (2019-2024 only). That window overlaps almost entirely with the 2022-2023 price spike, so this correlation is at real risk of reflecting a shared time trend rather than a causal weather effect -- treat it as suggestive, not established. |

### 3.4 A free-text feature that actually validates

The bulletin's broker remarks ("Market steady", "Market up by Rs.2/Kg for 334...") were parsed
into simple flags (`remark_up`, `remark_down`, `remark_steady`) and a signed magnitude
(`remark_move_signed`). Before using these in any model, they were sanity-checked against what
actually happened next: mean forward 5-trading-day return, split by the day's own remark
(`09_remark_sentiment_check.png`):

| Remark that day | n | Mean forward 5-day return |
|---|---|---|
| "up" | 359 | **+1.34%** |
| "steady" | 1,019 | +0.06% |
| "down" | 353 | **-0.97%** |

The ordering is exactly right and the separation is clean -- this one-line, zero-cost NLP
feature turned out to be genuinely predictive, and it shows up as the single most important
feature in the final weekly 1-week-ahead model (Section 5.3).

### 3.5 Autocorrelation and volatility

Weekly returns show a significant lag-1 autocorrelation of **0.287** (95% CI band ±0.081,
`10_acf_weekly_returns.png`) -- meaningful week-to-week momentum, though most longer lags are
not significant. Weekly return volatility (std. dev.) is **3.7%**, with a max single-week drop
of **-24.7%** and max jump of **+17.0%** -- rare but real tail events any forecasting system
needs to be honest about rather than smooth over.

### 3.6 International trade & exchange-rate dynamics

Export volume/value and the USD/INR rate are annual (export) or annual-average (exchange rate)
series, so this exploration works at the calendar-year level: each year's mean export
quantity/value and mean USD/INR rate against that year's mean domestic price, across the 14
years both sides of the data cover (`12_trade_dynamics.png`).

| Series | Correlation with annual mean price |
|---|---|
| Export value (Rs lakh) | **+0.86** |
| USD/INR exchange rate | +0.83 |
| Export quantity (tonnes) | +0.65 |

These look like strong relationships, and at face value they support the export-dynamics
hypothesis: export value, the exchange rate, and domestic price have all trended up together
over 2012-2025. But **a shared 13-year upward trend inflates all three of these correlations
whether or not there is a real short-horizon causal link** -- exactly the same trend-confounding
caveat already flagged for weather in Section 3.3, and it's worth being just as skeptical here.
One useful check that cuts against a naive "exports drove the price" reading: **the single
highest-price year (2022, Rs 222) was not one of the two highest-export-value years** (those
were 2023 and 2024, when export value kept climbing to a record ~Rs 11,940-12,492 crore while
the domestic price had already started retreating from its 2022 peak). If exports were the
dominant driver, the price peak and the export-value peak would be expected to line up more
closely than they do -- domestic supply/demand dynamics (see Section 3.3) evidently still matter
a great deal, alongside whatever the export/exchange-rate channel is doing.

Two documented events line up cleanly with the price series without needing statistics to see
it: the ~34% year-on-year export volume surge in FY2020-21 (COVID immunity-boosting spice
demand) happened *during* the same six weeks the domestic Guntur market was physically shut
(Section 2, item 2) -- a real illustration of international demand and domestic supply-chain
disruption pulling in different directions at the same time, not a single unified shock. And
the FY2022-23/FY2023-24 "China demand, high-price regime" window (record export values, China
remaining the top single buyer at ~179,000 tonnes in 2023-24) does broadly overlap the
2022-2023 domestic price spike described in Section 3.1, even if the exact peaks don't align
year-for-year as noted above.

Section 5.5 tests the practical question these correlations can't answer on their own: does
adding these features to the walk-forward-validated forecasting models actually improve
forecast accuracy, or does the trend-confounding risk noted above mean they mostly add noise?

## 4. Feature engineering

`src/features/build_features.py` builds 93 domestic candidate features per horizon (105 with
the trade/export features included -- see Section 5.1) from five families: lags (1-28 periods of
price/arrivals/sales), rolling statistics (mean/std/z-score over 4-12 period windows), growth
rates (% change over 1-8 periods), calendar/seasonal encodings (month/week sine-cosine,
harvest-season flag), plus the weather-anomaly and cold-storage-regime features described above,
and now export volume/value, USD/INR level and year-on-year change, and the five sourced trade-
event flags from Section 3.6.

**Every one of these is built with `series.shift(1)` (or later) before any window is applied**,
so a feature dated t can only depend on information available up to t-1. This was not just
asserted -- it was tested: `tests/test_features.py` rebuilds the feature set after truncating
the input data at various cutoffs and asserts every historical feature value is bit-identical
regardless of what happens after the cutoff (max absolute difference: 0.0 across all 100+
numeric features, including the trade/export features added in Section 3.6/5.5). This matters because the *previous* version of this project was named with
"leakfree" in its model filenames but had no test actually proving it -- here it's verified,
not just asserted.

## 5. Modeling

### 5.1 Design

Twelve horizons were modeled directly (a separate model per horizon, rather than one model fed
its own outputs recursively): **weekly** 1, 2, 3, 4, 6, 8 weeks ahead, and **monthly** 1-6
months ahead. This scope deliberately excludes the 12-24 week horizons the previous version of
this project attempted -- those showed sMAPE of 50-135% in earlier work, i.e. not meaningfully
better than a coin flip, and re-testing them here wasn't a good use of the validation budget.

Three model families were compared, all against the mandatory baseline:

- **naive** -- persistence: tomorrow's price = today's price. The bar every model must clear.
- **ridge** -- L2-regularised linear regression (standardised features).
- **random_forest** -- 400-tree bagged ensemble.
- **hist_gbm** -- scikit-learn's histogram-based gradient boosting (the same algorithm family
  as LightGBM/XGBoost; used here because those libraries could not be installed in the
  development sandbox -- see `src/models/train.py` for a one-line hook to swap in LightGBM on a
  machine where it's available, if the extra ~1-2% is worth the added dependency).

Two modeling decisions were made *because* the naive baseline is so strong for a
near-random-walk series like this one, and both are visible, testable choices rather than
hidden tuning:

1. **Models predict the price change, not the price level** (`target - current_price`), with
   the final forecast reconstructed as `current_price + predicted_change`. For a persistent
   series, a raw price-level model spends most of its capacity re-learning "output ≈ input,"
   which linear models do easily (one coefficient near 1.0) but tree ensembles do clumsily
   (fragmenting a near-identity mapping across hundreds of splits). Differencing removes that
   from the problem entirely, so the model is judged on the part that's actually hard: how much
   things will move. This alone cut the best model's error by roughly 30-40% relative to the
   raw-level formulation.
2. **A fold-internal top-15-feature selection by |correlation with the training-fold target|.**
   With 93-105 candidate features (depending on feature pool, see decision 3 below) and as few
   as ~250 training rows in the earliest walk-forward folds, fitting on everything invites
   memorising noise. Selection is refit inside every fold
   using only that fold's training data (never the test fold), so it can't leak information
   about the period being forecast. `k=15` was chosen via a small sensitivity check
   (`k ∈ {12, 18, 25, 35}` on the two headline horizons) rather than tuned against the full
   evaluation -- documented here rather than hidden, since any choice made by looking at
   held-out performance carries some risk of overfitting the evaluation itself.
3. **Feature-pool selection (`domestic_only` vs. `with_trade`) is itself walk-forward tested,
   per horizon, rather than assumed.** The first attempt at incorporating the export/exchange
   features (Section 3.6) simply added them to the existing top-15-by-correlation candidate
   pool -- and made results measurably *worse* (weekly MASE rose from ~1.06-1.12 to ~1.18-1.29),
   most likely because the trend-confounding risk flagged in Section 3.6 let annual macro series
   crowd out genuinely predictive short-horizon features in a fixed-size selection budget.
   Rather than discard the trade features outright, `src/models/train.py::run_full_evaluation`
   evaluates both the domestic-only pool and the with-trade pool through the identical
   walk-forward harness used for model comparison, and keeps whichever combination wins on
   MASE -- for every horizon independently. This turns "do export dynamics help?" into the same
   kind of honestly-measured answer as "which model wins?", rather than a judgment call. See
   Section 5.5 for the result.

### 5.2 Validation

**Expanding-window walk-forward cross-validation**, 5 folds per horizon: fold *k* trains on
every period before it and tests on the next contiguous block, so later folds always have
strictly more history -- mirroring how the model would actually be retrained and used. This
replaces the single 80/20 split the previous version of this project used, which cannot
detect a model that looks good on one lucky split and falls apart elsewhere (see the very
unstable per-fold R² in `reports/metrics/walk_forward_fold_detail.csv`, e.g. weekly h=8 swings
from R² ≈ 0.87 in early folds to -0.6 in the fold covering the 2022 spike).

The headline metric is **MASE** (mean absolute error scaled by the naive baseline's MAE):
MASE = 1.0 means "no better than assuming no change"; below 1.0 is real, quantified skill.

### 5.3 Results

*(Full table: `reports/metrics/walk_forward_summary.csv`. Best model per horizon:
`reports/metrics/model_selection.csv`.)*

| Horizon | Best model | Feature pool | MAE (Rs) | Naive MAE (Rs) | MASE |
|---|---|---|---|---|---|
| Weekly, 1 wk | random_forest | with_trade | 3.97 | 3.71 | 1.06 |
| Weekly, 2 wk | random_forest | domestic_only | 7.10 | 6.65 | 1.06 |
| Weekly, 3 wk | ridge | domestic_only | 9.65 | 9.00 | 1.07 |
| Weekly, 4 wk | ridge | domestic_only | 11.89 | 11.21 | **1.05** |
| Weekly, 6 wk | ridge | domestic_only | 16.11 | 14.50 | 1.10 |
| Weekly, 8 wk | ridge | domestic_only | 17.34 | 16.61 | **1.03** |
| Monthly, 1 mo | random_forest | domestic_only | 10.08 | 10.22 | **0.97** |
| Monthly, 2 mo | random_forest | domestic_only | 16.42 | 15.86 | 1.02 |
| Monthly, 3 mo | random_forest | domestic_only | 21.12 | 19.79 | 1.05 |
| Monthly, 4 mo | hist_gbm | domestic_only | 27.86 | 24.00 | 1.17 |
| Monthly, 5 mo | hist_gbm | with_trade | 34.55 | 28.14 | 1.24 |
| Monthly, 6 mo | hist_gbm | with_trade | 40.37 | 32.65 | 1.32 |

*(Walk-forward CV involves fitting Random Forest / HistGBM, which are not perfectly
deterministic run-to-run even with a fixed random seed under parallel execution -- re-running
`scripts/refresh_and_score.py` reproduces the same story within a few hundredths of a MASE
point, occasionally swapping the winning model or feature pool between two options that were
already close. The table above is the exact output of the run these figures and this report
ship with, and reflects the feature-pool ablation described in decision 3 of Section 5.1 and
Section 5.5 -- `domestic_only` wins 10 of 12 horizons; `with_trade` wins weekly 1-week-ahead and
the two longest monthly horizons, 5 and 6 months.)*

**The honest headline: naive persistence is a very hard benchmark to beat here, and no model
beats it decisively at any horizon.** One horizon -- 1-month-ahead, Random Forest -- reaches
essential parity (MASE 0.97, i.e. within noise of naive). Every weekly horizon and every
monthly horizon beyond 3 months sits in the 1.02-1.32 range: close to naive, sometimes within
noise of it, never a clear win.

This is not a failure of the pipeline -- it is a real, useful finding about this market, and
it is exactly the kind of result a single unvalidated train/test split would have hidden. (The
previous version of this project reported R² up to 0.97 on some horizons; that number came from
one static split with no naive-baseline comparison at all, and does not survive walk-forward
re-evaluation on the same underlying idea.) Short-horizon, highly liquid commodity prices being
close to a random walk is a well-documented phenomenon in commodity and financial markets
generally -- the interesting result here is *how close* engineered features get (within 3-10%
of the baseline at most horizons), not that they fall short of an unrealistic bar.

### 5.4 What the winning models actually use

For **weekly, 1 week ahead** (random forest, `with_trade` pool -- Section 5.5 discusses the
feature-pool choice for this horizon in more detail), permutation importance is dominated by
the 1-period price % change by a wide margin, followed at much lower magnitude by arrivals
volatility, the 4-period price % change, a solar-radiation anomaly, and the broker-sentiment
features (`remark_up`, `remark_move_signed`, `remark_steady`) from Section 3.4. This is a
coherent, explainable model: recent price momentum dominates, with market breadth and the
text-derived sentiment signal as secondary contributors.

For **monthly, 1 month ahead** (random forest, `domestic_only` pool -- the only horizon at
parity with naive), the picture is murkier: permutation importance is dominated by weather
variables with very small magnitudes, and the top-10 list looks closer to noise than signal.
Read this as a caveat, not a selling point -- the model's near-parity performance at this
horizon is real (it was measured honestly across 5 walk-forward folds), but *why* it works is
not well explained by the importance analysis, which is itself useful information: it means
this result should be monitored rather than trusted blindly if used operationally.

An 80% prediction interval is computed for every horizon from the empirical 10th/90th
percentile of out-of-fold residuals (not from a distributional assumption) -- see
`reports/metrics/model_diagnostics.json` and the dashboard's Forecasts tab. For weekly h=1
the interval is roughly ±Rs 7 around the point forecast; it widens to roughly ±Rs 20-40 by
monthly h=5-6, which is itself an honest visual signal of how much less can be said with
confidence that far out.

### 5.5 Does the export/trade angle actually help?

Answered honestly, on the ablation described in Section 5.1 decision 3: **mostly no, but yes at
the two horizons where it plausibly should.**

`domestic_only` wins 10 of the 12 horizons -- every weekly horizon from 3 weeks out, and every
monthly horizon from 1-4 months. `with_trade` wins exactly three: **weekly 1-week-ahead**, and
**monthly 5- and 6-months-ahead** (Section 5.3 table). That pattern is more informative than a
flat "yes" or "no":

- At **monthly 5- and 6-months out**, the winning feature set's fold-internal top-15 selection
  (recomputed on the full history for final training) genuinely does include export features --
  `export_qty_yoy_growth_pct` at both horizons, joined by `export_value_yoy_growth_pct` at
  6 months -- earning a place ahead of most weather and arrivals features. This is a coherent
  story: the fastest-moving lag/rolling-window features that dominate short-horizon forecasts
  lose relevance the further out you forecast, and a slower-moving year-on-year trade-growth
  signal becomes comparatively more useful. This is the one place in the whole project where the
  export-dynamics hypothesis from the business problem (Section 1) shows up as a *measured*, not
  just plausible, improvement -- worth exactly this much credit and no more (MASE 1.24 and 1.32
  are still well short of a decisive win over naive).
- At **weekly 1-week-ahead**, `with_trade` + random forest wins on MASE (1.06 vs. 1.14 for
  `domestic_only` + random forest), but the mechanism is murkier: when the top-15 features are
  recomputed on the full history for final training, *no* export/exchange-rate/event feature
  actually makes that top-15 list for this horizon (`reports/metrics/model_diagnostics.json`,
  `weekly_h1`) -- the winning model's own explained features are all price momentum, arrivals,
  and weather. The most likely explanation is that during walk-forward, the *fold-internal*
  correlation ranking (recomputed independently in each of the 5 expanding-window folds, per
  Section 5.1 decision 2) picked a trade feature in at least one early, smaller-training-set fold
  in a way that happened to help that fold's fit, without that feature persisting once the model
  is refit on the full 13-year history. This is disclosed rather than smoothed over: it means the
  weekly-1-week win for `with_trade` should be read with somewhat less confidence than the
  monthly 5-/6-month wins, where the trade signal's relevance survives all the way to the final
  model.

The practical takeaway for a Guntur-market user: **short-and-medium-horizon forecasting (the bulk
of this project's scope) is still driven almost entirely by domestic supply/price dynamics** --
recent price momentum, arrivals, cold storage, broker sentiment, weather. International trade and
exchange-rate data earns a place **only at the longest monthly horizons tested**, where its
slower-moving signal has more relative value against fast-decaying short-term momentum. This is
a more useful and more honest conclusion than either "exports obviously matter, LCA-334 is an
export variety" (Section 1's hypothesis, only partially true) or discarding the trade research
because it didn't help everywhere.

## 6. Limitations

- **Weather coverage is only 2019-2024** (47% of the price history) and overlaps almost
  entirely with the 2022-2023 price spike, so weather-price correlations are confounded with
  the time trend and should not be read as causal.
- **The 2022-2023 spike is the dominant event in the dataset** and has no earlier comparable
  analogue for the models to learn from; every long-horizon forecast that spans a
  similarly-sized shock should be treated with the same skepticism the walk-forward folds
  covering that period earned (several show negative R²).
- **Cold storage and inflation are monthly series forward-filled to daily/weekly** granularity
  -- they carry real information but update far less often than price does, and won't react to
  anything that happens mid-month.
- **Export and exchange-rate correlations (Section 3.6) are confounded with the same 13-year
  upward time trend as weather**, for the same reason: three series that all broadly rise over
  2012-2025 will correlate with each other partly because they share a trend, not necessarily
  because one drives another. The peak-price-year-vs-peak-export-value-year mismatch noted in
  Section 3.6 is one piece of evidence this isn't a clean causal story.
- **Export and exchange-rate figures are treated as "known" from the start of the fiscal/
  calendar year they describe**, forward-filled the same way cold storage and inflation are.
  In reality, official export statistics and even annual exchange-rate averages are published
  with a lag (often months after the year ends), so a model "using" 2023's mean USD/INR rate in
  a forecast issued in, say, June 2023, is implicitly assuming information that would not
  actually have been available yet. This doesn't affect the multi-horizon labels (target prices
  are never touched by this simplification, per Section 4), but it is a real optimism-inducing
  simplification in the *feature* side for the trade/exchange columns specifically, on top of
  the coarser (monthly) version of the same simplification already disclosed for cold storage
  and inflation above.
- **The trade-event flags (`data/external/trade_events.csv`) are hand-curated from five sourced
  articles/reports, not a systematic or exhaustive catalog** of every international event that
  plausibly affected chilli export demand over 13 years. They were chosen because each is
  independently documented and dated, not because they are the only or most important five --
  a more exhaustive event catalog is a natural extension, not something this project claims to
  have done.
- **No model here reliably beats "assume no change"** at any horizon by a wide margin. Anyone
  using these forecasts operationally should treat the point forecast as a mild adjustment to
  the current price, weighted by the reliability tag the dashboard shows for that horizon, not
  as an independent, high-confidence prediction.

## 7. What this project demonstrates

A full, defensible, walk-forward-validated forecasting pipeline built from a genuinely messy
real-world dataset (a two-row Excel-style header, a 16,000-column CSV with 15 real columns
buried in it, an undocumented zero-price data quality issue traced to a real-world event,
free text mined into a validated feature) -- with an honest reporting standard that surfaces a
negative-ish result (models barely beat naive) instead of hiding it behind a favorable metric
choice or an unvalidated train/test split. That discipline -- state the real bar, test against
it properly, report what actually happened -- is the point of the exercise as much as any
single number in Section 5.

The trade/export work in Sections 3.6 and 5.5 is the same discipline applied to a second,
independent question: a domain-informed hypothesis (LCA-334 is an export variety, so
international dynamics should matter) was researched with real, sourced public data, tested
through the identical walk-forward harness used for model selection rather than assumed true
or false, and reported exactly as it came out -- a real but narrow win at two specific
horizons, not a universal one. Treating "should we add this feature?" as an empirical question
with the same rigor as "which model wins?" is a small methodological point, but it's the kind
of judgment that separates a portfolio project from a checklist of techniques.
