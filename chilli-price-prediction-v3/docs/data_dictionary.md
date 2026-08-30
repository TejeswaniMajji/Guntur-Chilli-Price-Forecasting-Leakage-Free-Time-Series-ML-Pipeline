# Data Dictionary

## Raw sources (`data/raw/`)

### `Chilli.csv` -- Guntur mandi daily bulletin
Two-row header (variety, then grade). Parsed by `src/data/ingest.py::load_chilli_market`.
Key columns after flattening:

| Column | Meaning |
|---|---|
| `LCA_334_Best` | **Target.** Daily price, LCA-334 variety, "Best" grade, Rs/quintal-equiv. A `0` value has been re-coded to missing (see report §2, item 2 -- these were the Apr-May 2020 COVID lockdown days). |
| `LCA_334_Deluxe` / `_Med_Best` / `_Medium` | Other grades of the same variety, kept for context (not modeled). |
| `273_*`, `INDAM_*`, `TEJA_*`, `US_341_*`, `DD_*`, `GUN_BYD_355_*`, `GUN_BYD_5531_*`, `4884_*`, `Armour_*` | Other chilli varieties traded at the same mandi, each with the same grade breakdown. |
| `*_D_Grade`, `*_TALU` | Lower-grade / processed-form price series for several varieties. |
| `Arrivals_AC`, `NON_AC` | Air-cooled vs. non-air-cooled storage arrivals that day. |
| `New_Crop_Arrivals`, `Carry_Farward`, `Total_Arivals`, `SALES` | Supply-side volume figures (thousands-separators stripped, cast to numeric). |
| `Remarks` | Free-text broker commentary. Parsed into `remark_up`, `remark_down`, `remark_steady`, `remark_new_crop`, `remark_has_text`, `remark_move_signed` (signed Rs/Kg move magnitude when stated). |

### `weather_data_Dec_31_2024.csv` -- daily weather, 5 Andhra Pradesh districts
Filtered to `DISTRICT == "Guntur"` (2,192 of 10,960 rows). Columns prefixed `wx_` after
ingestion: `temperatures_min/max`, `precipitation_amount`, `solar_amount`,
`wind_morningMax/dayMax/average`, `relativeHumidity_max/min`, plus a derived `wx_temp_range`.
**Coverage: 2019-01-01 to 2024-12-31 only** -- 47% of the price history has no weather data.

### `Cold_Stock_pavan.csv` -- monthly cold-storage stock by location
1,552 rows (most of the file's 16,384 columns are blank Excel export artifacts and are
dropped). Aggregated to `cold_guntur_bags_lakh` (sum of "Guntur (Local)" + "Guntur
(Non-Local)"), `cold_national_bags_lakh` (sum across all 18 locations), and
`cold_guntur_share_pct` (Guntur's % of the national total). Monthly figures are forward-filled
onto the daily calendar in `src/data/ingest.py::expand_monthly_to_daily`.

### `Inflation_India.xlsx` -- annual CPI inflation, India, 1960-2024
Forward-filled onto the daily calendar the same way. `PAN_India_Cold_Stock.csv` is a single
cross-sectional snapshot (May 2017 vs. May 2018), not a time series, and is not used as a
model feature -- referenced only as context in the EDA.

## External sourced data (`data/external/`)

These three files were researched (not provided by the user) specifically to test whether
international trade/exchange-rate dynamics move the price of LCA-334, an export-oriented
variety -- see report §3.6/§5.5 for the analysis and honest result. Every row carries a
`source` column citing the specific article/report it came from.

### `india_chilli_export_stats.csv` -- India chilli export volume & value, FY2011-12 to FY2024-25
14 rows, one per Indian fiscal year (April-March). Source: Spices Board India data as reported
in ijsat.org (2025) "Chilli Export Dynamics in India."

| Column | Meaning |
|---|---|
| `fiscal_year` | Indian fiscal year label, e.g. `"2022-23"` (April 2022-March 2023). |
| `period_start` | The fiscal year's start date (`2022-04-01` for `"2022-23"`) -- used as the join/forward-fill anchor date. |
| `export_value_rs_lakh` | Total India chilli export value that fiscal year, Rs lakh. |
| `export_qty_tonnes` | Total India chilli export volume that fiscal year, tonnes. |
| `source` | Citation for that row. |

`src/data/ingest.py::load_export_stats` derives three additional columns at load time:
`export_value_per_tonne_rs_lakh` (value / quantity, a rough realised-price proxy),
`export_qty_yoy_growth_pct` and `export_value_yoy_growth_pct` (year-on-year % change).

### `usd_inr_annual_avg.csv` -- USD/INR exchange rate, annual average, 2012-2025
14 rows, one per calendar year. Sources: Forbes India "USD to INR History (1947-2025)"
(2012-2023), exchange-rates.org (2024 annual average), x-rates.com monthly averages (2025,
approximated as an annual mean since the year isn't complete in the source data -- flagged
explicitly in that row's `source` text).

| Column | Meaning |
|---|---|
| `year` | Calendar year. |
| `period_start` | `YYYY-01-01` -- the join/forward-fill anchor date. |
| `usd_inr_avg` | Mean USD/INR rate for that year. |
| `source` | Citation for that row. |

`src/data/ingest.py::load_exchange_rate` derives `usd_inr_yoy_change_pct` (year-on-year %
change) at load time.

### `trade_events.csv` -- hand-curated, sourced trade/macro event windows, 2020-2025
5 rows, each a dated event window with a `description` and a `source` citation:
`covid_export_demand_surge` (Apr 2020-Mar 2021), `covid_domestic_market_disruption`
(Apr-May 2020), `china_demand_high_price_regime` (Apr 2022-Mar 2024), `el_nino_2023`
(Jan-Dec 2023), `export_price_softening` (Apr 2024-Mar 2025). `src/data/ingest.py::
add_trade_event_flags` turns each row into an `event_{name}` binary column on the daily panel,
1 for every date inside that event's `[start_date, end_date]` window. **This is a small,
hand-curated set of five sourced events, not an exhaustive catalog** -- see report §6.

## Processed data (`data/processed/`)

| File | Grain | Description |
|---|---|---|
| `chilli_daily_panel.csv` | 1 row / trading day | Output of `src/data/ingest.py::build_daily_panel` -- the merged, cleaned daily table (112 columns) all downstream work is built from, now including the export/exchange/event columns above (all forward-filled from their annual/fiscal-year resolution onto the daily calendar, same mechanism as cold storage and inflation -- see report §6 for the timing-simplification caveat this implies). |
| `weekly_supervised.csv`, `monthly_supervised.csv` | 1 row / (period, horizon) | Output of `src/features/build_features.py::build_supervised_dataset`. Each row has 93 domestic candidate features (105 including the trade/export features -- see `feature_pool` below), a `horizon` column, and a `target` column (the actual future price for that horizon). `current_price` is the forward-filled last-known price at that period (see report §5.1 for why models predict `target - current_price` rather than `target` directly). |

## Model artifacts (`models/artifacts/`)

`{frequency}_h{horizon}.pkl` -- a joblib-serialized dict: `model` (fitted sklearn
estimator/pipeline), `features` (the exact ordered feature list it expects), `frequency`,
`horizon`, `model_name`, `feature_pool` (`"domestic_only"` or `"with_trade"` -- which candidate
feature set won walk-forward validation for this horizon, see report §5.1/§5.5),
`predict_delta` (whether predictions need `+ current_price` added back), and `n_train_rows`.
