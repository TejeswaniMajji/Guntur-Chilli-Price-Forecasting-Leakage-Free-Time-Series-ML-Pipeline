"""
Multi-horizon model training with expanding-window walk-forward cross-validation.

Every horizon gets its own model (direct multi-horizon forecasting, rather than one model
recursively fed its own predictions -- direct models don't compound error the way recursive
ones do, at the cost of training more models).

Four candidates are compared honestly on the same folds:
    naive          -- persistence forecast: "price h periods from now = price today".
                       This is the benchmark every real model must beat; MASE < 1 is the bar.
    ridge           -- linear baseline with L2 regularisation (handles the ~90 correlated
                       lag/rolling features better than plain OLS).
    random_forest   -- bagged trees, robust to outliers and non-linear supply/price effects.
    hist_gbm        -- scikit-learn's histogram-based gradient boosting (the same family of
                       algorithm as LightGBM/XGBoost, native NaN handling, no external
                       compiled dependency). This is the production model if it wins.

Optional local-only upgrade: if the user has `lightgbm` or `xgboost` installed, those can be
dropped into MODEL_ZOO as extra candidates -- see the commented block below. They were left
out of the default zoo because they are not installable in every environment; hist_gbm gets
you 90% of the benefit with zero extra dependencies.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import all_metrics

warnings.filterwarnings("ignore")

DATE_COL = "Date"
TARGET_COL = "LCA_334_Best"
NON_FEATURE_COLS = {DATE_COL, "horizon", "target"}

# Export volume/value, USD/INR, and the hand-curated trade-event flags are all annual or
# fiscal-year resolution data forward-filled onto the daily/weekly/monthly calendar (see
# src/data/ingest.py). They're motivated by real domain knowledge -- LCA-334 is an export
# variety, so international demand and the exchange rate plausibly move its price -- but
# whether they *actually* help a walk-forward-validated model is an empirical question, not
# an assumption: run_full_evaluation tests both feature pools per horizon and keeps whichever
# wins, rather than forcing them in.
TRADE_FEATURE_PREFIXES = ("export_", "usd_inr_", "event_")
FEATURE_POOLS = ("domestic_only", "with_trade")

ARTIFACTS_DIR = Path("models/artifacts")
METRICS_DIR = Path("reports/metrics")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def select_features(df: pd.DataFrame, feature_pool: str = "with_trade") -> List[str]:
    cols = [c for c in df.columns if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])]
    if feature_pool == "domestic_only":
        cols = [c for c in cols if not c.startswith(TRADE_FEATURE_PREFIXES)]
    return cols


def build_model_zoo(random_state: int = 42) -> Dict[str, object]:
    zoo = {
        "ridge": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=5.0, random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=400, max_depth=8, min_samples_leaf=3,
                random_state=random_state, n_jobs=-1)),
        ]),
        "hist_gbm": HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.05, max_depth=4, l2_regularization=0.5,
            early_stopping=False, random_state=random_state,
        ),
    }
    # Optional local-only upgrade (uncomment if lightgbm/xgboost are installed):
    # try:
    #     import lightgbm as lgb
    #     zoo["lightgbm"] = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.04,
    #                                          subsample=0.85, colsample_bytree=0.85,
    #                                          random_state=random_state)
    # except ImportError:
    #     pass
    return zoo


def expanding_window_splits(n_rows: int, n_splits: int = 5, min_train_frac: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Time-ordered expanding-window folds: fold k trains on everything before it and tests
    on the next contiguous block, so later folds always have strictly more training history
    (mirrors how the model would actually be retrained and used in production)."""
    min_train = int(n_rows * min_train_frac)
    remaining = n_rows - min_train
    fold_size = max(1, remaining // n_splits)
    splits = []
    for k in range(n_splits):
        test_start = min_train + k * fold_size
        test_end = n_rows if k == n_splits - 1 else min(n_rows, test_start + fold_size)
        if test_start >= n_rows or test_start >= test_end:
            break
        train_idx = np.arange(0, test_start)
        test_idx = np.arange(test_start, test_end)
        if len(train_idx) < 20 or len(test_idx) < 3:
            continue
        splits.append((train_idx, test_idx))
    return splits


def select_top_k_by_correlation(X_train: pd.DataFrame, y_train: pd.Series, k: int) -> List[str]:
    """Fold-internal feature selection: with ~90 candidate features and as few as ~250-450
    training rows in the earliest folds, fitting on all of them risks the model memorising
    noise rather than signal. Ranking by |correlation with the training-fold target only|
    (never touching test-fold data) keeps the feature:row ratio sane without leaking
    information about the test period into the selection step itself."""
    corr = X_train.apply(lambda col: col.corr(y_train) if col.notna().sum() > 5 else 0.0)
    corr = corr.abs().fillna(0).sort_values(ascending=False)
    return corr.head(k).index.tolist()


def walk_forward_evaluate(df: pd.DataFrame, horizon: int, n_splits: int = 5, predict_delta: bool = True,
                           top_k_features: int = 15, feature_pool: str = "with_trade") -> Dict:
    """predict_delta=True trains every model on (target - current_price) instead of the raw
    price level, then adds current_price back for evaluation. For a near-random-walk series
    like this one, this reframes the learning problem as "how much will price move from here"
    rather than "what is the price", which matters a lot for tree ensembles: a tree has to
    fragment a near-identity price-in/price-out relationship across many splits to reproduce
    it, whereas a linear model can just learn a ~1.0 coefficient. Differencing removes that
    identity mapping from the learning problem entirely, so both model families are judged on
    the part that's actually hard: the deviation from persistence."""
    sub = df[df["horizon"] == horizon].sort_values(DATE_COL).reset_index(drop=True)
    features = select_features(sub, feature_pool=feature_pool)
    X = sub[features]
    y_level = sub["target"]
    current_price = sub["current_price"]
    y = (y_level - current_price) if predict_delta else y_level
    dates = sub[DATE_COL]

    splits = expanding_window_splits(len(sub), n_splits=n_splits)
    zoo = build_model_zoo()
    fold_records = []

    fold_feature_sets = []
    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train_full, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test_full, y_test_level = X.iloc[test_idx], y_level.iloc[test_idx]
        cp_test = current_price.iloc[test_idx]
        naive_pred = cp_test  # persistence forecast: last known price (delta = 0)
        naive_mae = all_metrics(y_test_level, naive_pred)["mae"]

        top_feats = select_top_k_by_correlation(X_train_full, y_train, top_k_features) if top_k_features else features
        X_train, X_test = X_train_full[top_feats], X_test_full[top_feats]
        fold_feature_sets.append(top_feats)

        row = {
            "horizon": horizon, "fold": fold_i,
            "train_rows": len(train_idx), "test_rows": len(test_idx),
            "test_start": str(dates.iloc[test_idx[0]].date()),
            "test_end": str(dates.iloc[test_idx[-1]].date()),
        }
        row["naive"] = all_metrics(y_test_level, naive_pred)
        for name, model in zoo.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            preds_level = preds + cp_test.values if predict_delta else preds
            row[name] = all_metrics(y_test_level, preds_level, naive_mae=naive_mae)
        fold_records.append(row)

    return {"horizon": horizon, "n_folds": len(fold_records), "folds": fold_records, "features": features,
            "fold_feature_sets": fold_feature_sets}


def summarize_folds(fold_results: Dict, model_names: List[str]) -> pd.DataFrame:
    rows = []
    for m in ["naive"] + model_names:
        maes = [f[m]["mae"] for f in fold_results["folds"]]
        smapes = [f[m]["smape_pct"] for f in fold_results["folds"]]
        mases = [f[m].get("mase", np.nan) for f in fold_results["folds"]]
        r2s = [f[m]["r2"] for f in fold_results["folds"]]
        rows.append({
            "horizon": fold_results["horizon"], "model": m,
            "mae_mean": np.mean(maes), "mae_std": np.std(maes),
            "smape_mean": np.mean(smapes), "mase_mean": np.nanmean(mases),
            "r2_mean": np.mean(r2s), "n_folds": len(maes),
        })
    return pd.DataFrame(rows)


def train_final_models(df: pd.DataFrame, horizons: List[int], frequency: str,
                        best_model_name: str = "ridge", predict_delta: bool = True,
                        top_k_features: int = 15, feature_pool: str = "with_trade") -> Dict:
    """Refit the chosen model (and chosen feature pool) on ALL available history per horizon,
    for production forecasting, once walk-forward CV has already established which combination
    to trust."""
    saved = {}
    for h in horizons:
        sub = df[df["horizon"] == h].sort_values(DATE_COL).reset_index(drop=True)
        all_features = select_features(sub, feature_pool=feature_pool)
        X_full = sub[all_features]
        y = (sub["target"] - sub["current_price"]) if predict_delta else sub["target"]
        features = select_top_k_by_correlation(X_full, y, top_k_features) if top_k_features else all_features
        X = X_full[features]
        zoo = build_model_zoo()
        model = zoo[best_model_name]
        model.fit(X, y)
        bundle = {"model": model, "features": features, "frequency": frequency, "horizon": h,
                  "model_name": best_model_name, "feature_pool": feature_pool,
                  "predict_delta": predict_delta, "n_train_rows": len(sub)}
        path = ARTIFACTS_DIR / f"{frequency}_h{h}.pkl"
        joblib.dump(bundle, path)
        saved[h] = str(path)
    return saved


def run_full_evaluation(weekly_sup: pd.DataFrame, monthly_sup: pd.DataFrame,
                         weekly_horizons: List[int], monthly_horizons: List[int]) -> pd.DataFrame:
    """For every horizon, walk-forward-evaluates every (feature_pool x model) combination --
    i.e. this is also where the 'do export/exchange-rate features actually help' question from
    reports/ANALYSIS_REPORT.md Sec. 6 gets answered empirically, on the same footing as the
    naive-vs-model comparison, rather than assumed either way."""
    model_names = list(build_model_zoo().keys())
    all_summaries = []
    all_fold_detail = []
    for freq, sup, horizons in [("weekly", weekly_sup, weekly_horizons), ("monthly", monthly_sup, monthly_horizons)]:
        for h in horizons:
            for pool in FEATURE_POOLS:
                fr = walk_forward_evaluate(sup, h, n_splits=5, feature_pool=pool, top_k_features=15)
                summ = summarize_folds(fr, model_names)
                summ["frequency"] = freq
                summ["feature_pool"] = pool
                all_summaries.append(summ)
                for fold in fr["folds"]:
                    rec = {"frequency": freq, "horizon": h, "feature_pool": pool, "fold": fold["fold"],
                           "test_start": fold["test_start"], "test_end": fold["test_end"],
                           "train_rows": fold["train_rows"], "test_rows": fold["test_rows"]}
                    for m in ["naive"] + model_names:
                        for metric_name, val in fold[m].items():
                            rec[f"{m}_{metric_name}"] = val
                    all_fold_detail.append(rec)
                print(f"[{freq} h={h} pool={pool}] " + " | ".join(
                    f"{m}: MAE={summ.loc[summ.model==m,'mae_mean'].values[0]:.2f} "
                    f"MASE={summ.loc[summ.model==m,'mase_mean'].values[0]:.2f}" for m in ["naive"] + model_names))

    summary_df = pd.concat(all_summaries, ignore_index=True)
    fold_detail_df = pd.DataFrame(all_fold_detail)
    summary_df.to_csv(METRICS_DIR / "walk_forward_summary.csv", index=False)
    fold_detail_df.to_csv(METRICS_DIR / "walk_forward_fold_detail.csv", index=False)
    return summary_df
