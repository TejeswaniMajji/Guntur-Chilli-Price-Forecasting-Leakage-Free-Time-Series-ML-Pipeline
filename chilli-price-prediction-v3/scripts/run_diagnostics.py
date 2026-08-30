"""
Model diagnostics for the two headline horizons (weekly h=1, monthly h=1 -- the horizons
with the strongest validated skill). Produces:
  reports/figures/11-14_*.png     -- feature importance, actual-vs-predicted, residuals
  reports/metrics/model_diagnostics.json -- prediction-interval widths + top features
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.train import select_features, select_top_k_by_correlation, build_model_zoo, DATE_COL
from src.utils.viz import apply_style, CATEGORICAL, INK_MUTED, INK_SECONDARY

apply_style()
FIG_DIR = Path("reports/figures")
METRICS_DIR = Path("reports/metrics")

diagnostics = {}

sel = pd.read_csv(METRICS_DIR / "model_selection.csv")
DATA_PATHS = {"weekly": "data/processed/weekly_supervised.csv", "monthly": "data/processed/monthly_supervised.csv"}
FIG_START = {("weekly", 1): 11, ("monthly", 1): 14}
CASES = [(r.frequency, DATA_PATHS[r.frequency], int(r.horizon), r.model, r.feature_pool,
          FIG_START.get((r.frequency, int(r.horizon)), 99))
         for r in sel.itertuples()]
# still render full chart detail (importance / actual-vs-predicted / residuals) for the two
# headline horizons; every other horizon just gets the prediction-interval numbers.
DETAIL_HORIZONS = set(FIG_START.keys())

for freq, path, horizon, model_name, feature_pool, fig_start in CASES:
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    sub = df[df["horizon"] == horizon].sort_values(DATE_COL).reset_index(drop=True)
    all_features = select_features(sub, feature_pool=feature_pool)
    y_level = sub["target"]
    current_price = sub["current_price"]
    y_delta = y_level - current_price

    # last 20% as a genuine holdout for diagnostics (never seen during feature selection or fit)
    split = int(len(sub) * 0.8)
    train, test = sub.iloc[:split], sub.iloc[split:]
    y_train_delta = y_delta.iloc[:split]

    feats = select_top_k_by_correlation(train[all_features], y_train_delta, 15)
    X_train, X_test = train[feats], test[feats]

    model = build_model_zoo()[model_name]
    model.fit(X_train, y_train_delta)
    pred_delta = model.predict(X_test)
    pred_level = pred_delta + test["current_price"].values
    resid = test["target"].values - pred_level

    # --- feature importance ---
    if model_name == "ridge":
        coefs = model.named_steps["model"].coef_
        importance = pd.Series(np.abs(coefs), index=feats).sort_values(ascending=False)
        imp_label = "|standardised coefficient|"
    else:
        pi = permutation_importance(model, X_test, test["target"] - test["current_price"],
                                     n_repeats=8, random_state=42, scoring="neg_mean_absolute_error", n_jobs=-1)
        importance = pd.Series(pi.importances_mean, index=feats).sort_values(ascending=False)
        imp_label = "permutation importance (MAE increase)"
    importance = importance.head(10).iloc[::-1]

    if (freq, horizon) in DETAIL_HORIZONS:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance.index, importance.values, color=CATEGORICAL[0])
        ax.set_title(f"{freq.title()} h={horizon} ({model_name}, {feature_pool}): top 10 features")
        ax.set_xlabel(imp_label)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{fig_start}_{freq}_h{horizon}_feature_importance.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 6))
        lims = [min(test["target"].min(), pred_level.min()), max(test["target"].max(), pred_level.max())]
        ax.plot(lims, lims, color=INK_MUTED, linewidth=1, linestyle="--")
        ax.scatter(test["target"], pred_level, s=18, alpha=0.6, color=CATEGORICAL[0], edgecolors="none")
        ax.set_title(f"{freq.title()} h={horizon}: actual vs predicted (holdout)")
        ax.set_xlabel("Actual price")
        ax.set_ylabel("Predicted price")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{fig_start+1}_{freq}_h{horizon}_actual_vs_predicted.png")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.axhline(0, color=INK_MUTED, linewidth=0.8)
        ax.plot(test[DATE_COL], resid, color=CATEGORICAL[7], marker="o", markersize=3, linewidth=0.8)
        ax.set_title(f"{freq.title()} h={horizon}: holdout residuals over time (actual - predicted)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{fig_start+2}_{freq}_h{horizon}_residuals.png")
        plt.close(fig)

    q10, q50, q90 = np.percentile(resid, [10, 50, 90])
    diagnostics[f"{freq}_h{horizon}"] = {
        "model": model_name,
        "feature_pool": feature_pool,
        "n_holdout": int(len(test)),
        "holdout_mae": float(np.mean(np.abs(resid))),
        "top_features": importance.iloc[::-1].round(4).to_dict(),
        "prediction_interval_offsets_80pct": {"q10": float(q10), "q50_bias": float(q50), "q90": float(q90)},
        "note": "80% interval = point_forecast + [q10, q90]; q50 is the median residual (systematic bias, should be near 0).",
    }

with open(METRICS_DIR / "model_diagnostics.json", "w") as f:
    json.dump(diagnostics, f, indent=2)

print(json.dumps(diagnostics, indent=2))
