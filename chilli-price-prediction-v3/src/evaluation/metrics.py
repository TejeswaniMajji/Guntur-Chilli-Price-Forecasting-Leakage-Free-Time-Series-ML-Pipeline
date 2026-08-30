"""Forecast accuracy metrics. MASE is the headline metric for the report: it scales MAE by
the MAE of a naive (persistence) forecast, so 1.0 means 'no better than guessing today's
price will hold', and values well below 1.0 are the actual evidence of model skill."""
import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def smape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred))
    denom = np.where(denom == 0, np.nan, denom)
    return float(np.nanmean(2 * np.abs(y_true - y_pred) / denom) * 100)


def r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    ss_res = np.sum((y_true - np.asarray(y_pred)) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    if ss_tot == 0:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def mase(y_true, y_pred, naive_mae):
    if naive_mae == 0 or np.isnan(naive_mae):
        return float("nan")
    return float(mae(y_true, y_pred) / naive_mae)


def all_metrics(y_true, y_pred, naive_mae=None):
    out = {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred),
           "smape_pct": smape(y_true, y_pred), "r2": r2(y_true, y_pred)}
    if naive_mae is not None:
        out["mase"] = mase(y_true, y_pred, naive_mae)
    return out
