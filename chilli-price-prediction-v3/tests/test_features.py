"""Leakage tests: every feature must be identical whether or not future data exists."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.features.build_features import build_leakfree_features, resample, WEEKLY_FREQ, DATE_COL


@pytest.fixture(scope="module")
def daily_panel():
    path = Path("data/processed/chilli_daily_panel.csv")
    if not path.exists():
        pytest.skip("data/processed/chilli_daily_panel.csv not built yet -- run scripts/refresh_and_score.py first")
    return pd.read_csv(path, parse_dates=[DATE_COL])


def test_no_leakage_when_future_truncated(daily_panel):
    base_full = resample(daily_panel, WEEKLY_FREQ)
    feat_full = build_leakfree_features(base_full, "weekly")

    cutoff = base_full[DATE_COL].iloc[-30]
    daily_trunc = daily_panel[daily_panel[DATE_COL] <= cutoff]
    base_trunc = resample(daily_trunc, WEEKLY_FREQ)
    feat_trunc = build_leakfree_features(base_trunc, "weekly")

    # exclude the boundary row -- its rolling windows can legitimately differ (min_periods)
    common_dates = sorted(set(feat_trunc[DATE_COL]) & set(feat_full[DATE_COL]))[:-1]
    f1 = feat_full[feat_full[DATE_COL].isin(common_dates)].set_index(DATE_COL).sort_index()
    f2 = feat_trunc[feat_trunc[DATE_COL].isin(common_dates)].set_index(DATE_COL).sort_index()

    num_cols = [c for c in f1.columns if pd.api.types.is_numeric_dtype(f1[c]) and f1[c].dtype != bool]
    diff = (f1[num_cols].astype(float) - f2[num_cols].astype(float)).abs()
    max_diff = diff.max().max()
    assert max_diff < 1e-6, f"Feature leakage detected: max abs diff {max_diff}"


def test_current_price_never_null(daily_panel):
    base = resample(daily_panel, WEEKLY_FREQ)
    feat = build_leakfree_features(base, "weekly")
    assert feat["current_price"].isna().sum() == 0
