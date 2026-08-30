import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pipelines.scoring import run_scoring


@pytest.fixture(scope="module")
def forecasts():
    if not Path("models/artifacts").exists() or not any(Path("models/artifacts").glob("*.pkl")):
        pytest.skip("no trained models -- run scripts/refresh_and_score.py first")
    return run_scoring(save=False)


def test_all_horizons_scored(forecasts):
    from src.features.build_features import WEEKLY_HORIZONS, MONTHLY_HORIZONS
    expected = len(WEEKLY_HORIZONS) + len(MONTHLY_HORIZONS)
    assert len(forecasts) == expected


def test_predictions_are_finite(forecasts):
    assert forecasts["prediction"].notna().all()
    assert (forecasts["prediction"] > 0).all()  # a price forecast should never be non-positive


def test_forecast_dates_are_in_the_future(forecasts):
    import pandas as pd
    last_known = pd.read_csv("data/processed/chilli_daily_panel.csv", parse_dates=["Date"])["Date"].max()
    assert (pd.to_datetime(forecasts["forecast_date"]) > last_known).all()
