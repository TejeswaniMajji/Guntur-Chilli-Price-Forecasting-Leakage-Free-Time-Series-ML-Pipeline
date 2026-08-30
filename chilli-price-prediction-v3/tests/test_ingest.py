import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.ingest import load_chilli_market, TARGET_COL, DATE_COL


@pytest.fixture(scope="module")
def market():
    if not Path("data/raw/Chilli.csv").exists():
        pytest.skip("data/raw/Chilli.csv not present")
    return load_chilli_market()


def test_target_column_present(market):
    assert TARGET_COL in market.columns


def test_no_zero_prices(market):
    # 0 is not economically meaningful for a graded commodity (see docs/data_dictionary.md) --
    # it should have been re-coded to NaN during cleaning, never left as a literal 0.
    assert (market[TARGET_COL] == 0).sum() == 0


def test_dates_strictly_increasing(market):
    assert market[DATE_COL].is_monotonic_increasing
    assert market[DATE_COL].duplicated().sum() == 0


def test_remark_features_present(market):
    for col in ["remark_up", "remark_down", "remark_steady", "remark_move_signed"]:
        assert col in market.columns
