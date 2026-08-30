"""
Minimal prediction API over the same scoring pipeline the dashboard uses.

    uvicorn api.main:app --reload

Then see http://localhost:8000/docs for the interactive OpenAPI page.
"""
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pipelines.scoring import run_scoring

app = FastAPI(
    title="Guntur Chilli Price Forecast API",
    description="Multi-horizon forecasts for LCA-334 'Best' grade chilli price, Guntur mandi.",
    version="1.0.0",
)


class ForecastPoint(BaseModel):
    frequency: str
    horizon: int
    forecast_date: str
    current_price: float
    prediction: float
    pi_80_low: Optional[float] = None
    pi_80_high: Optional[float] = None
    model: str
    reliability: Optional[str] = None


_cache = {"df": None}


def _get_forecasts(refresh: bool = False) -> pd.DataFrame:
    if _cache["df"] is None or refresh:
        path = Path("reports/forecasts/latest_forecasts.csv")
        if path.exists() and not refresh:
            _cache["df"] = pd.read_csv(path, parse_dates=["forecast_date"])
        else:
            _cache["df"] = run_scoring(save=True)
    return _cache["df"]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/forecasts", response_model=List[ForecastPoint])
def get_forecasts(frequency: Optional[str] = None):
    """All current forecasts, optionally filtered to 'weekly' or 'monthly'."""
    df = _get_forecasts().copy()
    if frequency:
        if frequency not in {"weekly", "monthly"}:
            raise HTTPException(400, "frequency must be 'weekly' or 'monthly'")
        df = df[df["frequency"] == frequency]
    df["forecast_date"] = df["forecast_date"].astype(str)
    return df.where(pd.notnull(df), None).to_dict(orient="records")


@app.get("/forecasts/{frequency}/{horizon}", response_model=ForecastPoint)
def get_single_forecast(frequency: str, horizon: int):
    df = _get_forecasts()
    row = df[(df["frequency"] == frequency) & (df["horizon"] == horizon)]
    if row.empty:
        raise HTTPException(404, f"No model for frequency={frequency}, horizon={horizon}")
    rec = row.iloc[0].to_dict()
    rec["forecast_date"] = str(rec["forecast_date"])
    return {k: (None if pd.isna(v) else v) for k, v in rec.items()}


@app.post("/refresh", response_model=List[ForecastPoint])
def refresh_forecasts():
    """Re-run scoring against whatever is currently in data/processed/ (does NOT retrain
    models -- run scripts/refresh_and_score.py for a full retrain)."""
    df = _get_forecasts(refresh=True).copy()
    df["forecast_date"] = df["forecast_date"].astype(str)
    return df.where(pd.notnull(df), None).to_dict(orient="records")
