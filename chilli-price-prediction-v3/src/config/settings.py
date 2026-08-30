"""
Project-wide path conventions. Every module in src/ builds paths relative to the current
working directory rather than importing from here, on the assumption that everything is run
from the project root (as scripts/*.py and streamlit_app.py do) -- this file exists as the
single documented reference for those conventions, and as a convenient place to override them
via environment variables if you ever need to.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("CHILLI_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = Path(os.getenv("CHILLI_MODELS_DIR", PROJECT_ROOT / "models"))
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

REPORTS_DIR = Path(os.getenv("CHILLI_REPORTS_DIR", PROJECT_ROOT / "reports"))
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
FORECASTS_DIR = REPORTS_DIR / "forecasts"

LOGGING_LEVEL = os.getenv("CHILLI_LOG_LEVEL", "INFO")
