"""Data loading, cleaning and feature engineering.

Leakage rules followed in this module
-------------------------------------
* Every feature must be known at the moment a prediction is made.
  - ``hour``, ``day_of_week`` and the building are known from the calendar.
  - ``temperature`` and ``occupancy`` are treated as *known inputs* (a weather
    forecast and a class/timetable-based occupancy estimate).
  - ``lag_1h`` and ``lag_24h`` are the building's own consumption 1 hour and
    24 hours earlier - already measured at prediction time.
* The current hour's consumption (the target) is never used as a feature.
* The train/test split is chronological, never random.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BUILDINGS, DATA_PATH, REQUIRED_COLUMNS, TARGET, TEST_FRACTION

NUMERIC_INPUTS = ["temperature", "occupancy", TARGET, "electricity_rate"]


def building_column(name: str) -> str:
    """Name of the one-hot column for a building, e.g. ``building_computer_lab``."""
    return "building_" + re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


BUILDING_COLUMNS = [building_column(b) for b in BUILDINGS]
NUMERIC_FEATURES = [
    "hour", "day_of_week", "is_weekend", "hour_sin", "hour_cos",
    "temperature", "occupancy", "lag_1h", "lag_24h",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + BUILDING_COLUMNS


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------
def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the energy CSV and return a cleaned DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Generate it first with:\n"
            "  python -m src.generate_data"
        )
    return clean_data(pd.read_csv(path))


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean raw energy data.

    * Checks that all required columns exist.
    * Drops rows with an unreadable timestamp or an unknown building.
    * Converts numeric columns, turning invalid / negative values into NaN.
    * Removes duplicate (building, timestamp) rows.
    * Fills missing values by per-building interpolation, then the building median.

    A short report is stored in ``df.attrs["cleaning_report"]``.
    """
    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("The dataset is empty.")
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    df = raw.copy()
    rows_in = len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp", "building_name"])
    df["building_name"] = df["building_name"].astype(str).str.strip()

    unknown = ~df["building_name"].isin(BUILDINGS)
    dropped_unknown = int(unknown.sum())
    df = df[~unknown]

    for col in NUMERIC_INPUTS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Negative consumption / occupancy / rate are physically impossible.
    for col in ["occupancy", TARGET, "electricity_rate"]:
        df.loc[df[col] < 0, col] = np.nan

    before_dupes = len(df)
    df = df.drop_duplicates(subset=["building_name", "timestamp"], keep="first")
    dropped_duplicates = before_dupes - len(df)
    df = df.sort_values(["building_name", "timestamp"]).reset_index(drop=True)
    if df.empty:
        raise ValueError("No valid rows remain after cleaning the dataset.")

    imputed = int(df[NUMERIC_INPUTS].isna().sum().sum())
    for col in NUMERIC_INPUTS:
        by_building = df.groupby("building_name")[col]
        df[col] = by_building.transform(lambda s: s.interpolate(limit_direction="both"))
        df[col] = df[col].fillna(by_building.transform("median"))
        df[col] = df[col].fillna(df[col].median())
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' has no valid values.")
    df["occupancy"] = df["occupancy"].round().astype(int)

    df = df.sort_values(["timestamp", "building_name"]).reset_index(drop=True)
    df.attrs["cleaning_report"] = {
        "rows_in": rows_in,
        "rows_out": len(df),
        "dropped_unknown_building": dropped_unknown,
        "dropped_duplicates": dropped_duplicates,
        "imputed_values": imputed,
    }
    return df


# --------------------------------------------------------------------------
# Feature engineering
# --------------------------------------------------------------------------
def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features derived from the timestamp."""
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["day_of_week"] = out["timestamp"].dt.dayofweek          # Monday = 0
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)       # cyclical hour encoding
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the building's own consumption 1 h and 24 h earlier.

    Lags are looked up by *timestamp* (not by row position), so gaps in the data
    produce NaN instead of silently using the wrong reading.
    """
    out = df.reset_index(drop=True).copy()
    out["lag_1h"] = np.nan
    out["lag_24h"] = np.nan
    for _, idx in out.groupby("building_name").groups.items():
        part = out.loc[idx]
        series = pd.Series(part[TARGET].to_numpy(), index=pd.DatetimeIndex(part["timestamp"]))
        if series.index.has_duplicates:
            raise ValueError("Duplicate timestamps found for one building.")
        stamps = pd.DatetimeIndex(part["timestamp"])
        out.loc[idx, "lag_1h"] = series.reindex(stamps - pd.Timedelta(hours=1)).to_numpy()
        out.loc[idx, "lag_24h"] = series.reindex(stamps - pd.Timedelta(hours=24)).to_numpy()
    return out


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return the cleaned data with all features; rows without lags are dropped."""
    frame = add_lag_features(add_time_features(df))
    frame = frame.dropna(subset=["lag_1h", "lag_24h"])
    return frame.sort_values(["timestamp", "building_name"]).reset_index(drop=True)


def encode_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Build the model input matrix (always the same columns, in the same order)."""
    X = pd.DataFrame(index=frame.index)
    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(frame[col], errors="raise").astype(float)
    for name, col in zip(BUILDINGS, BUILDING_COLUMNS):
        X[col] = (frame["building_name"] == name).astype(float)
    return X[FEATURE_COLUMNS]


# --------------------------------------------------------------------------
# Chronological split
# --------------------------------------------------------------------------
def chronological_split(frame: pd.DataFrame, test_fraction: float = TEST_FRACTION):
    """Split by time: the **last** ``test_fraction`` of days becomes the test set.

    Returns ``(train, test, cutoff)``. Every training row is strictly earlier
    than the cutoff and every test row is on or after it, so the model never
    sees the future during training.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1.")
    days = np.sort(frame["timestamp"].dt.normalize().unique())
    if len(days) < 5:
        raise ValueError("At least 5 days of data are required for a train/test split.")
    cutoff_day = days[int(len(days) * (1 - test_fraction))]
    cutoff = pd.Timestamp(cutoff_day)
    train = frame[frame["timestamp"] < cutoff]
    test = frame[frame["timestamp"] >= cutoff]
    if train.empty or test.empty:
        raise ValueError("Chronological split produced an empty train or test set.")
    return train, test, cutoff
