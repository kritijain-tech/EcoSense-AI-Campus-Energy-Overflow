"""Prediction helpers built on top of the trained model bundle."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import BUILDINGS, MODEL_PATH, TARGET
from src.preprocess import encode_features, prepare_model_frame

# Column that stores each model's predictions in the prediction frame.
MODEL_COLUMNS = {
    "Random Forest": "pred_random_forest",
    "Gradient Boosting": "pred_gradient_boosting",
}


def load_bundle(path: Path | str = MODEL_PATH) -> dict:
    """Load the trained model bundle saved by ``src.train_model``."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Train it first with:\n"
            "  python -m src.train_model"
        )
    return joblib.load(path)


def build_prediction_frame(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Predict every row that has the lag features available.

    Adds one column per model, ``predicted_kwh`` (the best model), ``split``
    ("train" / "test") and ``cost_inr``. Predictions on ``train`` rows are
    *in-sample*; only ``test`` rows are a fair measure of accuracy.
    """
    frame = prepare_model_frame(df)
    X = encode_features(frame)[bundle["feature_columns"]]
    for name, model in bundle["models"].items():
        frame[MODEL_COLUMNS[name]] = np.clip(model.predict(X), 0, None)
    frame["predicted_kwh"] = frame[MODEL_COLUMNS[bundle["best_model_name"]]]
    cutoff = pd.Timestamp(bundle["split_cutoff"])
    frame["split"] = np.where(frame["timestamp"] >= cutoff, "test", "train")
    frame["cost_inr"] = frame[TARGET] * frame["electricity_rate"]
    return frame


def predict_single(bundle: dict, *, building: str, hour: int, day_of_week: int,
                   temperature: float, occupancy: int, lag_1h: float,
                   lag_24h: float) -> dict[str, float]:
    """What-if prediction for one hour. Returns ``{model name: predicted kWh}``."""
    if building not in BUILDINGS:
        raise ValueError(f"Unknown building: {building!r}")
    if not 0 <= hour <= 23:
        raise ValueError("hour must be between 0 and 23")
    if not 0 <= day_of_week <= 6:
        raise ValueError("day_of_week must be between 0 (Mon) and 6 (Sun)")
    if occupancy < 0 or lag_1h < 0 or lag_24h < 0:
        raise ValueError("occupancy and previous consumption cannot be negative")
    if not -10 <= temperature <= 55:
        raise ValueError("temperature must be between -10 and 55 deg C")

    row = pd.DataFrame([{
        "building_name": building,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": int(day_of_week >= 5),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "temperature": temperature,
        "occupancy": occupancy,
        "lag_1h": lag_1h,
        "lag_24h": lag_24h,
    }])
    X = encode_features(row)[bundle["feature_columns"]]
    return {name: float(max(model.predict(X)[0], 0.0)) for name, model in bundle["models"].items()}


def summarize_predictions(pred_df: pd.DataFrame, column: str = "predicted_kwh",
                          split: str = "test") -> pd.DataFrame:
    """Predicted vs actual totals per building (default: held-out test period)."""
    part = pred_df[pred_df["split"] == split] if split else pred_df
    if part.empty:
        return pd.DataFrame(columns=["building_name", "actual_kwh", "predicted_kwh",
                                     "error_kwh", "error_pct"])
    summary = (part.groupby("building_name")
               .agg(actual_kwh=(TARGET, "sum"), predicted_kwh=(column, "sum"))
               .reset_index())
    total = pd.DataFrame([{"building_name": "All Buildings",
                           "actual_kwh": summary["actual_kwh"].sum(),
                           "predicted_kwh": summary["predicted_kwh"].sum()}])
    summary = pd.concat([summary, total], ignore_index=True)
    summary["error_kwh"] = summary["predicted_kwh"] - summary["actual_kwh"]
    summary["error_pct"] = 100 * summary["error_kwh"] / summary["actual_kwh"]
    return summary.round(2)
