"""Train, evaluate and save the energy prediction models.

Two regressors are trained on the **training period only** and evaluated on a
later, unseen **test period** (chronological split):

* Random Forest Regressor
* Gradient Boosting Regressor

A naive "same as the previous hour" baseline is also reported, so the models'
value can be judged honestly. The better model is chosen by the lowest test
RMSE (ties broken by MAE). Nothing here is hardcoded - every number is
computed from the data at run time.

Usage::

    python -m src.train_model
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.config import MODEL_PATH, RANDOM_SEED, TARGET, TEST_FRACTION
from src.preprocess import (FEATURE_COLUMNS, chronological_split, encode_features,
                            load_data, prepare_model_frame)

MODEL_NAMES = ("Random Forest", "Gradient Boosting")


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, RMSE and R2 for a set of predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_models(seed: int = RANDOM_SEED) -> dict:
    """Create the two (untrained) models with sensible, un-tuned settings."""
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=150, max_depth=14, min_samples_leaf=5,
            n_jobs=-1, random_state=seed),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=3,
            subsample=0.8, random_state=seed),
    }


def train_and_evaluate(df: pd.DataFrame, test_fraction: float = TEST_FRACTION,
                       seed: int = RANDOM_SEED) -> dict:
    """Train both models and return a "bundle" dictionary with everything needed later."""
    frame = prepare_model_frame(df)
    train, test, cutoff = chronological_split(frame, test_fraction)
    X_train, y_train = encode_features(train), train[TARGET]
    X_test, y_test = encode_features(test), test[TARGET]

    models = build_models(seed)
    metrics, importances, test_predictions = {}, {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_predictions[name] = predictions
        metrics[name] = regression_metrics(y_test, predictions)
        importances[name] = dict(zip(FEATURE_COLUMNS, map(float, model.feature_importances_)))

    # Selection rule: lowest test RMSE, then lowest MAE.
    best_name = min(metrics, key=lambda n: (metrics[n]["rmse"], metrics[n]["mae"]))

    # Honest reference point: "next hour = previous hour".
    baseline = regression_metrics(y_test, test["lag_1h"])

    per_building = {}
    for building, part in test.assign(pred=test_predictions[best_name]).groupby("building_name"):
        per_building[building] = regression_metrics(part[TARGET], part["pred"])

    return {
        "models": models,
        "metrics": metrics,
        "baseline_metrics": baseline,
        "best_model_name": best_name,
        "selection_rule": "lowest test-set RMSE (ties broken by MAE)",
        "feature_columns": FEATURE_COLUMNS,
        "feature_importances": importances,
        "per_building_metrics": per_building,
        "split_cutoff": cutoff.isoformat(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_period": (str(train["timestamp"].min()), str(train["timestamp"].max())),
        "test_period": (str(test["timestamp"].min()), str(test["timestamp"].max())),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "seed": seed,
    }


def save_bundle(bundle: dict, path: Path | str = MODEL_PATH) -> Path:
    """Persist the bundle with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)
    return path


def format_report(bundle: dict) -> str:
    """Human-readable comparison table for the console / README."""
    lines = [f"{'Model':<20}{'MAE':>10}{'RMSE':>10}{'R2':>10}", "-" * 50]
    for name, m in bundle["metrics"].items():
        mark = "  <- best" if name == bundle["best_model_name"] else ""
        lines.append(f"{name:<20}{m['mae']:>10.3f}{m['rmse']:>10.3f}{m['r2']:>10.4f}{mark}")
    b = bundle["baseline_metrics"]
    lines.append(f"{'Baseline (prev hour)':<20}{b['mae']:>10.3f}{b['rmse']:>10.3f}{b['r2']:>10.4f}")
    return "\n".join(lines)


def main() -> None:
    df = load_data()
    bundle = train_and_evaluate(df)
    path = save_bundle(bundle)
    print(f"Train rows: {bundle['train_rows']:,} | Test rows: {bundle['test_rows']:,}")
    print(f"Train: {bundle['train_period'][0]} -> {bundle['train_period'][1]}")
    print(f"Test : {bundle['test_period'][0]} -> {bundle['test_period'][1]}\n")
    print(format_report(bundle))
    print(f"\nBest model ({bundle['selection_rule']}): {bundle['best_model_name']}")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
