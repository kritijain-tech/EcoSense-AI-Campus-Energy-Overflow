"""Isolation Forest based anomaly detection for hourly building consumption.

How it works
------------
1. For every (building, hour of day, weekday/weekend) we compute the *typical*
   (median) consumption. The ratio ``actual / typical`` tells how unusual a
   reading is for that time of day.
2. One Isolation Forest per building looks at the ratio, the occupancy level and
   the time of day. Readings that are easy to "isolate" get a high anomaly score.
3. The top ``contamination`` share of readings are flagged and labelled with a
   simple rule-based type (night-time, laboratory, low occupancy...).

IMPORTANT: flagged readings are *potential issues* - statistically unusual
values - NOT confirmed equipment faults.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.config import DEFAULT_CONTAMINATION, NIGHT_HOURS, RANDOM_SEED, TARGET

HIGH_RATIO = 1.4   # reading at least 40 % above typical
LOW_RATIO = 0.6    # reading at least 40 % below typical
LOW_OCCUPANCY_LEVEL = 0.6   # occupancy below 60 % of the building's busy level

ANOMALY_TYPES = [
    "High night-time consumption",
    "Laboratory usage spike",
    "High consumption at low occupancy",
    "Unusual consumption spike",
    "Unusually low consumption",
    "Unusual pattern",
]


def add_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``expected_kwh`` (typical value for that building/hour/day-type) and the ratio."""
    out = df.copy()
    out["hour"] = out["timestamp"].dt.hour
    out["is_weekend"] = (out["timestamp"].dt.dayofweek >= 5).astype(int)
    keys = ["building_name", "hour", "is_weekend"]
    out["expected_kwh"] = out.groupby(keys)[TARGET].transform("median").clip(lower=0.1)
    out["energy_ratio"] = out[TARGET] / out["expected_kwh"]
    return out


def _classify(row: pd.Series, occupancy_level: float) -> str:
    """Give a flagged reading a human-readable type."""
    ratio, hour = row["energy_ratio"], int(row["hour"])
    if ratio >= HIGH_RATIO:
        if hour in NIGHT_HOURS:
            return ANOMALY_TYPES[0]
        if (row["building_name"] == "Computer Lab" and not row["is_weekend"]
                and 8 <= hour <= 20):
            return ANOMALY_TYPES[1]
        if occupancy_level < LOW_OCCUPANCY_LEVEL:
            return ANOMALY_TYPES[2]
        return ANOMALY_TYPES[3]
    if ratio <= LOW_RATIO:
        return ANOMALY_TYPES[4]
    return ANOMALY_TYPES[5]


def detect_anomalies(df: pd.DataFrame, contamination: float = DEFAULT_CONTAMINATION,
                     seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Flag unusual readings.

    Returns a copy of ``df`` with extra columns: ``expected_kwh``, ``energy_ratio``,
    ``anomaly_score`` (higher = more unusual), ``is_anomaly`` (0/1),
    ``anomaly_type`` and ``anomaly_reason``.
    """
    if not 0 < contamination < 0.5:
        raise ValueError("contamination must be between 0 and 0.5 (e.g. 0.01 = 1 %).")
    if df.empty:
        raise ValueError("Cannot run anomaly detection on an empty dataset.")

    out = add_baseline(df).reset_index(drop=True)
    cap = out.groupby("building_name")["occupancy"].transform(lambda s: max(s.quantile(0.99), 1))
    out["occupancy_level"] = (out["occupancy"] / cap).clip(0, 1.5)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    features = ["energy_ratio", "occupancy_level", "hour_sin", "hour_cos"]

    out["anomaly_score"] = 0.0
    out["is_anomaly"] = 0
    for _, idx in out.groupby("building_name").groups.items():
        if len(idx) < 20:       # too little data for a meaningful model
            continue
        forest = IsolationForest(n_estimators=200, contamination=contamination,
                                 random_state=seed, n_jobs=-1)
        X = out.loc[idx, features]
        labels = forest.fit_predict(X)                       # -1 = anomaly
        out.loc[idx, "anomaly_score"] = -forest.score_samples(X)
        out.loc[idx, "is_anomaly"] = (labels == -1).astype(int)

    out["anomaly_type"] = ""
    out["anomaly_reason"] = ""
    flagged = out["is_anomaly"] == 1
    if flagged.any():
        out.loc[flagged, "anomaly_type"] = [
            _classify(row, row["occupancy_level"]) for _, row in out[flagged].iterrows()
        ]
        out.loc[flagged, "anomaly_reason"] = [
            f"{r[TARGET]:.1f} kWh vs typical {r['expected_kwh']:.1f} kWh "
            f"({r['energy_ratio']:.1f}x) with {int(r['occupancy'])} people present"
            for _, r in out[flagged].iterrows()
        ]
    return out.drop(columns=["hour_sin", "hour_cos"])


def anomaly_table(flagged: pd.DataFrame) -> pd.DataFrame:
    """Tidy table of flagged readings, most unusual first."""
    cols = ["timestamp", "building_name", TARGET, "expected_kwh", "energy_ratio",
            "occupancy", "temperature", "anomaly_type", "anomaly_score", "anomaly_reason"]
    table = flagged.loc[flagged["is_anomaly"] == 1, cols].copy()
    table["excess_kwh"] = table[TARGET] - table["expected_kwh"]
    return (table.sort_values("anomaly_score", ascending=False)
            .round({TARGET: 2, "expected_kwh": 2, "energy_ratio": 2,
                    "anomaly_score": 3, "excess_kwh": 2})
            .reset_index(drop=True))


def evaluate_against_injected(flagged: pd.DataFrame) -> dict[str, float] | None:
    """Precision / recall against the anomalies injected into the SYNTHETIC data.

    Returns ``None`` when the injected-anomaly labels are not available
    (e.g. real data). This is only a sanity check of the detector.
    """
    if "injected_anomaly" not in flagged.columns:
        return None
    truth = flagged["injected_anomaly"].astype(int)
    pred = flagged["is_anomaly"].astype(int)
    tp = int(((truth == 1) & (pred == 1)).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int(truth.sum()), 1)
    f1 = 2 * precision * recall / (precision + recall) if tp else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "flagged": int(pred.sum()), "injected": int(truth.sum()), "caught": tp}
