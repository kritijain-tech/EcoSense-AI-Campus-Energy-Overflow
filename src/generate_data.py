"""Synthetic campus electricity data generator.

Creates hourly electricity readings for five campus buildings. The data is
**synthetic**: it is produced from simple physical-looking rules (base load +
people-driven load + temperature-driven cooling load + noise) so the ML parts
of the project can be demonstrated without real meter data.

A small number of anomalies (night-time spikes, laboratory spikes and spikes
during low occupancy) are injected on purpose. They are recorded in the
``injected_anomaly`` columns, which are used ONLY to validate the anomaly
detector - they are never used as model inputs.

Usage (from the project root)::

    python -m src.generate_data
    python -m src.generate_data --days 120 --seed 42
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BUILDINGS, DATA_PATH, RANDOM_SEED, TARGET

START_DATE = "2026-04-01"
DEFAULT_DAYS = 120


@dataclass(frozen=True)
class BuildingProfile:
    """Parameters describing how one building uses electricity."""

    capacity: int                 # maximum people the building holds
    base_load_kwh: float          # always-on load (servers, lighting, standby)
    per_person_kwh: float         # extra load per person present
    cooling_kwh_per_deg: float    # cooling load per degree C above 24 C
    weekday_points: tuple         # (hour, occupancy fraction) key points
    weekend_points: tuple


PROFILES: dict[str, BuildingProfile] = {
    "Computer Lab": BuildingProfile(
        capacity=120, base_load_kwh=22.0, per_person_kwh=0.20, cooling_kwh_per_deg=1.5,
        weekday_points=((0, 0), (7, 0), (9, 0.5), (10, 0.9), (12, 0.85), (13, 0.6),
                        (15, 0.9), (17, 0.7), (19, 0.25), (21, 0.03), (23, 0)),
        weekend_points=((0, 0), (9, 0), (10, 0.2), (14, 0.3), (17, 0.15), (19, 0), (23, 0)),
    ),
    "Library": BuildingProfile(
        capacity=300, base_load_kwh=14.0, per_person_kwh=0.035, cooling_kwh_per_deg=1.1,
        weekday_points=((0, 0), (7, 0), (9, 0.35), (12, 0.7), (15, 0.85), (18, 0.7),
                        (20, 0.4), (21, 0.05), (23, 0)),
        weekend_points=((0, 0), (9, 0), (10, 0.3), (14, 0.5), (17, 0.4), (19, 0.1),
                        (20, 0), (23, 0)),
    ),
    "Classroom Block A": BuildingProfile(
        capacity=500, base_load_kwh=9.0, per_person_kwh=0.03, cooling_kwh_per_deg=1.4,
        weekday_points=((0, 0), (7, 0), (9, 0.8), (11, 0.95), (13, 0.55), (14, 0.9),
                        (16, 0.7), (17, 0.15), (18, 0), (23, 0)),
        weekend_points=((0, 0.02), (23, 0.02)),
    ),
    "Classroom Block B": BuildingProfile(
        capacity=450, base_load_kwh=8.0, per_person_kwh=0.03, cooling_kwh_per_deg=1.3,
        weekday_points=((0, 0), (8, 0), (10, 0.85), (12, 0.9), (13, 0.5), (15, 0.95),
                        (17, 0.75), (18, 0.2), (19, 0), (23, 0)),
        weekend_points=((0, 0), (10, 0.1), (14, 0.1), (16, 0), (23, 0)),
    ),
    "Administrative Block": BuildingProfile(
        capacity=180, base_load_kwh=12.0, per_person_kwh=0.07, cooling_kwh_per_deg=1.0,
        weekday_points=((0, 0), (8, 0), (9, 0.6), (10, 0.9), (13, 0.8), (14, 0.9),
                        (17, 0.8), (18, 0.2), (19, 0), (23, 0)),
        weekend_points=((0, 0.03), (23, 0.03)),
    ),
}


def _tariff(hour: int) -> float:
    """Illustrative time-of-use tariff in INR per kWh (not an official tariff)."""
    if 18 <= hour < 22:
        return 9.0   # evening peak
    if 6 <= hour < 18:
        return 7.5   # normal hours
    return 6.0       # off-peak night


def _generate_temperature(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    """Campus-wide outdoor temperature (deg C): season + daily cycle + weather noise."""
    n_days = len(timestamps) // 24
    day_index = np.arange(len(timestamps)) // 24
    hours = timestamps.hour.to_numpy()

    seasonal = 4.0 * np.sin(2 * np.pi * day_index / max(n_days, 1))
    daily_cycle = 6.0 * np.sin(2 * np.pi * (hours - 9) / 24)       # peaks mid-afternoon
    day_offset = rng.normal(0, 1.5, size=n_days)[day_index]         # warm / cool days

    # Smooth hour-to-hour noise (AR(1) process).
    noise = np.zeros(len(timestamps))
    for i in range(1, len(noise)):
        noise[i] = 0.8 * noise[i - 1] + rng.normal(0, 0.5)

    return np.clip(31 + seasonal + daily_cycle + day_offset + noise, 18, 46)


def _generate_building(name: str, timestamps: pd.DatetimeIndex, temperature: np.ndarray,
                       rng: np.random.Generator) -> pd.DataFrame:
    """Generate the normal (anomaly-free) readings of one building."""
    profile = PROFILES[name]
    hours = timestamps.hour.to_numpy()
    is_weekend = timestamps.dayofweek.to_numpy() >= 5

    weekday_xy = np.array(profile.weekday_points, dtype=float)
    weekend_xy = np.array(profile.weekend_points, dtype=float)
    frac_weekday = np.interp(hours, weekday_xy[:, 0], weekday_xy[:, 1])
    frac_weekend = np.interp(hours, weekend_xy[:, 0], weekend_xy[:, 1])
    fraction = np.where(is_weekend, frac_weekend, frac_weekday)

    # Day-to-day variation (exam days, absences...) + small random noise.
    n_days = len(timestamps) // 24
    day_factor = rng.normal(1.0, 0.07, size=n_days)[np.arange(len(timestamps)) // 24]
    occupancy = np.round(
        profile.capacity * fraction * day_factor * (1 + rng.normal(0, 0.08, len(timestamps)))
    )
    occupancy = np.clip(occupancy, 0, profile.capacity * 1.1).astype(int)

    occ_fraction = np.clip(occupancy / profile.capacity, 0, 1.1)
    cooling = (profile.cooling_kwh_per_deg * np.maximum(temperature - 24, 0)
               * (0.35 + 0.65 * occ_fraction))
    energy = profile.base_load_kwh + profile.per_person_kwh * occupancy + cooling
    energy = energy * (1 + rng.normal(0, 0.04, len(timestamps)))

    return pd.DataFrame({
        "timestamp": timestamps,
        "building_name": name,
        "temperature": np.round(temperature, 1),
        "occupancy": occupancy,
        TARGET: np.round(np.maximum(energy, 0.5), 3),
        "electricity_rate": [_tariff(h) for h in hours],
        "injected_anomaly": 0,
        "injected_anomaly_type": "",
    })


def _inject_anomalies(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Inject a small number of multi-hour anomaly events (about 0.5 % of rows)."""
    df = df.copy()
    first, last = df["timestamp"].min(), df["timestamp"].max()
    n_days = (last - first).days + 1

    def pick_start(allowed_hours: list[int], weekend: bool | None) -> pd.Timestamp:
        """Random timestamp whose hour (and optionally weekend flag) fits the event type."""
        while True:
            day = first.normalize() + pd.Timedelta(days=int(rng.integers(2, n_days - 1)))
            if weekend is not None and (day.dayofweek >= 5) != weekend:
                continue
            return day + pd.Timedelta(hours=int(rng.choice(allowed_hours)))

    events = []
    # 1) Night-time spikes in any building (equipment left running overnight).
    for _ in range(6):
        events.append(("night_high_consumption", str(rng.choice(BUILDINGS)),
                       pick_start([22, 23, 0, 1], None), int(rng.integers(3, 6)),
                       float(rng.uniform(2.0, 3.2))))
    # 2) Laboratory spikes during working hours.
    for _ in range(5):
        events.append(("lab_spike", "Computer Lab",
                       pick_start([9, 10, 11, 14, 15], False), int(rng.integers(2, 4)),
                       float(rng.uniform(1.8, 2.6))))
    # 3) High consumption although the building is almost empty (weekends).
    for _ in range(6):
        events.append(("high_consumption_low_occupancy",
                       str(rng.choice(["Library", "Classroom Block A", "Classroom Block B",
                                       "Administrative Block"])),
                       pick_start([10, 11, 12, 13], True), int(rng.integers(3, 7)),
                       float(rng.uniform(2.0, 2.8))))

    for event_type, building, start, duration, factor in events:
        mask = ((df["building_name"] == building)
                & (df["timestamp"] >= start)
                & (df["timestamp"] < start + pd.Timedelta(hours=duration)))
        df.loc[mask, TARGET] = (df.loc[mask, TARGET] * factor).round(3)
        df.loc[mask, "injected_anomaly"] = 1
        df.loc[mask, "injected_anomaly_type"] = event_type
    return df


def generate_energy_data(days: int = DEFAULT_DAYS, start: str = START_DATE,
                         seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return a synthetic hourly dataset for all campus buildings.

    Args:
        days: Number of days to simulate (at least 90 is recommended).
        start: First day of the simulation (YYYY-MM-DD).
        seed: Random seed - the same seed always gives the same dataset.
    """
    if days < 7:
        raise ValueError("days must be at least 7 to build weekly patterns.")
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(start=start, periods=days * 24, freq="h")
    temperature = _generate_temperature(timestamps, rng)

    frames = [_generate_building(name, timestamps, temperature, rng) for name in BUILDINGS]
    df = pd.concat(frames, ignore_index=True)
    df = _inject_anomalies(df, rng)
    return df.sort_values(["timestamp", "building_name"]).reset_index(drop=True)


def save_energy_data(df: pd.DataFrame, path: Path | str = DATA_PATH) -> Path:
    """Write the dataset to CSV, creating the folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic EcoSense AI dataset.")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help="days to simulate")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed")
    parser.add_argument("--output", type=Path, default=DATA_PATH, help="output CSV path")
    args = parser.parse_args()

    df = generate_energy_data(days=args.days, seed=args.seed)
    path = save_energy_data(df, args.output)
    print(f"Saved SYNTHETIC dataset: {path}")
    print(f"  rows: {len(df):,} | buildings: {df['building_name'].nunique()} | "
          f"period: {df['timestamp'].min():%Y-%m-%d} -> {df['timestamp'].max():%Y-%m-%d}")
    print(f"  injected anomalous hours: {int(df['injected_anomaly'].sum())} "
          f"({df['injected_anomaly'].mean():.2%})")


if __name__ == "__main__":
    main()
