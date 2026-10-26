"""Rule-based energy-saving recommendation engine.

Every recommendation separates three things so nobody mistakes a guess for a fact:

* **Observation**  - a number calculated from the data (verifiable).
* **Assumption**   - a possible explanation that the data alone cannot confirm.
* **Suggested check** - what a person could do on site to find out.

The engine takes the output of ``anomaly_detection.detect_anomalies``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.anomaly_detection import LOW_OCCUPANCY_LEVEL
from src.config import BUILDINGS, NIGHT_HOURS, TARGET

PRIORITY_ORDER = {"High": 0, "Medium": 1, "Low": 2}
PEAK_HOURS = (18, 19, 20, 21)


def _rec(building: str, priority: str, title: str, observation: str,
         assumption: str, check: str) -> dict:
    return {"building": building, "priority": priority, "title": title,
            "observation": observation, "assumption": assumption, "suggested_check": check}


def _hours(n: int) -> str:
    """'1 hour' / '5 hours'."""
    return f"{n} hour" + ("" if n == 1 else "s")


def _night_mask(df: pd.DataFrame) -> pd.Series:
    return df["timestamp"].dt.hour.isin(NIGHT_HOURS)


def _building_recommendations(df: pd.DataFrame, building: str,
                              night_ratio_all: dict[str, float]) -> list[dict]:
    """Recommendations for a single building's flagged data."""
    recs: list[dict] = []
    if len(df) < 48:
        return recs
    hours = df["timestamp"].dt.hour
    weekend = df["timestamp"].dt.dayofweek >= 5
    total_kwh = df[TARGET].sum()
    avg_rate = (df[TARGET] * df["electricity_rate"]).sum() / max(total_kwh, 1e-9)

    # 1) Night-time baseline load ------------------------------------------------
    night = df.loc[_night_mask(df), TARGET]
    day = df.loc[hours.between(10, 16) & ~weekend, TARGET]
    if len(night) and len(day) and day.mean() > 0:
        ratio = night.mean() / day.mean()
        peers = [v for b, v in night_ratio_all.items() if b != building]
        peer_median = float(np.median(peers)) if peers else ratio
        saving = 0.20 * night.sum()
        priority = "Medium" if ratio > 0.45 or ratio > 1.2 * peer_median else "Low"
        recs.append(_rec(
            building, priority, "Review the always-on night-time load",
            f"Night-time (22:00-05:59) consumption averages {night.mean():.1f} kWh/h, which is "
            f"{ratio:.0%} of the weekday daytime average ({day.mean():.1f} kWh/h). "
            f"The other buildings' median ratio is {peer_median:.0%}.",
            f"ASSUMPTION: part of this may be avoidable standby load (lighting, HVAC, idle "
            f"equipment). If 20% of it were avoided, it would save about {saving:,.0f} kWh "
            f"(~INR {saving * avg_rate:,.0f}) over this period - an estimate, not a measurement.",
            "Walk through the building after hours and list what is still switched on."))

    # 2) Night-time anomalies ----------------------------------------------------
    night_anoms = df[(df["is_anomaly"] == 1) & (df["anomaly_type"] == "High night-time consumption")]
    if len(night_anoms):
        excess = (night_anoms[TARGET] - night_anoms["expected_kwh"]).sum()
        dates = ", ".join(sorted({d.strftime("%d %b") for d in night_anoms["timestamp"]})[:4])
        recs.append(_rec(
            building, "High", "Investigate unusually high night-time consumption",
            f"{_hours(len(night_anoms))} of night-time consumption were flagged (e.g. around {dates}), using "
            f"about {excess:,.0f} kWh more than typical for those hours.",
            "ASSUMPTION: possible causes include equipment or air-conditioning left on, a "
            "scheduling change, or a metering problem. The data cannot tell which.",
            "Compare the flagged dates with the building's access / maintenance log."))

    # 3) Laboratory spikes -------------------------------------------------------
    lab_anoms = df[(df["is_anomaly"] == 1) & (df["anomaly_type"] == "Laboratory usage spike")]
    if len(lab_anoms):
        excess = (lab_anoms[TARGET] - lab_anoms["expected_kwh"]).sum()
        recs.append(_rec(
            building, "High", "Review laboratory equipment usage spikes",
            f"{_hours(len(lab_anoms))} of working-hour consumption were well above typical, "
            f"adding about {excess:,.0f} kWh.",
            "ASSUMPTION: possible causes are heavy equipment (servers, simulations, test rigs) "
            "or devices left running. Unconfirmed.",
            "Ask the lab in-charge what ran at those times and check equipment usage logs."))

    # 4) High consumption at low occupancy ----------------------------------------
    low_occ = df[(df["is_anomaly"] == 1) & (df["anomaly_type"] == "High consumption at low occupancy")]
    if len(low_occ):
        excess = (low_occ[TARGET] - low_occ["expected_kwh"]).sum()
        recs.append(_rec(
            building, "High", "Check high consumption while the building is lightly used",
            f"{_hours(len(low_occ))} had high consumption (about {excess:,.0f} kWh above typical) "
            f"although occupancy was below {LOW_OCCUPANCY_LEVEL:.0%} of the building's busy level.",
            "ASSUMPTION: lighting or cooling may be running for rooms nobody is using.",
            "Check whether rooms are being cooled/lit on weekends and whether timers are set."))

    # 5) Weekend vs weekday -------------------------------------------------------
    wk, we = df.loc[~weekend], df.loc[weekend]
    if len(wk) and len(we) and wk[TARGET].mean() > 0 and wk["occupancy"].mean() > 0:
        load_ratio = we[TARGET].mean() / wk[TARGET].mean()
        occ_ratio = we["occupancy"].mean() / wk["occupancy"].mean()
        if load_ratio > 0.7 and load_ratio > 1.8 * occ_ratio:
            recs.append(_rec(
                building, "Medium", "Weekend consumption is high compared with weekend use",
                f"Weekend load is {load_ratio:.0%} of the weekday load while weekend occupancy "
                f"is only {occ_ratio:.0%} of weekday occupancy.",
                "ASSUMPTION: some systems may follow the weekday schedule on weekends.",
                "Compare the building's weekend equipment/HVAC schedule with its real use."))

    # 6) Peak-tariff exposure ----------------------------------------------------
    peak = df[hours.isin(PEAK_HOURS)]
    if len(peak):
        share_kwh = peak[TARGET].sum() / total_kwh
        share_cost = (peak[TARGET] * peak["electricity_rate"]).sum() / (
            df[TARGET] * df["electricity_rate"]).sum()
        recs.append(_rec(
            building, "Low", "Look for flexible loads that could move out of the evening peak",
            f"Evening peak hours (18:00-21:59) carry {share_kwh:.0%} of consumption but "
            f"{share_cost:.0%} of the electricity cost under the illustrative tariff used here.",
            "ASSUMPTION: only loads that are actually flexible (e.g. pumps, battery chargers) "
            "can be shifted; classroom and lab loads usually cannot.",
            "List the loads running 18:00-22:00 and mark which could run earlier or later."))

    # 7) Compare with similar days ------------------------------------------------
    daily = df.groupby(df["timestamp"].dt.normalize())[TARGET].agg(["sum", "count"])
    daily = daily[daily["count"] == 24]["sum"]
    if len(daily) >= 14:
        last_day = daily.index[-1]
        same_type = daily[(daily.index.dayofweek == last_day.dayofweek) & (daily.index < last_day)]
        if len(same_type) >= 3:
            typical = same_type.median()
            diff = (daily.iloc[-1] - typical) / typical
            if abs(diff) >= 0.10:
                word = "higher" if diff > 0 else "lower"
                recs.append(_rec(
                    building, "Medium" if diff > 0 else "Low",
                    "Latest day differs from similar days",
                    f"{last_day:%A %d %b}: {daily.iloc[-1]:,.0f} kWh, {abs(diff):.0%} {word} than "
                    f"the median of {len(same_type)} earlier {last_day:%A}s ({typical:,.0f} kWh).",
                    "ASSUMPTION: could be weather, a special event or a schedule change.",
                    "Check the calendar and weather for that day before drawing conclusions."))

    # 8) Temperature sensitivity --------------------------------------------------
    daily_temp = df.groupby(df["timestamp"].dt.normalize())["temperature"].mean()
    full_days = daily.index
    if len(full_days) >= 14:
        corr = np.corrcoef(daily_temp.loc[full_days], daily.loc[full_days])[0, 1]
        if np.isfinite(corr) and corr > 0.4:
            recs.append(_rec(
                building, "Low", "Consumption follows the weather - review cooling set-points",
                f"Daily consumption and average temperature are positively correlated "
                f"(r = {corr:.2f}).",
                "ASSUMPTION: correlation is not causation, but it is consistent with "
                "cooling-driven load.",
                "Check thermostat set-points and the maintenance state of cooling equipment."))
    return recs


def generate_recommendations(flagged: pd.DataFrame, building: str | None = None) -> list[dict]:
    """Return recommendations (highest priority first) for one building or the whole campus."""
    if flagged.empty:
        return []
    night_ratio_all = {}
    for name, part in flagged.groupby("building_name"):
        day = part.loc[part["timestamp"].dt.hour.between(10, 16)
                       & (part["timestamp"].dt.dayofweek < 5), TARGET].mean()
        night = part.loc[_night_mask(part), TARGET].mean()
        if pd.notna(day) and day > 0 and pd.notna(night):
            night_ratio_all[name] = night / day

    targets = [building] if building and building in BUILDINGS else sorted(flagged["building_name"].unique())
    recs: list[dict] = []
    for name in targets:
        part = flagged[flagged["building_name"] == name]
        recs.extend(_building_recommendations(part, name, night_ratio_all))
    return sorted(recs, key=lambda r: (PRIORITY_ORDER[r["priority"]], r["building"]))


def recommendations_to_markdown(recs: list[dict], title: str = "EcoSense AI - Recommendations") -> str:
    """Plain-text (Markdown) export of the recommendations."""
    lines = [f"# {title}", "", "_Based on SYNTHETIC data. Observations are calculated; "
             "assumptions are not verified._", ""]
    for r in recs:
        lines += [f"## [{r['priority']}] {r['building']}: {r['title']}",
                  f"- **Observation:** {r['observation']}",
                  f"- **Assumption:** {r['assumption']}",
                  f"- **Suggested check:** {r['suggested_check']}", ""]
    if not recs:
        lines.append("No recommendations were generated.")
    return "\n".join(lines)
