"""EcoSense AI - Streamlit dashboard.

Run from the project root::

    streamlit run app.py

The dashboard works fully offline. If the dataset or trained model is missing it
creates them automatically on first start.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.anomaly_detection import (ANOMALY_TYPES, anomaly_table, detect_anomalies,
                                   evaluate_against_injected)
from src.config import (ANOMALY_NOTICE, BUILDINGS, DATA_PATH, DATASET_NOTICE,
                        DEFAULT_CONTAMINATION, MODEL_PATH, TARGET)
from src.generate_data import generate_energy_data, save_energy_data
from src.predict import (MODEL_COLUMNS, build_prediction_frame, load_bundle,
                         predict_single, summarize_predictions)
from src.preprocess import load_data
from src.recommendations import generate_recommendations, recommendations_to_markdown
from src.train_model import regression_metrics, save_bundle, train_and_evaluate

# --------------------------------------------------------------------------
# Look & feel
# --------------------------------------------------------------------------
BLUE, GREEN, NAVY = "#1565C0", "#2E9E5B", "#0D2B52"
ORANGE, GREY = "#E8590C", "#6B7C93"
ALL = "All Buildings"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
PAGES = ["Overview", "Energy Analytics", "ML Predictions", "Anomaly Detection",
         "Recommendations", "Reports"]
BUILDING_COLORS = {
    "Computer Lab": "#0D47A1", "Library": "#2E9E5B", "Classroom Block A": "#42A5F5",
    "Classroom Block B": "#9CCC65", "Administrative Block": "#00838F",
}
TYPE_COLORS = dict(zip(ANOMALY_TYPES, ["#6A1B9A", ORANGE, "#C62828", "#F9A825", "#00838F", GREY]))
PRIORITY_COLORS = {"High": "#C62828", "Medium": "#EF8F00", "Low": GREEN}

CSS = """
<style>
.block-container {padding-top: 1.2rem; max-width: 1400px;}
.hero {background: linear-gradient(120deg, #0D47A1 0%, #1565C0 55%, #2E9E5B 100%);
       color: #fff; padding: 1.2rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;}
.hero h1 {color: #fff; font-size: 1.7rem; margin: 0; padding: 0;}
.hero p {color: #E3F0FF; margin: .25rem 0 0 0; font-size: .95rem;}
.kpi {background: #fff; border-radius: 12px; padding: .9rem 1.1rem; color: #1B2A3A;
      box-shadow: 0 1px 6px rgba(13,43,82,.12); border-top: 4px solid #1565C0; min-height: 118px;}
.kpi-label {font-size: .74rem; text-transform: uppercase; letter-spacing: .06em; color: #6B7C93;}
.kpi-value {font-size: 1.75rem; font-weight: 700; line-height: 1.25;}
.kpi-sub {font-size: .8rem; color: #6B7C93;}
.pill {display: inline-block; padding: 2px 10px; border-radius: 999px; color: #fff;
       font-size: .75rem; font-weight: 600; margin-right: .4rem;}
.brand {font-size: 1.35rem; font-weight: 800; color: #0D2B52;}
.brand span {color: #2E9E5B;}
</style>
"""

_PLOTLY_HAS_WIDTH = "width" in inspect.signature(st.plotly_chart).parameters
_DATAFRAME_HAS_WIDTH = "width" in inspect.signature(st.dataframe).parameters


def show_chart(fig: go.Figure, height: int = 380) -> None:
    """Render a Plotly figure with the shared styling (works across Streamlit versions)."""
    fig.update_layout(
        template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        colorway=[BLUE, GREEN, "#42A5F5", "#9CCC65", "#00838F"],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    if _PLOTLY_HAS_WIDTH:
        st.plotly_chart(fig, width="stretch")
    else:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df: pd.DataFrame, height: int | None = None) -> None:
    """Render a DataFrame full-width (works across Streamlit versions)."""
    kwargs = {"hide_index": True}
    if height:
        kwargs["height"] = height
    if _DATAFRAME_HAS_WIDTH:
        st.dataframe(df, width="stretch", **kwargs)
    else:
        st.dataframe(df, use_container_width=True, **kwargs)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", color: str = BLUE) -> None:
    st.markdown(
        f'<div class="kpi" style="border-top-color:{color}"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div><div class="kpi-sub">{sub}</div></div>',
        unsafe_allow_html=True)


def fmt_kwh(value: float) -> str:
    return f"{value:,.0f} kWh"


def fmt_inr(value: float) -> str:
    return f"₹{value:,.0f}"


# --------------------------------------------------------------------------
# Data layer (cached)
# --------------------------------------------------------------------------
def ensure_artifacts() -> None:
    """Create the dataset / model on first start if they are missing."""
    if not DATA_PATH.exists():
        with st.spinner("Generating the synthetic dataset (first start only)..."):
            save_energy_data(generate_energy_data())
    if not MODEL_PATH.exists():
        with st.spinner("Training the ML models (first start only)..."):
            save_bundle(train_and_evaluate(load_data()))


@st.cache_data(show_spinner=False)
def get_data() -> pd.DataFrame:
    return load_data()


@st.cache_resource(show_spinner=False)
def get_bundle() -> dict:
    return load_bundle()


@st.cache_data(show_spinner="Scoring the data with the trained models...")
def get_predictions(df: pd.DataFrame) -> pd.DataFrame:
    return build_prediction_frame(df, get_bundle())


@st.cache_data(show_spinner="Running Isolation Forest...")
def get_flagged(df: pd.DataFrame, contamination: float) -> pd.DataFrame:
    return detect_anomalies(df, contamination=contamination)


def in_range(frame: pd.DataFrame, building: str, start, end) -> pd.DataFrame:
    """Rows of ``frame`` inside the sidebar filters."""
    mask = (frame["timestamp"] >= pd.Timestamp(start)) & \
           (frame["timestamp"] < pd.Timestamp(end) + pd.Timedelta(days=1))
    if building != ALL:
        mask &= frame["building_name"] == building
    return frame[mask]


def daily_totals(df: pd.DataFrame, value: str = TARGET, by_building: bool = False) -> pd.DataFrame:
    keys = [df["timestamp"].dt.normalize().rename("date")]
    if by_building:
        keys.append(df["building_name"])
    return df.groupby(keys)[value].sum().reset_index()


# --------------------------------------------------------------------------
# Pages
# --------------------------------------------------------------------------
def page_overview(ctx) -> None:
    hero("Campus Energy Overview",
         f"{ctx.building} · {ctx.start:%d %b %Y} → {ctx.end:%d %b %Y}")
    st.info(DATASET_NOTICE)
    df, pred, flagged = ctx.raw, ctx.pred, ctx.flagged
    if df.empty:
        st.warning("No data for the selected filters.")
        return

    total = df[TARGET].sum()
    cost = (df[TARGET] * df["electricity_rate"]).sum()
    test = pred[pred["split"] == "test"]
    n_anom = int(flagged["is_anomaly"].sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        kpi("Total consumption", fmt_kwh(total), f"{len(df):,} hourly readings")
    with c2:
        if test.empty:
            kpi("Predicted consumption", "—", "no held-out test days in this range", GREEN)
        else:
            p, a = test["predicted_kwh"].sum(), test[TARGET].sum()
            kpi("Predicted consumption", fmt_kwh(p),
                f"vs {fmt_kwh(a)} actual ({100 * (p - a) / a:+.1f}%) · test period", GREEN)
    with c3:
        kpi("Estimated cost", fmt_inr(cost), "illustrative tariff", BLUE)
    with c4:
        kpi("Average load", f"{total / df['timestamp'].nunique():,.1f} kWh/h",
            "per hour, all selected buildings", GREEN)
    with c5:
        kpi("Anomalies flagged", f"{n_anom}", f"{n_anom / len(df):.2%} of hours · potential issues",
            ORANGE)
    st.write("")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Consumption trend")
        daily = daily_totals(df)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily[TARGET], name="Actual (daily kWh)",
                                 mode="lines", line=dict(color=BLUE, width=2.5)))
        if not test.empty:
            dp = daily_totals(test, "predicted_kwh")
            fig.add_trace(go.Scatter(x=dp["date"], y=dp["predicted_kwh"], name="Predicted (test period)",
                                     mode="lines", line=dict(color=GREEN, width=2.5, dash="dash")))
            if daily["date"].min() < ctx.cutoff <= daily["date"].max():
                fig.add_shape(type="line", x0=ctx.cutoff, x1=ctx.cutoff, y0=0, y1=1,
                              yref="paper", line=dict(color=GREY, width=1, dash="dot"))
        fig.update_yaxes(title="kWh per day")
        show_chart(fig)
        st.caption("The dotted vertical line marks where the held-out test period starts; "
                   "predictions are only shown for days the model never saw during training.")
    with right:
        st.subheader("Building-wise consumption")
        by_b = df.groupby("building_name")[TARGET].sum().reset_index().sort_values(TARGET)
        fig = px.bar(by_b, x=TARGET, y="building_name", orientation="h", color="building_name",
                     color_discrete_map=BUILDING_COLORS)
        fig.update_layout(showlegend=False, xaxis_title="kWh", yaxis_title="")
        show_chart(fig)

    st.subheader("Latest detected anomalies")
    st.caption(ANOMALY_NOTICE)
    table = anomaly_table(flagged)
    if table.empty:
        st.success("No anomalies were flagged for the selected filters.")
    else:
        show_table(table[["timestamp", "building_name", TARGET, "expected_kwh", "energy_ratio",
                          "anomaly_type"]].head(8))


def page_analytics(ctx) -> None:
    hero("Energy Analytics", "Daily, weekly and hourly consumption patterns")
    df = ctx.raw
    if df.empty:
        st.warning("No data for the selected filters.")
        return
    df = df.assign(hour=df["timestamp"].dt.hour, dow=df["timestamp"].dt.dayofweek)
    tab1, tab2, tab3 = st.tabs(["Daily & weekly", "Hourly patterns", "Weather, occupancy & cost"])

    with tab1:
        st.subheader("Daily consumption")
        multi = ctx.building == ALL
        daily = daily_totals(df, by_building=multi)
        if multi:
            fig = px.bar(daily, x="date", y=TARGET, color="building_name",
                         color_discrete_map=BUILDING_COLORS)
        else:
            fig = px.line(daily, x="date", y=TARGET)
            fig.update_traces(line=dict(color=BLUE, width=2.5))
        fig.update_layout(yaxis_title="kWh per day", xaxis_title="", legend_title="")
        show_chart(fig)

        st.subheader("Weekly consumption")
        hide_partial = st.checkbox("Hide incomplete weeks", value=True)
        d = df.assign(week=df["timestamp"].dt.normalize()
                      - pd.to_timedelta(df["timestamp"].dt.dayofweek, unit="D"),
                      day=df["timestamp"].dt.normalize())
        weekly = d.groupby(["week", "building_name"])[TARGET].sum().reset_index()
        days_in_week = d.groupby("week")["day"].nunique()
        if hide_partial:
            weekly = weekly[weekly["week"].map(days_in_week) == 7]
        if weekly.empty:
            st.info("The selected range has no complete week - untick the box to see partial weeks.")
        else:
            fig = px.bar(weekly, x="week", y=TARGET, color="building_name",
                         color_discrete_map=BUILDING_COLORS)
            fig.update_layout(yaxis_title="kWh per week", xaxis_title="Week starting (Monday)",
                              legend_title="")
            show_chart(fig)

    with tab2:
        st.subheader("Average hourly profile: weekday vs weekend")
        prof = df.assign(day_type=np.where(df["dow"] >= 5, "Weekend", "Weekday")) \
                 .groupby(["day_type", "hour"])[TARGET].mean().reset_index()
        fig = px.line(prof, x="hour", y=TARGET, color="day_type", markers=True,
                      color_discrete_map={"Weekday": BLUE, "Weekend": GREEN})
        fig.update_layout(xaxis_title="Hour of day", yaxis_title="Mean kWh per hour", legend_title="")
        show_chart(fig)

        st.subheader("When is energy used? (mean kWh by weekday and hour)")
        pivot = df.pivot_table(index="dow", columns="hour", values=TARGET, aggfunc="mean")
        pivot = pivot.reindex(range(7))
        fig = go.Figure(go.Heatmap(z=pivot.values, x=pivot.columns, y=DAYS,
                                   colorscale=[[0, "#EAF3FB"], [0.5, "#64B5F6"], [1, "#0D47A1"]],
                                   colorbar=dict(title="kWh")))
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(xaxis_title="Hour of day")
        show_chart(fig, 360)

    with tab3:
        l, r = st.columns(2)
        sample = df.sample(min(len(df), 3000), random_state=1)
        with l:
            st.subheader("Temperature vs consumption")
            fig = px.scatter(sample, x="temperature", y=TARGET, color="building_name", opacity=0.5,
                             color_discrete_map=BUILDING_COLORS)
            fig.update_layout(xaxis_title="Temperature (°C)", yaxis_title="kWh per hour",
                              legend_title="")
            show_chart(fig)
        with r:
            st.subheader("Occupancy vs consumption")
            fig = px.scatter(sample, x="occupancy", y=TARGET, color="building_name", opacity=0.5,
                             color_discrete_map=BUILDING_COLORS)
            fig.update_layout(xaxis_title="People present", yaxis_title="kWh per hour",
                              legend_title="")
            show_chart(fig)
        st.subheader("Estimated electricity cost by building")
        cost = (df.assign(cost=df[TARGET] * df["electricity_rate"])
                .groupby("building_name")["cost"].sum().reset_index().sort_values("cost"))
        fig = px.bar(cost, x="cost", y="building_name", orientation="h", color="building_name",
                     color_discrete_map=BUILDING_COLORS)
        fig.update_layout(showlegend=False, xaxis_title="INR (illustrative tariff)", yaxis_title="")
        show_chart(fig, 320)


def page_ml(ctx) -> None:
    hero("ML Predictions", "Random Forest vs Gradient Boosting - trained on the past, tested on the future")
    bundle, pred_all = ctx.bundle, ctx.pred_all
    best = bundle["best_model_name"]
    metrics, base = bundle["metrics"], bundle["baseline_metrics"]
    st.info(
        f"**Chronological split:** trained on {bundle['train_period'][0][:10]} → "
        f"{bundle['train_period'][1][:10]} ({bundle['train_rows']:,} rows) and tested on "
        f"{bundle['test_period'][0][:10]} → {bundle['test_period'][1][:10]} "
        f"({bundle['test_rows']:,} rows). The test period is never used for training. "
        f"Best model rule: {bundle['selection_rule']}.")

    gain = 1 - metrics[best]["rmse"] / base["rmse"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Best model", best, "by test RMSE", GREEN)
    with c2:
        kpi("Test RMSE", f"{metrics[best]['rmse']:.2f} kWh", f"{gain:.0%} lower than baseline")
    with c3:
        kpi("Test MAE", f"{metrics[best]['mae']:.2f} kWh", "average absolute error", GREEN)
    with c4:
        kpi("Test R²", f"{metrics[best]['r2']:.3f}", "share of variance explained")

    st.subheader("Model comparison (test set)")
    rows = [{"Model": n, "MAE (kWh)": m["mae"], "RMSE (kWh)": m["rmse"], "R²": m["r2"],
             "Result": "Best (lowest RMSE)" if n == best else ""} for n, m in metrics.items()]
    rows.append({"Model": "Baseline: previous hour", "MAE (kWh)": base["mae"],
                 "RMSE (kWh)": base["rmse"], "R²": base["r2"], "Result": "Reference only"})
    table = pd.DataFrame(rows).round({"MAE (kWh)": 3, "RMSE (kWh)": 3, "R²": 4})
    show_table(table)
    l, r = st.columns(2)
    with l:
        err = pd.DataFrame([{"Model": n, "Metric": k, "kWh": v} for n, m in metrics.items()
                            for k, v in (("MAE", m["mae"]), ("RMSE", m["rmse"]))])
        fig = px.bar(err, x="Metric", y="kWh", color="Model", barmode="group",
                     color_discrete_map={"Random Forest": BLUE, "Gradient Boosting": GREEN})
        fig.update_layout(title="Error (lower is better)", legend_title="")
        show_chart(fig, 320)
    with r:
        r2 = pd.DataFrame({"Model": list(metrics), "R²": [m["r2"] for m in metrics.values()]})
        fig = px.bar(r2, x="Model", y="R²", color="Model",
                     color_discrete_map={"Random Forest": BLUE, "Gradient Boosting": GREEN})
        fig.update_layout(title="R² (higher is better)", showlegend=False)
        show_chart(fig, 320)

    st.subheader("Predicted vs actual")
    names = list(MODEL_COLUMNS)
    choice = st.radio("Model shown below", names, index=names.index(best), horizontal=True,
                      format_func=lambda n: f"{n} (best)" if n == best else n)
    col = MODEL_COLUMNS[choice]
    test = pred_all[pred_all["split"] == "test"]
    view = test if ctx.building == ALL else test[test["building_name"] == ctx.building]
    if view.empty:
        st.warning("No test-period data available.")
    else:
        n_days = view["timestamp"].dt.normalize().nunique()
        show_days = st.slider("Days to show (ending at the last test day)", 1, min(n_days, 21),
                              min(7, n_days))
        hourly = view.groupby("timestamp")[[TARGET, col]].sum().reset_index()
        hourly = hourly[hourly["timestamp"] >= hourly["timestamp"].max() - pd.Timedelta(days=show_days)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hourly["timestamp"], y=hourly[TARGET], name="Actual",
                                 line=dict(color=BLUE, width=2)))
        fig.add_trace(go.Scatter(x=hourly["timestamp"], y=hourly[col], name="Predicted",
                                 line=dict(color=GREEN, width=2, dash="dash")))
        fig.update_layout(title=f"{ctx.building}: hourly kWh", yaxis_title="kWh per hour")
        show_chart(fig)

        l, r = st.columns(2)
        with l:
            sample = view.sample(min(len(view), 1500), random_state=1)
            fig = px.scatter(sample, x=TARGET, y=col, opacity=0.5, color_discrete_sequence=[BLUE])
            top = float(max(sample[TARGET].max(), sample[col].max()))
            fig.add_shape(type="line", x0=0, y0=0, x1=top, y1=top, line=dict(color=GREEN, dash="dash"))
            fig.update_layout(title="Actual vs predicted (closer to the dashed line is better)",
                              xaxis_title="Actual kWh", yaxis_title="Predicted kWh")
            show_chart(fig, 340)
        with r:
            resid = view[col] - view[TARGET]
            fig = px.histogram(resid, nbins=50, color_discrete_sequence=[GREEN])
            fig.update_layout(title="Prediction errors (predicted - actual)", showlegend=False,
                              xaxis_title="kWh", yaxis_title="Hours")
            show_chart(fig, 340)

    l, r = st.columns(2)
    with l:
        st.subheader("Per-building test accuracy")
        rows = []
        for name, part in test.groupby("building_name"):
            m = regression_metrics(part[TARGET], part[col])
            rows.append({"Building": name, "MAE": m["mae"], "RMSE": m["rmse"], "R²": m["r2"]})
        show_table(pd.DataFrame(rows).round(3))
    with r:
        st.subheader("What drives the predictions?")
        imp = pd.Series(bundle["feature_importances"][choice]).sort_values().tail(8)
        fig = px.bar(x=imp.values, y=imp.index, orientation="h", color_discrete_sequence=[BLUE])
        fig.update_layout(xaxis_title="Importance", yaxis_title="")
        show_chart(fig, 320)
    st.caption("Previous-hour consumption is usually the strongest feature - consumption is "
               "persistent. Feature importance describes the model, not physical causation.")

    st.subheader("What-if prediction")
    st.caption("Estimate the consumption of one hour. Temperature and occupancy are treated as "
               "known inputs (a weather forecast and a timetable-based occupancy estimate).")
    raw = ctx.raw_all
    with st.form("what_if"):
        a, b, c = st.columns(3)
        building = a.selectbox("Building to predict", BUILDINGS,
                               index=BUILDINGS.index(ctx.building) if ctx.building in BUILDINGS else 0)
        hour = b.slider("Hour of day", 0, 23, 14)
        day_name = c.selectbox("Day of week", DAYS, index=1)
        d, e, f, g = st.columns(4)
        temp = d.number_input("Temperature (°C)", -10.0, 55.0, 34.0, 0.5)
        occ = e.number_input("People present", 0, 5000, 150, 10)
        lag1 = f.number_input("Consumption 1 h ago (kWh)", 0.0, 1000.0, 40.0, 1.0)
        lag24 = g.number_input("Consumption 24 h ago (kWh)", 0.0, 1000.0, 40.0, 1.0)
        go_predict = st.form_submit_button("Predict consumption")
    if go_predict:
        dow = DAYS.index(day_name)
        result = predict_single(bundle, building=building, hour=hour, day_of_week=dow,
                                temperature=temp, occupancy=int(occ), lag_1h=lag1, lag_24h=lag24)
        rate = raw.loc[raw["timestamp"].dt.hour == hour, "electricity_rate"].median()
        typical = raw[(raw["building_name"] == building) & (raw["timestamp"].dt.hour == hour)
                      & ((raw["timestamp"].dt.dayofweek >= 5) == (dow >= 5))][TARGET].median()
        m1, m2, m3 = st.columns(3)
        m1.metric("Random Forest", f"{result['Random Forest']:.1f} kWh")
        m2.metric("Gradient Boosting", f"{result['Gradient Boosting']:.1f} kWh")
        m3.metric(f"Estimated cost ({best})", fmt_inr(result[best] * rate))
        st.caption(f"For reference, the historical median for this building, hour and day type "
                   f"is {typical:.1f} kWh.")


def page_anomalies(ctx) -> None:
    hero("Anomaly Detection", "Isolation Forest finds unusual hours of electricity use")
    st.warning(ANOMALY_NOTICE)
    flagged = ctx.flagged
    if flagged.empty:
        st.warning("No data for the selected filters.")
        return
    an = flagged[flagged["is_anomaly"] == 1]
    excess = float((an[TARGET] - an["expected_kwh"]).clip(lower=0).sum())
    worst = an["building_name"].value_counts().idxmax() if len(an) else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi("Anomalous hours", f"{len(an)}", f"{len(an) / len(flagged):.2%} of hours", ORANGE)
    with c2:
        kpi("Excess energy", fmt_kwh(excess), "above typical, flagged hours only", BLUE)
    with c3:
        kpi("Most flagged building", worst, "potential hotspot to inspect", GREEN)
    with c4:
        kpi("Sensitivity", f"{ctx.contamination:.1%}", "set in the sidebar", BLUE)
    st.write("")

    view = ctx.building
    if view == ALL:
        view = st.selectbox("Building to inspect", BUILDINGS)
    bdf = flagged[flagged["building_name"] == view].sort_values("timestamp")
    st.subheader(f"{view}: consumption with flagged hours")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bdf["timestamp"], y=bdf[TARGET], mode="lines", name="Actual kWh",
                             line=dict(color=BLUE, width=1.4)))
    fig.add_trace(go.Scatter(x=bdf["timestamp"], y=bdf["expected_kwh"], mode="lines",
                             name="Typical kWh", line=dict(color=GREEN, width=1, dash="dot")))
    for kind, part in bdf[bdf["is_anomaly"] == 1].groupby("anomaly_type"):
        fig.add_trace(go.Scatter(
            x=part["timestamp"], y=part[TARGET], mode="markers", name=kind,
            marker=dict(size=9, color=TYPE_COLORS.get(kind, ORANGE), line=dict(width=1, color="white")),
            text=part["anomaly_reason"], hovertemplate="%{x}<br>%{text}<extra></extra>"))
    fig.update_layout(yaxis_title="kWh per hour")
    show_chart(fig, 420)

    l, m, r = st.columns(3)
    if len(an):
        with l:
            by_type = an["anomaly_type"].value_counts().reset_index()
            by_type.columns = ["type", "hours"]
            fig = px.bar(by_type, x="hours", y="type", orientation="h", color="type",
                         color_discrete_map=TYPE_COLORS)
            fig.update_layout(title="By type", showlegend=False, yaxis_title="")
            show_chart(fig, 300)
        with m:
            fig = px.histogram(an, x=an["timestamp"].dt.hour, nbins=24, color_discrete_sequence=[ORANGE])
            fig.update_layout(title="By hour of day", xaxis_title="Hour", yaxis_title="Anomalies")
            show_chart(fig, 300)
        with r:
            by_b = an["building_name"].value_counts().reset_index()
            by_b.columns = ["building", "hours"]
            fig = px.bar(by_b, x="building", y="hours", color="building",
                         color_discrete_map=BUILDING_COLORS)
            fig.update_layout(title="By building", showlegend=False, xaxis_title="")
            show_chart(fig, 300)

        st.subheader("Flagged hours (most unusual first)")
        show_table(anomaly_table(flagged).drop(columns=["temperature"]), height=380)
    else:
        st.success("No anomalies were flagged for the selected filters.")

    with st.expander("How does the detection work?"):
        st.markdown(
            "1. For every building, hour of day and weekday/weekend, the **typical** (median) "
            "consumption is calculated.\n"
            "2. The ratio *actual / typical*, the occupancy level and the time of day are given to an "
            "**Isolation Forest** (one per building). It isolates readings that are rare and "
            "different - they get a high anomaly score.\n"
            "3. The most unusual readings (share set by the sensitivity slider) are flagged and "
            "labelled with a simple rule (night-time, laboratory, low occupancy...).\n\n"
            "A flagged hour means *statistically unusual*, not *faulty*. A real cause needs an "
            "on-site check.")
    with st.expander("Sanity check against the anomalies injected into the synthetic data"):
        result = evaluate_against_injected(ctx.flagged_all)
        if result is None:
            st.write("No ground-truth labels are available for this dataset.")
        else:
            a, b, c = st.columns(3)
            a.metric("Precision", f"{result['precision']:.0%}")
            b.metric("Recall", f"{result['recall']:.0%}")
            c.metric("F1", f"{result['f1']:.2f}")
            st.caption(f"{result['caught']} of {result['injected']} injected anomalous hours were "
                       f"caught; {result['flagged']} hours were flagged in total. This is possible "
                       "only because the data is synthetic - with real data the true faults are unknown.")


def page_recommendations(ctx) -> None:
    hero("Energy-Saving Recommendations", "Calculated observations, clearly separated from assumptions")
    st.info("**Observation** = calculated from the data. **Assumption** = a possible explanation "
            "the data cannot confirm. **Suggested check** = how to verify it on site.")
    recs = generate_recommendations(ctx.flagged_dates, None if ctx.building == ALL else ctx.building)
    if not recs:
        st.success("Nothing noteworthy found for the selected filters (or too little data).")
        return
    counts = pd.Series([r["priority"] for r in recs]).value_counts()
    a, b, c = st.columns(3)
    a.metric("High priority", int(counts.get("High", 0)))
    b.metric("Medium priority", int(counts.get("Medium", 0)))
    c.metric("Low priority", int(counts.get("Low", 0)))

    for r in recs:
        with st.container(border=True):
            st.markdown(
                f'<span class="pill" style="background:{PRIORITY_COLORS[r["priority"]]}">'
                f'{r["priority"]}</span> **{r["building"]}** — {r["title"]}',
                unsafe_allow_html=True)
            st.markdown(f"**Observation (calculated):** {r['observation']}")
            st.markdown(f"**Assumption (not verified):** {r['assumption']}")
            st.markdown(f"**Suggested check:** {r['suggested_check']}")
    st.download_button("Download recommendations (Markdown)", recommendations_to_markdown(recs),
                       file_name="ecosense_recommendations.md", mime="text/markdown")


def page_reports(ctx) -> None:
    hero("Reports", "Predicted vs actual summary and downloadable results")
    st.info(DATASET_NOTICE)
    best = ctx.bundle["best_model_name"]

    st.subheader(f"Predicted vs actual consumption - held-out test period ({best})")
    summary = summarize_predictions(ctx.pred_all)
    if summary.empty:
        st.warning("No test-period predictions available.")
    else:
        show_table(summary.rename(columns={
            "building_name": "Building", "actual_kwh": "Actual (kWh)", "predicted_kwh": "Predicted (kWh)",
            "error_kwh": "Error (kWh)", "error_pct": "Error (%)"}))
        long = summary[summary["building_name"] != ALL].melt(
            id_vars="building_name", value_vars=["actual_kwh", "predicted_kwh"],
            var_name="Series", value_name="kWh")
        long["Series"] = long["Series"].map({"actual_kwh": "Actual", "predicted_kwh": "Predicted"})
        fig = px.bar(long, x="building_name", y="kWh", color="Series", barmode="group",
                     color_discrete_map={"Actual": BLUE, "Predicted": GREEN})
        fig.update_layout(xaxis_title="", legend_title="")
        show_chart(fig, 340)

    st.subheader("Downloads")
    st.caption(f"Reports cover the sidebar filters: **{ctx.building}**, "
               f"{ctx.start:%d %b %Y} → {ctx.end:%d %b %Y}.")
    fl = ctx.flagged
    analysis = fl[["timestamp", "building_name", "temperature", "occupancy", TARGET,
                   "electricity_rate", "expected_kwh", "is_anomaly", "anomaly_type"]].copy()
    analysis["cost_inr"] = analysis[TARGET] * analysis["electricity_rate"]
    preds = ctx.pred[["timestamp", "building_name", "predicted_kwh", "split"]]
    analysis = analysis.merge(preds, on=["timestamp", "building_name"], how="left")
    numeric_cols = analysis.select_dtypes("number").columns
    analysis[numeric_cols] = analysis[numeric_cols].round(3)
    anomalies = anomaly_table(fl)
    by_building = (fl.assign(cost=fl[TARGET] * fl["electricity_rate"])
                   .groupby("building_name")
                   .agg(total_kwh=(TARGET, "sum"), avg_kwh_per_hour=(TARGET, "mean"),
                        peak_kwh_per_hour=(TARGET, "max"), cost_inr=("cost", "sum"),
                        anomalous_hours=("is_anomaly", "sum")).round(2).reset_index())

    a, b, c = st.columns(3)
    a.download_button("Energy analysis (CSV)", analysis.to_csv(index=False).encode("utf-8"),
                      file_name="ecosense_energy_analysis.csv", mime="text/csv")
    b.download_button("Anomaly results (CSV)", anomalies.to_csv(index=False).encode("utf-8"),
                      file_name="ecosense_anomalies.csv", mime="text/csv")
    c.download_button("Building summary (CSV)", by_building.to_csv(index=False).encode("utf-8"),
                      file_name="ecosense_building_summary.csv", mime="text/csv")
    st.caption("`predicted_kwh` is empty for the first 24 hours (no previous-day value exists yet). "
               "`split = train` rows were seen by the model during training, so their predictions "
               "are in-sample and look better than they would on new data.")
    st.subheader("Building summary")
    show_table(by_building)


# --------------------------------------------------------------------------
# App shell
# --------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="EcoSense AI", page_icon="⚡", layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(CSS, unsafe_allow_html=True)
    ensure_artifacts()

    try:
        raw_all = get_data()
        bundle = get_bundle()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"Could not load the project data: {exc}")
        st.stop()
        return

    with st.sidebar:
        st.markdown('<div class="brand">⚡ EcoSense <span>AI</span></div>', unsafe_allow_html=True)
        st.caption("Campus energy monitoring, prediction & anomaly detection")
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        building = st.selectbox("Building", [ALL] + BUILDINGS)
        lo, hi = raw_all["timestamp"].min().date(), raw_all["timestamp"].max().date()
        picked = st.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi)
        if isinstance(picked, (tuple, list)):
            start, end = (picked[0], picked[-1]) if len(picked) else (lo, hi)
        else:
            start = end = picked
        contamination = st.slider("Anomaly sensitivity (% of hours flagged)", 0.2, 3.0,
                                  DEFAULT_CONTAMINATION * 100, 0.1) / 100
        st.divider()
        st.caption("Synthetic demo data · runs fully offline")

    flagged_all = get_flagged(raw_all, contamination)
    pred_all = get_predictions(raw_all)
    ctx = SimpleNamespace(
        building=building, start=start, end=end, contamination=contamination, bundle=bundle,
        cutoff=pd.Timestamp(bundle["split_cutoff"]),
        raw_all=raw_all, pred_all=pred_all, flagged_all=flagged_all,
        raw=in_range(raw_all, building, start, end),
        pred=in_range(pred_all, building, start, end),
        flagged=in_range(flagged_all, building, start, end),
        flagged_dates=in_range(flagged_all, ALL, start, end),
    )

    {"Overview": page_overview, "Energy Analytics": page_analytics, "ML Predictions": page_ml,
     "Anomaly Detection": page_anomalies, "Recommendations": page_recommendations,
     "Reports": page_reports}[page](ctx)


if __name__ == "__main__":
    main()
