"""
dashboard.py (app version)

Interactive dashboard: pick a vehicle and a ride date from dropdowns,
and the charts update to match.

IMPORTANT -- this version reads ONLY from small, pre-exported CSV
files in app/data/ (rides_summary.csv, battery_timeseries.csv,
vibration_timeseries.csv). It never touches the large raw .txt log
files, which stay private and local. This is what makes the app safe
and small enough to publish.

The CSVs are produced by src/export_dashboard_data.py, which DOES
read the raw files -- that script runs locally against private data
and produces these small, publishable summaries. Re-run it and
replace the files in app/data/ whenever new ride data is available.

How to run
----------
    pip install streamlit pandas matplotlib
    streamlit run app/dashboard.py
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import streamlit as st


plt.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
})

DATA_DIR = Path(__file__).resolve().parent / "data"

# Same threshold used in vehicle_health.py -- kept as a plain constant
# here since the app doesn't import the scoring modules at all.
VIBRATION_SPIKE_THRESHOLD = 2900.0

# Recency weighting, matching overall_vehicle_health.py's logic.
RECENCY_WEIGHT_GROWTH = 1.15


# ---------------------------------------------------------------------
# Load the small CSVs once, cached.
# ---------------------------------------------------------------------

@st.cache_data
def load_data():
    # device_id must be read as a string, not a number -- otherwise
    # pandas auto-detects it as int64 (since values like "1086344"
    # look purely numeric), which then breaks anywhere the code joins
    # device IDs into text (e.g. ', '.join(...) requires strings).
    id_dtypes = {"device_id": str, "vehicle_id": str}

    rides = pd.read_csv(DATA_DIR / "rides_summary.csv", dtype=id_dtypes,
                         parse_dates=["session_start", "session_end"])
    battery = pd.read_csv(DATA_DIR / "battery_timeseries.csv", dtype=id_dtypes,
                           parse_dates=["timestamp"])
    vibration = pd.read_csv(DATA_DIR / "vibration_timeseries.csv", dtype=id_dtypes)
    return rides, battery, vibration


def aggregate_overall_score(scores: list[float]) -> tuple[float, str]:
    """
    Same recency-weighted average + trend logic as
    overall_vehicle_health.py, reimplemented here directly (in plain
    Python, no pandas needed) so the app doesn't need that module as
    a dependency -- keeps the deployed app's dependency list minimal.
    """
    if not scores:
        return 0.0, "normal"

    weights = [RECENCY_WEIGHT_GROWTH ** i for i in range(len(scores))]
    overall = sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    if len(scores) < 2:
        return round(overall, 1), "stable"

    midpoint = len(scores) // 2
    older = scores[:midpoint] if midpoint > 0 else scores[:1]
    newer = scores[midpoint:] if midpoint > 0 else scores[1:]
    older_avg = sum(older) / len(older)
    newer_avg = sum(newer) / len(newer)

    if newer_avg > older_avg + 2:
        trend = "improving"
    elif newer_avg < older_avg - 2:
        trend = "worsening"
    else:
        trend = "stable"

    return round(overall, 1), trend


# ---------------------------------------------------------------------
# Chart-building functions.
# ---------------------------------------------------------------------

def chart_vehicle_health_trend(vehicle_rides: pd.DataFrame):
    vehicle_rides = vehicle_rides.sort_values("session_start")
    labels = vehicle_rides["session_start"].dt.strftime("%b %d").tolist()
    scores = vehicle_rides["vehicle_health_score"].tolist()

    overall, trend = aggregate_overall_score(scores)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, scores, marker="o", markersize=8, linewidth=2.2, color="#1f4e79")
    ax.axhline(overall, color="#c0392b", linestyle="--", linewidth=1.6, label=f"Overall: {overall}")
    for x, y in zip(labels, scores):
        ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Vehicle Health Score")
    ax.set_title(f"Trend: {trend.upper()}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    return fig


def chart_battery_swap(battery_ride: pd.DataFrame):
    battery_ride = battery_ride.sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(battery_ride["timestamp"], battery_ride["battery_1_soc_pct"],
            color="#1f4e79", linewidth=1.6, label="Battery Pack 1")
    ax.plot(battery_ride["timestamp"], battery_ride["battery_2_soc_pct"],
            color="#2e8b57", linewidth=1.6, label="Battery Pack 2")
    ax.set_ylabel("State of Charge (%)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    return fig


def chart_vibration_spikes(vibration_ride: pd.DataFrame):
    vibration_ride = vibration_ride.sort_values("minute_offset")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(vibration_ride["minute_offset"], vibration_ride["max_vibration_magnitude"],
            color="#555555", linewidth=1.0, alpha=0.8)

    spikes = vibration_ride[vibration_ride["above_threshold"]]
    ax.scatter(spikes["minute_offset"], spikes["max_vibration_magnitude"],
               color="#c0392b", s=30, zorder=5, label=f"Spikes ({len(spikes)})")
    ax.axhline(VIBRATION_SPIKE_THRESHOLD, color="#c0392b", linestyle="--", linewidth=1.4,
               label=f"Threshold ({int(VIBRATION_SPIKE_THRESHOLD)})")
    ax.set_xlabel("Minutes into ride")
    ax.set_ylabel("Max vibration per minute (raw units)")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    return fig


def chart_pattern_classification():
    """
    Illustrative example (synthetic data). Classification logic is
    reimplemented directly here (a simplified version of
    vibration_pattern.py's rules) so the app doesn't need that module
    -- this chart never uses real ride data, so a lightweight local
    copy is reasonable rather than adding a dependency for it.
    """
    scenarios = {
        "Isolated spike": ([10, 11, 9, 10, 40], "isolated_spike", "#e67e22"),
        "Persistent elevated": ([10, 11, 35, 36, 34], "persistent_elevated", "#c0392b"),
        "Worsening": ([10, 11, 30, 40, 50], "worsening", "#8e44ad"),
    }

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5), sharey=True)
    for ax, (title, (rates, label, color)) in zip(axes, scenarios.items()):
        baseline = sum(rates[:2]) / 2
        cutoff = baseline * 1.5
        bar_colors = [color if r > cutoff else "#95a5a6" for r in rates]
        x = [f"R{i+1}" for i in range(len(rates))]
        ax.bar(x, rates, color=bar_colors, width=0.6)
        ax.axhline(baseline, color="#1f4e79", linestyle="--", linewidth=1.1)
        ax.set_title(f"{title}\n\u2192 {label}", fontsize=9, fontweight="bold")
    axes[0].set_ylabel("Spike rate /1000 rows")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------
# The Streamlit page itself.
# ---------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Vehicle Health Dashboard", layout="wide")
    st.title("Vehicle & Battery Health Dashboard")

    rides, battery, vibration = load_data()

    if rides.empty:
        st.error("No ride data found in app/data/. Run export_dashboard_data.py first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        vehicle_id = st.selectbox("Vehicle", sorted(rides["vehicle_id"].unique()))
    with col2:
        vehicle_rides = rides[rides["vehicle_id"] == vehicle_id]
        ride_date = st.selectbox("Ride date", sorted(vehicle_rides["ride_date"].unique(), reverse=True))

    selected_ride = vehicle_rides[vehicle_rides["ride_date"] == ride_date].iloc[0]
    selected_device_id = selected_ride["device_id"]
    schema_tier = selected_ride["schema_tier"]

    devices_used = vehicle_rides["device_id"].unique()
    if len(devices_used) > 1:
        st.caption(
            f"This vehicle's history spans {len(devices_used)} different loggers: "
            f"{', '.join(sorted(devices_used))}. The trend below combines rides across "
            "all of them; scores from different logger generations may not be directly comparable."
        )

    st.caption(f"This ride was recorded by device **{selected_device_id}**")

    if schema_tier != "new":
        st.warning(
            f"This ride uses the '{schema_tier}' logger schema. The vibration threshold "
            "is calibrated for the newer hardware generation ('new' schema) -- Vehicle "
            "Health scores for this ride are not yet directly comparable to newer devices."
        )

    st.subheader(f"Vehicle Health Trend \u2014 {vehicle_id}")
    st.pyplot(chart_vehicle_health_trend(vehicle_rides))

    battery_ride = battery[(battery["vehicle_id"] == vehicle_id) & (battery["ride_date"] == ride_date)]
    vibration_ride = vibration[(vibration["vehicle_id"] == vehicle_id) & (vibration["ride_date"] == ride_date)]

    left, right = st.columns(2)
    with left:
        st.subheader("Battery Behaviour")
        if battery_ride.empty:
            st.info("No battery data for this ride.")
        else:
            st.pyplot(chart_battery_swap(battery_ride))
    with right:
        st.subheader("Vibration Spike Detection")
        if vibration_ride.empty:
            st.info("No vibration data for this ride.")
        else:
            st.pyplot(chart_vibration_spikes(vibration_ride))

    st.subheader("Distinguishing a Road Bump from a Real Vehicle Problem (illustrative)")
    st.pyplot(chart_pattern_classification())


if __name__ == "__main__":
    main()
