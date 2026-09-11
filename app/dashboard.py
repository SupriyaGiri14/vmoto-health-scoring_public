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
    # New: motor current data (only present for newer-schema devices --
    # file may be empty/missing entries for older-schema rides, handled
    # the same way as the other timeseries files).
    motor_path = DATA_DIR / "motor_timeseries.csv"
    if motor_path.exists() and motor_path.stat().st_size > 0:
        motor = pd.read_csv(motor_path, dtype=id_dtypes)
    else:
        motor = pd.DataFrame(columns=["vehicle_id", "device_id", "ride_date",
                                        "minute_offset", "max_current_per_throttle_ratio", "above_threshold"])
    return rides, battery, vibration, motor


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

    # Width grows with the number of rides, so labels stay readable
    # even for a vehicle with a long history (e.g. 22+ rides) instead
    # of cramming everything into a fixed-width chart.
    width = max(8, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(width, 4.5))
    ax.plot(labels, scores, marker="o", markersize=8, linewidth=2.2, color="#1f4e79")
    ax.axhline(overall, color="#c0392b", linestyle="--", linewidth=1.6, label=f"Overall: {overall}")
    for x, y in zip(labels, scores):
        ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Vehicle Health Score")
    ax.set_title(f"Trend: {trend.upper()}", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    # Vertical labels avoid overlapping when there are many dates.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
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


def chart_battery_multi_day(vehicle_battery: pd.DataFrame, num_days: int):
    """
    Shows battery behaviour across ALL of a vehicle's rides in one
    chart, using real timestamps on the x-axis (so overnight/between-
    ride gaps show up naturally as gaps in the line, not connected).
    Complements chart_battery_swap() (single ride) with the full
    multi-day picture -- useful for spotting whether a pack's
    behaviour is consistent or changing across many rides.
    """
    vehicle_battery = vehicle_battery.sort_values("timestamp")

    # Break the line at large gaps (between separate rides/days) so
    # matplotlib doesn't draw a misleading straight line connecting
    # the end of one day to the start of the next.
    timestamps = vehicle_battery["timestamp"].tolist()
    bat1 = vehicle_battery["battery_1_soc_pct"].tolist()
    bat2 = vehicle_battery["battery_2_soc_pct"].tolist()

    GAP_HOURS = 2
    for i in range(len(timestamps) - 1, 0, -1):
        gap_hours = (timestamps[i] - timestamps[i - 1]).total_seconds() / 3600
        if gap_hours > GAP_HOURS:
            timestamps.insert(i, timestamps[i - 1])
            bat1.insert(i, float("nan"))
            bat2.insert(i, float("nan"))

    # Width scales with how many days of history exist, same
    # reasoning as the trend chart -- but uses a more generous factor
    # since each day can show a dense zigzag swap pattern, not just
    # a single point.
    width = max(10, num_days * 1.0)
    fig, ax = plt.subplots(figsize=(width, 4.5))
    ax.plot(timestamps, bat1, color="#1f4e79", linewidth=1.2, label="Battery Pack 1")
    ax.plot(timestamps, bat2, color="#2e8b57", linewidth=1.2, label="Battery Pack 2")
    ax.set_ylabel("State of Charge (%)")
    ax.set_title(f"Battery Behaviour Across All Rides ({num_days} days)", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="best", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
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


# ---------------------------------------------------------------------
# NEW: Motor Health charts. Kept completely separate from the
# vibration-based Vehicle Health charts above -- this uses direct
# motor current data (only available on newer-schema devices) as a
# more precise replacement for the original throttle-vs-speed proxy.
# Nothing above this point was changed to add these.
# ---------------------------------------------------------------------

MOTOR_CURRENT_RATIO_THRESHOLD = 1.14  # matches motor_health.py's calibrated threshold


def chart_motor_health_trend(vehicle_rides: pd.DataFrame):
    """
    Motor Health Score per ride, same visual style as the Vehicle
    Health Trend chart. Rides with no motor current data (older
    logger schema) are simply skipped -- not plotted as a fake 0 or
    100, since there's genuinely no signal to show for them.
    """
    scored_rides = vehicle_rides.dropna(subset=["motor_health_score"]).sort_values("session_start")

    if scored_rides.empty:
        return None

    labels = scored_rides["session_start"].dt.strftime("%b %d").tolist()
    scores = scored_rides["motor_health_score"].tolist()

    width = max(8, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(width, 4))
    ax.plot(labels, scores, marker="o", markersize=8, linewidth=2.2, color="#8e44ad")
    for x, y in zip(labels, scores):
        ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center",
                    fontsize=9, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Motor Health Score")
    ax.set_title("Motor Health Trend (direct current data)", fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    plt.tight_layout()
    return fig


def chart_motor_current_detail(motor_ride: pd.DataFrame):
    """
    Same style as chart_vibration_spikes() -- shows the
    current-per-throttle ratio across one ride, with the threshold
    line and flagged high-ratio moments highlighted.
    """
    motor_ride = motor_ride.sort_values("minute_offset")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(motor_ride["minute_offset"], motor_ride["max_current_per_throttle_ratio"],
            color="#555555", linewidth=1.0, alpha=0.8)

    elevated = motor_ride[motor_ride["above_threshold"]]
    ax.scatter(elevated["minute_offset"], elevated["max_current_per_throttle_ratio"],
               color="#8e44ad", s=30, zorder=5, label=f"Elevated moments ({len(elevated)})")
    ax.axhline(MOTOR_CURRENT_RATIO_THRESHOLD, color="#8e44ad", linestyle="--", linewidth=1.4,
               label=f"Threshold ({MOTOR_CURRENT_RATIO_THRESHOLD})")
    ax.set_xlabel("Minutes into ride")
    ax.set_ylabel("Current \u00f7 throttle ratio")
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
# REAL pattern classification -- same rules as vibration_pattern.py's
# classify_vibration_pattern(), reimplemented locally (same reasoning
# as aggregate_overall_score() above: keeps the deployed app's
# dependency list minimal) but applied to the SELECTED vehicle's
# actual ride history, not a fixed synthetic example.
# ---------------------------------------------------------------------

ELEVATION_MULTIPLIER = 1.5
RECENT_WINDOW_SIZE = 3


def _percentile(values: list[float], fraction: float) -> float:
    values_sorted = sorted(values)
    n = len(values_sorted)
    if n == 1:
        return values_sorted[0]
    index = int(fraction * (n - 1))
    return values_sorted[index]


BASELINE_PERCENTILE = 0.25
CHRONIC_ELEVATION_FRACTION = 0.5
CHRONIC_MIN_SESSIONS = 6


def classify_real_pattern(rates: list[float]) -> tuple[str, str, float]:
    """
    Takes a vehicle's real spike rates, oldest to newest, and returns
    (classification, explanation, baseline_rate).

    Fixed after finding a real vehicle (119IAG) that was elevated on
    17 of 22 recorded rides. The original median-based baseline got
    contaminated by the many bad rides, understating the real
    severity. Baseline now uses the 25th percentile of earlier rides
    (closer to this vehicle's own genuinely good days), and a
    "chronically_elevated" result is returned first if most of the
    vehicle's FULL history is elevated, not just the recent window.
    """
    if len(rates) < 2:
        return "normal", "Only one ride available -- not enough history to detect a pattern yet.", (rates[0] if rates else 0.0)

    if len(rates) > RECENT_WINDOW_SIZE:
        earlier = rates[: len(rates) - RECENT_WINDOW_SIZE]
        recent = rates[-RECENT_WINDOW_SIZE:]
    else:
        earlier = rates[:-1]
        recent = rates[-1:]

    baseline = _percentile(earlier, BASELINE_PERCENTILE)
    cutoff = baseline * ELEVATION_MULTIPLIER

    if len(rates) >= CHRONIC_MIN_SESSIONS:
        elevated_count = sum(1 for r in rates if r > cutoff)
        fraction_elevated = elevated_count / len(rates)
        if fraction_elevated >= CHRONIC_ELEVATION_FRACTION:
            return (
                "chronically_elevated",
                f"{elevated_count} of {len(rates)} rides ({fraction_elevated:.0%}) are elevated "
                f"relative to this vehicle's own best/healthy readings -- looks like a "
                f"long-standing issue, not a recent development.",
                baseline,
            )

    recent_elevated = [r > cutoff for r in recent]
    earlier_normal = all(r <= cutoff for r in earlier)

    if not recent_elevated[-1]:
        return "normal", "Most recent ride is within normal range for this vehicle.", baseline

    if all(recent_elevated) and len(recent) >= 2:
        is_climbing = all(recent[i] < recent[i + 1] for i in range(len(recent) - 1))
        if is_climbing:
            return "worsening", f"Vibration has been elevated AND increasing across the last {len(recent)} rides -- looks like a developing issue.", baseline
        return "persistent_elevated", f"Vibration has been consistently elevated across the last {len(recent)} rides -- worth investigating.", baseline

    if recent_elevated[-1] and earlier_normal:
        return "isolated_spike", "Only the most recent ride is elevated; earlier rides were normal -- likely a rough road, not a vehicle issue.", baseline

    return "normal", "No clear persistent or worsening pattern detected.", baseline


def chart_real_pattern_classification(vehicle_rides: pd.DataFrame):
    """
    Same style as the illustrative chart above, but using the
    SELECTED vehicle's actual ride history -- one real result, not
    three hypothetical scenarios.
    """
    vehicle_rides = vehicle_rides.sort_values("session_start")
    rates = vehicle_rides["vibration_spike_rate_per_1000_rows"].tolist()
    labels = vehicle_rides["session_start"].dt.strftime("%b %d").tolist()

    classification, explanation, baseline = classify_real_pattern(rates)

    colors = {
        "normal": "#95a5a6",
        "isolated_spike": "#e67e22",
        "persistent_elevated": "#c0392b",
        "worsening": "#8e44ad",
        "chronically_elevated": "#7b241c",
    }
    bar_color = colors.get(classification, "#95a5a6")
    cutoff = baseline * ELEVATION_MULTIPLIER
    bar_colors = [bar_color if r > cutoff else "#95a5a6" for r in rates]

    # Width grows with the number of rides, same reasoning as the
    # trend chart above -- keeps date labels readable for vehicles
    # with a long ride history.
    width = max(9, len(labels) * 0.5)
    fig, ax = plt.subplots(figsize=(width, 4.5))
    ax.bar(labels, rates, color=bar_colors, width=0.6)
    ax.axhline(baseline, color="#1f4e79", linestyle="--", linewidth=1.3,
               label=f"Baseline: {baseline:.1f}")
    ax.set_ylabel("Vibration spike rate\n(per 1,000 rows)")
    ax.set_title(f"Real classification: {classification}", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    # Vertical labels avoid overlapping when there are many dates.
    plt.setp(ax.get_xticklabels(), rotation=90, ha="center")
    plt.tight_layout()
    return fig, classification, explanation


# ---------------------------------------------------------------------
# The Streamlit page itself.
# ---------------------------------------------------------------------

def main():
    st.set_page_config(page_title="Vehicle Health Dashboard", layout="wide")
    st.title("Vehicle & Battery Health Dashboard")

    rides, battery, vibration, motor = load_data()

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
    st.pyplot(chart_vehicle_health_trend(vehicle_rides), use_container_width=False)

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

    st.subheader(f"Battery Behaviour Across All Rides \u2014 {vehicle_id}")
    vehicle_battery = battery[battery["vehicle_id"] == vehicle_id]
    if vehicle_battery.empty:
        st.info("No battery data available for this vehicle across any ride.")
    else:
        num_days = vehicle_battery["ride_date"].nunique()
        st.pyplot(chart_battery_multi_day(vehicle_battery, num_days), use_container_width=False)
        st.caption(
            f"Showing {num_days} day(s) of battery data. Gaps in the line represent time "
            "between separate rides (e.g. overnight), not missing data within a ride."
        )

    st.subheader(f"Motor Health Trend \u2014 {vehicle_id}")
    st.caption(
        "New: uses direct motor current data (replaces the older throttle-vs-speed proxy). "
        "Only available for rides recorded on the newest logger schema -- rides without "
        "motor current data are simply not shown here, not faked."
    )
    motor_fig = chart_motor_health_trend(vehicle_rides)
    if motor_fig is None:
        st.info("No motor current data available for this vehicle yet (needs the newest logger schema).")
    else:
        st.pyplot(motor_fig, use_container_width=False)

    motor_ride = motor[(motor["vehicle_id"] == vehicle_id) & (motor["ride_date"] == ride_date)]
    if not motor_ride.empty:
        st.subheader("Motor Current Detail (selected ride)")
        st.pyplot(chart_motor_current_detail(motor_ride))

    st.subheader("Distinguishing a Road Bump from a Real Vehicle Problem (illustrative)")
    st.pyplot(chart_pattern_classification())

    st.subheader(f"Real Vibration Pattern \u2014 {vehicle_id}'s Actual Ride History")
    if len(vehicle_rides) < 2:
        st.info("Need at least 2 rides for this vehicle to detect a pattern -- only 1 available so far.")
    else:
        real_fig, real_classification, real_explanation = chart_real_pattern_classification(vehicle_rides)
        st.pyplot(real_fig, use_container_width=False)
        st.caption(f"**{real_classification}** \u2014 {real_explanation}")
        if len(vehicle_rides) <= 3:
            st.caption(
                "\u26a0\ufe0f Limited history: with only "
                f"{len(vehicle_rides)} rides, this can only distinguish "
                "\u2018normal\u2019 from \u2018isolated spike\u2019 -- not enough rides yet to "
                "detect a persistent or worsening pattern."
            )


if __name__ == "__main__":
    main()
