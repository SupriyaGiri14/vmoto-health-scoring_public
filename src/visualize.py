"""
visualize.py

Generates PNG charts from real fleet data, for reports/slides/dashboards.

This does not compute anything new -- it just visualizes results that
load_logs.py, battery_health.py, vehicle_health.py, and
vibration_pattern.py already calculate.

Requires matplotlib (see requirements.txt).

Usage
-----
Pass 2 or more log files for the SAME vehicle, ordered any way (they
get sorted by timestamp automatically):

    python3 src/visualize.py data/raw/1086344_2026-07-17.txt \\
                              data/raw/1086344_2026-07-27.txt \\
                              data/raw/1086344_2026-08-05.txt

This produces 4 PNG files in charts/:

  1_vehicle_health_trend.png          -- score per session + overall trend
  2_battery_swap_pattern.png          -- SOC over time (from the FIRST file given)
  3_vibration_spike_detection.png     -- vibration + threshold (from the FIRST file given)
  4_vibration_pattern_classification.png -- illustrative example (synthetic data,
                                             explains isolated/persistent/worsening)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # renders to files, no GUI window needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from load_logs import load_vmoto_log, NoVmotoDataError
from battery_health import _active_soc
from vehicle_health import compute_vehicle_health_score, _vibration_magnitude, VIBRATION_SPIKE_THRESHOLD
from overall_vehicle_health import ScoredSession, aggregate_vehicle_health
from vibration_pattern import VibrationSession, classify_vibration_pattern


# A consistent, clean look for every chart in this file.
plt.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#444444",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.6,
})


def plot_vehicle_health_trend(file_paths: list[str], output_path: str) -> None:
    """
    Chart 1: Vehicle Health Score for each session, plus the overall
    recency-weighted score and trend, using overall_vehicle_health.py.
    """
    sessions = []
    labels = []
    scores = []

    for path in file_paths:
        rows = load_vmoto_log(path)
        result = compute_vehicle_health_score(rows)
        sessions.append(ScoredSession(timestamp=rows[0].timestamp, score=result.score))
        labels.append(rows[0].timestamp.strftime("%b %d"))
        scores.append(result.score)

    overall = aggregate_vehicle_health(sessions)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(labels, scores, marker="o", markersize=9, linewidth=2.5,
            color="#1f4e79", label="Per-session score")
    ax.axhline(overall.overall_score, color="#c0392b", linestyle="--", linewidth=1.8,
               label=f"Overall (recency-weighted): {overall.overall_score}")
    for x, y in zip(labels, scores):
        ax.annotate(f"{y}", (x, y), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Vehicle Health Score")
    ax.set_title(f"Vehicle Health Score Trend\nTrend: {overall.trend.upper()}",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_battery_swap_pattern(file_path: str, output_path: str) -> None:
    """
    Chart 2: Both battery packs' SOC over time for one session -- shows
    the dual-battery hot-swap pattern (one pack draining while the
    other sits idle, then a swap when they trade roles).
    """
    rows = load_vmoto_log(file_path)
    soc_rows = [r for r in rows if _active_soc(r) is not None]

    times = [r.timestamp for r in soc_rows]
    bat1 = [r.battery_1_soc_pct for r in soc_rows]
    bat2 = [r.battery_2_soc_pct for r in soc_rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(times, bat1, color="#1f4e79", linewidth=1.8, label="Battery Pack 1")
    ax.plot(times, bat2, color="#2e8b57", linewidth=1.8, label="Battery Pack 2")
    ax.set_ylabel("State of Charge (%)")
    ax.set_title("Dual-Battery Behaviour Over One Ride", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_vibration_spike_detection(file_path: str, output_path: str) -> None:
    """
    Chart 3: Vibration magnitude over time, with the spike threshold
    line and the flagged spikes highlighted.
    """
    rows = load_vmoto_log(file_path)
    mags, times = [], []
    for r in rows:
        m = _vibration_magnitude(r)
        if m is not None:
            mags.append(m)
            times.append(r.timestamp)

    # Break the line at large time gaps so matplotlib doesn't draw a
    # misleading straight line across missing data.
    mags_plot, times_plot = list(mags), list(times)
    GAP_SECONDS = 5
    for i in range(len(times) - 1, 0, -1):
        if (times[i] - times[i - 1]).total_seconds() > GAP_SECONDS:
            times_plot.insert(i, times[i - 1])
            mags_plot.insert(i, float("nan"))

    spike_times = [t for t, m in zip(times, mags) if m >= VIBRATION_SPIKE_THRESHOLD]
    spike_mags = [m for m in mags if m >= VIBRATION_SPIKE_THRESHOLD]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(times_plot, mags_plot, color="#555555", linewidth=0.5, alpha=0.7)
    ax.scatter(spike_times, spike_mags, color="#c0392b", s=25, zorder=5,
               label=f"Spikes ({len(spike_times)})")
    ax.axhline(VIBRATION_SPIKE_THRESHOLD, color="#c0392b", linestyle="--", linewidth=1.5,
               label=f"Threshold ({int(VIBRATION_SPIKE_THRESHOLD)})")
    ax.set_ylabel("Vibration magnitude (raw units)")
    ax.set_title("Vibration Spike Detection", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_vibration_pattern_classification(output_path: str) -> None:
    """
    Chart 4: An illustrative example (synthetic data, not from a real
    file) showing the three classifications vibration_pattern.py can
    detect: isolated_spike, persistent_elevated, worsening.
    """
    def make_sessions(rates: list[float]) -> list[VibrationSession]:
        base = datetime(2026, 8, 1)
        return [
            VibrationSession(timestamp=base + timedelta(days=i * 2), spike_rate_per_1000_rows=r)
            for i, r in enumerate(rates)
        ]

    scenarios = {
        "Isolated spike\n(one rough road, not a vehicle issue)": [10, 11, 9, 10, 40],
        "Persistent elevated\n(same problem, every ride)": [10, 11, 35, 36, 34],
        "Worsening\n(likely a developing issue)": [10, 11, 30, 40, 50],
    }
    colors = {"isolated_spike": "#e67e22", "persistent_elevated": "#c0392b", "worsening": "#8e44ad"}

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

    for ax, (title, rates) in zip(axes, scenarios.items()):
        sessions = make_sessions(rates)
        result = classify_vibration_pattern(sessions)
        x = [f"Ride {i + 1}" for i in range(len(rates))]

        cutoff = result.baseline_rate * 1.5
        bar_colors = [
            colors.get(result.classification, "#c0392b") if r > cutoff else "#95a5a6"
            for r in rates
        ]

        ax.bar(x, rates, color=bar_colors, width=0.6)
        ax.axhline(result.baseline_rate, color="#1f4e79", linestyle="--", linewidth=1.3)
        ax.set_title(f"{title}\n\u2192 {result.classification}", fontsize=10.5, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)

    axes[0].set_ylabel("Vibration spike rate\n(per 1,000 rows)")
    fig.suptitle("Distinguishing a Road Bump from a Real Vehicle Problem",
                 fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Command-line entry point.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python3 visualize.py <file1.txt> <file2.txt> [more files...]")
        print("(Pass 2+ log files for the SAME vehicle, any order.)")
        raise SystemExit(1)

    file_paths = sys.argv[1:]

    # Sort by their own first timestamp, so charts always read
    # oldest -> newest regardless of the order given on the command line.
    def _first_timestamp(path: str) -> datetime:
        return load_vmoto_log(path)[0].timestamp

    try:
        file_paths = sorted(file_paths, key=_first_timestamp)
    except NoVmotoDataError as error:
        print(f"Error reading a file: {error}")
        raise SystemExit(1)

    output_dir = Path("charts")
    output_dir.mkdir(exist_ok=True)

    print("Generating chart 1: Vehicle Health trend...")
    plot_vehicle_health_trend(file_paths, str(output_dir / "1_vehicle_health_trend.png"))

    print("Generating chart 2: Battery swap pattern...")
    plot_battery_swap_pattern(file_paths[0], str(output_dir / "2_battery_swap_pattern.png"))

    print("Generating chart 3: Vibration spike detection...")
    plot_vibration_spike_detection(file_paths[0], str(output_dir / "3_vibration_spike_detection.png"))

    print("Generating chart 4: Vibration pattern classification (illustrative example)...")
    plot_vibration_pattern_classification(str(output_dir / "4_vibration_pattern_classification.png"))

    print(f"\nDone. Charts saved to {output_dir}/")
