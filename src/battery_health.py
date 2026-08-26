"""
battery_health.py

Computes a Battery Health Score (0-100) from the clean rows produced
by load_logs.py.

How the score works
--------------------
Start at 100 (perfect health) and subtract points for bad signals:

  - Voltage sag: a sudden drop in voltage between two readings close
    together in time. A healthy battery holds its voltage steady;
    sagging under load is an early sign of wear.

  - Fast SOC drop rate: the battery draining faster than expected
    (percent per minute). A worn battery can drain faster even under
    similar usage.

This is intentionally simple (rule-based, not statistical) so it's
easy to explain and easy to adjust the thresholds/penalties later
once real maintenance data exists to calibrate against.

Update: SOC drop-rate calculation, fixed
------------------------------------------
The first version of this check compared every consecutive pair of
rows directly. This produced false alarms: SOC is only reported in
whole percentages, but rows are logged roughly every 0.1 seconds. So
SOC often stays flat for several minutes, then ticks down by 1% at
some row -- and comparing THAT row to the one immediately before it
(0.1 seconds earlier) made it look like the drop happened almost
instantly, producing nonsense rates like "600% per minute".

Fix: SOC drop rate is now measured over ~1-minute windows instead of
row-to-row, which averages out this quantisation artifact.

Also new: the drop-rate check is now split into "riding" vs "idle"
periods, using speed as a proxy. A fast drop while riding (e.g. hard
acceleration) is normal and uses a higher threshold; a fast drop
while idle/stationary is more unusual and uses a lower threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from load_logs import LogRow


# ---------------------------------------------------------------------
# Tunable settings. Kept as plain constants at the top so they're easy
# to find and adjust without digging through the logic below.
# ---------------------------------------------------------------------

# A voltage drop bigger than this, between two readings, counts as
# one "sag event". (Units: volts)
VOLTAGE_SAG_THRESHOLD_V = 3.0

# Points subtracted from the score per sag event.
VOLTAGE_SAG_PENALTY = 2.0

# A SOC drop rate faster than this (percent per minute) counts as
# "abnormally fast" -- now split by context, since fast drain during
# hard riding is normal but fast drain while idle is more unusual.
#
# RIDING threshold: checked against real data (204 real riding
# windows across 3 files) -- the real 99th percentile came out at
# 2.0%/min, matching this value closely. Reasonably well-grounded.
#
# IDLE threshold: NOT well-grounded yet. Only 6 real idle windows
# were available to check against, nowhere near enough to set this
# properly. What little data exists suggests SOC drop rates cluster
# around 1.0%/min even at idle -- likely a measurement-resolution
# floor (SOC only moves in whole 1% steps, so any detected drop in a
# ~60s window reads as roughly "1%/min" minimum, regardless of the
# true underlying rate). The original 0.3%/min guess sat BELOW that
# floor and would likely flag most idle windows as false positives.
# Raised to 1.2%/min as a more cautious interim value -- still needs
# proper calibration once more genuine idle/parked data is available.
FAST_SOC_DROP_THRESHOLD_RIDING_PCT_PER_MIN = 2.0
FAST_SOC_DROP_THRESHOLD_IDLE_PCT_PER_MIN = 1.2

# A row counts as "riding" if speed is at or above this. Below this
# (or speed unknown) is treated as "idle" -- the more cautious
# assumption, since draining fast with no clear riding activity is
# the more unusual case worth flagging.
RIDING_SPEED_THRESHOLD_KPH = 3.0

# SOC drop rate is measured over windows of this length, rather than
# row-to-row, to avoid the quantisation artifact described above.
SOC_DROP_WINDOW_SECONDS = 60.0

# Points subtracted per minute that the drop rate exceeds the
# threshold above, added up across the whole file.
FAST_SOC_DROP_PENALTY_PER_MINUTE = 1.0

# The score never goes below this floor.
MINIMUM_SCORE = 0.0


# ---------------------------------------------------------------------
# The result we return: the score, plus the raw numbers that explain
# it. Returning the "why" alongside the score makes it possible to
# show supporting detail on the dashboard, not just a bare number.
# ---------------------------------------------------------------------

@dataclass
class BatteryHealthResult:
    score: float
    voltage_sag_events: int
    fast_soc_drop_minutes: float
    rows_used: int
    rows_missing_battery_data: int


# ---------------------------------------------------------------------
# Helper: for one row, get whichever battery pack is actually
# reporting data. We found that battery_1 and battery_2 are rarely
# populated at the same time -- usually only one pack is "active" at
# a given moment.
# ---------------------------------------------------------------------

def _active_voltage(row: LogRow) -> Optional[float]:
    if row.battery_1_voltage_v is not None:
        return row.battery_1_voltage_v
    return row.battery_2_voltage_v


def _active_soc(row: LogRow) -> Optional[float]:
    if row.battery_1_soc_pct is not None:
        return row.battery_1_soc_pct
    return row.battery_2_soc_pct


# ---------------------------------------------------------------------
# Main function.
# ---------------------------------------------------------------------

def compute_battery_health_score(rows: list[LogRow]) -> BatteryHealthResult:
    """
    Takes a list of LogRow (from one file / one ride session) and
    returns a BatteryHealthResult with the score and the numbers
    behind it.
    """
    # Only keep rows that actually have a voltage or SOC reading --
    # many rows in a file are missing battery data entirely (we saw
    # this a lot in the real files, e.g. only 50 of 25,097 rows had
    # battery_1_voltage_v). Working only with real readings keeps the
    # score from being thrown off by gaps.
    usable_rows = [
        row for row in rows
        if _active_voltage(row) is not None or _active_soc(row) is not None
    ]

    voltage_sag_events = 0

    # --- Voltage sag check (unchanged -- this one wasn't buggy) ---
    for previous_row, current_row in zip(usable_rows, usable_rows[1:]):
        seconds_between = (
            current_row.timestamp - previous_row.timestamp
        ).total_seconds()
        if seconds_between <= 0:
            continue

        previous_voltage = _active_voltage(previous_row)
        current_voltage = _active_voltage(current_row)
        if previous_voltage is not None and current_voltage is not None:
            drop = previous_voltage - current_voltage
            if drop >= VOLTAGE_SAG_THRESHOLD_V:
                voltage_sag_events += 1

    # --- SOC drop rate check, using ~1-minute windows ---
    #
    # Instead of comparing every consecutive row (which broke on SOC's
    # whole-percent quantisation -- see module docstring), we take ONE
    # representative SOC reading per ~60-second window (the LAST
    # reading seen within that window), then compare consecutive
    # window readings to each other. This means each comparison spans
    # a real ~1-minute-or-more gap, not a random 0.1-second gap.
    soc_rows = [row for row in usable_rows if _active_soc(row) is not None]

    fast_soc_drop_minutes = 0.0

    if soc_rows:
        session_start = soc_rows[0].timestamp

        # Group rows into windows, keeping the LAST row seen in each
        # window (so the window's "reading" reflects the end of that
        # time period, and any speed reading in that window tells us
        # whether the vehicle was riding or idle during it).
        window_last_row: dict[int, LogRow] = {}
        window_saw_riding_speed: dict[int, bool] = {}

        for row in soc_rows:
            seconds_since_start = (row.timestamp - session_start).total_seconds()
            window_index = int(seconds_since_start // SOC_DROP_WINDOW_SECONDS)

            window_last_row[window_index] = row

            is_riding = row.speed_kph is not None and row.speed_kph >= RIDING_SPEED_THRESHOLD_KPH
            window_saw_riding_speed[window_index] = (
                window_saw_riding_speed.get(window_index, False) or is_riding
            )

        # Walk through windows in order, comparing each window's
        # representative reading to the previous window's.
        ordered_window_indexes = sorted(window_last_row.keys())

        for previous_index, current_index in zip(
            ordered_window_indexes, ordered_window_indexes[1:]
        ):
            previous_row = window_last_row[previous_index]
            current_row = window_last_row[current_index]

            seconds_between = (
                current_row.timestamp - previous_row.timestamp
            ).total_seconds()
            if seconds_between <= 0:
                continue

            previous_soc = _active_soc(previous_row)
            current_soc = _active_soc(current_row)
            soc_drop = previous_soc - current_soc
            minutes_between = seconds_between / 60.0

            if soc_drop <= 0 or minutes_between <= 0:
                continue

            drop_rate_pct_per_min = soc_drop / minutes_between

            # Use the riding threshold if EITHER window saw riding
            # speed, otherwise treat the whole gap as idle (the more
            # cautious assumption).
            was_riding = (
                window_saw_riding_speed.get(previous_index, False)
                or window_saw_riding_speed.get(current_index, False)
            )
            threshold = (
                FAST_SOC_DROP_THRESHOLD_RIDING_PCT_PER_MIN
                if was_riding
                else FAST_SOC_DROP_THRESHOLD_IDLE_PCT_PER_MIN
            )

            if drop_rate_pct_per_min > threshold:
                fast_soc_drop_minutes += minutes_between

    # --- Turn the raw counts into a 0-100 score ---
    penalty = (
        voltage_sag_events * VOLTAGE_SAG_PENALTY
        + fast_soc_drop_minutes * FAST_SOC_DROP_PENALTY_PER_MINUTE
    )
    score = max(MINIMUM_SCORE, 100.0 - penalty)

    return BatteryHealthResult(
        score=round(score, 1),
        voltage_sag_events=voltage_sag_events,
        fast_soc_drop_minutes=round(fast_soc_drop_minutes, 1),
        rows_used=len(usable_rows),
        rows_missing_battery_data=len(rows) - len(usable_rows),
    )


# ---------------------------------------------------------------------
# Quick manual check when running this file directly.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from load_logs import load_vmoto_log, NoVmotoDataError

    if len(sys.argv) != 2:
        print("Usage: python3 battery_health.py <path-to-log-file.txt>")
        raise SystemExit(1)

    try:
        rows = load_vmoto_log(sys.argv[1])
    except NoVmotoDataError as error:
        print(f"Skipped: {error}")
        raise SystemExit(0)

    result = compute_battery_health_score(rows)

    print(f"Battery Health Score: {result.score} / 100")
    print(f"Voltage sag events: {result.voltage_sag_events}")
    print(f"Minutes with fast SOC drop: {result.fast_soc_drop_minutes}")
    print(f"Rows with usable battery data: {result.rows_used}")
    print(f"Rows missing battery data: {result.rows_missing_battery_data}")
