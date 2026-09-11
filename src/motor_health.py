"""
motor_health.py

Computes a Motor Health Score (0-100) from direct motor current data,
REPLACING the original plan's indirect proxy (comparing throttle
input to resulting speed) now that real motor current data is
available in newer-schema log files.

Why this replaces the throttle-vs-speed proxy
------------------------------------------------
The original PdM report (Section 2.4) proposed watching for "drift"
in the throttle-vs-speed relationship as an indirect sign of motor or
drivetrain wear -- if the same throttle input produces less speed
over time, something's dragging.

Now that vmoto_motor_current is available (newer logger schema
only), we have a DIRECT electrical measurement of how hard the motor
is working, which is a more precise signal than inferring it from
speed. A worn motor, dragging brake, or failing bearing all show up
the same way: the motor needs MORE current to deliver the SAME
throttle input, because something is adding mechanical resistance.

How it's calculated
---------------------
For every row with valid current and throttle > 5% (avoiding noise
near-zero throttle, where the ratio is unstable), compute:

    current_per_throttle = motor_current_a / throttle_pct

This ratio describes "how much electrical effort it took to deliver
this throttle position" -- a genuinely elevated ratio means the
motor needed unusually high current for a given throttle input,
which is what mechanical resistance/wear would look like.

Consecutive elevated readings are grouped into ONE event (same fix
already applied in vehicle_health.py -- a single hard-acceleration
burst spanning several rows should count once, not many times).

Score = 100 - (penalty x event rate per 1000 usable rows)

Threshold calibration
------------------------
MOTOR_CURRENT_RATIO_THRESHOLD (1.14) is the 99th percentile of
current_per_throttle, calculated from 552,822 real readings across
two newer-schema devices (16040936, 16116760) -- the only two
devices with motor current data available so far.

Caveat, same as other thresholds in this project: this is a
statistical outlier threshold, NOT yet validated against a real,
confirmed motor/drivetrain failure. No such failure has been
recorded in the fleet's maintenance data yet. This should be
revisited once real motor-fault maintenance records exist.

Data availability caveat
----------------------------
Motor current only exists in the newest logger schema tier. Vehicles
still running older loggers cannot be scored by this module at all --
compute_motor_health_score() will report zero usable rows for those
files, not a fabricated score.
"""

from __future__ import annotations

from dataclasses import dataclass

from load_logs import LogRow


# ---------------------------------------------------------------------
# Tunable settings.
# ---------------------------------------------------------------------

# 99th percentile of current_per_throttle, from 552,822 real readings
# across both newer-schema devices available so far (16040936,
# 16116760). See module docstring for caveats.
MOTOR_CURRENT_RATIO_THRESHOLD = 1.14

# Below this throttle position, the ratio is too noisy/unstable to be
# meaningful (dividing by a very small throttle value amplifies
# minor current fluctuations into huge, meaningless ratios).
MIN_THROTTLE_FOR_RATIO_PCT = 5.0

# Same event-grouping window as vehicle_health.py -- consecutive
# elevated readings within this many seconds count as ONE event.
MAX_EVENT_GAP_SECONDS = 1.0

# Points subtracted per event, per 1000 usable rows (a RATE, not a
# raw count -- same reasoning as vehicle_health.py: a longer ride
# shouldn't be penalised just for having more rows).
#
# Caveat: like vehicle_health.py's equivalent constant, this weight
# is a tunable assumption, not derived from real outcome data.
MOTOR_INEFFICIENCY_PENALTY_PER_1000_ROWS = 2.0

MINIMUM_SCORE = 0.0


@dataclass
class MotorHealthResult:
    score: float | None  # None if no motor current data is available at all
    inefficiency_events: int
    rows_used: int
    rows_with_current_data: int


def _current_per_throttle(row: LogRow) -> float | None:
    """
    Returns the current-per-throttle ratio for one row, or None if
    this row doesn't have the data needed to compute it.
    """
    if row.motor_current_a is None or row.throttle_pct is None:
        return None
    if row.throttle_pct <= MIN_THROTTLE_FOR_RATIO_PCT:
        return None
    return row.motor_current_a / row.throttle_pct


def _count_grouped_events(usable_rows: list[LogRow]) -> int:
    """
    Same grouping logic as vehicle_health.py's equivalent function --
    consecutive elevated readings within MAX_EVENT_GAP_SECONDS count
    as one event, not one per row.
    """
    event_count = 0
    currently_in_event = False
    previous_timestamp = None

    for row in usable_rows:
        ratio = _current_per_throttle(row)
        is_elevated = ratio is not None and ratio > MOTOR_CURRENT_RATIO_THRESHOLD

        gap_too_large = (
            previous_timestamp is not None
            and (row.timestamp - previous_timestamp).total_seconds() > MAX_EVENT_GAP_SECONDS
        )
        if gap_too_large:
            currently_in_event = False

        if is_elevated:
            if not currently_in_event:
                event_count += 1
            currently_in_event = True
        else:
            currently_in_event = False

        previous_timestamp = row.timestamp

    return event_count


def compute_motor_health_score(rows: list[LogRow]) -> MotorHealthResult:
    """
    Takes a list of LogRow (from one file / one ride session) and
    returns a MotorHealthResult. Returns score=None if the file has
    no motor current data at all (older logger schema) -- this is
    reported explicitly rather than faking a score of 100 or 0.
    """
    rows_with_current = sum(1 for r in rows if r.motor_current_a is not None)

    if rows_with_current == 0:
        return MotorHealthResult(
            score=None,
            inefficiency_events=0,
            rows_used=0,
            rows_with_current_data=0,
        )

    usable_rows = [r for r in rows if _current_per_throttle(r) is not None]

    inefficiency_events = _count_grouped_events(usable_rows)

    penalty = 0.0
    if usable_rows:
        event_rate_per_1000_rows = (inefficiency_events / len(usable_rows)) * 1000.0
        penalty = event_rate_per_1000_rows * MOTOR_INEFFICIENCY_PENALTY_PER_1000_ROWS

    score = max(MINIMUM_SCORE, 100.0 - penalty)

    return MotorHealthResult(
        score=round(score, 1),
        inefficiency_events=inefficiency_events,
        rows_used=len(usable_rows),
        rows_with_current_data=rows_with_current,
    )


# ---------------------------------------------------------------------
# Quick manual check when running this file directly.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from load_logs import load_vmoto_log, NoVmotoDataError

    if len(sys.argv) != 2:
        print("Usage: python3 motor_health.py <path-to-log-file.txt>")
        raise SystemExit(1)

    try:
        rows = load_vmoto_log(sys.argv[1])
    except NoVmotoDataError as error:
        print(f"Skipped: {error}")
        raise SystemExit(0)

    result = compute_motor_health_score(rows)

    if result.score is None:
        print("Motor Health Score: not available (no motor current data in this file -- older logger schema)")
    else:
        print(f"Motor Health Score: {result.score} / 100")
        print(f"Inefficiency events: {result.inefficiency_events}")
        print(f"Rows used (current + throttle > {MIN_THROTTLE_FOR_RATIO_PCT}%): {result.rows_used}")
        print(f"Rows with any current data: {result.rows_with_current_data}")
