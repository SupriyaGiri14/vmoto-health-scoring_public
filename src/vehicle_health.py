"""
vehicle_health.py

Computes a Vehicle Health Score (0-100) from the clean rows produced
by load_logs.py.

IMPORTANT - what this currently covers
---------------------------------------
The original plan called for four signals: odometer/service interval,
CAN fault flags, vibration, and misuse events (curb hits, potholes).

Checking the real ride-log files shows only ONE of these is actually
available right now:

  - Vibration (IMU accelerometer data)   -> AVAILABLE, used below
  - Odometer / service interval          -> NOT in any ride log file
  - CAN fault flags                      -> NOT in any ride log file
  - Misuse events (curb/pothole/falls)   -> NOT labelled yet

So this module only implements the vibration check for now. The
other three are left as clearly-marked gaps rather than faked with
placeholder numbers, so the score isn't silently misleading.

IMPORTANT - vibration threshold is calibrated for NEW-schema data
--------------------------------------------------------------------
Checking real vibration data across devices showed that raw IMU
magnitude is NOT directly comparable across logger generations --
even the SAME physical vehicle (116IAG) shows very different
vibration statistics depending on which logger was installed,
almost certainly due to a hardware/calibration difference between
old and new loggers, not a real difference in riding roughness.

Since future data will use the newer logger format (the one with
motor_current support), the threshold below is calibrated using
ONLY new-schema data, rather than trying to find one number that
works fairly across both old and new hardware. Scores computed from
OLDER-schema files should be treated as not directly comparable to
scores from newer files -- this is a known, accepted limitation
while the fleet transitions to the newer logger, not something this
code tries to correct for.

IMPORTANT - consecutive spikes are grouped into ONE event
--------------------------------------------------------------
IMU data is time-series data: a single physical bump (e.g. a
pothole) commonly triggers SEVERAL consecutive readings above the
threshold, not just one. The first version of this code counted
every individual above-threshold ROW as a separate "spike" -- so one
real pothole spanning 5 consecutive readings was counted as 5
spikes, not 1 event, significantly inflating the penalty.

Fix: consecutive above-threshold readings within
MAX_EVENT_GAP_SECONDS of each other are now grouped into a single
event before counting. Checked against real data (device 16116760):
this reduced 26 raw above-threshold rows down to 9 real grouped
events -- roughly a 3x difference, confirming this was a real,
practically significant bug, not just a theoretical one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from load_logs import LogRow


# ---------------------------------------------------------------------
# Tunable settings.
# ---------------------------------------------------------------------

# The IMU accelerometer values are in raw sensor units, not
# real-world g-force. This threshold was derived from real NEW-schema
# data (16116760_2026-08-07.txt, n=25,097 readings): the 99.9th
# percentile came out at 2,919. This is set at 2,900 -- close to,
# though very slightly BELOW, that percentile (not "just above" as an
# earlier version of this comment incorrectly stated).
#
# Caveat: only one new-schema file was available to check against --
# this should be re-validated once more new-schema files exist across
# more devices/vehicles.
VIBRATION_SPIKE_THRESHOLD = 2900.0

# How many seconds apart two above-threshold readings can be and
# still count as the SAME physical event (e.g. one pothole), rather
# than two separate events. Checked against real data: a genuine
# single bump typically shows as several consecutive above-threshold
# readings spanning well under 1 second (observed real events lasted
# up to ~0.4s). 1.0s is a deliberately generous cutoff above that.
MAX_EVENT_GAP_SECONDS = 1.0

# Points subtracted per EVENT (grouped, not raw row count -- see
# MAX_EVENT_GAP_SECONDS above), per 1000 usable rows (i.e. a RATE,
# not a raw count). Using a rate instead of a raw count matters a lot
# -- a file with more rows (a longer ride) will naturally contain
# more events even if the riding is equally smooth throughout, so
# scoring on raw count unfairly punishes longer files.
#
# Caveat: this 2.0 penalty weight is a tunable assumption, not
# derived from any real outcome data -- unlike the threshold above,
# which is at least grounded in a real percentile.
VIBRATION_SPIKE_PENALTY_PER_1000_ROWS = 2.0

MINIMUM_SCORE = 0.0


@dataclass
class VehicleHealthResult:
    score: float
    vibration_spike_events: int  # GROUPED events, not raw above-threshold rows
    rows_used: int

    # These stay None until real data exists to compute them from --
    # kept as explicit fields (rather than leaving them out) so the
    # dashboard/report can show "not available yet" instead of
    # silently missing a column.
    service_status: None = None
    active_can_faults: None = None


def _vibration_magnitude(row: LogRow) -> float | None:
    """
    Combines the three IMU acceleration axes (x, y, z) into a single
    number describing "how much total movement/shock" was measured
    at this moment -- the standard way to combine a 3-axis reading
    into one magnitude.
    """
    if row.imu_acc_x is None or row.imu_acc_y is None or row.imu_acc_z is None:
        return None
    return math.sqrt(
        row.imu_acc_x ** 2 + row.imu_acc_y ** 2 + row.imu_acc_z ** 2
    )


def _count_grouped_events(usable_rows: list[LogRow]) -> int:
    """
    Counts vibration EVENTS, not raw above-threshold rows. Walks
    through the rows in order; a run of consecutive above-threshold
    readings (each within MAX_EVENT_GAP_SECONDS of the previous one)
    counts as ONE event, regardless of how many individual rows it
    spans. A large time gap, or a reading that drops back below the
    threshold, ends the current event -- the next above-threshold
    reading after that starts a NEW event.
    """
    event_count = 0
    currently_in_event = False
    previous_timestamp = None

    for row in usable_rows:
        magnitude = _vibration_magnitude(row)
        is_above_threshold = magnitude >= VIBRATION_SPIKE_THRESHOLD

        gap_too_large = (
            previous_timestamp is not None
            and (row.timestamp - previous_timestamp).total_seconds() > MAX_EVENT_GAP_SECONDS
        )
        if gap_too_large:
            currently_in_event = False

        if is_above_threshold:
            if not currently_in_event:
                event_count += 1
            currently_in_event = True
        else:
            currently_in_event = False

        previous_timestamp = row.timestamp

    return event_count


def compute_vehicle_health_score(rows: list[LogRow]) -> VehicleHealthResult:
    """
    Takes a list of LogRow (from one file / one ride session) and
    returns a VehicleHealthResult with the score and the numbers
    behind it.
    """
    usable_rows = [row for row in rows if _vibration_magnitude(row) is not None]

    vibration_spike_events = _count_grouped_events(usable_rows)

    penalty = 0.0
    if usable_rows:
        spike_rate_per_1000_rows = (
            vibration_spike_events / len(usable_rows)
        ) * 1000.0
        penalty = spike_rate_per_1000_rows * VIBRATION_SPIKE_PENALTY_PER_1000_ROWS

    score = max(MINIMUM_SCORE, 100.0 - penalty)

    return VehicleHealthResult(
        score=round(score, 1),
        vibration_spike_events=vibration_spike_events,
        rows_used=len(usable_rows),
    )


# ---------------------------------------------------------------------
# Quick manual check when running this file directly.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from load_logs import load_vmoto_log, NoVmotoDataError

    if len(sys.argv) != 2:
        print("Usage: python3 vehicle_health.py <path-to-log-file.txt>")
        raise SystemExit(1)

    try:
        rows = load_vmoto_log(sys.argv[1])
    except NoVmotoDataError as error:
        print(f"Skipped: {error}")
        raise SystemExit(0)

    result = compute_vehicle_health_score(rows)

    print(f"Vehicle Health Score: {result.score} / 100")
    print(f"Vibration spike events: {result.vibration_spike_events}")
    print(f"Rows used: {result.rows_used}")
    print("Service status: not available (no odometer data in ride logs)")
    print("CAN faults: not available (no fault columns in ride logs)")
