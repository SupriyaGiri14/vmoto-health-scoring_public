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
# percentile came out at 2,919, so this is set just above that as a
# "genuinely rare event" cutoff for this hardware generation.
#
# Caveat: only one new-schema file was available to check against --
# this should be re-validated once more new-schema files exist across
# more devices/vehicles.
# This is a rough starting guess, not a calibrated value.
VIBRATION_SPIKE_THRESHOLD = 2900.0

# Points subtracted per spike, per 1000 usable rows (i.e. a RATE, not
# a raw count). Using a rate instead of a raw count matters a lot --
# a file with more rows (a longer ride) will naturally contain more
# spikes even if the riding is equally smooth throughout, so scoring
# on raw count unfairly punishes longer files.
VIBRATION_SPIKE_PENALTY_PER_1000_ROWS = 2.0

MINIMUM_SCORE = 0.0


@dataclass
class VehicleHealthResult:
    score: float
    vibration_spike_events: int
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


def compute_vehicle_health_score(rows: list[LogRow]) -> VehicleHealthResult:
    """
    Takes a list of LogRow (from one file / one ride session) and
    returns a VehicleHealthResult with the score and the numbers
    behind it.
    """
    usable_rows = [row for row in rows if _vibration_magnitude(row) is not None]

    vibration_spike_events = sum(
        1
        for row in usable_rows
        if _vibration_magnitude(row) >= VIBRATION_SPIKE_THRESHOLD
    )

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
