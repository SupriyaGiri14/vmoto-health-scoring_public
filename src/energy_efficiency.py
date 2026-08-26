"""
energy_efficiency.py

Computes energy efficiency -- how much battery charge (SOC) is used
per kilometre travelled -- from the clean rows produced by
load_logs.py.

Why this matters
-----------------
Voltage sag and SOC drop-rate (battery_health.py) catch SUDDEN
problems. Efficiency is different: it can catch a vehicle that's
GRADUALLY using more battery to cover the same distance over weeks --
worn bearings, underinflated tyres, brake drag, or battery capacity
loss can all show up this way, often before any other symptom is
visible.

How it's calculated
---------------------
  1. Estimate distance travelled by integrating speed over time:
     distance = sum of (average speed between two readings x time
     between them), skipping any large time gaps.

  2. Sum up total SOC CONSUMED across the session -- see "battery
     swaps" below for why this isn't just "SOC at start minus SOC at
     end".

  3. efficiency = total SOC consumed / distance travelled (unit: %
     SOC per km)

Battery swaps (important -- found in real data)
---------------------------------------------------
This fleet uses a dual hot-swappable battery design. Checking a real
ride confirmed riders swap packs mid-ride: pack 1 drains down, then
pack 2 comes online at a much higher SOC. A naive "SOC at start minus
SOC at end" calculation would be badly wrong here -- it would see
"pack 2 came online at 88%" as the battery suddenly REFILLING, when
really a fresh pack just took over.

Fix: instead of comparing start to end, this walks through every
consecutive pair of SOC readings and adds up only the DECREASES
(genuine consumption). Any INCREASE (a pack swap, or genuine
charging) is simply not counted as negative consumption -- it's
skipped, not subtracted. This correctly totals real consumption
across a ride with any number of pack swaps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from load_logs import LogRow
from battery_health import _active_soc


# Skip integrating distance across any gap longer than this many
# hours -- a large gap likely means the vehicle wasn't continuously
# moving (e.g. parked, or a break in logging), so treating it as
# continuous travel would overestimate distance.
MAX_INTEGRATION_GAP_HOURS = 0.1  # 6 minutes


@dataclass
class EnergyEfficiencyResult:
    distance_km: float
    soc_used_pct: float
    efficiency_pct_per_km: Optional[float]  # None if it couldn't be computed
    battery_swap_events: int
    rows_used: int


def compute_energy_efficiency(rows: list[LogRow]) -> EnergyEfficiencyResult:
    """
    Takes a list of LogRow (from one file / one ride session) and
    returns an EnergyEfficiencyResult.
    """
    # --- Estimate distance travelled ---
    speed_rows = [row for row in rows if row.speed_kph is not None]

    distance_km = 0.0
    for previous_row, current_row in zip(speed_rows, speed_rows[1:]):
        hours_between = (
            current_row.timestamp - previous_row.timestamp
        ).total_seconds() / 3600.0

        if hours_between <= 0 or hours_between > MAX_INTEGRATION_GAP_HOURS:
            continue

        average_speed = (previous_row.speed_kph + current_row.speed_kph) / 2
        distance_km += average_speed * hours_between

    # --- Sum up total SOC consumed, correctly handling battery swaps ---
    soc_rows = [row for row in rows if _active_soc(row) is not None]

    soc_used_pct = 0.0
    battery_swap_events = 0

    for previous_row, current_row in zip(soc_rows, soc_rows[1:]):
        previous_soc = _active_soc(previous_row)
        current_soc = _active_soc(current_row)

        if current_soc < previous_soc:
            soc_used_pct += previous_soc - current_soc
        elif current_soc > previous_soc:
            # A pack swap (or genuine charging) -- don't subtract this
            # as negative consumption, just note it happened.
            battery_swap_events += 1

    efficiency_pct_per_km = None
    if distance_km > 0 and soc_used_pct > 0:
        efficiency_pct_per_km = soc_used_pct / distance_km

    return EnergyEfficiencyResult(
        distance_km=round(distance_km, 2),
        soc_used_pct=round(soc_used_pct, 1),
        efficiency_pct_per_km=(
            round(efficiency_pct_per_km, 3) if efficiency_pct_per_km is not None else None
        ),
        battery_swap_events=battery_swap_events,
        rows_used=len(speed_rows),
    )


# ---------------------------------------------------------------------
# Quick manual check when running this file directly.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from load_logs import load_vmoto_log, NoVmotoDataError

    if len(sys.argv) != 2:
        print("Usage: python3 energy_efficiency.py <path-to-log-file.txt>")
        raise SystemExit(1)

    try:
        rows = load_vmoto_log(sys.argv[1])
    except NoVmotoDataError as error:
        print(f"Skipped: {error}")
        raise SystemExit(0)

    result = compute_energy_efficiency(rows)

    print(f"Distance travelled: {result.distance_km} km")
    print(f"SOC used (total, across any battery swaps): {result.soc_used_pct}%")
    print(f"Battery swap events detected: {result.battery_swap_events}")
    if result.efficiency_pct_per_km is not None:
        print(f"Efficiency: {result.efficiency_pct_per_km}% SOC per km")
    else:
        print("Efficiency: not available (insufficient data)")
