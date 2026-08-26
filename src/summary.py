"""
summary.py

Runs one log file through all three scorers (battery health, vehicle
health, synthetic events) and produces ONE clean result -- the kind
of row Hanlin's dashboard "Vehicle Health" table expects.

This file doesn't do any new calculation itself. It just calls the
existing functions from battery_health.py, vehicle_health.py, and
synthetic_labels.py, and puts their results together in one place.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from load_logs import load_vmoto_log, NoVmotoDataError, LogRow
from battery_health import compute_battery_health_score
from vehicle_health import compute_vehicle_health_score
from synthetic_labels import generate_synthetic_events


@dataclass
class VehicleSummary:
    # --- Identity ---
    device_id: str
    vehicle_id: str | None  # None until a device->vehicle map is supplied
    schema_tier: str

    # --- Session info ---
    session_start: datetime
    session_end: datetime
    total_rows: int

    # --- Scores (0-100) ---
    battery_health_score: float
    vehicle_health_score: float

    # --- Synthetic misuse-event counts (see synthetic_labels.py --
    # these are PLACEHOLDER counts, not real detections) ---
    synthetic_curb_up_count: int
    synthetic_curb_down_count: int
    synthetic_pothole_count: int
    synthetic_fall_count: int
    events_are_synthetic: bool  # always True right now -- see below

    # --- Fields the dashboard mockup expects but we cannot compute
    # yet, because the source data doesn't contain them. Kept as
    # explicit None (not omitted, not faked) so it's obvious to
    # anyone reading this that they're missing, not zero. ---
    next_service_km: None
    active_can_faults: None

    def to_dict(self) -> dict:
        """Convert to a plain dictionary -- handy for JSON output."""
        data = asdict(self)
        data["session_start"] = self.session_start.isoformat()
        data["session_end"] = self.session_end.isoformat()
        return data


def _device_to_vehicle_id(device_id: str, device_map: dict[str, str] | None) -> str | None:
    if device_map is None:
        return None
    return device_map.get(device_id)


def summarize_file(
    path: Path | str,
    device_map: dict[str, str] | None = None,
) -> VehicleSummary | None:
    """
    Reads one log file and returns a VehicleSummary combining all
    three scorers.

    device_map: optional dict of {device_id: vehicle_id}, e.g.
    {"16116760": "116IAG"}. If not supplied, vehicle_id stays None
    and only device_id is available -- see device_vehicle_map.py
    (not built yet) for where this mapping should eventually come
    from.

    Returns None (instead of raising) if the file has no vmoto data
    at all -- this makes it easy to process a whole folder of files
    and just skip the ones with nothing to score, without the whole
    batch crashing.
    """
    try:
        rows: list[LogRow] = load_vmoto_log(path)
    except NoVmotoDataError:
        return None

    if not rows:
        return None

    battery_result = compute_battery_health_score(rows)
    vehicle_result = compute_vehicle_health_score(rows)
    synthetic_events = generate_synthetic_events(rows)

    curb_up = sum(1 for e in synthetic_events if e.event_type == "curb_up")
    curb_down = sum(1 for e in synthetic_events if e.event_type == "curb_down")
    pothole = sum(1 for e in synthetic_events if e.event_type == "pothole")
    fall = sum(1 for e in synthetic_events if e.event_type == "fall")

    device_id = rows[0].device_id

    return VehicleSummary(
        device_id=device_id,
        vehicle_id=_device_to_vehicle_id(device_id, device_map),
        schema_tier=rows[0].schema_tier,
        session_start=rows[0].timestamp,
        session_end=rows[-1].timestamp,
        total_rows=len(rows),
        battery_health_score=battery_result.score,
        vehicle_health_score=vehicle_result.score,
        synthetic_curb_up_count=curb_up,
        synthetic_curb_down_count=curb_down,
        synthetic_pothole_count=pothole,
        synthetic_fall_count=fall,
        events_are_synthetic=True,
        next_service_km=None,
        active_can_faults=None,
    )


# ---------------------------------------------------------------------
# Quick manual check when running this file directly.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python3 summary.py <path-to-log-file.txt>")
        raise SystemExit(1)

    summary = summarize_file(sys.argv[1])

    if summary is None:
        print("No vmoto data in this file -- nothing to summarize.")
        raise SystemExit(0)

    print(json.dumps(summary.to_dict(), indent=2))
