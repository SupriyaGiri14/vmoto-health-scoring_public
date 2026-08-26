"""
device_vehicle_map.py

Loads data/device_vehicle_map.csv -- the lookup between a device_id
(the logger installed on a moped, e.g. "1086344") and its vehicle_id
(the actual moped, e.g. "116IAG").

Why this exists
-----------------
Several vehicles in the fleet have had TWO different loggers
installed over their lifetime (an older one, then a newer one). The
raw log files are named by device_id, not vehicle_id, so without this
mapping, the same physical moped's history looks like two unrelated
devices.

Source: the fleet device table shared by the team. Two vehicles
(118IAG, 122IAG) show a vehicle_type mismatch across their two
devices (vmoto_cpx vs vmoto_vs1) -- this is preserved as-is from the
source data and flagged as an open question, not corrected here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DeviceInfo:
    device_id: str
    vehicle_id: str
    fleet: str
    company: str
    vehicle_type: str


def load_device_vehicle_map(path: str | Path = "data/device_vehicle_map.csv") -> dict[str, DeviceInfo]:
    """
    Returns {device_id: DeviceInfo}.
    """
    mapping: dict[str, DeviceInfo] = {}
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            mapping[row["device_id"]] = DeviceInfo(
                device_id=row["device_id"],
                vehicle_id=row["vehicle_id"],
                fleet=row["fleet"],
                company=row["company"],
                vehicle_type=row["vehicle_type"],
            )
    return mapping


def device_to_vehicle_id(device_id: str, mapping: dict[str, DeviceInfo]) -> str:
    """
    Looks up the vehicle_id for a device_id. Falls back to the
    device_id itself (prefixed to make it obviously unmapped) if the
    device isn't in the map yet -- so a new/unrecognised device still
    shows up in the dashboard rather than silently disappearing.
    """
    info = mapping.get(device_id)
    if info is not None:
        return info.vehicle_id
    return f"unmapped-{device_id}"
