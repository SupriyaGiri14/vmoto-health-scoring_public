"""
load_logs.py

Reads a raw Vmoto ride-log file (the .txt files exported from the
loggers, e.g. "16116760_2026-08-07.txt") and turns it into a list of
clean, consistent rows we can compute health scores from.

Why this file exists
---------------------
The raw files are messy in three ways:

1. Missing values are written as the literal string "n/a" instead of
   being left blank or being a real number.
2. There are THREE different versions ("schema tiers") of the column
   layout, depending on which logger/firmware wrote the file:

     - OLD tier:  single battery, columns prefixed "vmoto_cpx_"
                  e.g. vmoto_cpx_bat_volt, vmoto_cpx_soc
     - MID tier:  dual battery, columns prefixed "vmoto_"
                  e.g. vmoto_bat_1_voltage, vmoto_bat_2_voltage
     - NEW tier:  same as MID, PLUS a direct motor current reading
                  e.g. vmoto_motor_current

3. The device ID (which logger/moped this file came from) is not
   inside the file at all -- it's only in the filename, e.g.
   "16116760_2026-08-07.txt" -> device_id "16116760".

This file's job is ONLY to clean and standardise the data. It does
NOT compute any health scores -- that happens in separate files
(battery_health.py, vehicle_health.py) that will use the clean rows
this file produces.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# 1. A simple, clear container for one clean row of data.
# ---------------------------------------------------------------------
#
# Using a dataclass just means: "a simple box with named fields",
# instead of a raw dictionary. It makes typos easier to catch, because
# your editor will warn you if you write row.battary_1_voltage_v by
# mistake instead of row.battery_1_voltage_v.

@dataclass
class LogRow:
    device_id: str
    timestamp: datetime
    schema_tier: str  # "old", "mid", or "new" -- useful for debugging

    # Battery signals. Kept as Optional[float] because a value may be
    # genuinely missing ("n/a" in the raw file) -- we use None for
    # that, never a fake number like 0 or -1.
    battery_1_voltage_v: Optional[float] = None
    battery_1_soc_pct: Optional[float] = None
    battery_2_voltage_v: Optional[float] = None
    battery_2_soc_pct: Optional[float] = None

    # Motor / drive signals
    motor_current_a: Optional[float] = None
    speed_kph: Optional[float] = None
    throttle_pct: Optional[float] = None

    # Environment / vehicle state
    ambient_temp_c: Optional[float] = None
    imu_acc_x: Optional[float] = None
    imu_acc_y: Optional[float] = None
    imu_acc_z: Optional[float] = None

    # Anything we didn't map yet, kept around just in case it's useful
    # later -- avoids silently throwing away data.
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------
# 2. Small helper: safely turn a raw string into a float or None.
# ---------------------------------------------------------------------

def _to_float(raw_value: str) -> Optional[float]:
    """
    Converts a raw CSV string into a float, or None if it's missing.

    The raw files use the literal text "n/a" for missing data. Some
    fields might also just be an empty string. Both should become
    None, not 0 -- treating "no reading" as "zero" would badly skew
    any health-score maths later (e.g. it would look like the battery
    voltage suddenly dropped to 0V, which isn't what happened).
    """
    if raw_value is None:
        return None
    value = raw_value.strip()
    if value == "" or value.lower() == "n/a":
        return None
    try:
        return float(value)
    except ValueError:
        # If something unexpected shows up (e.g. a stray letter),
        # treat it as missing rather than crashing the whole file.
        return None


# ---------------------------------------------------------------------
# 3. Small helper: parse the raw date + time columns into one
#    datetime object.
# ---------------------------------------------------------------------

def _parse_timestamp(date_str: str, time_str: str) -> datetime:
    """
    Raw files split date and time into two columns, like:
        date = "07.08.2026"
        time = "14:08:49:157"   (the last part is milliseconds)

    This combines them into one Python datetime.
    """
    day, month, year = date_str.split(".")
    hour, minute, second, millisecond = time_str.split(":")

    return datetime(
        year=int(year),
        month=int(month),
        day=int(day),
        hour=int(hour),
        minute=int(minute),
        second=int(second),
        microsecond=int(millisecond) * 1000,
    )


# ---------------------------------------------------------------------
# 4. Figure out which schema tier a file is, just from its header row.
# ---------------------------------------------------------------------

class NoVmotoDataError(ValueError):
    """
    Raised when a file has none of the expected vmoto_* columns at
    all. This is a KNOWN situation -- some files only ever logged
    base sensors (IMU/GPS/temperature) with no CAN-bus data, so this
    is not a surprising or corrupt file, just one with nothing to
    score battery/motor health from.
    """


def _detect_schema_tier(fieldnames: list[str]) -> str:
    """
    Looks at the column names in the file's header row and decides
    which of the known layouts it is.
    """
    if "vmoto_motor_current" in fieldnames:
        return "new"
    if "vmoto_bat_1_voltage" in fieldnames:
        return "mid"
    if "vmoto_cpx_bat_1_voltage" in fieldnames:
        # A 4th, "transitional" tier found in real fleet data (device
        # 1086564): dual battery, but still using the older "cpx"
        # column prefix rather than the newer plain "vmoto_" prefix.
        # No motor current, same as "mid".
        return "transitional"
    if "vmoto_cpx_bat_volt" in fieldnames:
        return "old"

    has_any_vmoto_column = any(name.startswith("vmoto") for name in fieldnames)
    if not has_any_vmoto_column:
        raise NoVmotoDataError(
            "This file has no vmoto_* columns at all -- only base "
            "sensors (IMU/GPS/temperature). There's no battery or "
            "motor data to read from it."
        )

    # There IS a vmoto_* column, but not one we recognise -- this is
    # genuinely unexpected and worth a loud error, since it might
    # mean a schema tier we haven't seen yet.
    raise ValueError(
        "Unrecognised vmoto column layout -- found vmoto_* columns "
        "that don't match any known schema tier (old / mid / transitional / new). "
        f"Columns were: {fieldnames}"
    )


# ---------------------------------------------------------------------
# 5. Extract the device_id from the filename.
# ---------------------------------------------------------------------

def _device_id_from_filename(path: Path) -> str:
    """
    Extracts the device_id from a filename. Most files look like
    "16116760_2026-08-07.txt" (device_id first, before the first
    underscore). But at least one real file uses a different
    convention: "logs_16116688_16116688_2026-08-20.txt" (a "logs_"
    prefix, then the device_id repeated twice).

    Rather than always taking the first underscore-separated part
    (which would incorrectly return "logs" for that file), this looks
    for the first part that is PURELY DIGITS -- device IDs seen so
    far are always numeric (e.g. "1086344", "16116760"), so this
    reliably finds the real device_id regardless of what prefix or
    repetition surrounds it.
    """
    parts = path.stem.split("_")
    for part in parts:
        if part.isdigit():
            return part

    # Fallback: no purely-numeric part found -- use the first part
    # rather than crashing, but this is an unexpected filename shape
    # worth investigating if it ever happens.
    return parts[0]


# ---------------------------------------------------------------------
# 6. The main function: read a whole file, return a list of clean rows.
# ---------------------------------------------------------------------

def load_vmoto_log(path: Path | str) -> list[LogRow]:
    """
    Reads one raw Vmoto log file and returns a list of LogRow objects,
    one per row in the original file, with:

      - missing values converted from "n/a" strings to None
      - old/mid/new schema columns all mapped onto the SAME set of
        clean field names, so the rest of the code never needs to
        care which schema tier a file came from
      - a proper datetime instead of separate date/time strings
      - the device_id pulled from the filename
    """
    path = Path(path)
    device_id = _device_id_from_filename(path)

    rows: list[LogRow] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        schema_tier = _detect_schema_tier(reader.fieldnames or [])

        for raw_row in reader:
            timestamp = _parse_timestamp(raw_row["date"], raw_row["time"])

            row = LogRow(
                device_id=device_id,
                timestamp=timestamp,
                schema_tier=schema_tier,
                ambient_temp_c=_to_float(raw_row.get("temp_C")),
                imu_acc_x=_to_float(raw_row.get("imu_acc_x")),
                imu_acc_y=_to_float(raw_row.get("imu_acc_y")),
                imu_acc_z=_to_float(raw_row.get("imu_acc_z")),
            )

            if schema_tier == "old":
                # Single battery pack, older naming.
                # We put its voltage/SOC into the "battery_1" slot so
                # that every schema tier ends up using the same
                # field names -- battery_2 just stays None for old
                # files, since they don't have a second pack.
                row.battery_1_voltage_v = _to_float(raw_row.get("vmoto_cpx_bat_volt"))
                row.battery_1_soc_pct = _to_float(raw_row.get("vmoto_cpx_soc"))
                row.speed_kph = _to_float(raw_row.get("vmoto_cpx_kph"))
                row.throttle_pct = _to_float(raw_row.get("vmoto_cpx_throttle"))

            elif schema_tier == "transitional":
                # Dual battery, but still using the "cpx" column
                # prefix rather than the newer plain "vmoto_" prefix.
                row.battery_1_voltage_v = _to_float(raw_row.get("vmoto_cpx_bat_1_voltage"))
                row.battery_1_soc_pct = _to_float(raw_row.get("vmoto_cpx_bat_1_soc"))
                row.battery_2_voltage_v = _to_float(raw_row.get("vmoto_cpx_bat_2_voltage"))
                row.battery_2_soc_pct = _to_float(raw_row.get("vmoto_cpx_bat_2_soc"))
                row.speed_kph = _to_float(raw_row.get("vmoto_cpx_kph"))
                row.throttle_pct = _to_float(raw_row.get("vmoto_cpx_throttle"))

            else:
                # "mid" and "new" tiers share the same dual-battery
                # column names.
                row.battery_1_voltage_v = _to_float(raw_row.get("vmoto_bat_1_voltage"))
                row.battery_1_soc_pct = _to_float(raw_row.get("vmoto_bat_1_soc"))
                row.battery_2_voltage_v = _to_float(raw_row.get("vmoto_bat_2_voltage"))
                row.battery_2_soc_pct = _to_float(raw_row.get("vmoto_bat_2_soc"))
                row.speed_kph = _to_float(raw_row.get("vmoto_kph"))
                row.throttle_pct = _to_float(raw_row.get("vmoto_throttle"))

                if schema_tier == "new":
                    # Only the newest tier has this column at all.
                    row.motor_current_a = _to_float(raw_row.get("vmoto_motor_current"))

            rows.append(row)

    return rows


# ---------------------------------------------------------------------
# 7. Quick manual check when running this file directly.
# ---------------------------------------------------------------------
#
# This lets you run:  python3 src/load_logs.py path/to/file.txt
# and see a small summary printed out, without writing a separate
# script just to try it.

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python3 load_logs.py <path-to-log-file.txt>")
        raise SystemExit(1)

    try:
        rows = load_vmoto_log(sys.argv[1])
    except NoVmotoDataError as error:
        print(f"Skipped: {error}")
        raise SystemExit(0)

    print(f"Loaded {len(rows)} rows")
    print(f"Device ID: {rows[0].device_id}")
    print(f"Schema tier: {rows[0].schema_tier}")
    print(f"First timestamp: {rows[0].timestamp}")
    print(f"Last timestamp: {rows[-1].timestamp}")

    have_battery = sum(1 for r in rows if r.battery_1_voltage_v is not None)
    have_motor_current = sum(1 for r in rows if r.motor_current_a is not None)
    print(f"Rows with battery_1_voltage_v: {have_battery}")
    print(f"Rows with motor_current_a: {have_motor_current}")
