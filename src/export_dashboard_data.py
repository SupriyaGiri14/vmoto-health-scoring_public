"""
export_dashboard_data.py

Processes every raw Vmoto log file in a folder and produces THREE
small CSV files, safe to publish (no raw sensor data, no huge files):

  1. rides_summary.csv       -- one row per ride/file: scores, dates,
                                 device + vehicle IDs.
  2. battery_timeseries.csv  -- SOC over time per ride, deduplicated
                                 to only real value changes (a full
                                 file can have 100,000+ raw rows but
                                 only a few hundred actual SOC
                                 changes).
  3. vibration_timeseries.csv -- vibration magnitude aggregated into
                                  1-minute buckets (max value per
                                  minute).

This is the ONLY step that ever touches the large raw .txt files
(which stay private, local, and gitignored). Everything downstream
-- including the deployed dashboard app -- only ever reads these
small CSVs, never the raw files.

Also joins in vehicle_id from data/device_vehicle_map.csv, so the
deployed app doesn't need to do that lookup or know about device IDs
being tied to physical loggers at all.

Usage
-----
    python3 src/export_dashboard_data.py data/raw/ app/data/
"""

from __future__ import annotations

import csv
from pathlib import Path

from load_logs import load_vmoto_log, NoVmotoDataError
from battery_health import compute_battery_health_score, _active_soc
from vehicle_health import compute_vehicle_health_score, _vibration_magnitude, VIBRATION_SPIKE_THRESHOLD
from device_vehicle_map import load_device_vehicle_map, device_to_vehicle_id


def export_all(
    input_folder: str,
    output_folder: str,
    device_map_path: str = "data/device_vehicle_map.csv",
) -> None:
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    device_map = load_device_vehicle_map(device_map_path)

    summary_rows = []
    battery_rows = []
    vibration_rows = []

    txt_files = sorted(input_path.glob("*.txt"))
    print(f"Found {len(txt_files)} files to process...")

    for file_path in txt_files:
        try:
            rows = load_vmoto_log(file_path)
        except NoVmotoDataError:
            print(f"  Skipped (no vmoto data): {file_path.name}")
            continue

        if not rows:
            continue

        device_id = rows[0].device_id
        vehicle_id = device_to_vehicle_id(device_id, device_map)
        schema_tier = rows[0].schema_tier
        ride_date = rows[0].timestamp.date().isoformat()
        session_start = rows[0].timestamp
        session_end = rows[-1].timestamp

        # --- Summary row ---
        battery_result = compute_battery_health_score(rows)
        vehicle_result = compute_vehicle_health_score(rows)
        spike_rate = (
            (vehicle_result.vibration_spike_events / vehicle_result.rows_used) * 1000
            if vehicle_result.rows_used else 0
        )

        summary_rows.append({
            "vehicle_id": vehicle_id,
            "device_id": device_id,
            "ride_date": ride_date,
            "session_start": session_start.isoformat(),
            "session_end": session_end.isoformat(),
            "schema_tier": schema_tier,
            "total_rows": len(rows),
            "battery_health_score": battery_result.score,
            "voltage_sag_events": battery_result.voltage_sag_events,
            "fast_soc_drop_minutes": battery_result.fast_soc_drop_minutes,
            "vehicle_health_score": vehicle_result.score,
            "vibration_spike_events": vehicle_result.vibration_spike_events,
            "vibration_spike_rate_per_1000_rows": round(spike_rate, 2),
        })
        print(f"  Processed: {file_path.name} (vehicle {vehicle_id}, device {device_id}, {ride_date})")

        # --- Battery time series (deduplicated on value change) ---
        last_bat1 = last_bat2 = "unset"
        for r in rows:
            if r.battery_1_soc_pct != last_bat1 or r.battery_2_soc_pct != last_bat2:
                battery_rows.append({
                    "vehicle_id": vehicle_id,
                    "device_id": device_id,
                    "ride_date": ride_date,
                    "timestamp": r.timestamp.isoformat(),
                    "battery_1_soc_pct": r.battery_1_soc_pct,
                    "battery_2_soc_pct": r.battery_2_soc_pct,
                })
                last_bat1, last_bat2 = r.battery_1_soc_pct, r.battery_2_soc_pct

        # --- Vibration time series (1-minute max buckets) ---
        bucket_max: dict[int, float] = {}
        for r in rows:
            mag = _vibration_magnitude(r)
            if mag is None:
                continue
            minute_index = int((r.timestamp - session_start).total_seconds() // 60)
            if minute_index not in bucket_max or mag > bucket_max[minute_index]:
                bucket_max[minute_index] = mag

        for minute_index in sorted(bucket_max.keys()):
            vibration_rows.append({
                "vehicle_id": vehicle_id,
                "device_id": device_id,
                "ride_date": ride_date,
                "minute_offset": minute_index,
                "max_vibration_magnitude": round(bucket_max[minute_index], 1),
                "above_threshold": bucket_max[minute_index] >= VIBRATION_SPIKE_THRESHOLD,
            })

    # --- Write CSVs ---
    _write_csv(output_path / "rides_summary.csv", summary_rows)
    _write_csv(output_path / "battery_timeseries.csv", battery_rows)
    _write_csv(output_path / "vibration_timeseries.csv", vibration_rows)

    print()
    print(f"rides_summary.csv:        {len(summary_rows)} rows")
    print(f"battery_timeseries.csv:   {len(battery_rows)} rows")
    print(f"vibration_timeseries.csv: {len(vibration_rows)} rows")
    print(f"\nSaved to {output_path}/ -- safe to commit, no raw sensor data included.")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python3 export_dashboard_data.py <input_folder> <output_folder>")
        raise SystemExit(1)

    export_all(sys.argv[1], sys.argv[2])
