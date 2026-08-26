"""
test_summary.py

Checks that summarize_file():
  - combines battery, vehicle, and synthetic event results correctly
  - returns None (not a crash) for files with no vmoto data
  - resolves vehicle_id when a device_map is supplied
  - leaves vehicle_id as None when no device_map is supplied
  - leaves next_service_km / active_can_faults as None (data not
    available), rather than a fake number
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from summary import summarize_file  # noqa: E402


def _write_new_schema_file(path: Path, num_rows: int = 5) -> None:
    """
    Writes a small, valid "new" schema fake log file -- enough rows
    for the loader and scorers to run without errors.
    """
    fieldnames = [
        "log_nr", "date", "time", "temp_C", "imu_acc_x", "imu_acc_y",
        "imu_acc_z", "vmoto_bat_1_voltage", "vmoto_bat_1_soc",
        "vmoto_bat_2_voltage", "vmoto_bat_2_soc", "vmoto_kph",
        "vmoto_throttle", "vmoto_motor_current",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(num_rows):
            writer.writerow(
                {
                    "log_nr": str(i),
                    "date": "7.8.2026",
                    "time": f"14:08:{49 + i}:000",
                    "temp_C": "23",
                    "imu_acc_x": "1000", "imu_acc_y": "1000", "imu_acc_z": "1000",
                    "vmoto_bat_1_voltage": "n/a", "vmoto_bat_1_soc": "n/a",
                    "vmoto_bat_2_voltage": "70.2", "vmoto_bat_2_soc": "99",
                    "vmoto_kph": "18.5", "vmoto_throttle": "60",
                    "vmoto_motor_current": "12.4",
                }
            )


def _write_no_vmoto_file(path: Path) -> None:
    fieldnames = ["log_nr", "date", "time", "temp_C", "imu_acc_x"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {"log_nr": "0", "date": "7.8.2026", "time": "14:08:49:000",
             "temp_C": "23", "imu_acc_x": "1"}
        )


class SummarizeFileTest(unittest.TestCase):
    def test_returns_summary_with_expected_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "16116760_sample.txt"
            _write_new_schema_file(path)

            summary = summarize_file(path)

        self.assertIsNotNone(summary)
        self.assertEqual(summary.device_id, "16116760")
        self.assertEqual(summary.schema_tier, "new")
        self.assertEqual(summary.total_rows, 5)
        self.assertIsInstance(summary.battery_health_score, float)
        self.assertIsInstance(summary.vehicle_health_score, float)
        self.assertTrue(summary.events_are_synthetic)

    def test_returns_none_for_file_with_no_vmoto_data(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "1086240_sample.txt"
            _write_no_vmoto_file(path)

            summary = summarize_file(path)

        self.assertIsNone(summary)

    def test_vehicle_id_is_none_without_device_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "16116760_sample.txt"
            _write_new_schema_file(path)

            summary = summarize_file(path)

        self.assertIsNone(summary.vehicle_id)

    def test_vehicle_id_resolves_with_device_map(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "16116760_sample.txt"
            _write_new_schema_file(path)

            summary = summarize_file(
                path, device_map={"16116760": "116IAG"}
            )

        self.assertEqual(summary.vehicle_id, "116IAG")

    def test_unmapped_device_id_stays_none(self) -> None:
        # A device_map is supplied, but it doesn't contain THIS
        # device's ID -- should stay None, not raise an error.
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "16116760_sample.txt"
            _write_new_schema_file(path)

            summary = summarize_file(
                path, device_map={"some_other_device": "999XYZ"}
            )

        self.assertIsNone(summary.vehicle_id)

    def test_unavailable_fields_stay_none(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "16116760_sample.txt"
            _write_new_schema_file(path)

            summary = summarize_file(path)

        self.assertIsNone(summary.next_service_km)
        self.assertIsNone(summary.active_can_faults)

    def test_to_dict_produces_json_safe_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "16116760_sample.txt"
            _write_new_schema_file(path)

            summary = summarize_file(path)
            data = summary.to_dict()

        # Timestamps should be strings (JSON can't hold datetime
        # objects directly), everything else should round-trip fine.
        self.assertIsInstance(data["session_start"], str)
        self.assertIsInstance(data["session_end"], str)


if __name__ == "__main__":
    unittest.main()
