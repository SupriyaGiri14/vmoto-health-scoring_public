"""
test_load_logs.py

Checks that load_vmoto_log() correctly handles all three schema
tiers (old / mid / new), missing values, and the timestamp/device_id
parsing.

Each test builds a small, FAKE log file (just 1-2 rows) in a
temporary folder, runs it through load_vmoto_log(), and checks the
output looks the way we expect. This is the same style of test as
the existing predictive-maintenance repo's test_adapter.py.
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

# Allow running this test file directly without installing the
# package -- adds the src/ folder to Python's search path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from load_logs import load_vmoto_log, NoVmotoDataError  # noqa: E402


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class LoadVmotoLogTest(unittest.TestCase):
    def test_old_schema_single_battery(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "1086344_sample.txt"
            _write_csv(
                path,
                fieldnames=[
                    "log_nr", "date", "time", "imu_pitch", "imu_roll",
                    "imu_yaw", "imu_acc_x", "imu_acc_y", "imu_acc_z",
                    "temp_C", "lux", "soc", "gps_lon", "gps_lat",
                    "gps_sat", "gps_alt", "gps_kmh", "gps_mph",
                    "gps_head", "gps_hdop", "update_frequency_hz",
                    "vmoto_cpx_soc", "vmoto_cpx_bat_volt",
                    "vmoto_cpx_kph", "vmoto_cpx_throttle",
                    "vmoto_cpx_pow_mode", "vmoto_cpx_blink_r",
                    "vmoto_cpx_blink_l", "vmoto_cpx_h_beam",
                ],
                rows=[
                    {
                        "log_nr": "0", "date": "17.7.2026",
                        "time": "06:26:28:694",
                        "imu_pitch": "1", "imu_roll": "2", "imu_yaw": "3",
                        "imu_acc_x": "10", "imu_acc_y": "20", "imu_acc_z": "30",
                        "temp_C": "23", "lux": "0", "soc": "91",
                        "gps_lon": "0", "gps_lat": "0", "gps_sat": "0",
                        "gps_alt": "0", "gps_kmh": "0", "gps_mph": "0",
                        "gps_head": "0", "gps_hdop": "0",
                        "update_frequency_hz": "10",
                        "vmoto_cpx_soc": "n/a",
                        "vmoto_cpx_bat_volt": "n/a",
                        "vmoto_cpx_kph": "n/a",
                        "vmoto_cpx_throttle": "n/a",
                        "vmoto_cpx_pow_mode": "n/a",
                        "vmoto_cpx_blink_r": "0",
                        "vmoto_cpx_blink_l": "0",
                        "vmoto_cpx_h_beam": "0",
                    },
                    {
                        "log_nr": "1", "date": "17.7.2026",
                        "time": "06:26:29:000",
                        "imu_pitch": "1", "imu_roll": "2", "imu_yaw": "3",
                        "imu_acc_x": "10", "imu_acc_y": "20", "imu_acc_z": "30",
                        "temp_C": "23", "lux": "0", "soc": "91",
                        "gps_lon": "0", "gps_lat": "0", "gps_sat": "0",
                        "gps_alt": "0", "gps_kmh": "0", "gps_mph": "0",
                        "gps_head": "0", "gps_hdop": "0",
                        "update_frequency_hz": "10",
                        "vmoto_cpx_soc": "88",
                        "vmoto_cpx_bat_volt": "70.5",
                        "vmoto_cpx_kph": "12.3",
                        "vmoto_cpx_throttle": "40",
                        "vmoto_cpx_pow_mode": "1",
                        "vmoto_cpx_blink_r": "0",
                        "vmoto_cpx_blink_l": "0",
                        "vmoto_cpx_h_beam": "0",
                    },
                ],
            )

            rows = load_vmoto_log(path)

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].device_id, "1086344")
        self.assertEqual(rows[0].schema_tier, "old")

        # First row's battery fields were "n/a" -> must become None,
        # never 0 or any other placeholder number.
        self.assertIsNone(rows[0].battery_1_voltage_v)
        self.assertIsNone(rows[0].battery_1_soc_pct)

        # Second row has real values -> must be parsed as floats and
        # mapped into the SAME field names as every other schema tier.
        self.assertEqual(rows[1].battery_1_voltage_v, 70.5)
        self.assertEqual(rows[1].battery_1_soc_pct, 88.0)

        # Old schema has no second battery pack or motor current at all.
        self.assertIsNone(rows[1].battery_2_voltage_v)
        self.assertIsNone(rows[1].motor_current_a)

        # Timestamp parsing check.
        self.assertEqual(rows[0].timestamp.year, 2026)
        self.assertEqual(rows[0].timestamp.month, 7)
        self.assertEqual(rows[0].timestamp.day, 17)
        self.assertEqual(rows[0].timestamp.hour, 6)
        self.assertEqual(rows[0].timestamp.microsecond, 694_000)

    def test_mid_schema_dual_battery_no_motor_current(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "1086344_sample_mid.txt"
            _write_csv(
                path,
                fieldnames=[
                    "log_nr", "date", "time", "temp_C", "imu_acc_x",
                    "imu_acc_y", "imu_acc_z",
                    "vmoto_bat_1_voltage", "vmoto_bat_1_soc",
                    "vmoto_bat_2_voltage", "vmoto_bat_2_soc",
                    "vmoto_kph", "vmoto_throttle",
                ],
                rows=[
                    {
                        "log_nr": "0", "date": "27.7.2026",
                        "time": "15:47:20:364",
                        "temp_C": "23", "imu_acc_x": "1",
                        "imu_acc_y": "2", "imu_acc_z": "3",
                        "vmoto_bat_1_voltage": "n/a",
                        "vmoto_bat_1_soc": "n/a",
                        "vmoto_bat_2_voltage": "70.2",
                        "vmoto_bat_2_soc": "99",
                        "vmoto_kph": "5.0",
                        "vmoto_throttle": "10",
                    },
                ],
            )

            rows = load_vmoto_log(path)

        self.assertEqual(rows[0].schema_tier, "mid")
        self.assertIsNone(rows[0].battery_1_voltage_v)
        self.assertEqual(rows[0].battery_2_voltage_v, 70.2)
        self.assertEqual(rows[0].battery_2_soc_pct, 99.0)
        self.assertIsNone(rows[0].motor_current_a)  # not present in "mid"

    def test_file_with_no_vmoto_columns_raises_clear_error(self) -> None:
        # Some real files only have base sensors (IMU/GPS/temperature)
        # and no vmoto_* CAN data at all -- e.g. device 1086240 on
        # 2026-07-11. Loading one of these should raise a clear,
        # specific error rather than crash with a confusing traceback
        # or silently return empty/wrong data.
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "1086240_sample_no_vmoto.txt"
            _write_csv(
                path,
                fieldnames=[
                    "log_nr", "date", "time", "imu_pitch", "imu_roll",
                    "imu_yaw", "imu_acc_x", "imu_acc_y", "imu_acc_z",
                    "temp_C", "lux", "soc",
                ],
                rows=[
                    {
                        "log_nr": "0", "date": "11.7.2026",
                        "time": "10:00:00:000",
                        "imu_pitch": "1", "imu_roll": "2", "imu_yaw": "3",
                        "imu_acc_x": "10", "imu_acc_y": "20", "imu_acc_z": "30",
                        "temp_C": "23", "lux": "0", "soc": "91",
                    },
                ],
            )

            with self.assertRaises(NoVmotoDataError):
                load_vmoto_log(path)

    def test_new_schema_has_motor_current(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "16116760_sample_new.txt"
            _write_csv(
                path,
                fieldnames=[
                    "log_nr", "date", "time", "temp_C",
                    "vmoto_bat_1_voltage", "vmoto_bat_1_soc",
                    "vmoto_bat_2_voltage", "vmoto_bat_2_soc",
                    "vmoto_kph", "vmoto_throttle", "vmoto_motor_current",
                ],
                rows=[
                    {
                        "log_nr": "0", "date": "7.8.2026",
                        "time": "14:08:49:157", "temp_C": "23",
                        "vmoto_bat_1_voltage": "n/a",
                        "vmoto_bat_1_soc": "n/a",
                        "vmoto_bat_2_voltage": "70.2",
                        "vmoto_bat_2_soc": "99",
                        "vmoto_kph": "18.5",
                        "vmoto_throttle": "60",
                        "vmoto_motor_current": "12.4",
                    },
                ],
            )

            rows = load_vmoto_log(path)

        self.assertEqual(rows[0].schema_tier, "new")
        self.assertEqual(rows[0].device_id, "16116760")
        self.assertEqual(rows[0].motor_current_a, 12.4)


if __name__ == "__main__":
    unittest.main()
