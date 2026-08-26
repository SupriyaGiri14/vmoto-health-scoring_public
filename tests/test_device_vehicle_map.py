"""
test_device_vehicle_map.py

Checks that load_device_vehicle_map() and device_to_vehicle_id():
  - correctly loads the real CSV
  - correctly maps a known device to its vehicle
  - a vehicle with two devices (a logger upgrade) maps both to the
    SAME vehicle_id
  - an unrecognised device falls back gracefully, not a crash
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from device_vehicle_map import load_device_vehicle_map, device_to_vehicle_id  # noqa: E402


MAP_PATH = Path(__file__).resolve().parent.parent / "data" / "device_vehicle_map.csv"


class DeviceVehicleMapTest(unittest.TestCase):
    def test_loads_real_csv(self) -> None:
        mapping = load_device_vehicle_map(MAP_PATH)
        self.assertIn("1086344", mapping)
        self.assertGreater(len(mapping), 5)

    def test_known_device_maps_to_correct_vehicle(self) -> None:
        mapping = load_device_vehicle_map(MAP_PATH)
        self.assertEqual(device_to_vehicle_id("1086344", mapping), "116IAG")

    def test_two_devices_same_vehicle_map_to_same_id(self) -> None:
        # 116IAG had two loggers over its lifetime: 1086344 (older)
        # and 16116760 (newer). Both should resolve to the same
        # vehicle_id, so the vehicle's history stays continuous.
        mapping = load_device_vehicle_map(MAP_PATH)
        self.assertEqual(
            device_to_vehicle_id("1086344", mapping),
            device_to_vehicle_id("16116760", mapping),
        )

    def test_unrecognised_device_falls_back_gracefully(self) -> None:
        mapping = load_device_vehicle_map(MAP_PATH)
        result = device_to_vehicle_id("99999999", mapping)
        self.assertEqual(result, "unmapped-99999999")


if __name__ == "__main__":
    unittest.main()
