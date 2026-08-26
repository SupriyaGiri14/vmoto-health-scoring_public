"""
test_vehicle_health.py

Checks that compute_vehicle_health_score() behaves sensibly:
  - smooth/steady IMU readings score high
  - large spikes lower the score
  - a LONGER file with the same spike RATE scores the same as a
    shorter one (this is the bug we caught and fixed -- scoring on
    raw spike count unfairly punished longer files)
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from load_logs import LogRow  # noqa: E402
from vehicle_health import compute_vehicle_health_score  # noqa: E402


def _row(seconds_offset: float, x: float, y: float, z: float) -> LogRow:
    return LogRow(
        device_id="test-device",
        timestamp=datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=seconds_offset),
        schema_tier="new",
        imu_acc_x=x,
        imu_acc_y=y,
        imu_acc_z=z,
    )


# A "calm" reading, magnitude well under the spike threshold.
CALM = (1000.0, 1000.0, 1000.0)  # magnitude ~1732

# A "spike" reading, magnitude well over the spike threshold (3200).
SPIKE = (3000.0, 3000.0, 3000.0)  # magnitude ~5196


class VehicleHealthTest(unittest.TestCase):
    def test_all_calm_readings_score_100(self) -> None:
        rows = [_row(i, *CALM) for i in range(20)]

        result = compute_vehicle_health_score(rows)

        self.assertEqual(result.vibration_spike_events, 0)
        self.assertEqual(result.score, 100.0)

    def test_spikes_lower_the_score(self) -> None:
        rows = [_row(i, *CALM) for i in range(18)]
        rows += [_row(18, *SPIKE), _row(19, *SPIKE)]

        result = compute_vehicle_health_score(rows)

        self.assertEqual(result.vibration_spike_events, 2)
        self.assertLess(result.score, 100.0)

    def test_same_spike_rate_scores_the_same_regardless_of_file_length(self) -> None:
        # Short file: 10 rows, 1 spike (10% spike rate)
        short_rows = [_row(i, *CALM) for i in range(9)]
        short_rows += [_row(9, *SPIKE)]

        # Long file: 100 rows, 10 spikes (same 10% spike rate)
        long_rows = []
        for i in range(100):
            if i % 10 == 0:
                long_rows.append(_row(i, *SPIKE))
            else:
                long_rows.append(_row(i, *CALM))

        short_result = compute_vehicle_health_score(short_rows)
        long_result = compute_vehicle_health_score(long_rows)

        # Same RATE of rough riding -> should produce the same score,
        # even though the long file has 10x more spike events in
        # absolute terms.
        self.assertEqual(short_result.score, long_result.score)


if __name__ == "__main__":
    unittest.main()
