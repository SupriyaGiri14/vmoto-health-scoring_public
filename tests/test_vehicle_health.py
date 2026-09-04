"""
test_vehicle_health.py

Checks that compute_vehicle_health_score() behaves sensibly:
  - smooth/steady IMU readings score high
  - large spikes lower the score
  - a LONGER file with the same spike RATE scores the same as a
    shorter one (this is the bug we caught and fixed -- scoring on
    raw spike count unfairly punished longer files)
  - CONSECUTIVE above-threshold readings (e.g. one pothole spanning
    several rows) count as ONE event, not one event per row -- this
    is the second bug that was caught and fixed
  - readings separated by a real time gap count as SEPARATE events
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

    def test_two_separated_spikes_count_as_two_events(self) -> None:
        # Two spikes far apart in time (50 seconds), each isolated by
        # calm readings -- these are two genuinely separate bumps.
        rows = [_row(i, *CALM) for i in range(18)]
        rows.append(_row(18, *SPIKE))
        rows += [_row(i, *CALM) for i in range(19, 68)]
        rows.append(_row(68, *SPIKE))

        result = compute_vehicle_health_score(rows)

        self.assertEqual(result.vibration_spike_events, 2)
        self.assertLess(result.score, 100.0)

    def test_consecutive_spikes_count_as_one_event(self) -> None:
        # This reproduces the real bug found in production data: a
        # single physical bump (e.g. a pothole) often triggers SEVERAL
        # consecutive above-threshold readings, not just one. These 5
        # consecutive spike rows (1 second apart, well within
        # MAX_EVENT_GAP_SECONDS) represent ONE bump and should be
        # counted as ONE event, not five.
        rows = [_row(i, *CALM) for i in range(10)]
        rows += [_row(i, *SPIKE) for i in range(10, 15)]  # 5 consecutive spike rows
        rows += [_row(i, *CALM) for i in range(15, 20)]

        result = compute_vehicle_health_score(rows)

        self.assertEqual(result.vibration_spike_events, 1)

    def test_same_spike_rate_scores_the_same_regardless_of_file_length(self) -> None:
        # Short file: 10 rows, 1 spike (10% spike rate)
        short_rows = [_row(i, *CALM) for i in range(9)]
        short_rows += [_row(9, *SPIKE)]

        # Long file: 100 rows, 10 ISOLATED spikes (same 10% spike
        # rate) -- each spike is surrounded by calm readings, so
        # grouping doesn't merge any of them together.
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
