"""
test_motor_health.py

Checks that compute_motor_health_score():
  - returns None (not a fake score) for files with no motor current
    data at all (older logger schema)
  - a healthy, consistent current-per-throttle ratio scores high
  - an abnormally high current-per-throttle ratio lowers the score
  - consecutive elevated readings count as ONE event, not one per row
    (same fix already applied in vehicle_health.py)
  - low-throttle rows are excluded (ratio is too noisy near zero)
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from load_logs import LogRow  # noqa: E402
from motor_health import compute_motor_health_score  # noqa: E402


def _row(seconds_offset: float, current: float | None, throttle: float | None) -> LogRow:
    return LogRow(
        device_id="test-device",
        timestamp=datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=seconds_offset),
        schema_tier="new",
        motor_current_a=current,
        throttle_pct=throttle,
    )


class MotorHealthTest(unittest.TestCase):
    def test_no_current_data_returns_none_score(self) -> None:
        # Rows with no motor_current_a at all -- e.g. an older-schema
        # file. Should report None, not fabricate a score.
        rows = [
            LogRow(device_id="d", timestamp=datetime(2026, 1, 1, 12, 0, i), schema_tier="mid")
            for i in range(5)
        ]

        result = compute_motor_health_score(rows)

        self.assertIsNone(result.score)
        self.assertEqual(result.rows_with_current_data, 0)

    def test_consistent_healthy_ratio_scores_100(self) -> None:
        # Current scales sensibly with throttle (ratio always ~0.3,
        # well under the threshold of 1.14) -- should score perfectly.
        rows = [_row(i, current=throttle * 0.3, throttle=throttle)
                for i, throttle in enumerate([20, 30, 40, 50, 60])]

        result = compute_motor_health_score(rows)

        self.assertEqual(result.inefficiency_events, 0)
        self.assertEqual(result.score, 100.0)

    def test_abnormally_high_ratio_lowers_score(self) -> None:
        # Most rows have a healthy ratio, but a few show current far
        # higher than the throttle position would suggest (ratio
        # ~2.5, well above the 1.14 threshold).
        rows = [_row(i, current=20 * 0.3, throttle=20) for i in range(10)]
        rows += [_row(i, current=50, throttle=20) for i in range(10, 13)]  # ratio = 2.5

        result = compute_motor_health_score(rows)

        self.assertGreater(result.inefficiency_events, 0)
        self.assertLess(result.score, 100.0)

    def test_consecutive_elevated_readings_count_as_one_event(self) -> None:
        # Same event-grouping fix as vehicle_health.py: 5 consecutive
        # elevated readings (1 second apart) should count as ONE
        # event, not five.
        rows = [_row(i, current=20 * 0.3, throttle=20) for i in range(10)]
        rows += [_row(i, current=50, throttle=20) for i in range(10, 15)]  # 5 consecutive elevated rows

        result = compute_motor_health_score(rows)

        self.assertEqual(result.inefficiency_events, 1)

    def test_low_throttle_rows_are_excluded(self) -> None:
        # Throttle at or below the minimum threshold (5%) should be
        # excluded entirely, even if current looks unusual -- the
        # ratio is too noisy to be meaningful that close to zero.
        rows = [_row(i, current=10.0, throttle=2.0) for i in range(10)]

        result = compute_motor_health_score(rows)

        self.assertEqual(result.rows_used, 0)
        self.assertEqual(result.score, 100.0)  # no usable rows -> no penalty applied


if __name__ == "__main__":
    unittest.main()
