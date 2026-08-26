"""
test_vibration_pattern.py

Checks that classify_vibration_pattern() correctly distinguishes:
  - a one-off rough ride (isolated_spike)
  - a real, ongoing elevated pattern (persistent_elevated)
  - a worsening trend over time (worsening)
  - normal riding (normal)
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vibration_pattern import (  # noqa: E402
    VibrationSession,
    classify_vibration_pattern,
)


def _session(days_ago: int, rate: float) -> VibrationSession:
    return VibrationSession(
        timestamp=datetime(2026, 8, 19) - timedelta(days=days_ago),
        spike_rate_per_1000_rows=rate,
    )


class ClassifyVibrationPatternTest(unittest.TestCase):
    def test_all_normal_sessions_classify_as_normal(self) -> None:
        sessions = [
            _session(days_ago=i, rate=10.0) for i in range(5)
        ]

        result = classify_vibration_pattern(sessions)

        self.assertEqual(result.classification, "normal")

    def test_single_recent_spike_after_normal_history_is_isolated(self) -> None:
        # Several normal rides, then ONE elevated ride most recently.
        sessions = [
            _session(days_ago=10, rate=10.0),
            _session(days_ago=8, rate=11.0),
            _session(days_ago=6, rate=9.0),
            _session(days_ago=0, rate=40.0),  # today, much higher
        ]

        result = classify_vibration_pattern(sessions)

        self.assertEqual(result.classification, "isolated_spike")

    def test_consistently_elevated_recent_sessions_are_persistent(self) -> None:
        # Normal history, then several recent rides ALL elevated (but
        # flat, not climbing) -- a real ongoing pattern, not a spike.
        sessions = [
            _session(days_ago=20, rate=10.0),
            _session(days_ago=18, rate=11.0),
            _session(days_ago=6, rate=35.0),
            _session(days_ago=4, rate=36.0),
            _session(days_ago=0, rate=34.0),
        ]

        result = classify_vibration_pattern(sessions)

        self.assertEqual(result.classification, "persistent_elevated")

    def test_climbing_recent_sessions_are_worsening(self) -> None:
        # Normal history, then recent rides that are BOTH elevated
        # AND increasing each time -- looks like something degrading.
        sessions = [
            _session(days_ago=20, rate=10.0),
            _session(days_ago=18, rate=11.0),
            _session(days_ago=6, rate=30.0),
            _session(days_ago=4, rate=40.0),
            _session(days_ago=0, rate=50.0),
        ]

        result = classify_vibration_pattern(sessions)

        self.assertEqual(result.classification, "worsening")

    def test_single_session_reports_normal_with_no_history(self) -> None:
        sessions = [_session(days_ago=0, rate=100.0)]

        result = classify_vibration_pattern(sessions)

        self.assertEqual(result.classification, "normal")
        self.assertEqual(result.sessions_used, 1)

    def test_recovering_after_one_bad_ride_is_not_worsening(self) -> None:
        # A vehicle that HAD a rough ride but has since returned to
        # normal should not be flagged as worsening or persistent.
        sessions = [
            _session(days_ago=10, rate=10.0),
            _session(days_ago=8, rate=45.0),  # one bad ride
            _session(days_ago=4, rate=11.0),  # back to normal
            _session(days_ago=0, rate=10.0),  # still normal
        ]

        result = classify_vibration_pattern(sessions)

        self.assertEqual(result.classification, "normal")

    def test_raises_on_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            classify_vibration_pattern([])


if __name__ == "__main__":
    unittest.main()
