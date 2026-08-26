"""
test_overall_vehicle_health.py

Checks that aggregate_vehicle_health():
  - does NOT punish a vehicle just for having more sessions (directly
    addresses Hanlin's "trip-based accumulation" concern)
  - weights recent sessions more heavily than old ones
  - correctly detects "improving" and "worsening" trends
  - a single bad session doesn't tank an otherwise healthy vehicle's
    overall score

And that group_sessions_by_day():
  - averages multiple same-day rides into one daily score
  - keeps different days as separate entries
  - produces output that still works correctly when fed into
    aggregate_vehicle_health()
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from overall_vehicle_health import (  # noqa: E402
    ScoredSession,
    aggregate_vehicle_health,
    group_sessions_by_day,
)


def _session(days_ago: int, score: float) -> ScoredSession:
    return ScoredSession(
        timestamp=datetime(2026, 8, 19) - timedelta(days=days_ago),
        score=score,
    )


class AggregateVehicleHealthTest(unittest.TestCase):
    def test_more_sessions_at_the_same_score_does_not_lower_the_result(self) -> None:
        # This is the direct check for Hanlin's Point 1: a vehicle
        # that consistently scores 90 across 3 sessions should have
        # roughly the same overall score as one that scores 90
        # across 10 sessions -- more usage at the SAME quality should
        # not drag the number down.
        few_sessions = [_session(days_ago=i, score=90.0) for i in range(3)]
        many_sessions = [_session(days_ago=i, score=90.0) for i in range(10)]

        few_result = aggregate_vehicle_health(few_sessions)
        many_result = aggregate_vehicle_health(many_sessions)

        self.assertAlmostEqual(few_result.overall_score, many_result.overall_score, places=1)
        self.assertEqual(few_result.overall_score, 90.0)
        self.assertEqual(many_result.overall_score, 90.0)

    def test_recent_sessions_count_more_than_old_ones(self) -> None:
        # Old session was bad (60), recent session is good (100).
        # The overall score should sit CLOSER to the recent one, not
        # dead in the middle -- because recent condition matters more.
        sessions = [
            _session(days_ago=30, score=60.0),
            _session(days_ago=0, score=100.0),
        ]

        result = aggregate_vehicle_health(sessions)

        midpoint = (60.0 + 100.0) / 2  # 80.0
        self.assertGreater(result.overall_score, midpoint)

    def test_worsening_trend_is_detected(self) -> None:
        sessions = [
            _session(days_ago=20, score=95.0),
            _session(days_ago=15, score=93.0),
            _session(days_ago=5, score=70.0),
            _session(days_ago=0, score=65.0),
        ]

        result = aggregate_vehicle_health(sessions)

        self.assertEqual(result.trend, "worsening")

    def test_improving_trend_is_detected(self) -> None:
        sessions = [
            _session(days_ago=20, score=65.0),
            _session(days_ago=15, score=70.0),
            _session(days_ago=5, score=93.0),
            _session(days_ago=0, score=95.0),
        ]

        result = aggregate_vehicle_health(sessions)

        self.assertEqual(result.trend, "improving")

    def test_stable_trend_when_scores_barely_change(self) -> None:
        sessions = [
            _session(days_ago=10, score=88.0),
            _session(days_ago=5, score=89.0),
            _session(days_ago=0, score=87.0),
        ]

        result = aggregate_vehicle_health(sessions)

        self.assertEqual(result.trend, "stable")

    def test_one_bad_session_does_not_tank_an_otherwise_healthy_vehicle(self) -> None:
        # Addresses Hanlin's Point 2 concern in spirit: a single rough
        # session (e.g. one bad pothole day) among many good ones
        # should only move the overall score a little, not collapse it.
        sessions = [
            _session(days_ago=i, score=95.0) for i in range(1, 6)
        ] + [_session(days_ago=0, score=40.0)]  # one bad recent session

        result = aggregate_vehicle_health(sessions)

        # Score should drop some (recent session counts more) but
        # stay well above the single bad session's own score.
        self.assertGreater(result.overall_score, 60.0)
        self.assertLess(result.overall_score, 95.0)

    def test_raises_on_empty_input(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_vehicle_health([])


class GroupSessionsByDayTest(unittest.TestCase):
    def test_multiple_same_day_sessions_are_averaged(self) -> None:
        # Two rides on the same day (morning bad, evening good) should
        # collapse into ONE entry for that day, with the average score.
        sessions = [
            ScoredSession(timestamp=datetime(2026, 8, 1, 8, 0), score=60.0),
            ScoredSession(timestamp=datetime(2026, 8, 1, 18, 0), score=90.0),
        ]

        daily = group_sessions_by_day(sessions)

        self.assertEqual(len(daily), 1)
        self.assertEqual(daily[0].timestamp.date(), datetime(2026, 8, 1).date())
        self.assertEqual(daily[0].score, 75.0)  # average of 60 and 90

    def test_different_days_stay_separate(self) -> None:
        sessions = [
            ScoredSession(timestamp=datetime(2026, 8, 1, 8, 0), score=60.0),
            ScoredSession(timestamp=datetime(2026, 8, 2, 9, 0), score=95.0),
        ]

        daily = group_sessions_by_day(sessions)

        self.assertEqual(len(daily), 2)
        scores_by_date = {s.timestamp.date(): s.score for s in daily}
        self.assertEqual(scores_by_date[datetime(2026, 8, 1).date()], 60.0)
        self.assertEqual(scores_by_date[datetime(2026, 8, 2).date()], 95.0)

    def test_result_is_sorted_oldest_to_newest(self) -> None:
        # Sessions passed in out of order should still come back
        # sorted by day.
        sessions = [
            ScoredSession(timestamp=datetime(2026, 8, 5, 8, 0), score=70.0),
            ScoredSession(timestamp=datetime(2026, 8, 1, 8, 0), score=60.0),
            ScoredSession(timestamp=datetime(2026, 8, 3, 8, 0), score=80.0),
        ]

        daily = group_sessions_by_day(sessions)
        dates = [s.timestamp.date() for s in daily]

        self.assertEqual(dates, sorted(dates))

    def test_daily_output_works_with_aggregate_vehicle_health(self) -> None:
        # The whole point of day-grouping is to feed its output into
        # aggregate_vehicle_health() for a per-day trend instead of a
        # per-ride trend. Confirm that pipeline works end to end.
        sessions = [
            ScoredSession(timestamp=datetime(2026, 8, 1, 8, 0), score=60.0),
            ScoredSession(timestamp=datetime(2026, 8, 1, 18, 0), score=90.0),
            ScoredSession(timestamp=datetime(2026, 8, 2, 9, 0), score=95.0),
        ]

        daily = group_sessions_by_day(sessions)
        result = aggregate_vehicle_health(daily)

        # 2 DAYS used, not 3 raw sessions.
        self.assertEqual(result.sessions_used, 2)
        # Aug 1 averaged to 75, Aug 2 is 95 -- clearly improving.
        self.assertEqual(result.trend, "improving")


if __name__ == "__main__":
    unittest.main()
