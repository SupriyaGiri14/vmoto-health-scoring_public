"""
test_battery_health.py

Checks that compute_battery_health_score() behaves sensibly:
  - a battery with steady voltage and normal SOC drain scores high
  - a battery with sudden voltage sags scores lower
  - a battery draining unusually fast scores lower
  - rows with no battery data at all are correctly ignored, not
    treated as a problem
  - the SOC quantisation artifact (many rows 0.1s apart, same SOC,
    then a 1% tick) is NOT falsely flagged as a huge drop rate
  - fast drain during riding vs during idle use different thresholds
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from load_logs import LogRow  # noqa: E402
from battery_health import compute_battery_health_score  # noqa: E402


def _row(
    minutes_offset: float,
    voltage: float | None,
    soc: float | None,
    speed_kph: float | None = None,
) -> LogRow:
    """
    Small helper to build a fake LogRow without repeating all the
    unrelated fields every time. Only battery_2 is used here, since
    that's the pack we saw actually active in most real files -- the
    scoring logic treats battery_1/battery_2 the same way either way.
    """
    return LogRow(
        device_id="test-device",
        timestamp=datetime(2026, 1, 1, 12, 0, 0) + timedelta(minutes=minutes_offset),
        schema_tier="new",
        battery_2_voltage_v=voltage,
        battery_2_soc_pct=soc,
        speed_kph=speed_kph,
    )


class BatteryHealthTest(unittest.TestCase):
    def test_steady_battery_scores_high(self) -> None:
        # Voltage barely moves, SOC drains slowly and normally over
        # 10 minutes of riding (0.5%/min, comfortably under the
        # riding threshold of 2.0%/min) -- this should look healthy.
        rows = [
            _row(0, 70.0, 90, speed_kph=25.0),
            _row(2, 69.9, 89, speed_kph=25.0),
            _row(4, 69.8, 88, speed_kph=25.0),
            _row(6, 69.9, 87, speed_kph=25.0),
            _row(8, 69.8, 86, speed_kph=25.0),
            _row(10, 69.7, 85, speed_kph=25.0),
        ]

        result = compute_battery_health_score(rows)

        self.assertEqual(result.voltage_sag_events, 0)
        self.assertEqual(result.score, 100.0)

    def test_sudden_voltage_drop_counts_as_sag_event(self) -> None:
        # Voltage drops sharply (70.0 -> 65.0, a 5V drop) between two
        # readings one minute apart -- this should be flagged.
        rows = [
            _row(0, 70.0, 90),
            _row(1, 65.0, 89),
        ]

        result = compute_battery_health_score(rows)

        self.assertEqual(result.voltage_sag_events, 1)
        self.assertLess(result.score, 100.0)

    def test_small_voltage_change_is_not_a_sag_event(self) -> None:
        # A small, normal dip (0.2V) should NOT be flagged as a sag --
        # only drops at or above the threshold count. SOC is held
        # constant here so this test checks ONLY the voltage-sag
        # logic, without also triggering the separate SOC-drop check.
        rows = [
            _row(0, 70.0, 90),
            _row(1, 69.8, 90),
        ]

        result = compute_battery_health_score(rows)

        self.assertEqual(result.voltage_sag_events, 0)
        self.assertEqual(result.score, 100.0)

    def test_fast_soc_drain_lowers_score_while_riding(self) -> None:
        # SOC falls from 90 to 70 in 2 minutes -- a 10%/minute drop
        # rate, far above the riding threshold (2.0%/min). Speed is
        # set to show this happened WHILE riding.
        rows = [
            _row(0, 70.0, 90, speed_kph=30.0),
            _row(2, 70.0, 70, speed_kph=30.0),
        ]

        result = compute_battery_health_score(rows)

        self.assertGreater(result.fast_soc_drop_minutes, 0)
        self.assertLess(result.score, 100.0)

    def test_moderate_soc_drain_is_fine_while_riding_but_flagged_while_idle(self) -> None:
        # Same drop rate (~1.67%/minute over 3 minutes) in both cases
        # -- this is BELOW the riding threshold (2.0%/min) but ABOVE
        # the idle threshold (1.2%/min). So it should only be flagged
        # in the idle case, demonstrating the two thresholds actually
        # differ in practice.
        riding_rows = [
            _row(0, 70.0, 90, speed_kph=30.0),
            _row(3, 70.0, 85, speed_kph=30.0),
        ]
        idle_rows = [
            _row(0, 70.0, 90, speed_kph=0.0),
            _row(3, 70.0, 85, speed_kph=0.0),
        ]

        riding_result = compute_battery_health_score(riding_rows)
        idle_result = compute_battery_health_score(idle_rows)

        self.assertEqual(riding_result.fast_soc_drop_minutes, 0.0)
        self.assertGreater(idle_result.fast_soc_drop_minutes, 0.0)

    def test_soc_quantisation_artifact_is_not_falsely_flagged(self) -> None:
        # This reproduces the real bug found in production data: SOC
        # stays flat for several minutes (many rows 0.1 seconds apart,
        # same value), then ticks down by 1% at some row. The OLD
        # (buggy) row-to-row comparison treated that as an instant
        # 1% drop in 0.1 seconds -- a nonsense ~600%/minute rate. The
        # windowed approach should NOT flag this.
        rows = []
        t = 0.0
        # 4 minutes flat at 90%, logged every 0.1 seconds worth of
        # simulated rows (using a coarser step here for test speed,
        # the principle is the same: MANY rows, same SOC value).
        for i in range(50):
            rows.append(_row(t, 70.0, 90.0, speed_kph=20.0))
            t += 0.1 / 60  # 0.1 seconds, expressed in minutes
        # Jump forward to represent several flat minutes passing...
        t = 4.0
        # ...then SOC ticks down by 1%, with the same tiny 0.1s gap
        # from the row right before it.
        rows.append(_row(t, 70.0, 89.0, speed_kph=20.0))
        rows.append(_row(t + (0.1 / 60), 70.0, 89.0, speed_kph=20.0))

        result = compute_battery_health_score(rows)

        # A genuine 1% drop over ~4 minutes is a perfectly normal
        # rate (0.25%/min) -- well under even the idle threshold, and
        # nowhere near the ~600%/minute the old bug would have
        # calculated from the raw 0.1s gap.
        self.assertEqual(result.fast_soc_drop_minutes, 0.0)
        self.assertEqual(result.score, 100.0)

    def test_rows_with_no_battery_data_are_ignored(self) -> None:
        # Rows where both battery_1 and battery_2 are missing (None)
        # should not be counted as "usable" and should not affect the
        # score -- this mirrors the real files, where most rows have
        # no battery reading at all.
        rows = [
            _row(0, 70.0, 90),
            LogRow(
                device_id="test-device",
                timestamp=datetime(2026, 1, 1, 12, 1, 0),
                schema_tier="new",
                # battery_1_* and battery_2_* left as default None
            ),
            _row(2, 69.9, 89),
        ]

        result = compute_battery_health_score(rows)

        self.assertEqual(result.rows_used, 2)
        self.assertEqual(result.rows_missing_battery_data, 1)


if __name__ == "__main__":
    unittest.main()
