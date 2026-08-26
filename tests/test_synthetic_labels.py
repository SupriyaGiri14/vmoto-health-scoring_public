"""
test_synthetic_labels.py

Checks that generate_synthetic_events():
  - only creates events at real vibration spikes, not everywhere
  - always marks every event as is_synthetic=True
  - is reproducible when given the same random seed
  - produces different results with different seeds (proves it's
    actually using randomness, not hardcoded)
"""

from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from load_logs import LogRow  # noqa: E402
from synthetic_labels import generate_synthetic_events, EVENT_TYPES  # noqa: E402


def _row(seconds_offset: float, x: float, y: float, z: float) -> LogRow:
    return LogRow(
        device_id="test-device",
        timestamp=datetime(2026, 1, 1, 12, 0, 0) + timedelta(seconds=seconds_offset),
        schema_tier="new",
        imu_acc_x=x,
        imu_acc_y=y,
        imu_acc_z=z,
    )


CALM = (1000.0, 1000.0, 1000.0)     # magnitude ~1732, below spike threshold
SPIKE = (3000.0, 3000.0, 3000.0)    # magnitude ~5196, above spike threshold


class SyntheticLabelsTest(unittest.TestCase):
    def test_no_events_generated_when_no_spikes(self) -> None:
        rows = [_row(i, *CALM) for i in range(20)]

        events = generate_synthetic_events(rows)

        self.assertEqual(len(events), 0)

    def test_one_event_generated_per_spike(self) -> None:
        rows = [_row(i, *CALM) for i in range(8)]
        rows += [_row(8, *SPIKE), _row(9, *SPIKE), _row(10, *SPIKE)]

        events = generate_synthetic_events(rows)

        self.assertEqual(len(events), 3)

    def test_every_event_is_marked_synthetic(self) -> None:
        rows = [_row(0, *SPIKE)]

        events = generate_synthetic_events(rows)

        self.assertTrue(all(event.is_synthetic for event in events))

    def test_event_type_is_always_a_known_type(self) -> None:
        rows = [_row(i, *SPIKE) for i in range(10)]

        events = generate_synthetic_events(rows)

        for event in events:
            self.assertIn(event.event_type, EVENT_TYPES)

    def test_same_seed_gives_identical_results(self) -> None:
        rows = [_row(i, *SPIKE) for i in range(10)]

        events_a = generate_synthetic_events(rows, random_seed=42)
        events_b = generate_synthetic_events(rows, random_seed=42)

        types_a = [event.event_type for event in events_a]
        types_b = [event.event_type for event in events_b]
        self.assertEqual(types_a, types_b)

    def test_different_seed_can_give_different_results(self) -> None:
        rows = [_row(i, *SPIKE) for i in range(20)]

        events_a = generate_synthetic_events(rows, random_seed=1)
        events_b = generate_synthetic_events(rows, random_seed=2)

        types_a = [event.event_type for event in events_a]
        types_b = [event.event_type for event in events_b]
        # With 20 events, two different seeds should not produce the
        # exact same sequence -- if they do, something's wrong with
        # how randomness is being used.
        self.assertNotEqual(types_a, types_b)


if __name__ == "__main__":
    unittest.main()
