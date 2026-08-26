"""
synthetic_labels.py

Generates FAKE, PLACEHOLDER curb/pothole/fall events, for testing the
dashboard's Misuse Score UI before real labelled data exists.

IMPORTANT - read this before using output from this file
-----------------------------------------------------------
Nothing in this file is a real detection algorithm. It does NOT
figure out where a curb hit or pothole actually happened. It just:

  1. Reuses the timestamps of real vibration spikes (so the fake
     events land at moments that at least LOOK plausible, since
     something did happen at those moments), and
  2. Randomly assigns an event type (curb up / curb down / pothole /
     fall) and severity to each one.

Every event this file produces has "is_synthetic": True attached to
it. Any code that reads this output (dashboard, reports, further
analysis) should check that flag and never present these events as
real fleet data.

This exists ONLY because the dashboard mockup shows columns like
"Curb ↑", "Curb ↓", "Pothole", "Falls" that need SOMETHING to
display before real labelled events are available. Once real labels
exist, this file should stop being used.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from load_logs import LogRow
from vehicle_health import _vibration_magnitude, VIBRATION_SPIKE_THRESHOLD


# The kinds of events the dashboard mockup expects to show.
EVENT_TYPES = ["curb_up", "curb_down", "pothole", "fall"]
SEVERITIES = ["low", "medium", "high"]

# "fall" should be much rarer than the others, matching the general
# shape of the example dashboard numbers Hanlin shared (a handful of
# falls vs. dozens of curb events). Weights are relative, not
# percentages -- curb_up is 4x as likely to be picked as fall.
EVENT_TYPE_WEIGHTS = {
    "curb_up": 4,
    "curb_down": 4,
    "pothole": 3,
    "fall": 1,
}


@dataclass
class SyntheticEvent:
    timestamp: object  # datetime, same type as LogRow.timestamp
    device_id: str
    event_type: str
    severity: str
    speed_kph: float | None
    is_synthetic: bool = True  # always True -- see module docstring


def generate_synthetic_events(
    rows: list[LogRow],
    random_seed: int | None = 42,
) -> list[SyntheticEvent]:
    """
    Finds moments where real vibration spikes occurred, and generates
    one fake, randomly-typed event per spike.

    random_seed: fixed by default (42) so running this twice on the
    same file gives the SAME fake events -- makes the dashboard
    demo-able and reproducible, rather than changing every run. Pass
    None if you want different random events each time.
    """
    rng = random.Random(random_seed)

    event_types = list(EVENT_TYPE_WEIGHTS.keys())
    weights = list(EVENT_TYPE_WEIGHTS.values())

    events: list[SyntheticEvent] = []

    for row in rows:
        magnitude = _vibration_magnitude(row)
        if magnitude is None or magnitude < VIBRATION_SPIKE_THRESHOLD:
            continue

        chosen_type = rng.choices(event_types, weights=weights, k=1)[0]
        chosen_severity = rng.choice(SEVERITIES)

        events.append(
            SyntheticEvent(
                timestamp=row.timestamp,
                device_id=row.device_id,
                event_type=chosen_type,
                severity=chosen_severity,
                speed_kph=row.speed_kph,
            )
        )

    return events


# ---------------------------------------------------------------------
# Quick manual check when running this file directly.
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from load_logs import load_vmoto_log, NoVmotoDataError

    if len(sys.argv) != 2:
        print("Usage: python3 synthetic_labels.py <path-to-log-file.txt>")
        raise SystemExit(1)

    try:
        rows = load_vmoto_log(sys.argv[1])
    except NoVmotoDataError as error:
        print(f"Skipped: {error}")
        raise SystemExit(0)

    events = generate_synthetic_events(rows)

    print(f"Generated {len(events)} SYNTHETIC events (not real data)")
    print()

    counts: dict[str, int] = {}
    for event in events:
        counts[event.event_type] = counts.get(event.event_type, 0) + 1

    for event_type, count in counts.items():
        print(f"  {event_type}: {count}")

    print()
    print("First 5 events:")
    for event in events[:5]:
        print(
            f"  {event.timestamp} | {event.event_type} | "
            f"{event.severity} | speed={event.speed_kph}"
        )
