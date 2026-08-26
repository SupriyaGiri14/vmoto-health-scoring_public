"""
vibration_pattern.py

Answers a question the vehicle_health.py score alone can't answer:
"is this vehicle's vibration elevated because of ONE rough ride (a
pothole day, a bad road), or because something on the vehicle
itself is actually getting worse?"

This directly addresses Hanlin's Vehicle Health concern #2: vibration
spikes are context-dependent (a pothole and a worn suspension can
look similar in a single ride), so we need to look at PATTERNS
across multiple rides for the SAME vehicle, not just one ride's
number in isolation.

How it works
------------
This compares a vehicle's vibration rate ONLY to its own history --
never to another vehicle. That matters because we separately found
that raw vibration numbers aren't reliably comparable ACROSS
different vehicles/logger generations (see vehicle_health.py). But
comparing a vehicle's own rides to its own past rides sidesteps that
problem entirely -- the comparison stays self-relative.

Classification logic (simple, explainable rules -- same style as the
rest of this project):

  - Compute a BASELINE rate: the median spike rate across all of this
    vehicle's sessions.
  - A session is "elevated" if its rate is notably above that
    baseline (more than ELEVATION_MULTIPLIER times higher).
  - Look at the most recent few sessions:
      * If ONLY the latest session is elevated, and the ones before
        it were normal -> "isolated_spike" (probably a rough road
        that day, not a vehicle problem)
      * If SEVERAL of the most recent sessions are ALL elevated ->
        "persistent_elevated" (a real, ongoing pattern worth
        investigating)
      * If the elevated sessions are trending upward over time ->
        "worsening" (looks like something degrading)
      * Otherwise -> "normal"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from statistics import median


@dataclass
class VibrationSession:
    """One session's vibration spike rate, with when it happened."""
    timestamp: datetime
    spike_rate_per_1000_rows: float


@dataclass
class VibrationPatternResult:
    classification: str  # "normal", "isolated_spike", "persistent_elevated", "worsening"
    baseline_rate: float
    most_recent_rate: float
    sessions_used: int
    explanation: str


# A session counts as "elevated" if its rate is at least this many
# times the vehicle's own baseline (median) rate. Starting estimate,
# same caveat as other thresholds in this project -- not yet
# calibrated against real maintenance outcomes.
ELEVATION_MULTIPLIER = 1.5

# How many of the most recent sessions to look at when deciding
# between "isolated_spike" and "persistent_elevated".
RECENT_WINDOW_SIZE = 3


def classify_vibration_pattern(
    sessions: list[VibrationSession],
) -> VibrationPatternResult:
    """
    Takes a vehicle's vibration spike rate across MULTIPLE sessions
    (same vehicle, ordered by time) and classifies whether any
    elevated vibration looks like a one-off event or a real pattern.
    """
    if not sessions:
        raise ValueError("Need at least one session to classify.")

    ordered = sorted(sessions, key=lambda s: s.timestamp)
    rates = [s.spike_rate_per_1000_rows for s in ordered]

    most_recent_rate = rates[-1]

    # Not enough history yet to say anything meaningful about a
    # pattern -- just report what we have.
    if len(ordered) < 2:
        return VibrationPatternResult(
            classification="normal",
            baseline_rate=rates[0],
            most_recent_rate=most_recent_rate,
            sessions_used=len(ordered),
            explanation="Only one session available -- not enough history to detect a pattern yet.",
        )

    # Split into "recent window" (what we're checking) and "earlier"
    # (what we're comparing it against). The baseline MUST come only
    # from the earlier sessions -- computing it from ALL sessions
    # (including the recent, possibly-elevated ones) would water down
    # the comparison, since the very spike we're trying to detect
    # would inflate the number we're comparing it to.
    if len(rates) > RECENT_WINDOW_SIZE:
        earlier = rates[: len(rates) - RECENT_WINDOW_SIZE]
        recent = rates[-RECENT_WINDOW_SIZE:]
    else:
        # Not enough sessions to cleanly separate "earlier" from
        # "recent" -- fall back to treating everything except the
        # very last session as the baseline.
        earlier = rates[:-1]
        recent = rates[-1:]

    baseline_rate = median(earlier)
    elevation_cutoff = baseline_rate * ELEVATION_MULTIPLIER

    recent_is_elevated = [rate > elevation_cutoff for rate in recent]
    earlier_was_normal = all(rate <= elevation_cutoff for rate in earlier)

    most_recent_is_elevated = recent_is_elevated[-1]

    if not most_recent_is_elevated:
        return VibrationPatternResult(
            classification="normal",
            baseline_rate=round(baseline_rate, 2),
            most_recent_rate=round(most_recent_rate, 2),
            sessions_used=len(ordered),
            explanation="Most recent session is within normal range for this vehicle.",
        )

    all_recent_elevated = all(recent_is_elevated)

    if all_recent_elevated and len(recent) >= 2:
        # Check if it's climbing over that window, not just flat-high.
        is_climbing = all(
            recent[i] < recent[i + 1] for i in range(len(recent) - 1)
        )
        if is_climbing:
            return VibrationPatternResult(
                classification="worsening",
                baseline_rate=round(baseline_rate, 2),
                most_recent_rate=round(most_recent_rate, 2),
                sessions_used=len(ordered),
                explanation=(
                    f"Vibration has been elevated AND increasing across the "
                    f"last {len(recent)} sessions -- looks like a developing issue, "
                    f"not just a rough road on one day."
                ),
            )
        return VibrationPatternResult(
            classification="persistent_elevated",
            baseline_rate=round(baseline_rate, 2),
            most_recent_rate=round(most_recent_rate, 2),
            sessions_used=len(ordered),
            explanation=(
                f"Vibration has been consistently elevated across the last "
                f"{len(recent)} sessions -- worth investigating as a possible "
                f"vehicle issue, not just a one-off rough ride."
            ),
        )

    if most_recent_is_elevated and earlier_was_normal:
        return VibrationPatternResult(
            classification="isolated_spike",
            baseline_rate=round(baseline_rate, 2),
            most_recent_rate=round(most_recent_rate, 2),
            sessions_used=len(ordered),
            explanation=(
                "Only the most recent session is elevated; earlier sessions "
                "were normal -- likely a rough road or one-off event, not a "
                "developing vehicle problem."
            ),
        )

    return VibrationPatternResult(
        classification="normal",
        baseline_rate=round(baseline_rate, 2),
        most_recent_rate=round(most_recent_rate, 2),
        sessions_used=len(ordered),
        explanation="No clear persistent or worsening pattern detected.",
    )
