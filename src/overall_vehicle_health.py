"""
overall_vehicle_health.py

Combines Vehicle Health scores from MULTIPLE sessions/files (for the
same vehicle) into ONE overall score -- the number the dashboard
would actually show per vehicle, rather than a score per file.

Why this file exists
---------------------
battery_health.py and vehicle_health.py score ONE session (one file)
at a time, starting fresh at 100 each time. That's correct and
intentional -- it avoids the problem where more usage automatically
drags the score down.

But the dashboard needs ONE number per vehicle, not one per file. So
we need a way to combine several session scores into an overall
picture, and it has to be done carefully:

  - Simply ADDING scores together would reintroduce the "more trips
    = worse score" problem.
  - Simply averaging ALL sessions equally would hide a vehicle that
    has recently started getting worse, because old healthy sessions
    would drag the average back up.
  - Using ONLY the most recent session would be noisy -- one unusually
    rough single ride could make an otherwise healthy vehicle look
    bad.

The approach used here: a RECENCY-WEIGHTED AVERAGE. Recent sessions
count more than older ones. This also helps distinguish "one bad
session" from "a real, ongoing problem": a single rough session gets
mostly averaged out, but if MANY recent sessions are consistently
low, the overall score will genuinely reflect that.

Day grouping
------------
Hanlin asked for the trend to be shown "per day" rather than per
ride. group_sessions_by_day() below groups multiple same-day rides
into ONE score per day (their average) before feeding that into the
same recency-weighting and trend logic. This gives a cleaner, less
noisy trend line -- one point per day instead of a jumpy point per
ride, especially useful for vehicles ridden many times a day.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ScoredSession:
    """One session's score, with when it happened -- the input to
    the aggregation below."""
    timestamp: datetime
    score: float


@dataclass
class OverallVehicleHealth:
    overall_score: float
    sessions_used: int
    most_recent_session_score: float
    trend: str  # "improving", "worsening", or "stable"


# How much more a session counts compared to the one before it, when
# sorted oldest to newest. A value of 1.0 would mean every session
# counts equally (a plain average). Higher values weight recent
# sessions more heavily. This is a starting estimate, same caveat as
# other thresholds in this project -- not yet calibrated against real
# outcomes.
RECENCY_WEIGHT_GROWTH = 1.15


def aggregate_vehicle_health(sessions: list[ScoredSession]) -> OverallVehicleHealth:
    """
    Takes a list of ScoredSession (one per file/ride for the SAME
    vehicle) and returns one combined OverallVehicleHealth.
    """
    if not sessions:
        raise ValueError("Need at least one session to aggregate.")

    # Sort oldest -> newest, so weighting can favour the end of the list.
    ordered = sorted(sessions, key=lambda s: s.timestamp)

    # Give each session a weight that grows the more recent it is.
    # Example with 3 sessions and growth 1.15:
    #   oldest   -> weight 1.15^0 = 1.00
    #   middle   -> weight 1.15^1 = 1.15
    #   newest   -> weight 1.15^2 = 1.32
    weights = [RECENCY_WEIGHT_GROWTH ** i for i in range(len(ordered))]

    weighted_sum = sum(s.score * w for s, w in zip(ordered, weights))
    total_weight = sum(weights)
    overall_score = weighted_sum / total_weight

    # Simple trend check: compare the average of the older half of
    # sessions to the average of the newer half. This is a rough
    # signal, not a precise statistical trend line -- good enough to
    # flag "getting better" vs "getting worse" at a glance.
    midpoint = len(ordered) // 2
    trend = "stable"
    if len(ordered) >= 2:
        older_half = ordered[:midpoint] if midpoint > 0 else ordered[:1]
        newer_half = ordered[midpoint:] if midpoint > 0 else ordered[1:]
        older_avg = sum(s.score for s in older_half) / len(older_half)
        newer_avg = sum(s.score for s in newer_half) / len(newer_half)

        if newer_avg > older_avg + 2:
            trend = "improving"
        elif newer_avg < older_avg - 2:
            trend = "worsening"

    return OverallVehicleHealth(
        overall_score=round(overall_score, 1),
        sessions_used=len(ordered),
        most_recent_session_score=ordered[-1].score,
        trend=trend,
    )


def group_sessions_by_day(sessions: list[ScoredSession]) -> list[ScoredSession]:
    """
    Groups multiple sessions from the SAME calendar day into ONE
    ScoredSession per day, using the average score across that day's
    rides.

    This is meant to be used BEFORE aggregate_vehicle_health(), when
    you want the overall score and trend to reflect "per day" rather
    than "per ride" -- useful for vehicles ridden many times a day,
    where a raw per-ride trend line would be noisy.

    The returned sessions are timestamped at midnight of their day
    (00:00:00), so aggregate_vehicle_health()'s recency-weighting and
    sorting still work correctly on the result.
    """
    scores_by_day: dict = defaultdict(list)

    for session in sessions:
        day = session.timestamp.date()
        scores_by_day[day].append(session.score)

    daily_sessions = []
    for day, scores in sorted(scores_by_day.items()):
        daily_average = sum(scores) / len(scores)
        daily_sessions.append(
            ScoredSession(
                timestamp=datetime(day.year, day.month, day.day),
                score=round(daily_average, 1),
            )
        )

    return daily_sessions


# ---------------------------------------------------------------------
# Quick manual check when running this file directly.
# ---------------------------------------------------------------------
#
# Usage: python3 overall_vehicle_health.py file1.txt file2.txt file3.txt
#
# Pass two or more log files for the SAME vehicle (different sessions/
# dates). This loads each one, scores it with vehicle_health.py, then
# combines all the session scores into one overall result.

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))

    from summary import summarize_file

    if len(sys.argv) < 3:
        print("Usage: python3 overall_vehicle_health.py <file1.txt> <file2.txt> [more files...]")
        print("(Needs at least 2 files, all from the SAME vehicle, to be useful.)")
        raise SystemExit(1)

    file_paths = sys.argv[1:]

    sessions = []
    for path in file_paths:
        result = summarize_file(path)
        if result is None:
            print(f"Skipping {path} -- no vmoto data in this file")
            continue
        sessions.append(ScoredSession(timestamp=result.session_start, score=result.vehicle_health_score))
        print(f"{path}: {result.session_start} -> vehicle health {result.vehicle_health_score}")

    if not sessions:
        print("\nNo usable sessions found -- nothing to aggregate.")
        raise SystemExit(0)

    print()
    print("--- Per-session (per-ride) view ---")
    overall = aggregate_vehicle_health(sessions)
    print(f"Overall vehicle health score: {overall.overall_score} / 100")
    print(f"Sessions used: {overall.sessions_used}")
    print(f"Most recent session score: {overall.most_recent_session_score}")
    print(f"Trend: {overall.trend}")

    print()
    print("--- Per-day view (same-day rides averaged together) ---")
    daily_sessions = group_sessions_by_day(sessions)
    for day_session in daily_sessions:
        print(f"  {day_session.timestamp.date()}: {day_session.score}")

    daily_overall = aggregate_vehicle_health(daily_sessions)
    print(f"Overall vehicle health score: {daily_overall.overall_score} / 100")
    print(f"Days used: {daily_overall.sessions_used}")
    print(f"Trend: {daily_overall.trend}")
