"""
active_battery.py

Determines which battery pack (1 or 2) is GENUINELY the active one at
each point in a ride -- shared logic used by battery_health.py and
energy_efficiency.py.

Why this exists
-----------------
battery_health.py originally used a simple rule: "prefer battery_1 if
it has any value, else battery_2". That works for most rows, since
usually only one pack reports at a time.

But checking a real file (1086344_2026-07-27.txt) during a
battery-swap event showed BOTH packs reporting simultaneously for an
extended period -- and during that period, battery_1's value stayed
FROZEN (stale, not actually changing) while battery_2 was the one
genuinely draining. The simple "prefer battery_1" rule would have
kept reporting the frozen, non-active value.

The fix: instead of a fixed preference, this tracks which pack's
value is ACTUALLY CHANGING between consecutive readings, and treats
that one as active. A pack that stops changing (even if its field is
still technically populated) is treated as no longer active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from load_logs import LogRow


@dataclass
class ResolvedBatteryReading:
    row: LogRow
    active_pack: Optional[int]  # 1, 2, or None if no data this row
    soc: Optional[float]
    voltage: Optional[float]


def resolve_active_battery_series(rows: list[LogRow]) -> list[ResolvedBatteryReading]:
    """
    Takes a chronologically-ordered list of LogRow and returns one
    ResolvedBatteryReading per row, using sequence-aware logic to
    pick whichever pack is genuinely active at that point -- not just
    whichever field happens to be populated.
    """
    results: list[ResolvedBatteryReading] = []

    active_pack: Optional[int] = None
    last_known_soc: dict[int, Optional[float]] = {1: None, 2: None}

    for row in rows:
        pack_soc = {1: row.battery_1_soc_pct, 2: row.battery_2_soc_pct}
        pack_voltage = {1: row.battery_1_voltage_v, 2: row.battery_2_voltage_v}

        available_packs = [p for p in (1, 2) if pack_soc[p] is not None]

        if not available_packs:
            results.append(
                ResolvedBatteryReading(row=row, active_pack=None, soc=None, voltage=None)
            )
            continue

        if len(available_packs) == 1:
            # Only one pack reporting -- it's active by definition,
            # same as the original simple logic for this common case.
            active_pack = available_packs[0]
        else:
            # BOTH packs reporting this row -- the rarer, trickier
            # case. Decide based on which pack's value has actually
            # CHANGED since we last saw it.
            changed_packs = [
                p for p in available_packs
                if last_known_soc[p] is not None and pack_soc[p] != last_known_soc[p]
            ]

            if len(changed_packs) == 1:
                # Exactly one pack changed -- that's clearly the
                # active one.
                active_pack = changed_packs[0]
            elif active_pack in available_packs:
                # Ambiguous (both or neither changed) -- keep
                # whichever pack we already believed was active,
                # rather than flip-flopping on ambiguous data.
                pass
            else:
                # No prior active pack to fall back on -- default to
                # the first available (matches the original simple
                # behaviour for this edge case).
                active_pack = available_packs[0]

        # Update what we last saw for BOTH packs, regardless of which
        # one is "active" -- needed to detect future changes.
        for p in (1, 2):
            if pack_soc[p] is not None:
                last_known_soc[p] = pack_soc[p]

        results.append(
            ResolvedBatteryReading(
                row=row,
                active_pack=active_pack,
                soc=pack_soc.get(active_pack),
                voltage=pack_voltage.get(active_pack),
            )
        )

    return results
