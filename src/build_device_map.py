"""
build_device_map.py

Builds data/device_vehicle_map.csv from the list below.

Why this exists
-----------------
Manually editing the CSV directly (adding a line via a text editor,
or via terminal commands with quotes) turned out to be error-prone --
edits kept silently failing to save. This script avoids that
entirely: the mapping lives here, in plain Python, and running this
script regenerates the CSV file fresh every time.

How to add a new device (e.g. when a moped gets a new logger)
-----------------------------------------------------------------
1. Add ONE new line to the DEVICES list below, following the same
   pattern as the existing entries.
2. Run this script:  python src/build_device_map.py
3. That's it -- data/device_vehicle_map.csv is now up to date.

No manual CSV editing, no terminal quoting issues.
"""

import csv
from pathlib import Path

# Each entry: (device_id, vehicle_id, fleet, company, vehicle_type)
#
# Some vehicles have MULTIPLE device entries -- this means that
# physical moped has had more than one logger installed on it over
# time (an older one, then a newer one). Both entries should point
# to the SAME vehicle_id, so the moped's full history stays connected
# even across a logger swap.
DEVICES = [
    ("012345",   "Example_V", "Examp",  "Public/Gu", "Example_Ve"),
    ("1086240",  "unknown",   "unknown","development","vmoto_cpx"),
    ("1086256",  "unknown",   "unknown","development","vmoto_vs1"),
    ("1086344",  "116IAG",    "Ring",   "Flink",      "vmoto_vs1"),
    ("1086352",  "117IAG",    "Ring",   "Flink",      "vmoto_cpx"),
    ("1086564",  "118IAG",    "Ring",   "Flink",      "vmoto_cpx"),
    ("1086572",  "122IAG",    "Markst", "Flink",      "vmoto_vs1"),
    ("1086692",  "119IAG",    "Markst", "Flink",      "vmoto_vs1"),
    ("16040936", "122IAG",    "Markst", "Flink",      "vmoto_cpx"),  # 122IAG's newest logger
    ("16113764", "118IAG",    "Ring",   "Flink",      "vmoto_vs1"),
    ("16113772", "122IAG",    "Markst", "Flink",      "vmoto_cpx"),
    ("16116688", "119IAG",    "Markst", "Flink",      "vmoto_vs1"),
    ("16116752", "117IAG",    "Ring",   "Flink",      "vmoto_cpx"),
    ("16116760", "116IAG",    "Ring",   "Flink",      "vmoto_vs1"),
]

OUTPUT_PATH = Path("data/device_vehicle_map.csv")


def build():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["device_id", "vehicle_id", "fleet", "company", "vehicle_type"])
        writer.writerows(DEVICES)

    print(f"Wrote {len(DEVICES)} device entries to {OUTPUT_PATH}")


if __name__ == "__main__":
    build()
