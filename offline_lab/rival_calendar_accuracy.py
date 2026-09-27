"""Measure rival_calendar's harvest-day rule against replay ground truth (#16).

THE MEASUREMENT

For each recorded HARVEST op of a seat, the tile standing there in the step
BEFORE the op is the crop harvested; its `planted_day` and crop name are the
replay's own fields, so truth is not modelled at all. The predicted payout day
is what `rival_calendar.harvest_events` derives from the same facts: the tile's
packed age is `day - planted_day - crop_age_origin(crop)`, and the rule says
`event = day - (age - origin) + crop_last_day(crop)`, which collapses to
`planted_day + crop_last_day(crop)`.

Per crop, two summary rows:
  - rule as written (`crop_last_day`): what the shipped code predicts;
  - the engine's own production law (what the observed ages say the day of
    collectable yield actually is), so the rule can be typed from evidence.

Ground truth = the day of the seat's own recorded HARVEST on that tile
(hands and farmer, both sources of harvest ops). The pairing uses the tile
state in the step BEFORE the harvest, so a replanted tile pairs each harvest
with the crop standing at that moment (the naive per-tile join pairs the
FIRST harvest with the SECOND planting and measures nothing — seen live:
melon mean delta +14 from exactly that error).

Run:  .venv/bin/python offline_lab/rival_calendar_accuracy.py --date 2026-09-10
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import duckdb

REPO = Path.home() / "Chista/kaggriculture-episodes-analyses"
RULES = {"WHEAT": dict(ongoing=False, first=2, maxy=4, interval=0, max_yield=6),
         "CARROT": dict(ongoing=False, first=2, maxy=3, interval=0, max_yield=4),
         "TOMATO": dict(ongoing=True, first=8, maxy=8, interval=1, max_yield=4),
         "STRAWBERRY": dict(ongoing=True, first=10, maxy=10, interval=2, max_yield=4),
         "MELON": dict(ongoing=False, first=10, maxy=12, interval=0, max_yield=6)}


def last_day(spec: dict) -> int:
    """`crop_last_day`, world/tile.py:84 — the day the tile stops being a PLANT."""
    if spec["ongoing"]:
        return spec["maxy"] + (spec["max_yield"] - 1) * max(1, spec["interval"])
    return spec["maxy"]


def first_collectable(spec: dict) -> int:
    """First day yield is collectable: one-shot WATER window start; ongoing
    first scheduled production. Both from the engine's own tables."""
    if spec["ongoing"]:
        return spec["maxy"]
    return (spec["maxy"] + 1) // 2


def measure(date: str, seat: int = 1) -> None:
    t = REPO / f"data/replays_parquet/{date}"
    con = duckdb.connect()
    rows = con.execute(f"""
        WITH harvests AS (
            SELECT h.episode_id, h.x, h.y, c.day AS hday, min(c.step) AS hstep
            FROM read_parquet('{t}/hands_steps.parquet') h
            JOIN read_parquet('{t}/city_steps.parquet') c USING (episode_id, step)
            WHERE h.player = {seat} AND h.op = 'HARVEST'
            GROUP BY h.episode_id, h.x, h.y, c.day
            UNION ALL
            SELECT f.episode_id, f.farmer_x, f.farmer_y, c.day, min(c.step)
            FROM read_parquet('{t}/farm_steps.parquet') f
            JOIN read_parquet('{t}/city_steps.parquet') c USING (episode_id, step)
            WHERE f.player = {seat} AND f.op = 'HARVEST'
            GROUP BY f.episode_id, f.farmer_x, f.farmer_y, c.day
        )
        SELECT d.crop, d.planted_day, hs.hday
        FROM harvests hs
        JOIN LATERAL (
            SELECT crop, planted_day FROM read_parquet('{t}/tiles_delta.parquet') d
            WHERE d.episode_id = hs.episode_id AND d.x = hs.x AND d.y = hs.y
              AND d.player = {seat} AND d.kind = 'PLANT' AND d.step < hs.hstep
            ORDER BY d.step DESC LIMIT 1
        ) d ON true
        WHERE d.crop IS NOT NULL AND d.planted_day IS NOT NULL
    """).fetchall()

    # Three predictors, one truth (the harvest day). The floor predicts the
    # EARLIEST collectable day — a predictor is correct on an event when its
    # day <= truth, and it never over-promises; the within-k scoring is its
    # tightness. The old rule (planted+crop_last_day) and the first-fix rule
    # (one-shot: window end) are scored the same way for the comparison.
    def floor_day(spec: dict) -> int:
        return (spec["maxy"] if spec["ongoing"] else first_collectable(spec))

    old_delta: Counter[int] = Counter()
    floor_gap: Counter[int] = Counter()      # truth - predicted (>=0 means safe)
    law_age: dict[str, Counter[int]] = defaultdict(Counter)
    for crop, planted, hday in rows:
        spec = RULES.get(str(crop))
        if spec is None:
            continue
        age = int(hday) - int(planted)
        law_age[str(crop)][age] += 1
        old_delta[age - last_day(spec)] += 1
        floor_gap[age - floor_day(spec)] += 1
    n = sum(old_delta.values())
    print(f"store {date}, seat {seat}: {n} recorded rival harvest events")
    if not n:
        print("nothing measured")
        return
    w = sum(v for k, v in old_delta.items() if abs(k) <= 1)
    print("[OLD rule: planted+crop_last_day]")
    print(f"  exact {old_delta[0]/n:6.1%}   within ±1 {w/n:6.1%}   (acceptance >= 90%)")
    safe = sum(v for k, v in floor_gap.items() if k >= 0)
    tight = sum(v for k, v in floor_gap.items() if 0 <= k <= 3)
    print("[NEW floor: first collectable day (window start / max_yield_day)]")
    print(f"  never-late {safe/n:6.1%}   within +3 days of truth {tight/n:6.1%}")
    print(f"  mean gap (truth - floor) "
          f"{sum(k*v for k, v in floor_gap.items())/n:+.2f} days")
    for crop, spec in RULES.items():
        ages = law_age[crop]
        ctotal = sum(ages.values())
        if not ctotal:
            continue
        top = sorted(ages.items(), key=lambda kv: -kv[1])[:5]
        cov = sum(nn for _, nn in top) / ctotal
        print(f"  {crop:11s} n={ctotal:6d}  top harvest ages {top} cover {cov:.0%}"
              f"   [floor day {floor_day(spec)}, old rule {last_day(spec)}]")


if __name__ == "__main__":
    date = sys.argv[sys.argv.index("--date") + 1] if "--date" in sys.argv else "2026-09-10"
    measure(date)
