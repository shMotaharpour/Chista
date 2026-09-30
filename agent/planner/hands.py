"""How many hands a day's work needs, from the owner's own regression.

The owner's `predict_min_hands` — a q=0.10 linear quantile fit over 53,520
day-rows of >60% win-rate agents (September 2026 replays) — is the ESTIMATE of
the day's helper count. It is deliberately free of any calendar feature: it reads
only the day's real workload and the unlocked geometry, which is what keeps it
useful on day 0 and on day 29 alike, and what keeps it from chasing the
Fibonacci wage.

It is an estimate, not a floor: the day layer still prices the day around it and
wsr still answers with the range it can carry, because a regression on other
agents' days is not a proof about OUR day.

`features` builds the arguments from the day's own chains the way the regression
counts them: the tile's cell decides the quadrant (the shed sits at (4,4)-(5,5),
so x >= 5 with y >= 5 is SE, x >= 5 with y < 5 is NE, x < 5 with y < 5 is NW,
x < 5 with y >= 5 is SW), and the ops decide the class of work.
"""
from __future__ import annotations

import math
from collections import Counter

#: The engine's hatch per quadrant: the tile a DROP from that quadrant lands on.
HATCH = {"NW": (4, 4), "NE": (5, 4), "SW": (4, 5), "SE": (5, 5)}

#: The ops the regression's `other_ops_cnt` counts: the field work.
FIELD_OPS = ("WATER", "PLANT", "DIG", "CARE", "HARVEST")

#: The owner's cap on the day's hands (2026-09-27): the estimate is used as the
#: day's hand count, and the day never asks for more than this many helpers.
MAX_HANDS: int = 16


def quadrant_of(cell) -> str:
    """Which 5x5 quadrant a cell is in, in the regression's own labels."""
    x, y = int(cell[0]), int(cell[1])
    if x >= 5:
        return "SE" if y >= 5 else "NE"
    return "SW" if y >= 5 else "NW"


def predict_min_hands(
    n_quads: int,
    drops_se: int = 0,
    drops_sw: int = 0,
    drops_nw: int = 0,
    drops_ne: int = 0,
    target_tiles_se: int = 0,
    target_tiles_ne: int = 0,
    target_tiles_nw: int = 0,
    fertilize_cnt: int = 0,
    feed_cnt: int = 0,
    other_ops_cnt: int = 0,
    collect_fert_cnt: int = 0,
) -> int:
    """Predict minimum helper hands (n_hands) required for a day's farm tasks.

    Linear quantile regression (q=0.10) fitted on 53,520 day-rows of top agents
    (>60% win rate) from September 2026 replays. It intentionally omits calendar
    day index features so the estimate generalizes across any match stage, driven
    purely by real workload and unlocked land geometry while guarding against
    Fibonacci wage inflation. Board geometry: 10x10, 0-based; the central 2x2 shed
    connects to all four quadrants at NW (4,4), NE (5,4), SW (4,5), SE (5,5).
    Returns the integer helper count, excluding the farmer.
    """
    score = (
        0.755
        + 1.604 * n_quads
        + 0.156 * drops_se
        + 0.143 * drops_sw
        + 0.089 * drops_nw
        + 0.084 * drops_ne
        + 0.072 * target_tiles_se
        + 0.034 * target_tiles_ne
        + 0.019 * target_tiles_nw
        + 0.039 * fertilize_cnt
        + 0.032 * feed_cnt
        + 0.025 * other_ops_cnt
        + 0.013 * collect_fert_cnt
    )
    return max(0, math.ceil(score))


def features(chains, quadrants: int = 1) -> dict:
    """The regression's arguments, counted off the day's own chains.

    `chains` is the planner's own `(cell, ops, entity)` sequence, so the counts are
    the work the day will actually try to do. A DROP is not declared by a chain: a
    chain that harvests banks its good, which is one delivery per working tile, so
    the drops are counted per quadrant from the tiles whose ops carry a harvest or
    a collect (the two ops that put something in the bag).
    """
    tiles = Counter()
    ops_count = Counter()
    for cell, ops, _entity in chains:
        names = [str(o) for o in ops]
        q = quadrant_of(cell)
        tiles[q] += 1
        for name in names:
            ops_count[name] += 1
    deliver = ops_count["HARVEST"] + ops_count["COLLECT_FERTILIZER"]
    drops = {q: (deliver if tiles[q] else 0) for q in ("SE", "SW", "NW", "NE")}
    field = sum(ops_count[op] for op in FIELD_OPS)
    return {
        "n_quads": int(quadrants),
        "drops_se": drops["SE"], "drops_sw": drops["SW"],
        "drops_nw": drops["NW"], "drops_ne": drops["NE"],
        "target_tiles_se": tiles["SE"], "target_tiles_ne": tiles["NE"],
        "target_tiles_nw": tiles["NW"],
        "fertilize_cnt": ops_count["FERTILIZE"],
        "feed_cnt": ops_count["FEED"],
        "other_ops_cnt": field,
        "collect_fert_cnt": ops_count["COLLECT_FERTILIZER"],
    }


def estimate(chains, quadrants: int = 1) -> int:
    """The owner's estimate for this day's chains, capped at `MAX_HANDS`.

    The count is the estimator's; the HOURS each hand starts at are the hourly
    secretary's (`plan` prices the day on the queue's own `hire_hours`, F040).
    """
    return min(MAX_HANDS, predict_min_hands(**features(chains, quadrants)))
