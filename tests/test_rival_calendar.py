"""Guards for the rival harvest calendar (#16): the rival's board is
public, so their scheduled payouts are a dated supply curve, and the
curve must price into the forecast's price path (#110's rival half).

The contract (owner, 2026-09-27): events sit on the FIRST day a
production can be collected and carry the FLOOR of units — a guarantee,
not a guess. `units_hint` is the fertilised expectation read off their
public tile.

Run:  .venv/bin/python -m tests.test_rival_calendar   (also under pytest)
"""

from __future__ import annotations

import numpy as np

from agent.belief.rival_calendar import (harvest_events, supply_curve,
                                         RivalHarvestEvent)
from agent.obs import decode_world
from agent.tile_dp.tile_state import TileState, TileZeroCode
from agent.world.model import Crop, TileKind


def _plant_obs(cells: dict[tuple[int, int], dict], day: int) -> dict:
    """An observation whose rival farm carries the given engine-shape tiles."""
    rival_tiles = [["LOCKED" for _ in range(10)] for _ in range(10)]
    for (x, y), tile in cells.items():
        rival_tiles[y][x] = dict(tile)
    return {
        "day": day, "hour": 0, "step": day * 24, "player": 0,
        "farms": [
            {"tiles": [[None] * 10 for _ in range(10)], "money": 3000,
             "farmer": [4, 4], "hands": [], "hires_today": 0},
            {"tiles": rival_tiles, "money": 3000, "farmer": [5, 4],
             "hands": [], "hires_today": 0},
        ],
        "market": {"inventory": {}, "prices": {}},
        "private": {"shed": {}, "inventories": [{}]},
        "town": {},
    }


def _melon(planted: int, yu: int, fert: int = -1) -> dict:
    return {"kind": "PLANT", "crop": "MELON", "planted_day": planted,
            "yield_units": yu, "watered_today": False,
            "fertilized_until_day": fert, "consecutive_unwatered": 0}


def _straw(planted: int, yu: int, fert: int = -1) -> dict:
    return {"kind": "PLANT", "crop": "STRAWBERRY", "planted_day": planted,
            "yield_units": yu, "watered_today": False,
            "fertilized_until_day": fert, "consecutive_unwatered": 0}


def _wheat(planted: int, yu: int, fert: int = -1) -> dict:
    return {"kind": "PLANT", "crop": "WHEAT", "planted_day": planted,
            "yield_units": yu, "watered_today": False,
            "fertilized_until_day": fert, "consecutive_unwatered": 0}


def _events(cells, day) -> list[RivalHarvestEvent]:
    return harvest_events(_plant_obs(cells, day), horizon=40)


def test_one_shot_floor_is_what_the_tile_carries_on_the_first_collectable_day() -> None:
    """MELON planted day 3, read day 10 (real age 7 > window start 6):
    everything it carries is collectable TODAY — the pre-fix module dated
    this +12 days late (the `st.age - origin` sign bug)."""
    evs = _events({(2, 2): _melon(3, 4)}, 10)
    assert [(e.day, e.units) for e in evs] == [(10, 4)], evs


def test_one_shot_future_tile_schedules_the_window_start_with_a_zero_floor() -> None:
    """MELON planted day 8, read day 10 (real age 2): the earliest payout is
    the window's start, day 14. Nothing is GUARANTEED (an unwatered melon
    yields 0) so the floor is 0; the hint carries the unwatered accrual."""
    evs = _events({(2, 2): _melon(8, 0)}, 10)
    assert [(e.day, e.units, e.units_hint) for e in evs] == [(14, 0, 1)], evs


def test_a_mature_one_shot_tile_is_collectable_today() -> None:
    evs = _events({(2, 2): _melon(-2, 6)}, 10)
    assert [(e.day, e.units) for e in evs] == [(10, 6)], evs


def test_ongoing_schedule_fires_every_remaining_interval_step() -> None:
    """STRAWBERRY planted day 0, read day 11 (real age 11): production ages
    10/12/14/16; age 10 is past and its units are in the tile, so the first
    remaining step (age 12 = day 13) carries the floor of 2, then 1 each."""
    evs = _events({(2, 2): _straw(0, 2)}, 11)
    assert [(e.day, e.units) for e in evs] == [(12, 2), (14, 1), (16, 1)], evs


def test_the_weed_deadline_is_never_an_event() -> None:
    """crop_last_day is the tile's death, not a payout (the rule the
    measurement rejected): a melon's schedule has exactly one event."""
    evs = _events({(2, 2): _melon(3, 4)}, 10)
    assert len(evs) == 1, evs


def test_the_fertilised_hint_reads_their_public_tile() -> None:
    """Their fertiliser is a public fact: while `fertilized_until_day`
    covers the event's day, a scheduled accrual is worth 2, not 1."""
    evs = _events({(2, 2): _straw(0, 2, fert=13)}, 11)
    assert evs[0].units == 2 and evs[0].units_hint == 4, evs   # 2 held + fert step
    assert evs[1].units == 1 and evs[1].units_hint == 1, evs   # day 14 uncovered
    evs = _events({(2, 2): _wheat(0, 0, fert=2)}, 1)
    assert evs[0].units == 0 and evs[0].units_hint == 2, evs


def test_the_curve_sums_the_floor_and_the_expected_view() -> None:
    obs = _plant_obs({(2, 2): _straw(0, 2, fert=13)}, 11)
    floor = supply_curve(obs, ("STRAWBERRY",), 40)
    expect = supply_curve(obs, ("STRAWBERRY",), 40, expected=True)
    assert floor[12, 0] == 2.0 and floor[14, 0] == 1.0, floor[:20, 0]
    assert expect[12, 0] == 4.0 and expect[14, 0] == 1.0, expect[:20, 0]


def test_an_already_past_payout_never_reappears() -> None:
    """A one-shot tile past its whole life decodes as a WEED and schedules
    nothing; a young tile's only event is its window start."""
    from agent.world.tile import TileHourZero
    dead = {"kind": "PLANT", "crop": "MELON", "planted_day": -20,
            "yield_units": 6, "watered_today": False,
            "fertilized_until_day": -1, "consecutive_unwatered": 0}
    assert TileHourZero.decode(dead, 10).kind is TileKind.WEED
    evs = _events({(1, 1): dead}, 10)
    assert evs == [], evs


def test_the_dated_curve_moves_the_path_on_the_day_it_lands() -> None:
    """The wire: a DATED curve, not a flat residual.

    `forecast`'s walk adds the rival's supply per turn. A flat residual spreads
    one number over every day, so it moves the whole path; a calendar entry is
    dated, so the path BEFORE its day must be untouched and the path from it on
    must be lower. The curve is built by hand here, not through
    `harvest_events`: this guard is about the FORECAST's consumption of a dated
    curve. CARROT, because its path is the one that responds to supply (MELON's
    quote is flat at 250-280 whatever the supply — a guard written on it cannot
    fail)."""
    from agent.belief.market import forecast, PRODUCTS

    day, horizon = 3, 30
    obs = _plant_obs({}, day)
    col = tuple(PRODUCTS).index("CARROT")
    landing = day + 4
    curve = np.zeros((horizon, len(PRODUCTS)))
    curve[landing, col] = 6.0

    cfg = {"farmHandCostMult": 1}
    plain = forecast(obs, days=horizon, config=cfg)

    def first_differing(landing_day: int) -> int:
        """The first day whose CARROT price moves when the supply lands there."""
        c = np.zeros((horizon, len(PRODUCTS)))
        c[landing_day, col] = 6.0
        moved = forecast(obs, days=horizon, rival_supply=c, config=cfg)
        return next(i for i in range(day, horizon)
                    if float(moved.price_of("CARROT", i))
                    != float(plain.price_of("CARROT", i)))

    dated = forecast(obs, days=horizon, rival_supply=curve, config=cfg)
    first = first_differing(landing)
    # The path is untouched until the supply has accumulated into the quote
    # table's next band, so the day it bites is a fixed lag AFTER the landing
    # day. The lag is absolute on purpose: a curve shifted one day late moves
    # this day by one, and a relation between two landing days would not see it
    # (the shift applies to both). Measured on this fixture: 6 units of CARROT
    # on day 7 first move the path on day 11.
    assert first - landing == 4, (
        f"a supply on day {landing} bit on day {first}: expected the measured "
        f"4-day accumulation lag")
    assert [float(dated.price_of("CARROT", d)) for d in range(day, first)] == \
        [float(plain.price_of("CARROT", d)) for d in range(day, first)]
    after_plain = [float(plain.price_of("CARROT", d)) for d in range(first, horizon)]
    after_dated = [float(dated.price_of("CARROT", d)) for d in range(first, horizon)]
    assert any(b > a for a, b in zip(after_dated, after_plain)), \
        "the scheduled supply must lower the path from where it bites"

    # The same total, spread flat, moves the days BEFORE the payout as well —
    # which is the whole reason the dated form exists.
    flat = forecast(obs, days=horizon, residual={"CARROT": float(curve.sum())},
                    config=cfg)
    assert [float(flat.price_of("CARROT", d)) for d in range(day, first)] != \
        [float(plain.price_of("CARROT", d)) for d in range(day, first)]
