"""Guards for the rival harvest calendar (#16): the rival's board is
public, so their scheduled payouts are a dated supply curve, and the
curve must price into the forecast's price path (#110's rival half).

Run:  .venv/bin/python -m tests.test_rival_calendar   (also under pytest)
"""

from __future__ import annotations

import numpy as np

from agent.belief.rival_calendar import (harvest_events, supply_curve,
                                         RivalHarvestEvent)
from agent.obs import decode_world
from agent.tile_dp.tile_state import TileState, TileZeroCode
from agent.world.model import Crop, TileKind


def _key_for(kind: TileKind, crop, age: int, yield_units: int = 0) -> int:
    st = TileState(kind=kind, crop=crop, animal=None, structure=None,
                   age=age, consec=0, unfed=0, fert_left=0, care_bank=0,
                   yield_units=yield_units)
    return st.pack().code if hasattr(st.pack(), "code") else int(st.pack())


def _obs_with_rival_board(cells: dict[tuple[int, int], int], day: int,
                          goods_order) -> dict:
    """An observation whose rival farm carries the given packed keys."""
    rival_tiles = [[("LOCKED" if True else None) for _ in range(10)]
                   for _ in range(10)]
    # the observation exposes the RIVAL board as strings to us; build it
    # from the packed keys through the decode's own inverse: pack -> the
    # engine's tile dict. The decode reads `farms[1]["tiles"]` raw values,
    # so give it real engine-shape dicts where a tile is planted.
    from agent.world.model import Crop
    for (x, y), key in cells.items():
        st = TileState.unpack(TileZeroCode(key))
        crop = (st.crop.value if hasattr(st.crop, "value") else str(st.crop))
        rival_tiles[y][x] = {"kind": "PLANT", "crop": crop,
                             "age": st.age, "yield_units": st.yield_units,
                             "watered_today": False,
                             "fertilized_until_day": 0,
                             "planted_day": day - st.age - 6}
    for y in range(10):
        for x in range(10):
            if not isinstance(rival_tiles[y][x], dict):
                rival_tiles[y][x] = "LOCKED"
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


def _melon_key(age: int, units: int) -> int:
    st = TileState(kind=TileKind.PLANT, crop=Crop.MELON, animal=None,
                   structure=None, age=age, consec=0, unfed=0, fert_left=0,
                   care_bank=0, yield_units=units)
    return int(st.pack())


def test_a_planted_rival_tile_lands_on_its_payout_day() -> None:
    """MELON at its origin age (6): planted today, payout on day+12
    (crop_last_day counts from planting)."""
    from agent.world.tile import crop_last_day, crop_age_origin
    from agent.world.model import Crop
    day = 3
    units = 6
    key = _melon_key(crop_age_origin(Crop.MELON), units)
    obs = _obs_with_rival_board({(2, 2): key}, day, ("WHEAT", "MELON"))
    events = harvest_events(obs, horizon=30)
    expected_day = day + crop_last_day(Crop.MELON)
    assert any(ev.day == expected_day and ev.good == "MELON"
               and ev.units == units for ev in events), events
    curve = supply_curve(obs, ("MELON",), horizon=30)
    assert curve[expected_day, 0] == units


def test_an_already_past_payout_never_reappears() -> None:
    """A tile whose window passed (age past last day) schedules nothing."""
    key = _melon_key(20, 6)          # past melon's last day (12)
    obs = _obs_with_rival_board({(1, 1): key}, 5, ("WHEAT", "MELON"))
    curve = supply_curve(obs, ("MELON",), horizon=30)
    assert curve.sum() == 0.0, curve


def test_the_curve_sums_across_tiles_and_days() -> None:
    """Two melon tiles on different schedules produce two bumps."""
    day = 0
    obs = _obs_with_rival_board({(1, 1): _melon_key(6, 6),
                                 (5, 5): _melon_key(6, 5)}, 13, ("MELON",))
    curve = supply_curve(obs, ("MELON",), horizon=30)
    assert abs(curve[:, 0].sum() - 11.0) < 1e-9, curve[:, 0]
    assert (curve[:, 0] > 0).sum() >= 1


def _carrot_key(age: int, units: int) -> int:
    st = TileState(kind=TileKind.PLANT, crop=Crop.CARROT, animal=None,
                   structure=None, age=age, consec=0, unfed=0, fert_left=0,
                   care_bank=0, yield_units=units)
    return int(st.pack())


def test_the_dated_curve_moves_the_path_on_the_day_it_lands() -> None:
    """The wire: a DATED curve, not a flat residual.

    `forecast`'s walk adds the rival's supply per turn. A flat residual spreads
    one number over every day, so it moves the whole path; a calendar entry is
    dated, so the path BEFORE its day must be untouched and the path from it on
    must be lower. That difference is what the wire buys, and it is exactly what
    a flat spread fails.

    The curve is built by hand here, not through `harvest_events`: this guard is
    about the FORECAST's consumption of a dated curve, and the calendar's own
    fixture arithmetic is guarded by the three checks above. CARROT, because its
    path is the one that responds to supply (MELON's quote is flat at 250-280
    whatever the supply — a guard written on it cannot fail).
    """
    from agent.belief.market import forecast, PRODUCTS

    day, horizon = 3, 30
    obs = _obs_with_rival_board({}, day, ("CARROT",))
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


def test_a_wrong_shaped_curve_is_refused() -> None:
    """`rival_supply` is a `(days, 9)` contract, and a wrong shape raises."""
    from agent.belief.market import forecast

    obs = _obs_with_rival_board({}, 0, ("MELON",))
    try:
        forecast(obs, days=30, rival_supply=np.zeros((5, 3)),
                 config={"farmHandCostMult": 1})
    except ValueError:
        pass
    else:
        raise AssertionError("a wrong-shaped rival_supply must be refused")


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} rival-calendar checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
