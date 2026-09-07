"""Fixture tests for world.mechanics — every rule compared against the ENGINE.

Run: cd /chista/Chista/ChistaAgent && . .venv/bin/activate && python -m tests.test_mechanics
"""
from __future__ import annotations

from kaggle_environments.envs.kaggriculture import kaggriculture as engine

from world import mechanics as M


def plant(crop, planted_day, **kw):
    t = {"kind": "PLANT", "crop": crop, "planted_day": planted_day,
         "watered_today": False, "consecutive_unwatered": 0, "yield_units": 0}
    t.update(kw)
    return t


def goose(placed_day, **kw):
    t = {"kind": "COOP", "animal": "GOOSE", "placed_day": placed_day,
         "fed_today": False, "cared_today": False, "consecutive_unfed": 0,
         "yield_units": 0, "pending_care_bonus": 0, "fertilizer_available": False}
    t.update(kw)
    return t


def test_tables_are_the_engine_tables():
    assert M.CROPS is engine.CROPS
    assert M.ANIMALS is engine.ANIMALS
    assert M.market_price is engine.market_price
    assert M.ANIMALS["GOOSE"]["interval"] == 1 and M.ANIMALS["GOOSE"]["max_held"] == 4
    assert M.ANIMALS["COW"]["first_yield_day"] == 8 and M.ANIMALS["COW"]["max_held"] == 6
    assert M.ANIMALS["SHEEP"]["interval"] == 3
    assert M.CROPS["MELON"]["max_yield_day"] == 12 and M.CROPS["MELON"]["max_yield"] == 6


def test_market_price_matches_engine():
    for item in engine.PRODUCTS:
        for inv in (0, 5000, 10000, 15000, 20000):
            assert M.market_price(item, inv) == engine.market_price(item, inv)
    assert M.market_price("MELON", 30000) == 1   # floor


def test_is_animal_tile_uses_engine_guard():
    # engine L811: `isinstance(tile, dict) and "animal" in tile` — kind irrelevant
    fake = {"animal": "GOOSE"}                     # no "kind" key at all
    assert M.is_animal_tile(fake)
    assert not M.is_animal_tile({"kind": "COOP"})  # structure without animal
    assert M.is_animal_tile(goose(0))


def test_water_bonus_window_matches_engine_formula():
    # engine L440: window_start = (max_yield_day + 1) // 2
    for crop, cd in engine.CROPS.items():
        lo, hi = M.water_bonus_window(crop)
        assert lo == (cd["max_yield_day"] + 1) // 2
        assert hi == cd["max_yield_day"]
    assert M.water_bonus_window("MELON") == (6, 12)
    assert M.water_bonus_window("WHEAT") == (2, 4)


def test_production_due_matches_engine_refresh():
    """Engine L828-829 evaluates at refresh for next_day = day+1:
    (next_day - placed - first) >= 0 and % interval == 0.
    Our query with `day` reproduces the production the NEXT refresh grants."""
    g = goose(placed_day=0)
    # end-of-day-3 refresh (next_day=4): first egg
    assert M.animal_production_due(g, day=3)
    # and daily after (interval 1)
    assert M.animal_production_due(g, day=4)
    assert M.animal_production_due(g, day=5)
    assert not M.animal_production_due(g, day=2)
    # cow placed day 0: milk at end-of-day-7 refresh (next_day=8), then every 2
    c = {"kind": "PASTURE", "animal": "COW", "placed_day": 0}
    due = [d for d in range(0, 14) if M.animal_production_due(c, d)]
    assert due == [7, 9, 11, 13], due
    # sheep: first at next_day=6 -> query day 5, then +3
    s = {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 0}
    due = [d for d in range(0, 15) if M.animal_production_due(s, d)]
    assert due == [5, 8, 11, 14], due


def test_animal_pending_yield_matches_engine_payout():
    # engine L831-833: min(max_held, yield + 1 + bank if fed)
    assert M.animal_pending_yield(goose(0)) == 1                    # base only
    g2 = goose(0, fed_today=True, pending_care_bonus=2)
    assert M.animal_pending_yield(g2) == 3                          # 0 + 1 + 2
    g3 = goose(0, fed_today=True, pending_care_bonus=9)
    assert M.animal_pending_yield(g3) == 4                          # max_held cap
    g4 = goose(0, fed_today=False, pending_care_bonus=5)
    assert M.animal_pending_yield(g4) == 1                          # bank not consumed
    g5 = goose(0, fed_today=True, yield_units=3, pending_care_bonus=1)
    assert M.animal_pending_yield(g5) == 4                          # 3+1+1, cap 4


def test_feed_care_gates_match_engine():
    # engine L505: FEED returns if fed_today; L519: CARE returns if cared_today
    assert M.animal_needs_feed(goose(0))
    assert not M.animal_needs_feed(goose(0, fed_today=True))
    assert M.animal_needs_care(goose(0))
    assert not M.animal_needs_care(goose(0, cared_today=True))


def test_hire_cost_matches_engine_fib():
    # engine L690-691: mult * fib(n_already_today); _fib(0)=1, _fib(1)=1
    for n in range(6):
        assert M.hire_cost(n) == engine._hire_cost(n), n
    assert [M.hire_cost(n) for n in range(4)] == [1, 1, 2, 3]
    assert sum(M.hire_cost(n) for n in range(4)) == 7


def test_one_time_crops_list():
    assert set(M.ONE_TIME_CROPS) == {"WHEAT", "CARROT", "MELON"}


def test_new_plant_facts_we_discovered():
    """Documented engine behaviors that our earlier imagined model missed:
    one-time crops START with yield_units=1; planting day counts as unwatered."""
    p = engine._new_plant("WHEAT", 0, 24)
    assert p["yield_units"] == 1
    assert p["consecutive_unwatered"] == 1
    o = engine._new_plant("STRAWBERRY", 0, 24)
    assert o["yield_units"] == 0   # ongoing starts at 0


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
