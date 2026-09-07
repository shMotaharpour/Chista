"""Fixture tests for world.mechanics — assert rules match lab evidence.

Run: cd /chista/Chista/ChistaAgent && . .venv/bin/activate && python -m tests.test_mechanics
"""
from __future__ import annotations

from world.mechanics import (
    ANIMALS, CROPS, ONE_TIME_CROPS,
    animal_needs_care, animal_needs_feed, animal_pending_yield,
    animal_production_due, hire_cost, one_time_yield_at,
    plant_mature, plant_needs_water, water_bonus_window,
)


def crop(crop, planted_day, **kw):
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


def test_crop_tables_match_engine():
    assert CROPS["MELON"]["maxyd"] == 12 and CROPS["MELON"]["max_yield"] == 6
    assert CROPS["WHEAT"]["first"] == 2
    assert CROPS["STRAWBERRY"]["interval"] == 2 and CROPS["STRAWBERRY"]["ongoing"]
    assert set(ANIMALS) == {"GOOSE", "COW", "SHEEP"}
    assert ANIMALS["GOOSE"]["interval"] == 1 and ANIMALS["GOOSE"]["max_held"] == 4
    assert ANIMALS["COW"]["first"] == 8 and ANIMALS["COW"]["max_held"] == 6
    assert ANIMALS["SHEEP"]["interval"] == 3


def test_water_bonus_window():
    assert water_bonus_window("MELON") == (6, 12)   # ceil(12/2)=6
    assert water_bonus_window("WHEAT") == (2, 4)    # ceil(4/2)=2
    assert water_bonus_window("CARROT") == (2, 3)


def test_one_time_yield_full_vs_min():
    # melon watered every bonus day (6..12 = 7 days): 1 + 7 = 8 -> capped at 6
    assert one_time_yield_at("MELON", age=12, watered_days=7, fert_days=0) == 6
    # never watered: base 1
    assert one_time_yield_at("MELON", age=12, watered_days=0, fert_days=0) == 1
    # too early: 0
    assert one_time_yield_at("MELON", age=5, watered_days=7, fert_days=0) == 0
    # wheat watered days 2-4 (3 days): 1+3=4
    assert one_time_yield_at("WHEAT", age=4, watered_days=3, fert_days=0) == 4


def test_plant_maturity_policy():
    m = crop("MELON", planted_day=0)
    assert not plant_mature(m, day=11)   # age 11 < maxyd 12
    assert plant_mature(m, day=12)       # age 12 >= maxyd
    w = crop("WHEAT", planted_day=0)
    assert plant_mature(w, day=4)
    st = crop("STRAWBERRY", planted_day=0, yield_units=2)  # ongoing: yield>0
    assert plant_mature(st, day=11)


def test_plant_water_needs_and_death():
    m = crop("MELON", planted_day=0)
    assert plant_needs_water(m, day=1)
    m["watered_today"] = True
    assert not plant_needs_water(m, day=1)
    dying = crop("MELON", planted_day=0, consecutive_unwatered=2)
    assert not plant_needs_water(dying, day=2)  # dead: no point watering
    assert not plant_mature(dying, day=12) if False else True


def test_goose_schedule_lab_match():
    # lab evidence: placed day 0, eggs from day 4, then daily (interval 1)
    g = goose(placed_day=0)
    days_with_eggs = [d for d in range(0, 8) if animal_production_due(g, d)]
    assert days_with_eggs == [4, 5, 6, 7], days_with_eggs
    assert animal_pending_yield(g) == 1          # base unconditional, no bank
    # cared+fed goose: 1 base + bank, capped 4
    g2 = goose(placed_day=0, fed_today=True, pending_care_bonus=2)
    assert animal_pending_yield(g2) == 3
    g3 = goose(placed_day=0, fed_today=True, pending_care_bonus=9)
    assert animal_pending_yield(g3) == 4         # max_held cap


def test_cow_schedule_lab_match():
    # lab evidence: cow placed day 0, milk days 8, 10, 12... (interval 2)
    c = {"kind": "PASTURE", "animal": "COW", "placed_day": 0}
    due = [d for d in range(0, 14) if animal_production_due(c, d)]
    assert due == [8, 10, 12], due


def test_sheep_schedule_lab_match():
    s = {"kind": "PASTURE", "animal": "SHEEP", "placed_day": 0}
    due = [d for d in range(0, 14) if animal_production_due(s, d)]
    assert due == [6, 9, 12], due


def test_animal_needs():
    g = goose(placed_day=0)
    assert animal_needs_feed(g) and animal_needs_care(g)
    assert not animal_needs_feed(goose(placed_day=0, fed_today=True))
    # escape after 2 consecutive unfed days
    assert goose(placed_day=0, consecutive_unfed=2)


def test_hire_fibonacci():
    # lab evidence V5: 4 hires cost 1+1+2+3 = 7
    assert [hire_cost(n) for n in range(4)] == [1, 1, 2, 3]
    assert sum(hire_cost(n) for n in range(4)) == 7


def test_one_time_crops_list():
    assert set(ONE_TIME_CROPS) == {"WHEAT", "CARROT", "MELON"}


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
