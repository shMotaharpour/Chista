"""Tests for world.state — State read-views over the live engine observation.

Run: cd /chista/Chista/ChistaAgent && . .venv/bin/activate && python -m tests.test_state
"""
from __future__ import annotations

from kaggle_environments import make

from world import mechanics as M
from world.state import State, quadrant_of

P = {"farmer": ["PASS"], "hands": [], "market": []}


def make_env(seed: int):
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed},
               debug=False)
    env.reset(2)
    return env


def test_state_reads_live_obs():
    env = make_env(70)
    env.step([{"farmer": ["PASS"], "hands": [], "market":
               [["BUY_SEED", "MELON", 2], ["BUY_ANIMAL", "GOOSE", 1]]}, P])
    obs = env.state[0].observation
    s = State.from_obs(obs)
    assert s.day == 0 and s.money == obs.farms[0]["money"]
    assert isinstance(s.market_prices, dict) and s.market_prices
    assert s.seeds.get("MELON", 0) == 2


def test_quadrant_names():
    assert quadrant_of(0, 0) == "NW"
    assert quadrant_of(6, 2) == "NE"
    assert quadrant_of(2, 7) == "SW"
    assert quadrant_of(7, 7) == "SE"
    assert quadrant_of(4, 4) == "NW"   # NW = x<5 and y<5


def test_unlocked_per_quadrant_not_symmetric():
    """Land opens per quadrant in fixed order (NE first) — after buying NE the
    unlocked set is NW+NE (whole top rows), not a centered square."""
    env = make_env(70)
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_LAND"]]}, P])
    s = State.from_obs(env.state[0].observation)
    assert s.unlocked == ["NW", "NE"]
    assert s.is_unlocked_tile(6, 2)     # NE tile
    assert not s.is_unlocked_tile(2, 6) # SW still locked
    assert not s.is_unlocked_tile(7, 7) # SE still locked
    assert s.is_unlocked_tile(4, 4)     # NW


def test_spatial_lookups():
    env = make_env(70)
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 1]]}, P])
    env.step([{"farmer": ["PLANT", "MELON"], "hands": [], "market": []}, P])
    s = State.from_obs(env.state[0].observation)
    p = s.plant_at(4, 4)
    assert p is not None and p.crop == "MELON" and p.age == 0
    assert s.plant_at(2, 2) is None
    assert s.empty_at(2, 2)
    assert not s.empty_at(4, 4)
    assert not s.empty_at(6, 6)         # locked quadrant: not "empty" for us


def test_animal_lookup_after_setup():
    from lab.goose3_demo import run_goose3
    env = run_goose3(days=3)
    o = env.steps[-1][0].observation
    s = State.from_obs(o)
    animals = [(xy, s.animal_at(*xy)) for xy, t in s.iter_animals()]
    assert len(animals) == 3
    assert all(v.animal == "GOOSE" for _, v in animals)


def test_watering_view_updates():
    env = make_env(70)
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 1]]}, P])
    env.step([{"farmer": ["PLANT", "MELON"], "hands": [], "market": []}, P])
    s = State.from_obs(env.state[0].observation)
    assert s.plant_at(4, 4).watered_today is False
    env.step([{"farmer": ["WATER"], "hands": [], "market": []}, P])
    s2 = State.from_obs(env.state[0].observation)
    assert s2.plant_at(4, 4).watered_today is True


def test_harvestable_respects_maturity():
    env = make_env(70)
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}, P])
    env.step([{"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, P])
    s = State.from_obs(env.state[0].observation)
    # engine HARVEST gate (L449, L453): yield>0 AND age>=first_yield_day
    t = s.tile_at(4, 4)
    age = M.plant_age(t, s.day)
    assert t["yield_units"] > 0 and age < M.CROPS["WHEAT"]["first_yield_day"]


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
