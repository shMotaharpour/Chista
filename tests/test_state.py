"""Tests for world.state — State views over the live engine observation.

Run: cd /chista/Chista/ChistaAgent && . .venv/bin/activate && python -m tests.test_state
"""
from __future__ import annotations

from kaggle_environments import make

from world.state import State

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


def test_plants_and_animals_views():
    env = make_env(70)
    # plant 2 melons manually via market + farmer ops
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 2]]}, P])
    # farmer at (4,4): PLANT then step off and PLANT again
    env.step([{"farmer": ["PLANT", "MELON"], "hands": [], "market": []}, P])
    env.step([{"farmer": ["WEST"], "hands": [], "market": []}, P])
    env.step([{"farmer": ["PLANT", "MELON"], "hands": [], "market": []}, P])
    s = State.from_obs(env.state[0].observation)
    plants = s.plants()
    assert len(plants) == 2, plants
    assert all(p.crop == "MELON" for p in plants)
    assert any(p.x == 4 and p.y == 4 for p in plants)
    assert any(p.x == 3 and p.y == 4 for p in plants)


def test_animal_views_after_setup():
    # use the working 3-goose demo to get real coops
    from lab.goose3_demo import run_goose3
    env = run_goose3(days=3)
    o = env.steps[-1][0].observation
    s = State.from_obs(o)
    animals = s.animal_tiles()
    assert len(animals) == 3, animals
    assert {a.animal for a in animals} == {"GOOSE"}
    assert s.animals_needing_feed() or all(a.fed_today for a in animals)


def test_empty_tiles_respects_unlocked():
    env = make_env(70)
    s = State.from_obs(env.state[0].observation)
    empties = s.empty_unlocked_tiles()
    assert all(x < 5 and y < 5 for x, y in empties)   # NW only at start
    assert len(empties) == 25                          # full 5x5 NW incl. spawn


def test_watering_views_update():
    env = make_env(70)
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 1]]}, P])
    env.step([{"farmer": ["PLANT", "MELON"], "hands": [], "market": []}, P])
    s = State.from_obs(env.state[0].observation)
    assert len(s.plants_needing_water()) == 1
    env.step([{"farmer": ["WATER"], "hands": [], "market": []}, P])
    s2 = State.from_obs(env.state[0].observation)
    assert len(s2.plants_needing_water()) == 0


def test_harvestable_respects_maturity():
    env = make_env(70)
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}, P])
    env.step([{"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, P])
    s = State.from_obs(env.state[0].observation)
    assert len(s.harvestable_plants()) == 0    # day 0: wheat not mature
    assert len(s.plants()) == 1


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
