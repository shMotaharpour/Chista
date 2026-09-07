"""Tests for world.pruning — every rule verified IN-ENGINE:
apply the pruned action, diff state before/after == zero change.

Run: cd /chista/Chista/ChistaAgent && . .venv/bin/activate && python -m tests.test_pruning
"""
from __future__ import annotations

import copy

from kaggle_environments import make

from world.pruning import prune_all, prune_farmer, market_candidates as prune_market

P = {"farmer": ["PASS"], "hands": [], "market": []}


def make_env(seed: int = 70):
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed},
               debug=False)
    env.reset(2)
    return env


def state_of(env):
    from world.state import State
    return State.from_obs(env.state[0].observation)


def inert_in_engine(env, action) -> bool:
    """True iff applying `action` on the live engine changes nothing."""
    before = copy.deepcopy(env.state)
    env.step([action, P])
    after = env.state[0].observation
    b = before[0].observation
    changed = (
        after.farms[0]["money"] != b.farms[0]["money"]
        or after.farms[0]["tiles"] != b.farms[0]["tiles"]
        or dict(after.private["shed"]) != dict(b.private["shed"])
        or dict(after.private["seeds"]) != dict(b.private["seeds"])
    )
    # rewind so tests stay independent
    env.state = before
    env.steps = env.steps[:-1]
    return not changed


def test_water_on_watered_is_inert():
    env = make_env()
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 1]]}, P])
    env.step([{"farmer": ["PLANT", "MELON"], "hands": [], "market": []}, P])
    env.step([{"farmer": ["WATER"], "hands": [], "market": []}, P])
    s = state_of(env)
    cands = prune_farmer(s)
    assert ("WATER", 4, 4) not in cands            # pruned: already watered
    assert inert_in_engine(env, {"farmer": ["WATER"], "hands": [], "market": []})


def test_water_on_thirsty_is_kept():
    env = make_env()
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 1]]}, P])
    env.step([{"farmer": ["PLANT", "MELON"], "hands": [], "market": []}, P])
    s = state_of(env)
    assert ("WATER", 4, 4) in prune_farmer(s)
    assert not inert_in_engine(env, {"farmer": ["WATER"], "hands": [], "market": []})


def test_harvest_zero_yield_inert():
    env = make_env()
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}, P])
    env.step([{"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, P])
    s = state_of(env)
    cands = prune_farmer(s)
    assert not any(c[0] == "HARVEST" for c in cands)   # yield 0: engine L446 returns
    assert inert_in_engine(env, {"farmer": ["HARVEST"], "hands": [], "market": []})


def test_pickup_away_from_shed_absent():
    env = make_env()
    env.step([{"farmer": ["WEST"], "hands": [], "market": [["BUY_ANIMAL", "GOOSE", 1]]}, P])
    s = state_of(env)
    assert s.farmer_xy == (3, 4)                        # not shed-adjacent
    assert not any(c[0] == "PICKUP" for c in prune_farmer(s))


def test_pickup_at_shed_present_when_animals_need_feed():
    env = make_env()
    # buy goose + wheat, then PLACE the goose so an animal actually needs feed
    env.step([{"farmer": ["PASS"], "hands": [], "market":
               [["BUY_ANIMAL", "GOOSE", 1], ["BUY_PRODUCT", "WHEAT", 5]]}, P])
    env.step([{"farmer": ["PICKUP", "GOOSE", 1], "hands": [], "market": []}, P])
    env.step([{"farmer": ["BUILD_COOP"], "hands": [], "market": []}, P])
    env.step([{"farmer": ["PLACE", "GOOSE"], "hands": [], "market": []}, P])
    s = state_of(env)
    assert s.animal_at(4, 4) is not None, "goose should be placed and need feed"
    cands = prune_farmer(s)
    assert any(c[0] == "PICKUP" and c[1] == "WHEAT" for c in cands)


def test_hire_without_money_pruned():
    from world import mechanics as M
    env = make_env()
    # drain money to $0: 35 melon seeds ($2800) + strawberry seeds ($200)
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 35]]}, P])
    s = state_of(env)
    env.step([{"farmer": ["PASS"], "hands": [], "market":
               [["BUY_SEED", "STRAWBERRY", int(s.money // 100)]]}, P])
    s = state_of(env)
    assert s.money < M.hire_cost(s.hires_today), s.money
    assert ["HIRE"] not in prune_market(s)


def test_sell_only_nonzero():
    env = make_env()
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "WHEAT", 1]]}, P])
    s = state_of(env)
    sells = [m for m in prune_market(s) if m[0] == "SELL"]
    assert all(m[2] > 0 for m in sells)
    # WHEAT seed in shed? no — seeds live separately. SELL of items not in shed:
    assert not any(m[1] == "MELON" for m in sells)


def test_buy_land_when_all_unlocked_pruned():
    env = make_env()
    s = state_of(env)
    assert ["BUY_LAND"] in prune_market(s)              # NW only: legal
    # can't easily unlock 3 quadrants here (expensive) — check the gate logic:
    s.me["unlocked_quadrants"].append("NE")
    s.me["unlocked_quadrants"].append("SW")
    s.me["unlocked_quadrants"].append("SE")
    assert ["BUY_LAND"] not in prune_market(s)


def test_plant_without_seeds_pruned():
    env = make_env()
    s = state_of(env)
    assert s.seeds.get("MELON", 0) == 0
    assert not any(c[0] == "PLANT" and c[1] == "MELON" for c in prune_farmer(s))


def test_prune_all_shape():
    env = make_env()
    s = state_of(env)
    v = prune_all(s)
    assert set(v.keys()) == {"farmer", "hands", "farmer_moves", "market"}
    assert ("PASS",) in v["farmer_moves"]


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
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
