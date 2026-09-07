"""Tests for world.rollback — determinism of snapshot/restore on the live engine.

Run: cd /chista/Chista/ChistaAgent && . .venv/bin/activate && python -m tests.test_rollback
"""
from __future__ import annotations

from kaggle_environments import make

from world.rollback import hypothetical, restore, snapshot

P = {"farmer": ["PASS"], "hands": [], "market": []}
BUY = {"farmer": ["PASS"], "hands": [], "market": [["BUY_SEED", "MELON", 2]]}


def make_env(seed: int):
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed},
               debug=False)
    env.reset(2)
    return env


def weed_count(env):
    return sum(1 for row in env.state[0].observation.farms[0]["tiles"]
               for t in row if isinstance(t, dict) and t.get("kind") == "WEED")


def test_snapshot_restore_money():
    env = make_env(70)
    env.step([BUY, P])
    snap = snapshot(env)
    money_at_snap = env.state[0].observation.farms[0]["money"]
    env.step([BUY, P])
    env.step([BUY, P])
    assert env.state[0].observation.farms[0]["money"] < money_at_snap
    restore(env, snap)
    assert env.state[0].observation.farms[0]["money"] == money_at_snap


def test_restore_matches_fresh_episode():
    """Restoring to turn N must reproduce the same future as a fresh run to N+1
    (day-keyed RNG makes the engine deterministic)."""
    env = make_env(70)
    for _ in range(48):
        env.step([P, P])
    snap = snapshot(env)
    env.step([BUY, P])          # divergent path
    env.step([P, P])
    restore(env, snap)
    env.step([P, P])            # same length path, no buy
    money_rolled = env.state[0].observation.farms[0]["money"]

    fresh = make_env(70)
    for _ in range(49):
        fresh.step([P, P])
    money_fresh = fresh.state[0].observation.farms[0]["money"]
    assert money_rolled == money_fresh, (money_rolled, money_fresh)


def test_hypothetical_leaves_live_state_untouched():
    env = make_env(70)
    env.step([P, P])
    snap = snapshot(env)
    live_money = env.state[0].observation.farms[0]["money"]
    result = hypothetical(env, snap, [BUY, P])
    hyp_money = result[0].observation.farms[0]["money"]
    assert hyp_money < live_money          # hypothetical spent money
    assert env.state[0].observation.farms[0]["money"] == live_money  # live intact


def test_market_inventory_identical_after_restore():
    env = make_env(123)
    for _ in range(24):
        env.step([P, P])
    snap = snapshot(env)
    inv_before = dict(env.state[0].observation.market["inventory"])
    env.step([BUY, P])
    restore(env, snap)
    inv_after = dict(env.state[0].observation.market["inventory"])
    assert inv_before == inv_after


def test_multi_seed_determinism():
    for seed in (7, 70, 123):
        env = make_env(seed)
        for _ in range(24):
            env.step([P, P])
        snap = snapshot(env)
        env.step([P, P])
        restore(env, snap)
        env.step([P, P])
        t_rolled = env.state[0].observation.farms[0]["tiles"][3][7]
        fresh = make_env(seed)
        for _ in range(25):
            fresh.step([P, P])
        t_fresh = fresh.state[0].observation.farms[0]["tiles"][3][7]
        assert t_rolled == t_fresh, (seed, t_rolled, t_fresh)


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
