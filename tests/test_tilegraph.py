"""Tests for tilegraph — atoms verified against engine movement reality.

Run: cd /chista/Chista/ChistaAgent && . .venv/bin/activate && python -m tests.test_tilegraph
"""
from __future__ import annotations

from kaggle_environments import make

from world.tilegraph import (
    action_turns, at_shed, dist, dist_to_shed, spawn_position,
    units_spawn_positions,
)

P = {"farmer": ["PASS"], "hands": [], "market": []}


def test_known_distances():
    assert dist((4, 4), (0, 0)) == 8
    assert dist((4, 4), (9, 9)) == 10
    assert dist((3, 4), (4, 4)) == 1


def test_shed_distances():
    assert dist_to_shed((4, 4)) == 0
    assert dist_to_shed((5, 5)) == 0
    assert dist_to_shed((0, 0)) == 8
    assert at_shed((4, 5)) and not at_shed((3, 3))


def test_action_turns():
    assert action_turns("WATER") == 1
    assert action_turns("PICKUP", item_types=1) == 1   # n of one type: free
    assert action_turns("PICKUP", item_types=2) == 2   # two types: two ops
    assert action_turns("DROP", item_types=3) == 1     # DROP: whole hand, 1 op


def test_engine_movement_matches_dist():
    """Walk the farmer step-by-step toward a target; arrival turn must equal
    the predicted Manhattan distance (verified against the LIVE engine)."""
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 70},
               debug=False)
    env.reset(2)
    start = tuple(env.state[0].observation.farms[0]["farmer"])
    goal = (0, 0)
    predicted = dist(start, goal)

    def move_toward(pos):
        x, y = pos
        if y > goal[1]:
            return ["NORTH"]
        if y < goal[1]:
            return ["SOUTH"]
        if x > goal[0]:
            return ["WEST"]
        return ["EAST"]

    turns = 0
    while tuple(env.state[0].observation.farms[0]["farmer"]) != goal:
        pos = env.state[0].observation.farms[0]["farmer"]
        env.step([{"farmer": move_toward(tuple(pos)), "hands": [], "market": []}, P])
        turns += 1
        assert turns <= predicted + 2, "farmer never arrived"
    assert turns == predicted, (turns, predicted)


def test_engine_locked_tiles_passable():
    """Engine docs: locked tiles are passable. Walk into the locked quadrant;
    dist still Manhattan."""
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 70},
               debug=False)
    env.reset(2)
    start = tuple(env.state[0].observation.farms[0]["farmer"])  # (4,4)
    goal = (5, 4)   # NE quadrant, locked at start
    predicted = dist(start, goal)
    turns = 0
    while tuple(env.state[0].observation.farms[0]["farmer"]) != goal:
        env.step([{"farmer": ["EAST"], "hands": [], "market": []}, P])
        turns += 1
        if turns > predicted + 1:
            break
    assert tuple(env.state[0].observation.farms[0]["farmer"]) == (5, 4)


def test_spawn_distribution():
    """Farmer always (4,4); hands fill min-occupancy tiles NWSE (engine L533)."""
    # farmer alone => (4,4)
    assert units_spawn_positions(0) == [(4, 4)]
    # 1 hand => (5,4) [next NWSE, occupancy tied at 0]
    assert units_spawn_positions(1) == [(4, 4), (5, 4)]
    # 2 hands => (4,5); 3 hands => (5,5)
    assert units_spawn_positions(2) == [(4, 4), (5, 4), (4, 5)]
    assert units_spawn_positions(3) == [(4, 4), (5, 4), (4, 5), (5, 5)]
    # more hands than tiles: cycle back to min occupancy (5,4 has 1... all 1)
    # => first NWSE again
    assert units_spawn_positions(4)[4] == (4, 4)


def test_spawn_position_matches_engine():
    """Hire hands in the live engine and check where they actually spawn."""
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": 70},
               debug=False)
    env.reset(2)
    env.step([{"farmer": ["PASS"], "hands": [], "market": [["HIRE"]] * 4}, P])
    hands = [tuple(h) for h in env.state[0].observation.farms[0]["hands"]]
    predicted = units_spawn_positions(len(hands))[1:]
    assert hands == predicted, (hands, predicted)


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
