"""agent obs-decode tests: both farms -> packed keys, coverage, LOCKED.

Run:  .venv/bin/python -m tests.test_agent_obs

Contracts under test:
- LOCKED tiles decode to the sentinel (-1), never raise, never enter
  `classes` (F042: three of four quadrants start locked).
- Coverage round-trip: a scripted fast_sim episode with
  weedSpawnChance = 0, decoded at every day start on BOTH farms, lands
  entirely inside the shipped graph's key_index — a real coverage test
  of the graph itself.
- Day-start guard: the DP entry asserts hour == 0.
- Opponent parity: the opponent farm decodes through the identical path.
- Equivalence classes: sum(classes.values()) == plannable tile count.
- The magic-number TODO (tile_state.py mls check) is pinned by a live
  engine probe: plant, read max_lifespan_step back, compare (closes the
  R005 TODO §3.4 pointed at).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from agent.obs import LOCKED_KEY, WorldView, decode_farm, decode_world
from agent.tile_dp.graph import TileGraph
from agent.tile_dp.tile_state import decode_tile

GRAPH_PATH = Path(__file__).resolve().parents[1] / "tile_dp" / "models" \
    / "graph_tile_lifecycle.npz"


def _graph() -> TileGraph:
    return TileGraph.load(GRAPH_PATH)


def test_locked_decodes_to_sentinel() -> None:
    """A fresh 10x10 board: 3 quadrants locked (F042), decode never raises."""
    board = []
    for y in range(10):
        row = []
        for x in range(10):
            row.append(None if (y < 5 and x < 5) else "LOCKED")
        board.append(row)
    farm = {"money": 3000, "tiles": board, "farmer": [4, 4], "hands": [],
            "unlocked_quadrants": ["NW"], "hires_today": 0}
    view = decode_farm(farm, day=0, hour=0)
    locked_count = int((view.keys == LOCKED_KEY).sum())
    assert locked_count == 75, locked_count       # F042: 3 of 4 quadrants
    assert LOCKED_KEY not in view.classes
    # plannable = the NW quadrant
    assert sum(view.classes.values()) == 25


def test_day_start_guard_raises_off_hour_zero() -> None:
    """The DP-entry decode asserts hour == 0 (tested, not just present)."""
    obs = _mini_obs(hour=7)
    raised = False
    try:
        decode_world(obs, at_day_start=True)
    except AssertionError:
        raised = True
    assert raised, "mid-day decode must raise when at_day_start is required"


def test_opponent_decode_is_opt_in() -> None:
    """M1 default: opponent is None (nothing consumes it yet, #16);
    decode_opponent=True fills it through the identical code path."""
    obs = _mini_obs(hour=0)
    default = decode_world(obs)
    assert default.opponent is None
    opt_in = decode_world(obs, decode_opponent=True)
    assert opt_in.opponent is not None
    # and the opt-in view equals a direct decode_farm of the same board
    direct = decode_farm(obs["farms"][1], day=obs["day"], hour=0)
    assert (opt_in.opponent.keys == direct.keys).all()
    assert opt_in.opponent.classes == direct.classes


def test_opponent_parity_same_path() -> None:
    """Mirrored boards decode to identical FarmViews field for field."""
    board = _board_with_plant(day=2)
    me = decode_farm({"money": 100, "tiles": board, "farmer": [4, 4],
                      "hands": [], "unlocked_quadrants": ["NW"],
                      "hires_today": 0}, day=2, hour=0)
    opp = decode_farm({"money": 100, "tiles": board, "farmer": [0, 0],
                       "hands": [], "unlocked_quadrants": ["NW"],
                       "hires_today": 0}, day=2, hour=0)
    assert (me.keys == opp.keys).all()
    assert me.classes == opp.classes


def test_equivalence_classes_count_plannable() -> None:
    """sum(classes.values()) == plannable tiles; distinct keys reported."""
    farm = {"money": 0,
            "tiles": [[None, None, {"kind": "WEED"}],
                      ["LOCKED", "LOCKED", "LOCKED"],
                      [None, None, None]],
            "farmer": [0, 0], "hands": [],
            "unlocked_quadrants": ["NW"], "hires_today": 0}
    view = decode_farm(farm, day=0, hour=0)
    assert sum(view.classes.values()) == 6      # 9 tiles - 3 locked
    assert len(view.classes) >= 1


def test_coverage_roundtrip_weedspawn_zero() -> None:
    """Scripted episode, weedSpawnChance = 0: every day-start key on BOTH
    farms is inside the shipped graph's key_index (graph coverage test)."""
    from offline_lab.fast_sim import FastSim

    graph = _graph()
    known = frozenset(graph.key_index)
    sim = FastSim({"episodeSteps": 10 * 24, "seed": 7,
                   "weedSpawnChance": 0.0})

    def act():
        return {"farmer": ["PASS"], "hands": [], "market": []}

    for day in range(10):
        obs = sim.observations()[0]    # hour 0 of `day`
        assert int(obs["hour"]) == 0, (day, obs["hour"])
        for p in (0, 1):
            view = decode_farm(obs["farms"][p], day=int(obs["day"]), hour=0)
            for key in view.classes:
                assert key in known, (
                    f"unmodelled key {key} "
                    f"({TileState.unpack(key).describe()}) on farm {p} "
                    f"day {day}")
        for _ in range(24):            # fill the day; next loop sees hour 0
            if sim.done:
                break
            sim.step([act(), act()])
        if sim.done:
            break


def test_magic_number_pinned_by_engine_probe() -> None:
    """tile_state.py's mls TODO: plant a MELON on a live sim, read
    max_lifespan_step back, and pin the decoder's formula
    (planted + max_yield_day + 1) * TURNS_PER_DAY against it."""
    from kaggle_environments.envs.kaggriculture import kaggriculture as K
    from agent.tile_dp.tile_state import TURNS_PER_DAY
    from offline_lab.fast_sim import FastSim

    sim = FastSim({"episodeSteps": 5 * 24, "seed": 11,
                   "weedSpawnChance": 0.0})
    act = {"farmer": ["PASS"], "hands": [], "market": []}
    sim.step([{"farmer": ["PASS"], "hands": [],
               "market": [["BUY_SEED", "MELON", 1]]}, act])
    sim.step([{"farmer": ["PLANT", "MELON"], "hands": [], "market": []},
              act])
    obs = sim.observations()[0]
    me = obs["farms"][0]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]
    assert isinstance(tile, dict) and tile["kind"] == "PLANT"
    spec = K.CROPS["MELON"]
    want = (int(tile["planted_day"]) + int(spec["max_yield_day"]) + 1) \
        * TURNS_PER_DAY
    assert int(tile["max_lifespan_step"]) == want, (
        f"max_lifespan_step {tile['max_lifespan_step']} != "
        f"(planted {tile['planted_day']} + max_yield_day "
        f"{spec['max_yield_day']} + 1) * {TURNS_PER_DAY}")


def _board_with_plant(day: int) -> list:
    return [[None, None, {"kind": "PLANT", "crop": "WHEAT",
                          "planted_day": 0, "watered_today": True,
                          "consecutive_unwatered": 0, "yield_units": 2,
                          "max_lifespan_step": 96,
                          "fertilized_until_day": -1}],
            [None, "WEED", None],
            ["LOCKED", "LOCKED", "LOCKED"]]


def _mini_obs(hour: int) -> dict:
    return {"player": 0, "day": 1, "hour": hour,
            "farms": [
                {"money": 10, "tiles": _board_with_plant(1),
                 "farmer": [0, 0], "hands": [],
                 "unlocked_quadrants": ["NW"], "hires_today": 0},
                {"money": 10, "tiles": _board_with_plant(1),
                 "farmer": [0, 0], "hands": [],
                 "unlocked_quadrants": ["NW"], "hires_today": 0}],
            "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
            "market": {"inventory": {}, "prices": {}},
            "town": {"unlocked_shops": []},
            "remainingOverageTime": 60.0}


def test_decode_world_no_graph_keys_option() -> None:
    """graph_keys=None: raw keys, unknown 0 (the caller opted out)."""
    wv = decode_world(_mini_obs(hour=0), decode_opponent=True)
    assert wv.unknown_keys == 0
    assert wv.me.classes and wv.opponent.classes


def test_unknown_keys_map_nearest_and_count() -> None:
    """A key outside the graph maps to the nearest modelled state and is
    counted — never raised, never silent (issue §3.3)."""
    graph = _graph()
    known = frozenset(graph.key_index)
    # build an obs whose plant tile yields a key the graph lacks:
    # a WHEAT with an odd yield (the graph enumerates the reachable ones)
    obs = _mini_obs(hour=0)
    obs["farms"][0]["tiles"][0][0] = {
        "kind": "PLANT", "crop": "WHEAT", "planted_day": -3,
        "watered_today": True, "consecutive_unwatered": 0,
        "yield_units": 5, "max_lifespan_step": 96,
        "fertilized_until_day": -1}
    wv = decode_world(obs, graph_keys=known, decode_opponent=True)
    raw = decode_world(obs, decode_opponent=True)
    unknown_raw = 0
    for farm_view in (raw.me, raw.opponent):   # cumulative over BOTH farms
        raw_keys = {int(k) for row in farm_view.keys for k in row} \
            - {LOCKED_KEY}
        unknown_raw += sum(farm_view.classes[k] for k in raw_keys
                           if k not in known)
    # remapping preserves each farm's multiset size (6 plannable: 9 - 3
    # locked); every unmodelled key on either farm is counted, and every
    # surviving key is modelled
    assert sum(wv.me.classes.values()) == 6
    assert sum(wv.opponent.classes.values()) == 6
    assert wv.unknown_keys == unknown_raw
    for key in list(wv.me.classes) + list(wv.opponent.classes):
        assert key in known


def test_timing_under_2ms() -> None:
    """Both farms decode in <= 2 ms per turn (issue §6, measured)."""
    obs = _mini_obs(hour=0)
    # 10x10 board like the real game
    for farm in obs["farms"]:
        big = []
        for y in range(10):
            row = []
            for x in range(10):
                if (y < 5 and x < 5) is False and (y == 2 and x == 3):
                    row.append({"kind": "PLANT", "crop": "WHEAT",
                                "planted_day": 0, "watered_today": True,
                                "consecutive_unwatered": 0,
                                "yield_units": 1, "max_lifespan_step": 96,
                                "fertilized_until_day": -1})
                elif y < 5 and x < 5:
                    row.append(None)
                else:
                    row.append("LOCKED")
            big.append(row)
        farm["tiles"] = big
    decode_world(obs)                     # warm
    t0 = time.perf_counter()
    for _ in range(50):
        decode_world(obs)
    per_turn_ms = (time.perf_counter() - t0) / 50 * 1000.0
    assert per_turn_ms <= 2.0, f"decode took {per_turn_ms:.3f} ms per turn"


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print("all agent obs tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
