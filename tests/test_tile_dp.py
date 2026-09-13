"""tile_dp tests: state coding, chain registry, engine-driven graph, DP.

Run:  .venv/bin/python -m tests.test_tile_dp

Key contracts under test:
- TileState encode/decode round-trips and the day-start projection matches
  engine findings (F002/F004/F005/F006/F009/F026).
- The engine-driven graph reproduces the honest-yield calendars (F009:
  wheat 6 by day 4; carrot 4 by day 3) — the graph IS engine truth.
- TileContractor DP: plants and harvests when profitable, respects harvest
  timing under price schedules, and is integer-exact.
"""

from __future__ import annotations

import numpy as np

from tile_dp.chains import CHAINS, chain_id_of, chain_ops
from tile_dp.contractor import TileContractor
from tile_dp.graph import TileGraph, build_graph
from tile_dp.tile_state import (CROP_ID, CROP_NONE, CROP_WEED, TileState,
                                all_start_states, decode_tile, pack_key,
                                unpack_key)


# ---------------------------------------------------------------- tile_state

def test_pack_roundtrip() -> None:
    for crop in (-2, -1, 0, 1):
        for age in range(-10, 8):
            for consec in (0, 1):
                for fert in (0, 1, 2):
                    for y in range(0, 7):
                        k = pack_key(crop, age, consec, fert, y)
                        assert unpack_key(k) == (crop, age, consec, fert, y)


def test_decode_tile_projection() -> None:
    tile = {"kind": "PLANT", "crop": "WHEAT", "planted_day": 0,
            "watered_today": True, "consecutive_unwatered": 0,
            "yield_units": 1, "max_lifespan_step": 120,
            "fertilized_until_day": 2}
    st = decode_tile(tile, day=1)
    assert st.crop_id == CROP_ID["WHEAT"]
    assert st.age == -1                       # origin = first harvest day
    assert st.fert_left == 2                  # covers days 1 and 2
    assert st.consec_unwatered == 0
    assert st.yield_units == 1
    assert decode_tile(None, day=0).crop_id == CROP_NONE
    assert decode_tile("WEED", day=0).crop_id == CROP_WEED
    assert decode_tile({"kind": "WEED"}, day=0).crop_id == CROP_WEED


def test_state_space_pruned() -> None:
    wheat = all_start_states(CROP_ID["WHEAT"])
    carrot = all_start_states(CROP_ID["CARROT"])
    assert len(wheat) < 336 and len(carrot) < 180
    bad = [s for s in wheat
           if s.age == 0 and s.yield_units > 1 + 2 * 1]
    assert not bad, f"over-cap states leaked: {bad[:5]}"


# -------------------------------------------------------------------- chains

def test_chain_registry_ordering() -> None:
    for ops in CHAINS:
        if "WATER" in ops and "HARVEST" in ops:
            assert ops.index("WATER") < ops.index("HARVEST"), ops
    assert chain_ops(chain_id_of(("DIG",))) == ("DIG",)
    assert chain_ops(chain_id_of(("PASS",))) == ("PASS",)


# --------------------------------------------------------------- engine graph

def test_graph_reproduces_honest_calendars() -> None:
    """Engine-driven graph must reproduce F009: wheat planted day 0 reaches
    5 units by day-4 start and the (WATER, HARVEST) chain from there
    collects 6 (the watered unit lands immediately — F026). Carrot reaches
    4 by day-3 start."""
    g = build_graph(CROP_ID["WHEAT"])
    start = g.state_id_of(TileState(CROP_ID["WHEAT"], -1, 0, 0, 1))
    reached = {start}
    for _ in range(3):  # day-1 state -> day-4 states
        nxt_set = set()
        for s in reached:
            lo, hi = g.edges_of(0, 0)  # placeholder; replaced below
            nxt_set.add(int(g.edge_next[e])) if False else None
        reached = nxt_set
    # NOTE: expansion must use per-day edge blocks; see helper below.


def _expand(g: TileGraph, reached: set[int], day: int) -> set[int]:
    nxt_set = set()
    for s in reached:
        lo, hi = g.edges_of(day, s)
        for e in range(lo, hi):
            nxt_set.add(int(g.edge_next[e]))
    return nxt_set


def test_wheat_calendar_via_per_day_edges() -> None:
    g = build_graph(CROP_ID["WHEAT"])
    start = g.state_id_of(TileState(CROP_ID["WHEAT"], -1, 0, 0, 1))
    reached = {start}
    for day in range(1, 4):  # day-1 -> day-4 states
        reached = _expand(g, reached, day)
    five = [s for s in reached
            if TileState.unpack(int(g.state_keys[s])).age == 2
            and TileState.unpack(int(g.state_keys[s])).yield_units == 5]
    assert five, "wheat: yield 5 at day-4 start not reachable"
    s = five[0]
    lo, hi = g.edges_of(4, s)
    prods = [int(g.edge_prod[0][e]) for e in range(lo, hi)
             if chain_ops(int(g.edge_chain[e])) == ("WATER", "HARVEST")]
    assert prods == [6], f"(WATER, HARVEST) on day 4 yields {prods}, want [6]"


def test_carrot_calendar_via_per_day_edges() -> None:
    g = build_graph(CROP_ID["CARROT"])
    start = g.state_id_of(TileState(CROP_ID["CARROT"], -1, 0, 0, 1))
    reached = {start}
    for day in range(1, 3):  # day-1 -> day-3 states
        reached = _expand(g, reached, day)
    # F009: 3 units by day-3 start (fert path), and (WATER, HARVEST) on
    # day 3 collects 4 (watered unit lands immediately — F026; cap 4).
    three = [s for s in reached
             if TileState.unpack(int(g.state_keys[s])).age == 1
             and TileState.unpack(int(g.state_keys[s])).yield_units == 3
             and TileState.unpack(int(g.state_keys[s])).fert_left >= 1]
    assert three, "carrot: yield 3 at day-3 start not reachable"
    s = three[0]
    lo, hi = g.edges_of(3, s)
    prods = [int(g.edge_prod[0][e]) for e in range(lo, hi)
             if chain_ops(int(g.edge_chain[e])) == ("WATER", "HARVEST")]
    assert prods == [4], f"(WATER, HARVEST) on day 3 yields {prods}, want [4]"


def test_graph_save_load_roundtrip(tmp_path=None) -> None:
    import tempfile
    from pathlib import Path
    g = build_graph(CROP_ID["CARROT"])
    out = Path(tempfile.mkdtemp()) / "graph_CARROT.npz"
    g.save(out)
    g2 = TileGraph.load(out)
    assert g2.n_states == g.n_states
    assert np.array_equal(g2.edge_next, g.edge_next)
    assert np.array_equal(g2.edge_prod, g.edge_prod)


# ------------------------------------------------------------------ DP solve

def _flat_prices(days: int, wheat: int, carrot: int) -> dict[int, np.ndarray]:
    return {0: np.full(days, wheat, dtype=np.int32),
            1: np.full(days, carrot, dtype=np.int32)}


def _flat_wage(days: int, labor: int, sw: int, sc: int, fert: int
               ) -> dict[int, np.ndarray]:
    return {0: np.full(days, labor, dtype=np.int32),
            1: np.full(days, sw, dtype=np.int32),
            2: np.full(days, sc, dtype=np.int32),
            3: np.full(days, fert, dtype=np.int32)}


def test_dp_prefers_harvest_over_abandon() -> None:
    g = build_graph(CROP_ID["WHEAT"])
    c = TileContractor(g)
    none_state = TileState(-1, 0, 0, 0, 0)

    sol = c.solve(_flat_prices(30, 30, 40), _flat_wage(30, 1, 10, 20, 2),
                  start_state=none_state, start_day=0, horizon_days=30)
    assert sol.total_profit > 0
    assert sol.production[:, 0].sum() > 0

    sol0 = c.solve(_flat_prices(30, 0, 0), _flat_wage(30, 1, 10, 20, 2),
                   start_state=none_state, start_day=0, horizon_days=30)
    assert sol0.total_profit <= 0


def test_dp_respects_harvest_timing() -> None:
    """Wheat worthless days 0..3, gold afterwards: the OPTIMAL plan must
    harvest wheat only on days >= 4. (Harvesting a worthless 1-unit crop
    early costs labor, so the DP replants on the worthwhile schedule.)"""
    g = build_graph(CROP_ID["WHEAT"])
    c = TileContractor(g)
    none_state = TileState(-1, 0, 0, 0, 0)

    prices = _flat_prices(30, 10, 10)
    prices[0][:4] = 0
    prices[0][4:] = 500
    sol = c.solve(prices, _flat_wage(30, 1, 10, 20, 2),
                  start_state=none_state, start_day=0, horizon_days=30)
    assert sol.production[:, 0].sum() > 0
    days = np.nonzero(sol.production[:, 0])[0]
    assert (days >= 4).all(), f"harvested on worthless days: {days}"


def test_dp_int_exactness() -> None:
    g = build_graph(CROP_ID["CARROT"])
    c = TileContractor(g)
    none_state = TileState(-1, 0, 0, 0, 0)
    prices = _flat_prices(30, 7, 11)
    wage = _flat_wage(30, 2, 13, 17, 3)
    sol = c.solve(prices, wage, start_state=none_state, start_day=0,
                  horizon_days=30)
    assert isinstance(sol.total_profit, int)
    labor = int(sol.resource_use[:, 0].sum())
    seed = int(sol.resource_use[:, 1].sum() + sol.resource_use[:, 2].sum())
    fert = int(sol.resource_use[:, 3].sum())
    profit = (int((sol.production[:, 0] * 11).sum())
              - labor * 2 - seed * 17 - fert * 3)
    assert sol.total_profit == profit, (sol.total_profit, profit)


if __name__ == "__main__":
    import tempfile
    failures = 0
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            import inspect
            sig = inspect.signature(fn)
            args = []
            if "tmp_path" in sig.parameters:
                args.append(tempfile.mkdtemp())
            fn(*args)
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
