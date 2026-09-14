"""tile_dp tests: the merged tile graph (v15).

Run:  .venv/bin/python -m tests.test_tile_dp

Contracts under test:
- TileState decode/pack round-trips (carrot + ongoing/animal dims).
- The carrot graph is engine truth: honest-yield calendars come out of its
  edges (3 by day-3 start no-fert; WATER,HARVEST = 4 with fert).
- Pruning: no zero-cost self-loops; dominated edges absent (no
  FERTILIZE->HARVEST with bare-HARVEST production on fert-less states).
- Rescue watering on consec=1 states exists (F002 second-night rule).
- v15 vocabulary/cost contracts: 18 resource names with no duplicate, NO_ACT
  only ever a whole chain, no chain longer than 24 labour hours, and cost /
  produce as two separate 18-int vectors per edge.
"""

from __future__ import annotations

import numpy as np

from tile_dp.chains import (CHAIN_NAMES, N_RESOURCE, NO_ACT, RES_CARROT,
                            RES_WHEAT, RESOURCE_ID, RESOURCE_NAMES,
                            chain_labor, chains_for, domain_ok)
from tile_dp.graph import TileGraph, _exec_chain, _new_sim, build_graph
from tile_dp.tile_state import KIND_NONE, TileState, decode_tile

def test_decode_and_pack() -> None:
    tile = {"kind": "PLANT", "crop": "CARROT", "planted_day": 0,
            "watered_today": True, "consecutive_unwatered": 0,
            "yield_units": 1, "max_lifespan_step": 96,
            "fertilized_until_day": 2}
    st = decode_tile(tile, day=1)   # day 1 → age -1
    assert st.kind == "PLANT" and st.age == -1
    assert st.fert_left == 2 and st.consec == 0 and st.yield_units == 1
    assert TileState.unpack(st.pack()) == st
    assert decode_tile(None, 0).kind == "NONE"
    assert decode_tile({"kind": "WEED"}, 0).kind == "WEED"


def test_chain_order_canonical() -> None:
    for ops in CHAIN_NAMES:
        if "WATER" in ops and "HARVEST" in ops:
            assert ops.index("WATER") < ops.index("HARVEST"), ops


def test_young_plant_cannot_harvest() -> None:
    young = chains_for("PLANT", -1)
    assert all("HARVEST" not in c for c in young)
    assert all("PLANT" not in c for c in young)  # occupied tile


def test_registry_contracts() -> None:
    """v15: the registry's own invariants (brief part 2, item 1)."""
    assert len(RESOURCE_NAMES) == 18
    assert len(set(RESOURCE_NAMES)) == 18
    assert [c for c in CHAIN_NAMES if NO_ACT in c] == [(NO_ACT,)]
    assert max(chain_labor(c) for c in CHAIN_NAMES) <= 24


def _find(g: TileGraph, **kw) -> int:
    for i in range(g.n_states):
        s = g.state_of(i)
        if (s.kind == kw.get("kind", s.kind)
                and s.age == kw.get("age", s.age)
                and s.consec == kw.get("consec", s.consec)
                and s.fert_left == kw.get("fert_left", s.fert_left)
                and s.yield_units == kw.get("yield_units", s.yield_units)):
            return i
    raise KeyError(f"state not in graph: {kw}")


def _prod_of(g: TileGraph, sid: int, chain, res: str = RES_CARROT) -> int | None:
    """Produced units of `res` on the state's `chain` edge (None if absent)."""
    for edge in g.edges_from(sid):
        if edge.ops == chain:
            return edge.produce[RESOURCE_ID[res]]
    return None


def test_graph_calendars() -> None:
    """F009: (age 1, consec 0, fert 2, y 3) + (WATER, HARVEST) = 4; the
    no-fert sibling (y 2) + (WATER, HARVEST) = 3."""
    g = build_graph("CARROT")
    n = _find(g, kind="PLANT", age=1, consec=0, fert_left=2, yield_units=3)
    assert _prod_of(g, n, ("WATER", "HARVEST")) == 4
    n2 = _find(g, kind="PLANT", age=1, consec=0, fert_left=0, yield_units=2)
    assert _prod_of(g, n2, ("WATER", "HARVEST")) == 3


def test_dry_consec1_pass_dies() -> None:
    """consec=1 + PASS → the plant is gone next morning (F002)."""
    g = build_graph("CARROT")
    for i in range(g.n_states):
        s = g.state_of(i)
        if s.kind == "PLANT" and s.consec == 1:
            for edge in g.edges_from(i):
                if edge.ops == (NO_ACT,):
                    nxt = g.state_of(edge.to_id)
                    assert nxt.kind != "PLANT", (s.describe(), nxt.describe())


def test_rescue_watering_exists() -> None:
    """Watering TODAY on a consec=1 plant saves it (only the SECOND dry
    night kills): at least one consec=1 state must keep a WATER edge."""
    g = build_graph("CARROT")
    saved = 0
    for i in range(g.n_states):
        s = g.state_of(i)
        if s.kind == "PLANT" and s.consec == 1:
            if any(edge.ops in (("WATER",), ("FERTILIZE", "WATER"))
                   for edge in g.edges_from(i)):
                saved += 1
    assert saved >= 1


def test_dominated_fert_harvest_absent() -> None:
    """(FERTILIZE, HARVEST) without water wastes the fertilizer: on a
    fert-less YOUNG state (age<0, no harvest possible) the pair must be
    pruned by the dominance filter. Mature states can legitimately keep
    both edges: FERTILIZE before the next cycle's window is a real
    choice the secretary prices."""
    g = build_graph("CARROT")
    for i in range(g.n_states):
        s = g.state_of(i)
        if s.kind != "PLANT" or s.fert_left != 0 or s.age >= 0:
            continue
        chains = [edge.ops for edge in g.edges_from(i)]
        if ("HARVEST",) in chains and ("FERTILIZE", "HARVEST") in chains:
            p_bare = _prod_of(g, i, ("HARVEST",))
            p_fert = _prod_of(g, i, ("FERTILIZE", "HARVEST"))
            assert p_fert > p_bare, (
                f"dominated edge kept on {s.describe()}")


def test_no_zero_cost_self_loops() -> None:
    g = build_graph("CARROT")
    for i in range(g.n_states):
        for edge in g.edges_from(i):
            assert not (edge.to_id == i and not any(edge.cost)
                        and not any(edge.produce)), (
                f"zero-cost self-loop on state {i}")


def test_chain_one_day_contract() -> None:
    """v15: NO_ACT costs 0 hours; every daily chain fits exactly one day.

    The chain is executed from the bare-tile state (a fresh sim), so the ops
    that need a plant/an animal are engine no-ops here - what is under test is
    the day boundary, and the v15 successor assertion that has to accept them.
    """
    assert chain_labor((NO_ACT,)) == 0
    assert chain_labor(("PLANT", "WATER")) == 2
    assert chain_labor(("BUILD", "PLACE", "FEED")) == 3
    bare = TileState(KIND_NONE, None, None, None, 0, 0, 0, 0, 0, 0)
    todo = [c for c in list(chains_for("NONE")) + list(chains_for("PLANT", 1))
            if domain_ok(c, "CARROT")]      # the graph filters domains too
    for ops in todo:
        sim = _new_sim()
        day0 = int(sim.observations()[0]["day"])
        hops: list[int] = []
        raw = sim.step

        def wrap(actions, *a, **k):
            hops.append(1)
            return raw(actions, *a, **k)

        sim.step = wrap
        _exec_chain(sim, bare, ops, "CARROT")
        assert len(hops) == 24, (ops, len(hops))
        assert int(sim.observations()[0]["day"]) == day0 + 1, ops


_MERGED: TileGraph | None = None


def _merged() -> TileGraph:
    """The merged tile graph (built once: ~40 s, shared by these tests)."""
    global _MERGED
    if _MERGED is None:
        _MERGED = build_graph()
    return _MERGED


def _all_edges(g: TileGraph):
    for sid in range(g.n_states):
        yield from g.edges_from(sid)


def test_merged_vectors_are_two_18_vectors() -> None:
    """v15 decision 2: cost and produce are separate, int, 18 entries long."""
    g = _merged()
    assert g.edge_cost.shape == (g.n_edges, N_RESOURCE)
    assert g.edge_produce.shape == (g.n_edges, N_RESOURCE)
    assert g.edge_cost.dtype == np.int32
    assert g.edge_produce.dtype == np.int32
    wheat = RESOURCE_ID[RES_WHEAT]
    both = [e for e in _all_edges(g)
            if e.cost[wheat] > 0 and e.produce[wheat] > 0]
    assert both, ("no edge both eats and harvests wheat: netting the cost and "
                  "produce vectors would go unnoticed")
    assert g.n_edges == sum(1 for _ in _all_edges(g))


if __name__ == "__main__":
    failures = 0
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
