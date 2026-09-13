"""tile_dp tests: carrot daily state-action graph (v2 generalized).

Run:  .venv/bin/python -m tests.test_tile_dp

Contracts under test:
- TileState decode/pack round-trips (carrot + ongoing/animal dims).
- The carrot graph is engine truth: honest-yield calendars come out of
  its edges (3 by day-3 start no-fert; WATER,HARVEST = 4 with fert).
- Pruning: no zero-cost self-loops; dominated edges absent (no
  FERTILIZE->HARVEST with bare-HARVEST production on fert-less states).
- Rescue watering on consec=1 states exists (F002 second-night rule).
"""

from __future__ import annotations

from tile_dp.chains import CHAIN_NAMES, chain_ops, chains_for
from tile_dp.graph import TileGraph, build_graph
from tile_dp.tile_state import TileState, decode_tile


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


def _prod_of(g: TileGraph, sid: int, chain) -> int | None:
    lo, hi = g.edges_of(sid)
    for e in range(lo, hi):
        if chain_ops(int(g.edge_chain[e])) == chain:
            return int(g.edge_prod[e])
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
            lo, hi = g.edges_of(i)
            for e in range(lo, hi):
                cid = int(g.edge_chain[e])
                if chain_ops(cid) == ("PASS",):
                    nxt = g.state_of(int(g.edge_next[e]))
                    assert nxt.kind != "PLANT", (s.describe(), nxt.describe())


def test_rescue_watering_exists() -> None:
    """Watering TODAY on a consec=1 plant saves it (only the SECOND dry
    night kills): at least one consec=1 state must keep a WATER edge."""
    g = build_graph("CARROT")
    saved = 0
    for i in range(g.n_states):
        s = g.state_of(i)
        if s.kind == "PLANT" and s.consec == 1:
            lo, hi = g.edges_of(i)
            if any(chain_ops(int(g.edge_chain[e])) in
                   (("WATER",), ("FERTILIZE", "WATER"))
                   for e in range(lo, hi)):
                saved += 1
    assert saved >= 1


def test_dominated_fert_harvest_absent() -> None:
    """(FERTILIZE, HARVEST) without water wastes the fertilizer: on a
    fert-less state it must be pruned by the dominance filter."""
    g = build_graph("CARROT")
    for i in range(g.n_states):
        s = g.state_of(i)
        if s.kind != "PLANT" or s.fert_left != 0:
            continue
        lo, hi = g.edges_of(i)
        chains = [chain_ops(int(g.edge_chain[e])) for e in range(lo, hi)]
        if ("HARVEST",) in chains and ("FERTILIZE", "HARVEST") in chains:
            p_bare = _prod_of(g, i, ("HARVEST",))
            p_fert = _prod_of(g, i, ("FERTILIZE", "HARVEST"))
            assert p_fert > p_bare, (
                f"dominated edge kept on {s.describe()}")


def test_no_zero_cost_self_loops() -> None:
    g = build_graph("CARROT")
    for i in range(g.n_states):
        lo, hi = g.edges_of(i)
        for e in range(lo, hi):
            nid = int(g.edge_next[e])
            total_use = sum(int(g.edge_use[r][e]) for r in range(3))
            assert not (nid == i and total_use == 0
                        and int(g.edge_prod[e]) == 0), (
                f"zero-cost self-loop on state {i}")


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
