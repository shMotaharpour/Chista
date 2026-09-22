"""tile_dp tests: the merged tile graph.

Run:  .venv/bin/python -m tests.test_tile_dp

Contracts under test:
- TileState decode/pack round-trips (carrot + ongoing/animal dims).
- The carrot graph is engine truth: honest-yield calendars come out of its
  edges (3 by day-3 start no-fert; WATER,HARVEST = 4 with fert).
- Chain shape: every op is a known op; inside one tile episode (a DIG, or a PLANT
  after a HARVEST, starts a new one) no op repeats, WATER precedes HARVEST (F009)
  and a PLACE follows its BUILD.
- Pruning: the only product-less self-loop a node keeps is the idle day (#84);
  dominated edges absent (no FERTILIZE->HARVEST with bare-HARVEST production on
  fert-less states).
- Dominance is componentwise on BOTH vectors (2026-09-14): a difference in one
  cost or one produce component keeps both edges; only a componentwise-<= cost
  with a componentwise->= produce prunes. Nothing is netted.
- CARE is never offered without FEED (a no-op on its own).
- Rescue watering on consec=1 states exists (F002 second-night rule).
- vocabulary/cost contracts: 18 resource names with no duplicate, NO_ACTION
  only ever a whole chain, no chain longer than 24 labour hours, and cost /
  produce as two separate 18-int vectors per edge.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.tile_dp.chains import CHAIN_NAMES, NO_ACTION, TILE_OPS
from agent.world.model import RESOURCE_NAMES, TileKind
from offline_lab.build.chains import chains_for, domain_ok
from offline_lab.build.graph import (_act, _dominates, _exec_chain, _new_sim,
                                     _tile_and_day, build_graph)
from offline_lab.build.ledger import chain_labor
from agent.world.model import RES_WHEAT
from agent.world.model import RES_FERTILIZER
from agent.world.model import RES_LABOR
from agent.world.model import RESOURCE_ID
from agent.world.model import N_RESOURCE
from agent.artifact import artifact_path
from agent.tile_dp.graph import Edge, TileGraph
from agent.tile_dp.tile_state import KIND_NONE, TileState, decode_tile

# The resource names this file compares vectors on, as `world.model` spells them.
RES_CARROT = "CARROT"
RES_MELON = "MELON"
RES_SEED_CARROT = "SEED_CARROT"
RES_SEED_WHEAT = "SEED_WHEAT"

_CARROT: TileGraph | None = None


def _carrot() -> TileGraph:
    """The carrot graph: built once per run, shared by the tests that need it
    (owner's item 13: five independent builds cost the suite ~0.8 s)."""
    global _CARROT
    if _CARROT is None:
        _CARROT = build_graph("CARROT")
    return _CARROT


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
    """The op order inside a chain is an order the engine can run in one day.

    Two invariants (the WATER-before-HARVEST rule of the old version is the
    second one, and it stays true only until a DIG separates the two): every
    chain is drawn from one canonical op list without repeating an op, and
    WATER comes before HARVEST while no DIG stands between them - watering the
    plant you harvest the same day is what the +1 yield of F009 pays for, while
    (HARVEST, DIG, PLANT, WATER) waters the NEXT plant (owner's item 4).
    """
    for ops in CHAIN_NAMES:
        if not ops:
            continue
        # Every op is a known worker or market op.
        for op in ops:
            assert op in TILE_OPS, (ops, op)
        # Split the chain into one-tile episodes: a DIG starts a new tile (the old
        # plant is gone, a new one may be planted and watered), and so does a PLANT
        # after a HARVEST - a one-shot crop is cleared by its own harvest
        # (kaggriculture.py:464-468), so the tile is bare again and the day's
        # canonical order restarts there, exactly as it does after a DIG.
        segments: list[list[str]] = [[]]
        for op in ops:
            if op == "DIG":
                segments.append([])
                continue
            if op == "PLANT" and "HARVEST" in segments[-1]:
                segments.append([])
            segments[-1].append(op)
        for seg in segments:
            # No op repeats inside one segment (one water per plant day...).
            assert len(set(seg)) == len(seg), ops
            # WATER waters the plant it harvests: it precedes HARVEST - the
            # +1 yield of F009 (owner's item 4). (HARVEST, DIG, PLANT, WATER)
            # waters the NEXT plant and is legal because the DIG separates.
            if "WATER" in seg and "HARVEST" in seg:
                assert seg.index("WATER") < seg.index("HARVEST"), ops
            # PLACE lands an animal in a structure: either the same day's
            # BUILD (build before place) or the tile's own empty structure.
            if "PLACE" in seg:
                place = seg.index("PLACE")
                if "BUILD" in seg:
                    assert seg.index("BUILD") < place, ops
        # One PLANT per chain even across DIGs: the registry never replants
        # twice in one day (two plants would need two seeds and two hours
        # beyond the model's one-crop-per-day shape).
        assert sum(op == "PLANT" for op in ops) <= 1, ops


def test_young_plant_cannot_harvest() -> None:
    """A young plant is not harvested, and nothing is planted into an occupied
    tile: a chain may only PLANT after it DIGs that tile free (owner's item 4).
    """
    young = chains_for(TileKind.PLANT, -1)
    assert all("HARVEST" not in c for c in young)
    assert all(not c or c[0] != "PLANT" for c in young)
    assert any(c and c[0] == "DIG" and "PLANT" in c for c in young)


def test_registry_contracts() -> None:
    """The registry's own invariants (brief part 2, item 1)."""
    assert len(RESOURCE_NAMES) == 18
    assert len(set(RESOURCE_NAMES)) == 18
    assert [c for c in CHAIN_NAMES if c == NO_ACTION] == [NO_ACTION]
    assert all(NO_ACTION not in c for c in CHAIN_NAMES)   # only ever a whole chain
    assert max(chain_labor(c) for c in CHAIN_NAMES) <= 24
    # CARE without FEED is a no-op: it must not even be a registry entry
    assert not [c for c in CHAIN_NAMES if "CARE" in c and "FEED" not in c]
    assert ("FEED", "CARE") in CHAIN_NAMES      # the legal pair stays


def _edge(to_id: int, cost: dict[str, int], produce: dict[str, int]) -> Edge:
    """Synthetic edge for the dominance unit tests (order = RESOURCE_ID)."""
    c = [0] * N_RESOURCE
    p = [0] * N_RESOURCE
    for name, units in cost.items():
        c[RESOURCE_ID[name]] = units
    for name, units in produce.items():
        p[RESOURCE_ID[name]] = units
    return Edge(0, to_id, 0, 0, tuple(c), tuple(p))


def test_dominance_compares_components() -> None:
    """A difference in ONE component keeps both edges (2026-09-14): 1 wheat is
    not 1 melon, a carrot seed is not a wheat seed, one collected fertilizer is
    not nothing. Only componentwise-<= cost with componentwise->= produce prunes.
    """
    t = 7
    wheat = _edge(t, {RES_LABOR: 1}, {RES_WHEAT: 1})
    melon = _edge(t, {RES_LABOR: 1}, {RES_MELON: 1})
    assert not _dominates(wheat, melon) and not _dominates(melon, wheat)
    cseed = _edge(t, {RES_SEED_CARROT: 1}, {RES_CARROT: 1})
    wseed = _edge(t, {RES_SEED_WHEAT: 1}, {RES_WHEAT: 1})
    assert not _dominates(cseed, wseed) and not _dominates(wseed, cseed)
    collect = _edge(t, {RES_LABOR: 1}, {RES_FERTILIZER: 1})
    idle = _edge(t, {}, {})
    assert not _dominates(idle, collect) and not _dominates(collect, idle)
    assert _dominates(idle, _edge(t, {RES_LABOR: 1}, {}))          # cheaper
    assert _dominates(_edge(t, {RES_LABOR: 1}, {RES_WHEAT: 2}),    # more yield
                      _edge(t, {RES_LABOR: 1}, {RES_WHEAT: 1}))


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
    g = _carrot()
    n = _find(g, kind="PLANT", age=1, consec=0, fert_left=2, yield_units=3)
    assert _prod_of(g, n, ("WATER", "HARVEST")) == 4
    n2 = _find(g, kind="PLANT", age=1, consec=0, fert_left=0, yield_units=2)
    assert _prod_of(g, n2, ("WATER", "HARVEST")) == 3


def test_dry_consec1_pass_dies() -> None:
    """consec=1 + PASS → the plant is gone next morning (F002)."""
    g = _carrot()
    for i in range(g.n_states):
        s = g.state_of(i)
        if s.kind == "PLANT" and s.consec == 1:
            for edge in g.edges_from(i):
                if edge.ops == (NO_ACTION,):
                    nxt = g.state_of(edge.to_id)
                    assert nxt.kind != "PLANT", (s.describe(), nxt.describe())


def test_rescue_watering_exists() -> None:
    """Watering TODAY on a consec=1 plant saves it (only the SECOND dry
    night kills): at least one consec=1 state must keep a WATER edge."""
    g = _carrot()
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
    choice the day prices."""
    g = _carrot()
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


def test_the_idle_day_is_kept_where_the_day_cannot_move_the_tile() -> None:
    """A product-less self-loop is a no-op, and the idle day is the exception (#84):
    it is the decision to decline. So the states a day cannot move keep their empty
    chain, and no OTHER product-less self-loop may exist (a 2-hour DIG+BUILD of the
    tile's own structure is not one).
    """
    g = _carrot()
    stuck = [i for i in range(g.n_states)
             if any(e.to_id == i for e in g.edges_from(i) if e.ops == NO_ACTION)]
    assert {g.state_of(i).kind for i in stuck} == {TileKind.NONE, TileKind.WEED}, (
        f"the idle day is missing where a day cannot move the tile: {stuck}")
    for i in range(g.n_states):
        for edge in g.edges_from(i):
            if edge.to_id != i or any(edge.produce):
                continue
            assert edge.ops == NO_ACTION, (
                f"product-less self-loop on state {i}: {edge.name}")
            assert not any(edge.cost), f"the idle day is not free on state {i}"


def test_chain_one_day_contract() -> None:
    """NO_ACTION costs 0 hours; every daily chain fits exactly one day.

    The chain is executed from the bare-tile state (a fresh sim), so the ops
    that need a plant/an animal are engine no-ops here - what is under test is
    the day boundary, and the successor assertion that has to accept them.
    """
    assert chain_labor((NO_ACTION,)) == 0
    assert chain_labor(("PLANT", "WATER")) == 2
    assert chain_labor(("BUILD_COOP", "PLACE", "FEED")) == 3
    bare = TileState(KIND_NONE, None, None, None, 0, 0, 0, 0, 0, 0)
    todo = [c for c in list(chains_for(TileKind.NONE)) + list(chains_for(TileKind.PLANT, 1))
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


_SHIPPED: TileGraph | None = None


def _shipped() -> TileGraph:
    """The merged graph the agent actually ships, loaded — not rebuilt.

    These four guards are about the SHAPE of the merged graph: two int vectors
    of `N_RESOURCE`, a FEED that eats a wheat without producing one, a
    COLLECT_FERTILIZER that produces, and no CARE without a FEED beside it.
    Every one of them is a property of the artifact, and the artifact is what
    the agent loads at run time — so reading it is not a weaker check than
    rebuilding, it is the check that matches what ships.

    Rebuilding cost 235 s of a 563 s suite (42 % of the whole run) and bought
    nothing these assertions could not make in 1.35 s, because nothing here
    ever compared the rebuild against the artifact. The builder is still
    covered: `test_build_is_deterministic`, the seed-invariance pair and the
    calendars all drive `build_graph` directly, on the cheap per-entity graphs.

    What is NOT covered by anything, before or after this change, is whether
    the shipped artifact is what today's builder would produce. That needs a
    test of its own and it is not this one.
    """
    global _SHIPPED
    if _SHIPPED is None:
        _SHIPPED = TileGraph.load(artifact_path("tile_graph", ".npz"))
    return _SHIPPED


def _all_edges(g: TileGraph):
    for sid in range(g.n_states):
        yield from g.edges_from(sid)


def test_merged_vectors_are_two_18_vectors() -> None:
    """The cost and produce vectors are separate, int, 18 entries long."""
    g = _shipped()
    assert g.edge_cost.shape == (g.n_edges, N_RESOURCE)
    assert g.edge_produce.shape == (g.n_edges, N_RESOURCE)
    assert g.edge_cost.dtype == np.int32
    assert g.edge_produce.dtype == np.int32
    assert g.n_edges == sum(1 for _ in _all_edges(g))
    # A chain that spends one resource and produces another keeps the two vectors
    # apart: FEED eats a wheat and the animal's own product is what comes back.
    wheat = RESOURCE_ID[RES_WHEAT]
    fed = [e for e in _all_edges(g) if e.ops and "FEED" in e.ops]
    assert fed, "no edge feeds an animal"
    assert all(e.cost[wheat] == 1 for e in fed)
    assert all(e.produce[wheat] == 0 for e in fed)


@pytest.mark.xfail(reason="the cross-domain layering gap (#89): no PLANT-state edge carries "
                          "a BUILD, so no edge both eats and harvests wheat", strict=True)
def test_an_edge_can_both_eat_and_harvest_wheat() -> None:
    """The strongest witness that the vectors are never netted: one edge that spends a
    wheat on FEED and harvests wheat on the same day, which only a chain that harvests the
    crop and then builds on the freed tile can have. It is missing from the graph (#89),
    and this test is what caught it.
    """
    g = _shipped()
    wheat = RESOURCE_ID[RES_WHEAT]
    assert [e for e in _all_edges(g)
            if e.cost[wheat] > 0 and e.produce[wheat] > 0]


def test_fert_collect_edges_survive() -> None:
    """F023: one COLLECT_FERTILIZER per animal per day is real produce, so the
    shipped graph must carry edges whose fertilizer produce is 1."""
    g = _shipped()
    fert = RESOURCE_ID[RES_FERTILIZER]
    collect = [e for e in _all_edges(g)
               if "COLLECT_FERTILIZER" in e.ops and e.produce[fert] > 0]
    assert collect, "no edge collects fertilizer"
    assert all(e.produce[fert] == 1 for e in collect)


def test_no_care_without_feed_in_graph() -> None:
    """No edge may run CARE without FEED in the same chain (a one-hour no-op)."""
    g = _shipped()
    bad = [e.ops for e in _all_edges(g)
           if "CARE" in e.ops and "FEED" not in e.ops]
    assert not bad, bad[:3]


def test_build_is_deterministic() -> None:
    """Two builds of the same restricted graph agree edge for edge.

    The build walks a seeded sim and a deterministic BFS, so a second run
    must intern the same states in the same order and produce the same CSR
    arrays; anything else means hidden state (an RNG read, dict ordering)
    leaked into the artifact. Byte-level: the packed keys and the cost /
    produce matrices, not just the counts.
    """
    a = build_graph("CARROT")
    b = build_graph("CARROT")
    assert (a.state_keys == b.state_keys).all()
    assert (a.edge_offsets == b.edge_offsets).all()
    assert (a.edge_next == b.edge_next).all()
    assert (a.edge_chain == b.edge_chain).all()
    assert (a.edge_entity == b.edge_entity).all()
    assert (a.edge_cost == b.edge_cost).all()
    assert (a.edge_produce == b.edge_produce).all()
    assert (a.edge_steps == b.edge_steps).all()


def _graph_arrays(g: TileGraph) -> tuple:
    """The artifact's own arrays, in a fixed order, for byte comparison."""
    return (g.state_keys, g.edge_offsets, g.edge_next, g.edge_chain,
            g.edge_entity, g.edge_cost, g.edge_produce, g.edge_steps)


_ARRAY_NAMES = ("state_keys", "edge_offsets", "edge_next", "edge_chain",
                "edge_entity", "edge_cost", "edge_produce", "edge_steps")


def _sim_factory(seed: int, weed_spawn: float):
    """A drop-in `graph._new_sim` with another seed (and optionally weeds)."""
    from offline_lab.fast_sim import FastSim

    def make() -> FastSim:
        return FastSim({"episodeSteps": 30 * 24, "seed": seed,
                        "weedSpawnChance": weed_spawn})
    return make


def _build_with(factory, entity: str) -> TileGraph:
    """Run the SHIPPED builder under another sim factory.

    `_new_sim` is a module-level constant (one 30-day sim, seed 4242, no
    weeds). Swapping it re-runs the real `build_graph` - the same BFS, the
    same executor, the same assertions - under a different RNG seed,
    instead of re-implementing the build beside it.
    """
    from offline_lab.build import graph as G

    real = G._new_sim
    G._new_sim = factory
    try:
        return G.build_graph(entity)
    finally:
        G._new_sim = real


def _arrays_equal(a, b) -> bool:
    """Byte-compare two CSR columns, tolerating different shapes."""
    return a.shape == b.shape and bool((a == b).all())


def test_seed_invariance_no_rng_in_tile_transitions() -> None:
    """Issue #24 class A: with `weedSpawnChance = 0.0`, NO tile transition
    may depend on the RNG - so the graph must come out byte-identical
    across seeds.

    Every other check in this suite runs at the builder's single hard seed
    (4242), so a transition that silently read the RNG would stay
    consistent with itself. This re-runs the SHIPPED builder at several
    seeds and compares the artifact arrays byte for byte.

    Teeth: two controls inside the test — the same comparison on WHEAT (a
    different graph) must report a difference, and a copy of the graph with
    one edge's produce raised must be rejected on content alone (so an
    equal-shaped comparison that ignores values cannot pass). The RNG's reach
    into tile transitions is measured in `test_weed_rng_reaches_tile_transitions`.
    """
    base = _carrot()
    base_arrays = _graph_arrays(base)
    for seed in (1, 7, 99):
        other = _build_with(_sim_factory(seed, 0.0), "CARROT")
        for name, a, b in zip(_ARRAY_NAMES, base_arrays, _graph_arrays(other)):
            assert _arrays_equal(a, b), (
                f"seed {seed} changed {name}: the no-weed graph must not "
                f"depend on the RNG (issue #24 class A)")
    # Control 1: a DIFFERENT graph must be rejected (different shape/content).
    wheat = build_graph("WHEAT")
    assert not all(_arrays_equal(a, b) for a, b
                   in zip(base_arrays, _graph_arrays(wheat))), (
        "the byte comparison used above cannot tell two different graphs "
        "apart, so its equality proves nothing")
    # Control 2: the CONTENT path alone, on an identically-shaped graph (one
    # edge's produce raised): a control that only exercises the shape mismatch
    # would not notice a comparison that ignores equal-shaped arrays.
    from dataclasses import replace

    ep = base.edge_produce.copy()
    ep[0, 0] = ep[0, 0] + 1
    nudged = replace(base, edge_produce=ep)
    assert not _arrays_equal(base.edge_produce, nudged.edge_produce), (
        "the comparison misses a changed value on an identically-shaped "
        "array, so the seed equality above proves nothing")


def test_weed_rng_reaches_tile_transitions() -> None:
    """The premise the seed-invariance check rests on, measured directly.

    F045 draws weeds from the RNG; the engine seeds that draw per
    (episode seed, day): `random.Random((seed * 1_000_003) ^ day)`
    (kaggriculture.py:871, inside `_end_of_day`). So with
    `weedSpawnChance = 0.0` a bare tile stays bare for every seed - which
    is the only reason the graph can be seed-independent - while a
    non-zero spawn turns some tiles into WEED.

    Measured on this engine, 12 idle days: spawn 0.0 → 0 of 24 seeds show a
    tile (every one still bare); spawn 0.2 → 22 of 24 are WEED. Without this,
    "the graph does not depend on the RNG" is a claim nobody has seen fail.
    """
    def idle(seed: int, spawn: float, days: int = 12):
        from offline_lab.fast_sim import FastSim
        from offline_lab.build.graph import _act, _tile_and_day

        sim = FastSim({"episodeSteps": 30 * 24, "seed": seed,
                       "weedSpawnChance": spawn})
        for _ in range(days * 24):
            sim.step([_act(["PASS"]), _act(["PASS"])])
        tile, _day = _tile_and_day(sim)
        # a bare tile is `None` in the raw engine; the decoder maps it to NONE
        return tile if tile is None else tile.get("kind")

    seeds = tuple(range(24))
    quiet = {seed: idle(seed, 0.0) for seed in seeds}
    assert set(quiet.values()) == {None}, (
        f"a tile changed with weeds off, so the builder's own premise is "
        f"wrong: {quiet}")
    loud = {seed: idle(seed, 0.2) for seed in seeds}
    assert any(kind == "WEED" for kind in loud.values()), (
        f"a non-zero weedSpawnChance changed no tile in {loud}, so the RNG "
        f"does not reach tile transitions and the invariance claim above is "
        f"vacuous")


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
