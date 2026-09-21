"""The decline edge: every node keeps its idle day (#84).

Run:  .venv/bin/python -m pytest tests/test_tile_dp_decline.py

The shipped graph is what the DP reads, so it is the artifact under test: a rebuild
that loses a node's decline would ship a graph the planner cannot switch a tile off
with, and the loss would only show up as a plan that spends on a tile it should have
left alone.

The law these guards pin (offline_lab/build/graph.py, `_is_noop_edge`): a self-loop
that produces nothing is a no-op, EXCEPT the empty chain - declining is a decision.
The exception bites in exactly the four day-invariant states (bare, weed, empty coop,
empty pasture): every other node's idle day moves the tile, so its idle edge is not a
self-loop at all.
"""

from __future__ import annotations

import pytest

from agent.artifact import artifact_path
from agent.tile_dp.chains import NO_ACTION
from agent.tile_dp.graph import Edge, TileGraph
from agent.tile_dp.tile_state import (KIND_EMPTY_COOP, KIND_EMPTY_PASTURE, KIND_NONE,
                                      KIND_WEED, TileState)
from agent.world.model import N_RESOURCE, RESOURCE_ID
from offline_lab.build.chains import chain_id_of
from offline_lab.build.graph import (NoDeclineEdge, _assert_decline, _is_noop_edge,
                                     _prune)

_GRAPH: TileGraph | None = None
_ZERO: tuple[int, ...] = tuple([0] * N_RESOURCE)
_BARE = TileState(KIND_NONE, None, None, None, 0, 0, 0, 0, 0, 0)


def _cost(**units: int) -> tuple[int, ...]:
    """An 18-wide cost vector from the resources the test cares about."""
    row = [0] * N_RESOURCE
    for name, value in units.items():
        row[RESOURCE_ID[name]] = value
    return tuple(row)


def _edge(from_id: int, to_id: int, ops: tuple[str, ...],
          cost: tuple[int, ...] | None = None) -> Edge:
    return Edge(from_id, to_id, chain_id_of(ops), 6, cost or _ZERO, _ZERO)


def _graph() -> TileGraph:
    """The shipped graph, loaded once per run."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = TileGraph.load(artifact_path("tile_graph", ".npz"))
    return _GRAPH


def _idle(g: TileGraph, state_id: int) -> list[Edge]:
    """The node's idle-day edges (the empty chain)."""
    return [e for e in g.edges_from(state_id) if e.ops == NO_ACTION]


def _states_of_kind(g: TileGraph, kind) -> list[int]:
    return [s for s in range(g.n_states) if g.state_of(s).kind == kind]


def test_every_node_keeps_its_idle_day() -> None:
    """One decline per node, free and producing nothing."""
    g = _graph()
    for s in range(g.n_states):
        idle = _idle(g, s)
        assert len(idle) == 1, (
            f"{g.state_of(s).describe()}: {len(idle)} idle edges, expected 1")
        edge = idle[0]
        assert not any(edge.cost) and not any(edge.produce), (
            f"{g.state_of(s).describe()}: the idle edge is not free: "
            f"cost {list(edge.cost)}, produce {list(edge.produce)}")


def test_the_idle_day_is_a_self_loop_only_where_the_day_cannot_move_the_tile() -> None:
    """The four day-invariant states are the whole exception, and nothing else is."""
    g = _graph()
    loops = [s for s in range(g.n_states)
             if any(e.to_id == s for e in _idle(g, s))]
    assert {g.state_of(s).kind for s in loops} == {
        KIND_NONE, KIND_WEED, KIND_EMPTY_COOP, KIND_EMPTY_PASTURE}, (
            f"idle self-loops on unexpected kinds: "
            f"{[(s, g.state_of(s).kind) for s in loops]}")
    assert len(loops) == 4, f"one state per kind, got {loops}"
    # Everywhere else the idle day moves the tile, which is why the sweep never saw it.
    for s in range(g.n_states):
        if s in loops:
            continue
        assert all(e.to_id != s for e in _idle(g, s)), (
            f"{g.state_of(s).describe()}: the idle day is a self-loop on a "
            f"day-moving state")


def test_the_same_structure_rebuild_stays_pruned() -> None:
    """A DIG+BUILD of the tile's OWN structure is a 2-hour self-loop with nothing
    produced: the DP never needs it (owner's reading of #84), and the idle edge
    dominates it. So the only product-less self-loop in the graph is the idle chain.
    """
    g = _graph()
    for s in range(g.n_states):
        for edge in g.edges_from(s):
            if edge.to_id == s and not any(edge.produce):
                assert edge.ops == NO_ACTION, (
                    f"{g.state_of(s).describe()}: product-less self-loop {edge.name}")
    # Named where it matters: the two empty structures, whose own rebuild is the case
    # the sweep used to let through (it only caught the zero-cost ones).
    for kind, own in ((KIND_EMPTY_COOP, "BUILD_COOP"),
                      (KIND_EMPTY_PASTURE, "BUILD_PASTURE")):
        for s in _states_of_kind(g, kind):
            for edge in g.edges_from(s):
                if edge.to_id == s:
                    assert edge.ops == NO_ACTION, (
                        f"{g.state_of(s).describe()}: {edge.name} lands on itself")
                assert not (edge.ops and edge.ops[0] == "DIG" and own in edge.ops
                            and edge.to_id == s), (
                    f"{g.state_of(s).describe()}: DIG+{own} kept as a self-loop")


def test_the_sweep_keeps_the_idle_day_and_drops_the_same_structure_rebuild() -> None:
    """The law itself, on two edges of one node - no build, no artifact.

    A pasture that DIGs and rebuilds its own pasture is a 2-hour self-loop: the sweep
    drops it (and dominance would drop it again behind the free idle edge), while the
    idle day survives.
    """
    idle = _edge(7, 7, NO_ACTION)
    rebuild = _edge(7, 7, ("DIG", "BUILD_PASTURE"), _cost(LABOR=2))
    assert not _is_noop_edge(7, idle), "the idle day is a decision, not a no-op"
    assert _is_noop_edge(7, rebuild), "DIG+BUILD on the same structure is a no-op"
    kept, dropped, _ = _prune(7, [idle, rebuild])
    assert kept == [idle] and dropped == 1, kept


def test_a_node_without_a_decline_fails_the_build() -> None:
    """The build's post-condition: a node that kept no free idle edge is refused."""
    # A node that kept its decline passes, whatever else it kept.
    _assert_decline(7, _BARE, [_edge(7, 8, ("PLANT", "WATER"), _cost(LABOR=2)),
                               _edge(7, 7, NO_ACTION)])
    with pytest.raises(NoDeclineEdge):
        _assert_decline(7, _BARE, [_edge(7, 8, ("PLANT", "WATER"), _cost(LABOR=2)),
                                   _edge(7, 8, ("WATER",), _cost(LABOR=1))])
    with pytest.raises(NoDeclineEdge):
        # An idle edge that is not free is not a decline the DP can take for nothing.
        _assert_decline(7, _BARE, [_edge(7, 7, NO_ACTION, _cost(LABOR=1))])
