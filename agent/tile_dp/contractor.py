"""TileContractor — the pricing oracle: a backward DP over the tile graph, plus forward
plan recovery.

The graph (`agent/artifact/tile_graph.npz`) is day-invariant and position-invariant: every
tile, on every day, faces the same state-action graph. One backward sweep therefore prices
the whole board — no per-tile solve and no per-day rebuild. A loop over tiles in the sweep
means that property has been lost.

    V_d(s) = max over edges e out of s of
                 produce(e)·p_d − cost(e)·w_d + V_{d+1}(next(e))

The season ends with no liquidation, so the terminal row of `V` is zero. The argmax is not
stored; plans are recovered forward, for the tiles we own.

Two contracts come with the graph:

- **R006 — non-negativity.** The graph is dominance-pruned, and that pruning preserves the
  optimum only while `p >= 0` and `w >= 0` componentwise. A negative component makes a
  pruned edge the true optimum and this DP silently wrong, so the check runs on entry.
- **No empty edge slices.** `np.maximum.reduceat` returns the element *at* the index for an
  empty group instead of an identity, so a state with no out-edges would price as garbage.
  Checked at construction.

Cost and production are two matrices and are never netted: they are priced by different
vectors (`produce` at prices, `cost` at wages).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from agent.world.rules import TURNS_PER_DAY
from agent.world.model import N_RESOURCE, RESOURCE_ID, RES_LABOR
from agent.tile_dp.graph import TileGraph

# The season's days. Shed goods at the end are worth nothing: there is no liquidation, so
# the horizon's value is exactly zero.
HORIZON_DAYS = 30
# The sweep's arithmetic dtype: the edge matrices are held in it, so numpy does not
# upcast the whole sweep on every call.
DTYPE = np.float32
LABOR_ID = RESOURCE_ID[RES_LABOR]


def _check_non_negative(name: str, values: np.ndarray) -> None:
    """R006's guard: the pruned graph is only optimal for non-negative vectors.

    The failure it prevents is silent, so the check is not skipped for speed.
    """
    if values.size and float(np.min(values)) < 0.0:
        worst = float(np.min(values))
        raise ValueError(
            f"R006: {name} has a negative component ({worst}); the tile graph "
            "is dominance-pruned and that pruning is optimality-preserving "
            "only while every price and wage is >= 0 componentwise. A negative "
            "component makes a pruned edge the true optimum, so this DP would "
            "return a silently wrong plan: clamp the duals onto the non-negative "
            "orthant before pricing.")


@dataclass(frozen=True)
class PricedBoard:
    """What one pricing call gives back.

    `values` is the whole sweep, with the terminal row zero. `columns` and `produce` are one
    row per owned tile, in the order `owned_states` was given, and `plans` are the recovered
    schedules.
    """

    values: np.ndarray            # (days + 1, n_states) float32 — V[d, s]
    reduced_cost: float           # max over owned tiles of V_0(s): the signal
    columns: np.ndarray           # (n_owned, N_RESOURCE) int64 — cost summed
    produce: np.ndarray           # (n_owned, N_RESOURCE) int64 — output summed
    plans: list                   # per owned tile: [(day, state_id, chain_id)]
    tile_values: np.ndarray       # (n_owned,) float64 — V_0 of each owned tile
    days: int
    # Per-day coefficients of each plan, (n_owned, days, N_RESOURCE) int64:
    # labour[d] = per_day_cost[:, :, LABOR_ID], inputs = per_day_cost, output =
    # per_day_produce. `columns` / `produce` are their sums.
    per_day_cost: np.ndarray = field(repr=False, default=None)
    per_day_produce: np.ndarray = field(repr=False, default=None)
    # (n_owned, days) int8 — the entity code of each day's chosen edge, i.e.
    # what a constructive op (PLANT / BUILD / PLACE) names; 0 = none.
    per_day_entity: np.ndarray = field(repr=False, default=None)
    # The per-day edge rewards the sweep computed, kept because plan recovery
    # re-reads exactly these numbers (and re-deriving them per tile would be
    # the per-tile loop the architecture forbids anyway).
    rewards: np.ndarray = field(repr=False, default=None)   # (days, n_edges)


class TileContractor:
    """The sweep's buffers, cast once.

    `price_board()` builds one per call, which is the convenient path for a one-off pricing;
    a caller that prices every turn holds one contractor and reuses it, so the casts and the
    slice check happen once per process instead of once per turn.
    """

    def __init__(self, graph: TileGraph, days: int = HORIZON_DAYS) -> None:
        self.graph = graph
        self.days = int(days)
        self.n_states = int(graph.n_states)
        self.edge_offsets = np.ascontiguousarray(graph.edge_offsets, dtype=np.intp)
        self.edge_next = np.ascontiguousarray(graph.edge_next, dtype=np.intp)
        self.edge_chain = np.ascontiguousarray(graph.edge_chain, dtype=np.int64)
        self.edge_starts = self.edge_offsets[:-1]
        # Cast once at load: casting per call would upcast the whole sweep.
        self.EP = np.ascontiguousarray(graph.edge_produce, dtype=DTYPE)
        self.EC = np.ascontiguousarray(graph.edge_cost, dtype=DTYPE)
        self._assert_no_empty_slices()

    # ---- guards -------------------------------------------------------
    def _assert_no_empty_slices(self) -> None:
        """Every state must own at least one edge (see the module docstring).

        The `reduceat` trap: an empty group prices as the element at that index, a plausible
        wrong number instead of an error.
        """
        widths = np.diff(self.edge_offsets)
        empty = np.flatnonzero(widths == 0) if widths.size else []
        if len(empty):
            raise ValueError(
                f"state {int(empty[0])} owns zero out-edges: "
                "np.maximum.reduceat returns the element at that index for an "
                "empty group, so its value would be silently wrong")

    def _as_dual(self, values, name: str) -> np.ndarray:
        """A dual vector -> `(days, N_RESOURCE)` float32, R006-checked.

        Accepts either the per-day matrix or a single `(N_RESOURCE,)` vector
        applied to every day (the flat-vector path the tests use).
        """
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            if arr.shape[0] != N_RESOURCE:
                raise ValueError(
                    f"{name}: expected {N_RESOURCE} components, got {arr.shape}")
            arr = np.broadcast_to(arr, (self.days, N_RESOURCE))
        if arr.ndim != 2 or arr.shape != (self.days, N_RESOURCE):
            raise ValueError(
                f"{name}: expected shape ({self.days}, {N_RESOURCE}) or "
                f"({N_RESOURCE},), got {arr.shape}")
        _check_non_negative(name, arr)
        return np.ascontiguousarray(arr, dtype=DTYPE)

    # ---- the kernel ---------------------------------------------------
    def sweep(self, p, w) -> tuple[np.ndarray, np.ndarray]:
        """Backward sweep: `(days+1, n_states)` values and per-day rewards."""
        return self._sweep(self._as_dual(p, "prices"), self._as_dual(w, "wages"))

    def _sweep(self, p: np.ndarray, w: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        n_edges = int(self.edge_next.size)
        rewards = np.empty((self.days, n_edges), dtype=DTYPE)
        # The terminal row is the season's own end: no liquidation, so shed goods are worth
        # nothing and V[days] is zero.
        V = np.zeros((self.days + 1, self.n_states), dtype=DTYPE)
        for d in range(self.days - 1, -1, -1):
            # Produce and cost are priced by their own vector, in two passes. Folding them
            # into one wider matvec measures slower: the wider gemv loses to BLAS dispatch.
            r = self.EP @ p[d] - self.EC @ w[d]
            rewards[d] = r
            cand = r + V[d + 1][self.edge_next]
            V[d] = np.maximum.reduceat(cand, self.edge_starts)
        return V, rewards

    # ---- plan recovery ------------------------------------------------
    def _recover(self, V: np.ndarray, rewards: np.ndarray, owned: np.ndarray):
        """Forward walk for every owned tile, in lockstep over the days.

        The sweep does not keep the argmax, so plans are recovered forward. The tiles walk
        together: each day gathers the candidates for all owned tiles once and segments them
        once, instead of one small argmax per tile per day — per-tile slices pay full numpy
        dispatch cost and do not fit.

        Ties break to the lowest edge index, so a plan is reproducible run to run.
        """
        days = self.days
        n_owned = int(owned.size)
        per_day_cost = np.zeros((n_owned, days, N_RESOURCE), dtype=np.int64)
        per_day_produce = np.zeros_like(per_day_cost)
        rows = np.empty((days, n_owned), dtype=np.intp)
        states_at = np.empty((days, n_owned), dtype=np.intp)
        entities_at = np.empty((n_owned, days), dtype=np.int8)
        lane = np.arange(n_owned, dtype=np.intp)
        states = np.asarray(owned, dtype=np.intp).copy()
        for d in range(days):
            starts = self.edge_starts[states]
            sizes = self.edge_offsets[states + 1] - starts
            block = np.cumsum(sizes) - sizes          # each tile's slice start
            edge_ix = np.repeat(starts, sizes) + (
                np.arange(int(sizes.sum()), dtype=np.intp)
                - np.repeat(block, sizes))
            segment = np.repeat(lane, sizes)
            cand = rewards[d][edge_ix] + V[d + 1][self.edge_next[edge_ix]]
            best = np.maximum.reduceat(cand, block)
            hits = np.flatnonzero(cand == best[segment])
            chosen = edge_ix[hits[np.searchsorted(segment[hits], lane)]]
            rows[d] = chosen
            states_at[d] = states
            # Transposed on purpose: every per-day array the board publishes is
            # (n_owned, days).
            entities_at[:, d] = self.graph.edge_entity[chosen]
            # Per-day coefficients, not only the sum: a coupling layer works on
            # labour[d], inputs[r][d] and produce[r][d].
            per_day_cost[:, d, :] = self.graph.edge_cost[chosen]
            per_day_produce[:, d, :] = self.graph.edge_produce[chosen]
            states = self.edge_next[chosen]
        chain = self.edge_chain[rows]
        plans = [[(d, int(states_at[d, i]), int(chain[d, i]))
                  for d in range(days)] for i in range(n_owned)]
        columns = per_day_cost.sum(axis=1)
        produced = per_day_produce.sum(axis=1)
        return (columns, produced, plans,
                V[0, np.asarray(owned, dtype=np.intp)].astype(np.float64),
                per_day_cost, per_day_produce, entities_at)

    def price(self, p, w, owned_states) -> PricedBoard:
        """Price the board: one sweep for every owned tile, plus their columns.

        `owned_states` are graph state ids, one per tile we own.
        """
        p = self._as_dual(p, "prices")
        w = self._as_dual(w, "wages")
        owned = np.asarray(list(owned_states), dtype=np.int64)
        if owned.size:
            if int(owned.min()) < 0 or int(owned.max()) >= self.n_states:
                raise ValueError(
                    f"owned state id out of range 0..{self.n_states - 1}: "
                    f"{int(owned.min())}..{int(owned.max())}")
        V, rewards = self._sweep(p, w)
        (columns, produced, plans, tile_values, per_day_cost,
         per_day_produce, per_day_entity) = self._recover(V, rewards, owned)
        # With no owned tile there is no column and nothing to price; the
        # signal is defined as 0 rather than as the max of an empty set.
        reduced_cost = float(tile_values.max()) if owned.size else 0.0
        return PricedBoard(values=V, reduced_cost=reduced_cost,
                           columns=columns, produce=produced, plans=plans,
                           tile_values=tile_values, days=self.days,
                           per_day_cost=per_day_cost,
                           per_day_produce=per_day_produce,
                           per_day_entity=per_day_entity,
                           rewards=rewards)


def price_board(graph: TileGraph, p, w, owned_states: Sequence[int],
                days: int = HORIZON_DAYS) -> PricedBoard:
    """One pricing call: the sweep plus the recovered columns.

    Builds a `TileContractor`, so the casts pay once per call; a caller pricing every turn
    should hold the contractor instead.
    """
    return TileContractor(graph, days=days).price(p, w, owned_states)
