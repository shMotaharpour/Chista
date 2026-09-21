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
        #: Edge-cost matrices with a walk charged inside, by distance (see
        #: `_travel_edge_costs`). Built once per distance, not once per round.
        self._travel_costs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        #: The edges a walk is charged on — those with a positive LABOUR cost —
        #: as a 0/1 float32 vector, so the distance enters the sweep's rewards
        #: as one elementwise term and one base gemv serves every distance
        #: (see `_sweep_from`).
        self._travel_mask = np.ascontiguousarray(
            (np.asarray(graph.edge_cost)[:, LABOR_ID] > 0).astype(DTYPE))
        self._assert_no_empty_slices()

    def _travel_edge_costs(self, travel_hours: int) -> tuple[np.ndarray, np.ndarray]:
        """The graph's edge costs with `travel_hours` charged on every worked day.

        The walk to a tile is paid once per day the plan puts a worker on it (the
        farm is cleared every night and the farmer respawns on a shed door, F040),
        and it lands on that day's LABOUR column. `0` returns the graph's own
        table, untouched.

        It has to be charged BEFORE the argmax. An edge priced without its walk is
        not an edge the farm can run, so a DP that chooses under one price vector
        while its caller prices the choice under another solves a different
        problem: its value is not the class's best dual-priced plan, the
        reduced-cost test is no longer about the master's LP, and the Lagrangian
        bound `y·b + Σ N_c v_c` can come out BELOW the objective it is supposed to
        bound. Measured on a real board: bound 33,765.5 against an objective of
        34,008.8 (-0.72 %), which is not a bound.

        Whole hours by construction: a walk is a Manhattan distance.
        """
        hours = int(travel_hours)
        if hours <= 0:
            return self.graph.edge_cost, self.EC
        cached = self._travel_costs.get(hours)
        if cached is None:
            cost = np.asarray(self.graph.edge_cost, dtype=np.int64).copy()
            works = cost[:, LABOR_ID] > 0
            cost[works, LABOR_ID] += hours
            cached = (np.ascontiguousarray(cost),
                      np.ascontiguousarray(cost, dtype=DTYPE))
            self._travel_costs[hours] = cached
        return cached

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
    def sweep(self, p, w, travel_hours: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Backward sweep: `(days+1, n_states)` values and per-day rewards."""
        _ec_int, ec = self._travel_edge_costs(travel_hours)
        return self._sweep(self._as_dual(p, "prices"), self._as_dual(w, "wages"), ec)

    def _sweep(self, p: np.ndarray, w: np.ndarray, ec: np.ndarray | None = None
               ) -> tuple[np.ndarray, np.ndarray]:
        ec = self.EC if ec is None else ec
        rewards = np.empty((self.days, int(self.edge_next.size)), dtype=DTYPE)
        for d in range(self.days - 1, -1, -1):
            # Produce and cost are priced by their own vector, in two passes. Folding them
            # into one wider matvec measures slower: the wider gemv loses to BLAS dispatch.
            rewards[d] = self.EP @ p[d] - ec @ w[d]
        return self._backward(rewards), rewards

    def _base_rewards(self, p: np.ndarray, w: np.ndarray) -> np.ndarray:
        """`EP @ p[d] - EC @ w[d]` for every day: the sweep's distance-free half.

        The walk only ever lands on the LABOUR column of the worked edges
        (`_travel_edge_costs`), so the edge cost of a distance-`h` group is
        `EC + h·mask` and its rewards are `base - h·(mask · w[:, LABOUR])` — one
        gemv per day for every distance instead of one per distance.
        """
        rewards = np.empty((self.days, int(self.edge_next.size)), dtype=DTYPE)
        for d in range(self.days):
            rewards[d] = self.EP @ p[d] - self.EC @ w[d]
        return rewards

    def _sweep_from(self, base: np.ndarray, w: np.ndarray, travel_hours: int
                    ) -> tuple[np.ndarray, np.ndarray]:
        """The backward sweep for one distance, given `_base_rewards`."""
        hours = int(travel_hours)
        if hours <= 0:
            rewards = base
        else:
            term = self._travel_mask[None, :] * w[:, LABOR_ID][:, None]
            rewards = base - np.float32(hours) * term
        return self._backward(rewards), rewards

    def _backward(self, rewards: np.ndarray) -> np.ndarray:
        """The day loop: every state takes the best of its own out-edges."""
        n_edges = int(self.edge_next.size)
        if rewards.shape != (self.days, n_edges):
            raise ValueError(
                f"rewards: expected shape ({self.days}, {n_edges}), "
                f"got {rewards.shape}")
        # The terminal row is the season's own end: no liquidation, so shed goods are worth
        # nothing and V[days] is zero.
        V = np.zeros((self.days + 1, self.n_states), dtype=DTYPE)
        for d in range(self.days - 1, -1, -1):
            cand = rewards[d] + V[d + 1][self.edge_next]
            V[d] = np.maximum.reduceat(cand, self.edge_starts)
        return V

    # ---- plan recovery ------------------------------------------------
    def _recover(self, V: np.ndarray, rewards: np.ndarray, owned: np.ndarray,
                 ec_int: np.ndarray | None = None):
        """Forward walk for every owned tile, in lockstep over the days.

        The sweep does not keep the argmax, so plans are recovered forward. The tiles walk
        together: each day gathers the candidates for all owned tiles once and segments them
        once, instead of one small argmax per tile per day — per-tile slices pay full numpy
        dispatch cost and do not fit.

        Ties break to the lowest edge index, so a plan is reproducible run to run.
        """
        days = self.days
        n_owned = int(owned.size)
        ec = self.graph.edge_cost if ec_int is None else ec_int
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
            # labour[d], inputs[r][d] and produce[r][d]. `ec` is the graph's own
            # table, or the one with this class's walk charged inside it.
            per_day_cost[:, d, :] = ec[chosen]
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

    def price(self, p, w, owned_states, travel_hours: int = 0) -> PricedBoard:
        """Price the board: one sweep for every owned tile, plus their columns.

        `owned_states` are graph state ids, one per tile we own. `travel_hours` is
        the walk this group of tiles pays on every day it is worked, charged on
        the labour column BEFORE the argmax (see `_travel_edge_costs`): a caller
        that prices tiles at different distances passes each group its own.
        """
        hours = int(travel_hours)
        return self.price_many(p, w, {hours: owned_states})[hours]

    def price_many(self, p, w, owned_by_distance: dict[int, Sequence[int]]
                   ) -> dict[int, PricedBoard]:
        """Price every distance off ONE base sweep.

        A round prices one group per distinct distance, and each group's sweep
        differs from the next only by the walk on the labour column. The gemv
        that dominates the sweep — `EP @ p[d]` and `EC @ w[d]` over ~12,000 edges
        and 20 days — is therefore the same work nine times over; measured on a
        day-0 board, 9 distances cost 9 sweeps of 5.8 ms each.

        `_base_rewards` computes that half once and `_sweep_from` adds each
        distance's own term, so the per-distance cost is one elementwise pass
        plus the day loop. The result is the same sweep `price` returns for that
        distance: `tests/test_tile_dp_contractor.py` pins them against each
        other, term by term.
        """
        prices = self._as_dual(p, "prices")
        wages = self._as_dual(w, "wages")
        base = self._base_rewards(prices, wages)
        out: dict[int, PricedBoard] = {}
        for raw_hours, states in owned_by_distance.items():
            hours = int(raw_hours)
            owned = np.asarray(list(states), dtype=np.int64)
            if owned.size:
                if int(owned.min()) < 0 or int(owned.max()) >= self.n_states:
                    raise ValueError(
                        f"owned state id out of range 0..{self.n_states - 1}: "
                        f"{int(owned.min())}..{int(owned.max())}")
            ec_int, _ec = self._travel_edge_costs(hours)
            V, rewards = self._sweep_from(base, wages, hours)
            (columns, produced, plans, tile_values, per_day_cost,
             per_day_produce, per_day_entity) = self._recover(V, rewards, owned,
                                                              ec_int)
            # With no owned tile there is no column and nothing to price; the
            # signal is defined as 0 rather than as the max of an empty set.
            out[hours] = PricedBoard(
                values=V,
                reduced_cost=float(tile_values.max()) if owned.size else 0.0,
                columns=columns, produce=produced, plans=plans,
                tile_values=tile_values, days=self.days,
                per_day_cost=per_day_cost, per_day_produce=per_day_produce,
                per_day_entity=per_day_entity, rewards=rewards)
        return out


def price_board(graph: TileGraph, p, w, owned_states: Sequence[int],
                days: int = HORIZON_DAYS, travel_hours: int = 0) -> PricedBoard:
    """One pricing call: the sweep plus the recovered columns.

    Builds a `TileContractor`, so the casts pay once per call; a caller pricing every turn
    should hold the contractor instead.
    """
    return TileContractor(graph, days=days).price(p, w, owned_states,
                                                  travel_hours=travel_hours)
