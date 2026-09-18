"""TileContractor — the pricing oracle: a vectorised backward DP over the tile
graph, plus forward plan recovery (issue #11, M2/D2-4).

The graph (`agent/artifact/tile_graph.npz`) is **day-invariant and
position-invariant**: every tile on the board, on every day, faces the same
state-action graph. So one backward sweep prices the whole board at once —
there is no per-tile solve and no per-day graph rebuild. A loop over tiles in
the sweep means the property the architecture rests on has been lost.

The recurrence (F029: the season ends with no liquidation, so `V_days ≡ 0`):

    V_d(s) = max over edges e out of s of
                 produce(e)·p_d − cost(e)·w_d + V_{d+1}(next(e))

Three array ops per day, over the shipped CSR: matvec, add, segmented max. The
argument max is **not** stored (that would be 30 × 15708 policies instead of an
80 KB table); plans are recovered forward, for the tiles we actually own.

Two contracts come with the graph:

- **R006 — non-negativity.** The graph is dominance-pruned, which is
  optimality-preserving only while `p >= 0` and `w >= 0` componentwise. The
  check runs on entry, every call: a negative component makes the pruned graph
  silently wrong, with no symptom to notice.
- **No empty edge slices.** `np.maximum.reduceat` returns the element *at* the
  index for an empty group rather than an identity, so a state with no
  out-edges would price as garbage instead of raising. Checked at construction.

Cost and production are kept as two separate matrices and are never netted:
they are priced by different vectors (`produce` at prices, `cost` at wages).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from agent.world.rules import TURNS_PER_DAY
from agent.tile_dp.ledger import RES_LABOR, N_RESOURCE, RESOURCE_ID
from agent.tile_dp.graph import TileGraph

# F029: 720 turns of 24 = 30 days, and shed goods at the end are worth
# nothing — there is no liquidation day, so the horizon's value is exactly 0.
HORIZON_DAYS = 30
# The sweep's arithmetic dtype (issue #11 §2): float32 keeps the two 18217×18
# matrices at ~1.1 MB each and stops numpy upcasting per call.
DTYPE = np.float32
LABOR_ID = RESOURCE_ID[RES_LABOR]


def _check_non_negative(name: str, values: np.ndarray) -> None:
    """R006's guard: the pruned graph is only optimal for non-negative vectors.

    Runs in dev mode AND in fast mode (R004): the check is one pass over 18
    numbers, and the failure it prevents is silent, so there is nothing to be
    saved by skipping it.
    """
    if values.size and float(np.min(values)) < 0.0:
        worst = float(np.min(values))
        raise ValueError(
            f"R006: {name} has a negative component ({worst}); the tile graph "
            "is dominance-pruned and that pruning is optimality-preserving "
            "only while every price and wage is >= 0 componentwise. A negative "
            "component makes a pruned edge the true optimum, so this DP would "
            "return a silently wrong plan. The master (#12) must clamp its "
            "duals onto the non-negative orthant before pricing.")


@dataclass(frozen=True)
class PricedBoard:
    """What the master (#12) needs back from one pricing call.

    `values` is the whole sweep — `(days + 1, n_states)`, with the terminal
    row identically zero. `plans` are the recovered columns' schedules,
    `columns` their resource use and `produce` their output, one row per owned
    tile, in the order `owned_states` was given.
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
    """The sweep's buffers, cast once: the runtime prices the board every day.

    `price_board()` builds one of these per call, which is the convenient path
    for tests and one-off pricing; a caller that prices every turn (the
    replanner) holds one contractor and reuses it, so the float32 casts and the
    slice check happen once per process rather than once per turn.
    """

    def __init__(self, graph: TileGraph, days: int = HORIZON_DAYS) -> None:
        self.graph = graph
        self.days = int(days)
        self.n_states = int(graph.n_states)
        self.edge_offsets = np.ascontiguousarray(graph.edge_offsets, dtype=np.intp)
        self.edge_next = np.ascontiguousarray(graph.edge_next, dtype=np.intp)
        self.edge_chain = np.ascontiguousarray(graph.edge_chain, dtype=np.int64)
        self.edge_starts = self.edge_offsets[:-1]
        # C-contiguous float32 once at load (issue #11 §2): 18217×18 is 1.1 MB
        # per matrix, and casting per call would upcast the whole sweep.
        self.EP = np.ascontiguousarray(graph.edge_produce, dtype=DTYPE)
        self.EC = np.ascontiguousarray(graph.edge_cost, dtype=DTYPE)
        self._assert_no_empty_slices()

    # ---- guards -------------------------------------------------------
    def _assert_no_empty_slices(self) -> None:
        """Every state must own at least one edge (see the module docstring).

        This is the `reduceat` trap: an empty group prices as the element at
        that index, i.e. as a plausible wrong number rather than an error.
        Asserted at load, not discovered as a silent wrong value.
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
        # V[days] is the terminal condition itself, not a modelling choice:
        # F029 — no liquidation, shed goods are worth nothing at season end.
        V = np.zeros((self.days + 1, self.n_states), dtype=DTYPE)
        for d in range(self.days - 1, -1, -1):
            # Two matvecs, priced by their own vector — and measured: folding
            # produce and cost into one [EP | −EC] @ [p ; w] looks cheaper (one
            # pass over the matrices instead of two) and is not. On this box it
            # cost 85-126 ms a sweep against 8.5-10.4 ms here, because the wider
            # gemv loses to BLAS thread dispatch. The two-matvec form is the
            # measured one; do not "optimise" it back.
            r = self.EP @ p[d] - self.EC @ w[d]
            rewards[d] = r
            cand = r + V[d + 1][self.edge_next]
            V[d] = np.maximum.reduceat(cand, self.edge_starts)
        return V, rewards

    # ---- plan recovery ------------------------------------------------
    def _recover(self, V: np.ndarray, rewards: np.ndarray, owned: np.ndarray):
        """Forward walk for every owned tile, in lockstep over the days.

        The sweep does not keep the argmax (30 × 15708 policies is not worth
        an 80 KB value table), so plans are recovered forward.

        The tiles walk TOGETHER: on day `d` the candidate array is gathered once
        for all owned tiles and segmented once, instead of 30 × n_owned small
        argmaxes. That is not a micro-optimisation: measured on the dev box
        (issue #11 §7), the obvious per-tile loop costs **20.8 ms for 100
        tiles** against this issue's 5 ms ceiling, because every ~25-element
        slice pays full numpy dispatch cost. The lockstep form is what fits.

        Ties break to the **lowest edge index** — `searchsorted` takes the first
        element equal to the segment maximum — so a plan is reproducible run to
        run. A plan that varied would make the arena's measurements
        unrepeatable.
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
            # transposed on purpose: every per-day array the board
            # publishes is (n_owned, days), and this one was (days, n_owned).
            # It never showed until the rung priced MORE than one tile
            # (measured: 100 tiles -> IndexError at [column, 0]).
            entities_at[:, d] = self.graph.edge_entity[chosen]
            # Per-day coefficients, not only the sum: the master (#12) couples
            # on labour[d], inputs[r][d] and produce[r][d].
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

        `owned_states` are graph state ids, one per tile we own — produced by
        `agent/obs.py`'s decode (#10), which already maps keys the graph does
        not model onto their nearest modelled neighbour.
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
    """One pricing call: the sweep plus the recovered columns (issue #11 §5).

    Builds a `TileContractor`, so the float32 casts pay once per call. A caller
    pricing every turn should hold the contractor instead — see the class.
    """
    return TileContractor(graph, days=days).price(p, w, owned_states)
