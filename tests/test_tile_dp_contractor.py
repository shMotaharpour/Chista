"""TileContractor tests (issue #11): the sweep, R006, the calendars, the plan.

Run:  .venv/bin/python -m tests.test_tile_dp_contractor

Contracts under test:
- `V_30 ≡ 0` (F029: no liquidation) and zero prices + zero wages ⇒ `V ≡ 0`.
- Monotonicity in `p` and in `w` (a sign error is otherwise invisible).
- The vectorised sweep equals a scalar DP bit-for-bit on one entity's graph,
  and equals an exhaustive path enumeration over a 6-day horizon.
- R006: a negative price or a negative wage raises, naming the rule.
- The `reduceat` empty-slice case is refused at construction, not priced.
- The engine's honest calendars come out of the graph: F009 one-shot maxima
  (wheat 6 by day 4, carrot 4 by day 3, melon 6 by day 10) and the ongoing
  ready days of F027 / nights of F014 (tomato 7-10, strawberry 9/11/13/15).
- Plans are deterministic, and every recovered plan is a legal dispatch.
- Budget (issue #11 §7): the sweep and 100 recoveries stay inside the ceiling.

The integer duals the exactness tests use are deliberate: with small integer
prices and wages every reward and every value is an exactly representable
float32, so "equal" is a statement about the kernel's indexing, not about
floating-point rounding. Real-valued duals are exercised by the calendar tests.
"""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np

from tile_dp.chains import N_RESOURCE, RESOURCE_ID, chain_ops
from tile_dp.contractor import HORIZON_DAYS, TileContractor, price_board
from tile_dp.graph import build_graph
from tile_dp.tile_state import KIND_NONE, KIND_PLANT, TileState

from kaggle_environments.envs.kaggriculture import kaggriculture as K

REPO = Path(__file__).resolve().parents[1]
GRAPH_PATH = REPO / "tile_dp" / "models" / "graph_tile_lifecycle.npz"

# Acceptance ceilings from issue #11 §7 / §5.
SWEEP_CEILING_MS = 15.0
RECOVERY_CEILING_MS = 5.0

_GRAPH = None
_WHEAT = None


def _graph():
    """The shipped merged graph, loaded once per run."""
    global _GRAPH
    if _GRAPH is None:
        from tile_dp.graph import TileGraph
        _GRAPH = TileGraph.load(GRAPH_PATH)
    return _GRAPH


def _wheat():
    """The wheat-only view (26 states / 187 edges): small enough to enumerate."""
    global _WHEAT
    if _WHEAT is None:
        _WHEAT = build_graph("WHEAT")
    return _WHEAT


def _flat(resource: str, amount: float = 1.0) -> np.ndarray:
    """A flat price vector: `amount` on one resource, zero elsewhere."""
    p = np.zeros(N_RESOURCE)
    p[RESOURCE_ID[resource]] = float(amount)
    return p


def _integer_duals() -> tuple[np.ndarray, np.ndarray]:
    """Small integer duals: exact in float32, so equality is bit-for-bit."""
    p = np.zeros(N_RESOURCE)
    p[RESOURCE_ID["WHEAT"]] = 25
    p[RESOURCE_ID["CARROT"]] = 35
    p[RESOURCE_ID["EGG"]] = 60
    w = np.zeros(N_RESOURCE)
    w[RESOURCE_ID["LABOR_HOURS"]] = 3
    w[RESOURCE_ID["FERTILIZER"]] = 7
    return p, w


def _zero_duals() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(N_RESOURCE), np.zeros(N_RESOURCE)


def _owned(resource: str, age: int | None = None) -> list[int]:
    """Owned state ids: the bare tile, or a plant of `resource` at `age`."""
    graph = _graph()
    if age is None:
        return [graph.state_id_of(TileState(KIND_NONE, None, None, None,
                                            0, 0, 0, 0, 0, 0))]
    state = TileState(KIND_PLANT, resource, None, None, age, 0, 0, 0, 0, 0)
    return [graph.state_id_of(state)]


# --------------------------------------------------------------- invariants

def test_terminal_row_is_zero() -> None:
    """F029: the season ends with no liquidation, so V[days] ≡ 0."""
    graph = _graph()
    board = price_board(graph, _flat("WHEAT"), np.zeros(N_RESOURCE), [0])
    assert board.values.shape == (HORIZON_DAYS + 1, graph.n_states)
    assert np.array_equal(board.values[HORIZON_DAYS],
                          np.zeros(graph.n_states, dtype=board.values.dtype))


def test_zero_duals_price_everything_at_zero() -> None:
    """Every edge reward is zero, so every value is — a total indexing check.

    The recovered plan still walks edges (the DP is indifferent, so the ties
    fall to the lowest edge index) and still reports resource USE in
    `columns`; what must vanish is every VALUE, and the plan's value with it.
    """
    p, w = _zero_duals()
    board = price_board(_graph(), p, w, [0, 3, 4])
    assert np.array_equal(board.values, np.zeros_like(board.values))
    assert board.reduced_cost == 0.0
    plan_value = float(np.sum(board.per_day_produce[0] * p)
                       - np.sum(board.per_day_cost[0] * w))
    assert plan_value == 0.0
    assert int(board.produce.sum()) == 0


def test_monotone_in_prices_and_in_wages() -> None:
    """Raising a price cannot lower any value; lowering a wage cannot raise it.

    Sign errors survive every equality test in this file and die here.
    """
    graph = _graph()
    p0, w0 = _integer_duals()
    base = price_board(graph, p0, w0, [0]).values

    p_hi = p0.copy()
    p_hi[RESOURCE_ID["WHEAT"]] += 5
    assert np.all(price_board(graph, p_hi, w0, [0]).values >= base)

    w_lo = w0.copy()
    w_lo[RESOURCE_ID["LABOR_HOURS"]] -= 1
    assert np.all(price_board(graph, p0, w_lo, [0]).values >= base)

    p_lo = p0.copy()
    p_lo[RESOURCE_ID["WHEAT"]] -= 5
    assert np.all(price_board(graph, p_lo, w0, [0]).values <= base)


# ------------------------------------------------------------------- R006

def test_negative_price_raises_r006() -> None:
    """A negative price makes the pruned graph silently wrong: it must raise."""
    p, w = _integer_duals()
    p[RESOURCE_ID["WHEAT"]] = -1.0
    try:
        price_board(_graph(), p, w, [0])
    except ValueError as exc:
        assert "R006" in str(exc), exc
        assert "prices" in str(exc), exc
    else:
        raise AssertionError("a negative price priced the board without a word")


def test_negative_wage_raises_r006() -> None:
    """Same rule on the wage side, and per-day vectors are checked too."""
    p, w = _integer_duals()
    w[RESOURCE_ID["FERTILIZER"]] = -0.5
    try:
        price_board(_graph(), p, w, [0])
    except ValueError as exc:
        assert "R006" in str(exc), exc
        assert "wages" in str(exc), exc
    else:
        raise AssertionError("a negative wage priced the board without a word")

    # the per-day form is checked on every day, not only on day 0
    daily = np.tile(np.zeros(N_RESOURCE), (HORIZON_DAYS, 1))
    daily[7, RESOURCE_ID["WHEAT"]] = -2.0
    try:
        price_board(_graph(), daily, np.zeros(N_RESOURCE), [0])
    except ValueError as exc:
        assert "R006" in str(exc), exc
    else:
        raise AssertionError("a negative day-7 price went unnoticed")


def test_empty_edge_slice_is_refused() -> None:
    """`reduceat` prices an empty group as the element at that index."""
    graph = _graph()
    offsets = np.array(graph.edge_offsets, dtype=np.int64)
    offsets[1] = offsets[0]                      # state 0 loses its out-edges
    broken = dataclasses.replace(graph, edge_offsets=offsets)
    try:
        TileContractor(broken)
    except ValueError as exc:
        assert "zero out-edges" in str(exc), exc
    else:
        raise AssertionError("an empty edge slice was accepted silently")

    # the shipped graph itself must have none — asserted, not assumed
    assert int(np.diff(graph.edge_offsets).min()) > 0


# ------------------------------------------------------- the kernel is right

def _scalar_values(graph, p, w, days: int) -> np.ndarray:
    """The recurrence written out: one state at a time, plain Python loops.

    Deliberately the slowest possible expression of the same definition — the
    vectorised kernel is only trustworthy if it agrees with this.
    """
    P = np.asarray(p, dtype=np.float64)
    W = np.asarray(w, dtype=np.float64)
    if P.ndim == 1:
        P = np.tile(P, (days, 1))
    if W.ndim == 1:
        W = np.tile(W, (days, 1))
    V = np.zeros((days + 1, graph.n_states), dtype=np.float64)
    for d in range(days - 1, -1, -1):
        for s in range(graph.n_states):
            lo, hi = graph.edges_of(s)
            best = None
            for row in range(lo, hi):
                reward = 0.0
                for k in range(N_RESOURCE):
                    reward += float(graph.edge_produce[row][k]) * P[d][k]
                    reward -= float(graph.edge_cost[row][k]) * W[d][k]
                cand = reward + V[d + 1][int(graph.edge_next[row])]
                if best is None or cand > best:
                    best = cand
            V[d][s] = best
    return V


def test_sweep_matches_a_scalar_dp_on_one_entity() -> None:
    """Bit-for-bit: the four array ops are the triple loop, exactly."""
    graph = _wheat()
    p, w = _integer_duals()
    days = 8
    V, _ = TileContractor(graph, days=days).sweep(p, w)
    scalar = _scalar_values(graph, p, w, days)
    # integer duals keep every value exactly representable in float32, so this
    # equality is about indexing, not about rounding
    assert np.all(scalar == np.round(scalar)), "the oracle is not integral"
    assert np.array_equal(V, np.asarray(scalar, dtype=V.dtype))


def _enumerate_best(graph, start: int, days: int,
                    rewards: np.ndarray) -> float:
    """Exhaustive search over every edge sequence of `days` days."""
    best = -np.inf

    def walk(day: int, state: int, total: float) -> None:
        nonlocal best
        if day == days:
            best = max(best, total)
            return
        lo, hi = graph.edges_of(state)
        row_rewards = rewards[day]
        for row in range(lo, hi):
            walk(day + 1, int(graph.edge_next[row]),
                 total + float(row_rewards[row]))

    walk(0, start, 0.0)
    return best


def test_sweep_matches_an_exhaustive_search() -> None:
    """Every path of a 6-day horizon, against the DP's chosen chains."""
    graph = _wheat()
    p, w = _integer_duals()
    days = 6
    contractor = TileContractor(graph, days=days)
    P = np.tile(p, (days, 1)).astype(np.float64)
    W = np.tile(w, (days, 1)).astype(np.float64)
    rewards = np.array([[sum(float(graph.edge_produce[row][k]) * P[d][k]
                             - float(graph.edge_cost[row][k]) * W[d][k]
                             for k in range(N_RESOURCE))
                         for row in range(graph.n_edges)]
                        for d in range(days)], dtype=np.float64)
    start = graph.state_id_of(TileState(KIND_NONE, None, None, None,
                                        0, 0, 0, 0, 0, 0))
    best = _enumerate_best(graph, start, days, rewards)

    board = contractor.price(p, w, [start])
    # the plan the DP committed to, valued at the same duals it was priced with
    plan_value = float(np.sum(board.per_day_produce[0] * p)
                       - np.sum(board.per_day_cost[0] * w))
    assert board.tile_values[0] == best, (board.tile_values[0], best)
    assert plan_value == best, (plan_value, best)


# ------------------------------------------------------- engine calendars

def test_honest_one_shot_calendars() -> None:
    """F009's honest maxima, read off the graph by the DP itself.

    The horizon admits exactly one cycle, so there is no replant/throughput
    choice to make and the plan is the single-cycle maximum: wheat 6 by day 4,
    carrot 4 by day 3, melon 6 by day 10.
    """
    graph = _graph()
    w = np.zeros(N_RESOURCE)
    for resource, days, harvest_day, units in (("WHEAT", 5, 4, 6),
                                               ("CARROT", 4, 3, 4),
                                               ("MELON", 11, 10, 6)):
        board = price_board(graph, _flat(resource), w, _owned(resource),
                            days=days)
        crop = RESOURCE_ID[resource]
        per_day = board.per_day_produce[0][:, crop]
        assert int(per_day.sum()) == units, (resource, per_day)
        assert [int(d) for d in np.flatnonzero(per_day)] == [harvest_day], \
            (resource, per_day)
        assert board.tile_values[0] == float(units), (resource, units)


def test_ongoing_ready_days_and_nights() -> None:
    """F014's production nights and F027's ready days, from the graph alone.

    The tile is a plant of the crop on the first day-start it exists (the
    planting day itself is intra-day), so day `d` of the sweep is the engine's
    day `d + 1`. The stock produced on night `n` is on the tile at day-start
    `n + 1` and is harvested that day — so the sweep's harvest days ARE F014's
    nights, and the engine's ready days are those plus one.
    """
    graph = _graph()
    w = np.zeros(N_RESOURCE)
    for resource, nights, ready in (("TOMATO", (7, 8, 9, 10), (8, 9, 10, 11)),
                                    ("STRAWBERRY", (9, 11, 13, 15),
                                     (10, 12, 14, 16))):
        days = ready[-1]
        # a strictly decreasing path: with a flat vector, holding stock is
        # value-free (F022) and the harvest day is one of several ties
        p = np.zeros((days, N_RESOURCE))
        p[:, RESOURCE_ID[resource]] = 1.0 - 0.01 * np.arange(days)
        first_day_start_age = 1 - int(K.CROPS[resource]["max_yield_day"])
        board = price_board(graph, p, w,
                            _owned(resource, age=first_day_start_age),
                            days=days)
        per_day = board.per_day_produce[0][:, RESOURCE_ID[resource]]
        assert [int(d) for d in np.flatnonzero(per_day)] == list(nights), \
            (resource, per_day)
        assert [int(d) + 1 for d in np.flatnonzero(per_day)] == list(ready), \
            (resource, ready)


# ------------------------------------------------------------ the plan itself

def test_plan_is_deterministic() -> None:
    """Same graph, same duals, same plan — twice, byte-identical."""
    graph = _graph()
    p, w = _integer_duals()
    a = price_board(graph, p, w, [0, 3, 4])
    b = price_board(graph, p, w, [0, 3, 4])
    assert a.values.tobytes() == b.values.tobytes()
    assert a.plans == b.plans
    assert a.columns.tobytes() == b.columns.tobytes()


def test_plan_follows_the_argmax_it_priced() -> None:
    """The walk's edge is the argmax of `reward + V_next` at every step."""
    graph = _graph()
    p, w = _integer_duals()
    days = 30
    contractor = TileContractor(graph, days=days)
    board = contractor.price(p, w, [0])
    for d, state, chain_id in board.plans[0]:
        lo, hi = graph.edges_of(state)
        cand = board.rewards[d][lo:hi] + board.values[d + 1][
            np.asarray(graph.edge_next[lo:hi], dtype=np.intp)]
        row = lo + int(np.argmax(cand))
        assert int(graph.edge_chain[row]) == chain_id, (d, state, chain_id)
        assert chain_ops(int(graph.edge_chain[row])) == chain_ops(chain_id)


def test_empty_owned_board_prices_nothing() -> None:
    """No owned tile: no column, no plan, and a defined (zero) signal."""
    board = price_board(_graph(), _flat("WHEAT"), np.zeros(N_RESOURCE), [])
    assert board.columns.shape == (0, N_RESOURCE)
    assert board.plans == [] and board.reduced_cost == 0.0


# ------------------------------------------------------------------ budget

def test_budget_sweep_and_recovery() -> None:
    """Issue #11 §7: sweep ≤ 15 ms, 100 tile recoveries ≤ 5 ms.

    Best of three runs: this is a shared box, so the floor is the honest
    number, and a single reading carries another process's contention.
    """
    graph = _graph()
    p, w = _integer_duals()
    contractor = TileContractor(graph)
    contractor.sweep(p, w)                                   # warm the pages

    solo = min(_time_ms(contractor.sweep, p, w) for _ in range(3))
    owned = list(range(0, min(100, graph.n_states)))
    one = min(_time_ms(contractor.price, p, w, [0]) for _ in range(3))
    hundred = min(_time_ms(contractor.price, p, w, owned) for _ in range(3))
    recovery = max(0.0, hundred - one)
    print(f"  sweep {solo:.2f} ms  price(1 tile) {one:.2f} ms  "
          f"price(100 tiles) {hundred:.2f} ms  recovery {recovery:.2f} ms")
    assert solo <= SWEEP_CEILING_MS, f"sweep {solo:.2f} ms"
    assert recovery <= RECOVERY_CEILING_MS, f"recovery {recovery:.2f} ms"


def _time_ms(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - t0) * 1000.0


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
    print("all tile contractor tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
