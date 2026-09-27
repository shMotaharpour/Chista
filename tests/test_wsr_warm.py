"""Warm-memory guards. Each guard was broken and seen red on the code it
protects (R007); the defects they cover are the #206/#207 family:

- the retired `_SEARCH_CACHE` returned ANSWERS with a key that dropped
  drop_by and the time columns — two different days shared one entry;
- a warm seed whose turns violated the task windows poisoned the row
  (the 'DROP at hour 10 with deadline 9' probe).

The contract agreed with the owner: the memory stores SEEDS (never answers),
the key is the (category x quadrant) logistic signature + the hands-per-hour
vector, every returned seed passes check_route + compile_route, and the pool
is a hard gate.

Run:  .venv/bin/python -m tests.test_wsr_warm   (also under pytest)
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from agent.wsr import beam as B, tasks as T, warm as W
from agent.wsr.emit import check_route


def _two_water_day():
    grid = [((2, 2), ("WATER",), "WHEAT"), ((2, 3), ("WATER",), "WHEAT")]
    tasks = T.build(grid, available={})
    day = B.Day(chains=tuple(grid), available={}, hire_times=(1,))
    return day, tasks


def test_memory_stores_and_returns_a_seed_for_a_same_signature_day():
    day, tasks = _two_water_day()
    W.memory()._entries.clear()
    result = B.search(day, tasks, hands=0, max_hands=0)
    W.remember(day, tasks, result)
    # a SECOND solve at the same pool must receive the stored route as a seed:
    cands = W.candidates_for(day, tasks, 0)
    assert cands, "a stored route for this exact signature must be offered"
    for c in cands:
        assert c.pool == 0
        assert not check_route(day, tasks, c), (
            "a seed that fails check_route must never leave the memory")


def test_a_seed_never_replaces_the_answer():
    """The #206 defect: a cache hit returned an ANSWER. Here the memory only
    seeds: with a stored route, the search still runs and its answer is its
    own — and for a trivially small day identical to a cold solve."""
    day, tasks = _two_water_day()
    W.memory()._entries.clear()
    cold = B.search(day, tasks, hands=0, max_hands=0)
    W.remember(day, tasks, cold)
    warm_hit = B.search(day, tasks, hands=0, max_hands=0)
    assert warm_hit.complete and len(warm_hit.route) == tasks.n
    assert len(warm_hit.route) == len(cold.route), (
        "the seed must not change the answer on a day the search solves fully")


def test_pool_is_a_hard_gate():
    day, tasks = _two_water_day()
    W.memory()._entries.clear()
    result = B.search(day, tasks, hands=0, max_hands=0)
    W.remember(day, tasks, result)
    # a 1-hand day has a different pool AND a different hands-per-hour vector:
    day1 = B.Day(chains=tuple([((2, 2), ("WATER",), "WHEAT"),
                               ((2, 3), ("WATER",), "WHEAT")]),
                 available={}, hire_times=(1, 2))
    cands = W.candidates_for(day1, tasks)
    assert all(c.pool == 1 for c in cands), (
        "a 0-pool entry must never be offered to a 1-pool question")


def test_a_window_violating_route_is_never_stored():
    """The poison guard: a route whose turns sit outside [earliest, latest]
    must not become a seed — the probe that caught the #206-era row death."""
    day, tasks = _two_water_day()
    W.memory()._entries.clear()
    poisoned = B.Result(pool=0, route=[(0, tasks.ids[0], 0),
                                       (23, tasks.ids[1], 0)],
                        complete=False)
    # force-store without verification, as an offline build could:
    W.memory().store(day, tasks, poisoned, verified=False)
    assert W.candidates_for(day, tasks) == [], (
        "unverified entries are never offered — the memory stores verified seeds only")


def test_lookup_distance_prefers_the_closer_signature():
    """Two stored entries, one day in between: the nearest signature must be
    the first candidate."""
    day, tasks = _two_water_day()
    W.memory()._entries.clear()
    # entry 1: exactly this day's signature
    exact = B.search(day, tasks, hands=0, max_hands=0)
    W.remember(day, tasks, exact)
    # entry 2: a different signature (a 1-tile day)
    small_grid = [((2, 2), ("WATER",), "WHEAT")]
    small_tasks = T.build(small_grid, available={})
    small_day = B.Day(chains=tuple(small_grid), available={}, hire_times=(1,))
    small = B.search(small_day, small_tasks, hands=0, max_hands=0)
    W.remember(small_day, small_tasks, small)
    cands = W.candidates_for(day, tasks, 0)
    assert cands, "the exact entry must be offered"
    first = cands[0]
    # the exact-signature seed must be the one that re-projects LOSSLESSLY
    # (all its placements survive) — the 1-tile day's cannot
    assert len(first.route) == 2, (
        f"expected the exact signature's 2 placements, got {len(first.route)}")


def test_the_counters_are_honest():
    day, tasks = _two_water_day()
    W.memory()._entries.clear()
    W.memory().hits = 0
    W.memory().misses = 0
    B.search(day, tasks, hands=0, max_hands=0)      # empty memory: a miss
    assert W.memory().misses == 1 and W.memory().hits == 0
    result = B.search(day, tasks, hands=0, max_hands=0)
    W.remember(day, tasks, result)
    B.search(day, tasks, hands=0, max_hands=0)      # now a hit
    assert W.memory().hits == 1


def test_retired_cache_is_a_noop_not_a_cache():
    day, tasks = _two_water_day()
    B.clear_search_cache()
    r1 = B.search(day, tasks, hands=0, max_hands=0)
    B.clear_search_cache()
    r2 = B.search(day, tasks, hands=0, max_hands=0)
    assert r1 is not r2, (
        "the retired `_SEARCH_CACHE` must not return object-identical answers")


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} warm-memory checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
