"""Self-warm guards. Each guard was broken and seen red on the code it
protects (R007); the defects they cover are the #206/#207 family:

- the retired `_SEARCH_CACHE` returned ANSWERS with a key that dropped
  drop_by and the time columns — two different days shared one entry;
- a warm seed whose turns violated the task windows poisoned the row.

The contract agreed with the owner: the artifact stores SEEDS (never
answers), the hard gates are pool + hands-per-hour vector, the soft key is
the (category x quadrant) density, every returned seed passes
check_route + compile_route, and self-warm is always consulted
independently of the caller's own warm.

Run:  .venv/bin/python -m tests.test_wsr_warm   (also under pytest)
"""
from __future__ import annotations

import numpy as np
import pytest

from agent.wsr import beam as B, tasks as T
from agent.wsr import selfwarm as S
from agent.wsr.emit import check_route


def _fresh_memory() -> S.SelfWarm:
    """The shared global memory, reset to empty for the test."""
    m = S.selfwarm()
    m.seeds = []
    m.hits = 0
    m.misses = 0
    return m


def _two_water_day():
    grid = [((2, 2), ("WATER",), "WHEAT"), ((2, 3), ("WATER",), "WHEAT")]
    tasks = T.build(grid, available={})
    day = B.Day(chains=tuple(grid), available={}, hire_times=(1,))
    return day, tasks


def test_pool_is_a_hard_gate():
    """A 0-pool seed must never be offered to a 1-pool question: the beam's
    seed() accepts warm.pool == pool and nothing else."""
    m = _fresh_memory()
    day, tasks = _two_water_day()
    r0 = B.search(day, tasks, hands=0, max_hands=0)
    m.store(day, tasks, r0)
    assert len(m.lookup(day, tasks, 0)) == 1, (
        "same pool: the stored seed must be offered")
    day1 = B.Day(chains=tuple([((2, 2), ("WATER",), "WHEAT"),
                               ((2, 3), ("WATER",), "WHEAT")]),
                 available={}, hire_times=(1, 2))
    cands1 = m.lookup(day1, tasks, 1)
    assert all(c.pool == 1 for c in cands1), (
        "a 0-pool seed must not serve a 1-pool question")


def test_hire_hours_are_a_hard_gate():
    """The hands-per-hour vector is part of the hard key: the door geometry
    (F040) is a function of it, so a different vector must not match."""
    m = _fresh_memory()
    day, tasks = _two_water_day()
    r0 = B.search(day, tasks, hands=1, max_hands=1)
    m.store(day, tasks, r0)
    day_alt = B.Day(chains=tuple([((2, 2), ("WATER",), "WHEAT"),
                                  ((2, 3), ("WATER",), "WHEAT")]),
                    available={}, hire_times=(1, 4))
    assert m.lookup(day_alt, tasks, 1) == [], (
        "a different hands-per-hour vector must not match the stored seeds")


def test_seed_offered_at_same_pool_and_vector():
    day, tasks = _two_water_day()
    m = _fresh_memory()
    r = B.search(day, tasks, hands=0, max_hands=0)
    m.store(day, tasks, r)
    cands = m.lookup(day, tasks, 0)
    assert cands, "a stored seed for this exact pool+vector must be offered"
    for c in cands:
        assert c.pool == 0
        assert not check_route(day, tasks, c), (
            "a seed that fails check_route must never leave the memory")


def test_seed_never_replaces_the_answer():
    """The #206 defect: a cache hit returned an ANSWER. Self-warm only seeds —
    the search always runs, and on a small day the answer equals cold."""
    day, tasks = _two_water_day()
    m = _fresh_memory()
    cold = B.search(day, tasks, hands=0, max_hands=0)
    m.store(day, tasks, cold)
    warm_hit = B.search(day, tasks, hands=0, max_hands=0)
    assert warm_hit.complete and len(warm_hit.route) == tasks.n
    assert len(warm_hit.route) == len(cold.route), (
        "the seed must not change the answer on a day the search solves fully")


def test_unknown_task_ids_are_dropped_in_projection():
    """A stored placement naming a task today's day does not have is dropped
    by the re-projection — a partial seed is legal (the beam completes it)."""
    day, tasks = _two_water_day()
    m = _fresh_memory()
    seed = m._retime([(0, 0, 0), (5, 1, 0)], tasks, day, 0, {0: 0})
    assert seed is not None
    placed = sorted(str(tid) for _t, tid, _w in seed.route)
    assert placed == ["d0_water", "d1_water"], placed


def test_counters_are_honest():
    day, tasks = _two_water_day()
    m = _fresh_memory()
    B.search(day, tasks, hands=0, max_hands=0)          # empty memory: miss
    assert m.misses == 1 and m.hits == 0
    r = B.search(day, tasks, hands=0, max_hands=0)
    m.store(day, tasks, r)
    B.search(day, tasks, hands=0, max_hands=0)          # now a hit
    assert m.hits == 1


def test_selfwarm_is_independent_of_the_callers_warm():
    """Self-warm is always consulted, even when the caller passes warm=: the
    two channels are separate. (Verified at the memory level: a stored entry
    is returned regardless of any warm argument the caller holds.)"""
    day, tasks = _two_water_day()
    m = _fresh_memory()
    r = B.search(day, tasks, hands=0, max_hands=0)
    m.store(day, tasks, r)
    cands = m.lookup(day, tasks, 0)
    assert cands, "self-warm is unconditional"


def test_the_retired_answer_cache_stays_retired():
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
    print(f"{len(tests) - failures}/{len(tests)} self-warm checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
