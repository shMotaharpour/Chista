"""Bench the turn budget: a full 720-turn episode, per-turn timing.

Run from the repo root:

    .venv/bin/python -m bench.bench_turn_budget [seeds]

Plays the agent spine against `random` on the official path
(offline.kaggle_env.run_episode), records every turn's self_ms and prints
the p50/p95/p99/max table plus the bank draw. Acceptance (#9): bank
drawn = 0 s, max <= 400 ms; this table is the baseline later milestones
compare against, and SAFETY / FLOOR_S for the bank policy derive from it
(not yet applied in M1 - the bank policy lands with the replanner).

It also times the tile-DP sweep (issue #11 §7), which is the one heavy
thing the replanner rung does: `timing_solo` with nothing else running and
`timing_contended` with a competing thread in the same turn. The second is
the number that matters - a competing thread took 59 % of our compute in the
#20 measurement (§A4), and a budget sized against the solo number is sized
against a game nobody plays. The direction is asserted (contended >= solo):
a contended reading that came back faster means the measurement is broken,
not that load helps.
"""

from __future__ import annotations

import statistics
import sys
import threading
import time

import numpy as np

from agent.main import agent


def bench(seeds: int = 3) -> int:
    from offline.kaggle_env import run_episode

    all_ms: list[float] = []
    first_turns: list[float] = []
    for seed in range(seeds):
        turn_times: list[float] = []

        def timed_agent(obs, config=None):
            t0 = time.perf_counter()
            action = agent(obs, config)
            turn_times.append((time.perf_counter() - t0) * 1000.0)
            return action

        env = run_episode([timed_agent, "random"],
                          configuration={"episodeSteps": 720, "seed": seed})
        rewards = [s.reward for s in env.steps[-1]]
        print(f"seed {seed}: rewards {rewards}, turns {len(turn_times)}, "
              f"max {max(turn_times):.1f} ms")
        all_ms.extend(turn_times)
        first_turns.append(turn_times[0])

    all_ms.sort()
    p50 = statistics.median(all_ms)
    p95 = all_ms[int(0.95 * len(all_ms)) - 1]
    p99 = all_ms[int(0.99 * len(all_ms)) - 1]
    mx = all_ms[-1]
    print(f"\nturns {len(all_ms)}  p50 {p50:.2f} ms  p95 {p95:.2f} ms  "
          f"p99 {p99:.2f} ms  max {mx:.2f} ms")
    print(f"first-turn (cold) max: {max(first_turns):.2f} ms")
    bank_drawn = max(0.0, (mx - 1000.0)) / 1000.0
    print(f"bank drawn: {bank_drawn:.3f} s (from the max turn; "
          f"0.000 = untouched)")
    return 0


class _Contender(threading.Thread):
    """A competing thread: an opponent deliberating in the same turn.

    numpy releases the GIL, so this is real contend-for-a-core load, not a
    cosmetic sleep - the same shape as the #20 measurement, where a competing
    thread took 59 % of our compute.
    """

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stop = threading.Event()

    def run(self) -> None:
        rng = np.random.default_rng(0)          # the same work every run
        a = rng.random((512, 512))
        b = rng.random((512, 512))
        while not self.stop.is_set():
            a @ b

    def __enter__(self) -> "_Contender":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()
        self.join(timeout=1.0)


def bench_sweep(runs: int = 5) -> int:
    """The tile-DP sweep: solo and contended, direction asserted.

    Only the SWEEP is timed (the plan recovery is measured by
    tests/test_tile_dp_contractor.py): the sweep is what every replan pays for
    the whole board at once.
    """
    from agent.tile_dp.chains import N_RESOURCE, RESOURCE_ID
    from agent.tile_dp.contractor import TileContractor
    from agent.tile_dp.graph import TileGraph
    from pathlib import Path

    graph = TileGraph.load(Path("agent/tile_dp/models/graph_tile_lifecycle.npz"))
    contractor = TileContractor(graph)
    p = np.zeros(N_RESOURCE)
    w = np.zeros(N_RESOURCE)
    p[RESOURCE_ID["WHEAT"]] = 25           # the engine's own pivot price (F034)
    contractor.sweep(p, w)                 # warm the pages before timing

    solo = min(_ms(contractor.sweep, p, w) for _ in range(runs))
    with _Contender():
        contended = min(_ms(contractor.sweep, p, w) for _ in range(runs))
    print(f"timing_solo      sweep {solo:.2f} ms")
    print(f"timing_contended sweep {contended:.2f} ms "
          f"(ratio {contended / solo:.2f})")
    assert contended >= solo, (
        f"contended {contended:.2f} ms < solo {solo:.2f} ms: the measurement "
        "is broken (load does not make work faster)")
    return 0


def _ms(fn, *args) -> float:
    t0 = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - t0) * 1000.0


if __name__ == "__main__":
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    status = bench(seeds)
    status |= bench_sweep()
    raise SystemExit(status)
