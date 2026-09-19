"""Branch purity and the RNG-from-state invariant for world.fast_sim.

Run:  .venv/bin/python -m tests.test_world_branch_purity

Why this exists: the shipped interpreter rebuilds its RNG every day from
`random.Random((seed * 1_000_003) ^ day)` (F045) — the draw sequence is a
function of (seed, day), not of a stream position. Everything built on
clone()/what_if() (branch-and-bound, DP, MCTS, scenario sweeps) is only sound
while that holds: if a future kaggle-environments release moves to one
advancing stream, branches would silently share or inherit its position and
every rollout would be quietly wrong.

The invariant tested here is implementation-independent: a branch taken at
turn k and continued with a suffix must equal a fresh replay of prefix+suffix.
"""
from __future__ import annotations

import json
from typing import Any

from offline_lab.fast_sim import FastSim

from tests.test_world_parity import action_for, _clean

BRANCH_AT = 48   # two day boundaries
TOTAL = 96

CONFIG = {"seed": 20260912, "episodeSteps": TOTAL, "weedSpawnChance": 0.05}
PASS_ACTION = {"farmer": ["PASS"], "hands": [], "market": []}


def _pairs(n: int) -> list[tuple[dict, dict]]:
    return [(action_for(i, 0), action_for(i, 1)) for i in range(n)]


def _play(sim: FastSim, pairs) -> None:
    """Apply actions in order, stopping at episode end (like FastSim.run)."""
    for a0, a1 in pairs:
        if sim.done:
            return
        sim.step([dict(a0), dict(a1)])


def state_of(sim: FastSim) -> str:
    """Full state signature: status, reward and every observation field."""
    return json.dumps([
        {"status": s.status, "reward": s.reward, "obs": _clean(dict(s.observation))}
        for s in sim.state
    ], sort_keys=True, default=str)


def _fresh() -> FastSim:
    sim = FastSim(dict(CONFIG), validate="fast")
    sim.reset()
    return sim


def test_branch_equals_fresh_replay() -> None:
    """A clone continued with a suffix == a from-scratch replay of prefix+suffix.

    This is the invariant clone()/what_if()/DP depend on: RNG position must be
    re-derived from (seed, day), not carried in a mutable stream.
    """
    pairs = _pairs(TOTAL)
    sim = _fresh()
    _play(sim, pairs[:BRANCH_AT])

    branch = sim.clone()
    _play(branch, pairs[BRANCH_AT:])

    replay = _fresh()
    _play(replay, pairs)

    assert state_of(branch) == state_of(replay), (
        "clone()+suffix diverged from a fresh replay: the RNG is not a pure "
        "function of the episode state"
    )
    assert branch.money() == replay.money()


def test_branches_are_isolated_and_deterministic() -> None:
    """Exploring branches must not touch the parent, and twins must agree."""
    pairs = _pairs(TOTAL)
    base = _fresh()
    _play(base, pairs[:BRANCH_AT])
    before = state_of(base)

    tail = _pairs(12)
    for _ in range(5):
        throwaway = base.clone()
        _play(throwaway, tail)
        del throwaway

    assert state_of(base) == before, "exploring a clone mutated the parent state"

    left, right = base.clone(), base.clone()
    _play(left, tail)
    _play(right, tail)
    assert state_of(left) == state_of(right), (
        "two identical branches disagree: clone() leaks shared mutable state"
    )


def test_what_if_matches_a_manual_branch() -> None:
    """The convenience wrapper must be exactly clone()+step(), and side-effect free."""
    pairs = _pairs(TOTAL)
    base = _fresh()
    _play(base, pairs[:BRANCH_AT])
    before = state_of(base)

    hypothetical = base.what_if([pairs[BRANCH_AT][0], pairs[BRANCH_AT][1]], horizon=3)

    manual = base.clone()
    _play(manual, [pairs[BRANCH_AT]])
    for _ in range(2):
        if not manual.done:
            manual.observations()  # same call order as what_if()
            manual.step([PASS_ACTION, dict(PASS_ACTION)])

    assert state_of(base) == before, "what_if() mutated the source simulator"
    assert state_of(hypothetical) == state_of(manual), (
        "what_if() disagrees with clone()+step()"
    )


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
    print("all branch-purity tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
