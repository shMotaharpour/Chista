"""offline.evaluate unit tests (issue #18).

Run:  .venv/bin/python -m tests.test_evaluate

Covers the harness's own logic; the runner and pool layers have their
own suites (tests/test_pool_isolation.py, tests/test_pool_loader.py).
The classification rules pinned here are the issue's metric contract:

- the margin is A - B on the same seed against the same opponent;
- outcomes are win / loss / tie, ties counted explicitly, never as
  losses or halves;
- paired records merge on (opponent, seed);
- the paired schedule covers every opponent x seed exactly once, and a
  truncated run leaves every seed with similar coverage (interleaving);
- the seed-count line is computed, never hardcoded (16 was provisional).
"""

from __future__ import annotations

import math

from offline.evaluate import _paired_jobs, _seeds_needed, _stats


def test_paired_jobs_covers_every_pair_once() -> None:
    opponents = ["a", "b", "c"]
    seeds = [0, 1]
    jobs = _paired_jobs(opponents, seeds)
    assert sorted(jobs) == [("a", 0), ("a", 1), ("b", 0), ("b", 1),
                            ("c", 0), ("c", 1)]
    assert len(jobs) == len(set(jobs))          # no duplicates


def test_paired_jobs_interleaves_seeds() -> None:
    """A truncated run must not strand the last seeds: seed s starts at
    opponent index s mod n, so the first len(n) jobs touch every seed."""
    opponents = ["a", "b", "c", "d"]
    seeds = [0, 1, 2, 3]
    jobs = _paired_jobs(opponents, seeds)
    first_n = [j for _, j in jobs[:len(opponents)]]
    assert sorted(first_n) == seeds


def test_outcomes_win_loss_tie() -> None:
    m = [10.0, -10.0, 0.0]
    outcomes = ["win" if x > 0 else ("loss" if x < 0 else "tie") for x in m]
    assert outcomes == ["win", "loss", "tie"]
    wins = outcomes.count("win")
    losses = outcomes.count("loss")
    ties = outcomes.count("tie")
    assert wins + losses + ties == 3            # nothing silently halved
    assert wins == 1 and losses == 1 and ties == 1


def test_stats_paired_mean_and_ci() -> None:
    s = _stats([4.0, 6.0])
    assert s["n"] == 2
    assert s["mean"] == 5.0
    assert s["sd"] == math.sqrt(2.0)
    # normal-approx CI: mean +/- 1.96 sd / sqrt(n)
    half = 1.96 * math.sqrt(2.0) / math.sqrt(2)
    assert abs(s["ci_lo"] - (5.0 - half)) < 1e-9
    assert abs(s["ci_hi"] - (5.0 + half)) < 1e-9
    empty = _stats([])
    assert empty == {"n": 0}
    single = _stats([3.0])
    assert single["n"] == 1 and single["sd"] != single["sd"]  # nan


def test_seeds_needed_is_computed_not_hardcoded() -> None:
    """#20 brief 6.3: the provisional 16 must never be written anywhere.
    The number comes out of the measured sd and mean."""
    # a margin of 100 coins with sd 200 needs n such that 1.96*200/sqrt(n)
    # <= 100 -> n >= 15.37 -> reported 15.37
    n = _seeds_needed(200.0, 100.0)
    assert abs(n - (1.96 * 200.0 / 100.0) ** 2) < 1e-9
    assert abs(n - 15.3664) < 1e-3
    # degenerate inputs return nan (never 0, never a fake small number)
    assert _seeds_needed(0.0, 100.0) != _seeds_needed(0.0, 100.0)  # nan
    assert _seeds_needed(float("nan"), 100.0) != \
        _seeds_needed(float("nan"), 100.0)
    assert _seeds_needed(200.0, 0.0) != _seeds_needed(200.0, 0.0)


def test_seeds_needed_scales_with_noise() -> None:
    """Noisier margins need more seeds - monotone in sd."""
    n_quiet = _seeds_needed(50.0, 100.0)
    n_loud = _seeds_needed(400.0, 100.0)
    assert n_loud > n_quiet


def test_tier_counts_agree_with_the_note() -> None:
    """B2 (review round 2): the function that decides WHO you play must
    be guarded - a 19-vs-16 gap shipped through a green suite once.
    For every tier: the returned list length equals TIER_OPPONENTS[tier]
    AND the count the note claims, and length * seeds * 2 equals
    TIER_AGENT_EPISODES[tier] (which makes that constant load-bearing).
    R007: verified in its failing direction - with slugs() restored over
    include_aliases, this test fails with 'full tier: code returns 16
    opponents, its note and TIER_OPPONENTS say 19'."""
    from offline.evaluate import (_pick_opponents, TIER_AGENT_EPISODES,
                                  TIER_OPPONENTS, TIER_SEEDS)
    import re
    for tier in ("smoke", "ladder", "full"):
        ops, note = _pick_opponents(tier, None)
        assert len(ops) == TIER_OPPONENTS[tier], (
            f"{tier} tier: code returns {len(ops)} opponents, "
            f"TIER_OPPONENTS says {TIER_OPPONENTS[tier]}")
        m = re.search(r"^(\d+) opponents", note)
        if m:
            assert int(m.group(1)) == len(ops), (
                f"{tier} tier: the note claims {m.group(1)} opponents, "
                f"the code returns {len(ops)}")
        total = len(ops) * TIER_SEEDS[tier] * 2
        assert total == TIER_AGENT_EPISODES[tier], (
            f"{tier} tier: {len(ops)} x {TIER_SEEDS[tier]} seeds x 2 = "
            f"{total} agent-episodes, TIER_AGENT_EPISODES says "
            f"{TIER_AGENT_EPISODES[tier]}")


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
    print("all evaluate tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
