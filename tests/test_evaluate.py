"""offline.evaluate unit tests (issue #18).

Run:  .venv/bin/python -m tests.test_evaluate

Covers the harness's own logic; the pool loader has its own suite
(tests/test_pool_loader.py).
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

from offline_lab.evaluate import _paired_jobs, _seeds_needed, _stats


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
    from offline_lab.evaluate import (_pick_opponents, TIER_AGENT_EPISODES,
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


def test_bank_seconds_is_the_policy_draw() -> None:
    """F046: 1 free second per turn, billed per turn -> the bank draw is
    the SUM of overruns, not the worst turn's.

    Self-review fix: the worker reported `max(0, worst_turn - 1 s)`
    under the name `bank_drawn_s`. Two 1.4 s turns draw 0.8 s from the
    60 s bank; the old formula called that 0.4 s — an understated draw
    under the policy's own name (the bench prints that weaker reading,
    but labels it "from the max turn"). This test pins both numbers.
    """
    from offline_lab.runner import bank_seconds

    drawn, worst = bank_seconds([1400.0, 1400.0, 200.0])
    assert abs(drawn - 0.8) < 1e-9, drawn
    assert abs(worst - 0.4) < 1e-9, worst
    assert bank_seconds([500.0, 999.0]) == (0.0, 0.0)
    assert bank_seconds([]) == (0.0, 0.0)


def test_scoreboard_writer_quotes_and_labels() -> None:
    """The writer must never emit a row the guard test above would reject:
    a comma-bearing label is QUOTED (the defect that corrupted two rows
    was a label written by hand without quoting), and an absent label is
    synthesised so no run is unidentifiable evidence.
    """
    import csv
    import tempfile
    from pathlib import Path

    from offline_lab import evaluate as E

    real = E.SCOREBOARD
    with tempfile.TemporaryDirectory() as d:
        E.SCOREBOARD = Path(d) / "scoreboard.csv"
        try:
            E._append_scoreboard({"run_id": "x", "tier": "smoke",
                                  "a": "A", "b": "B"})
            E._append_scoreboard({"run_id": "y", "tier": "smoke",
                                  "a": "A", "b": "B",
                                  "label": "hand-written, with commas, 3 of them"})
        finally:
            E.SCOREBOARD = real
        rows = list(csv.reader((Path(d) / "scoreboard.csv").open()))
    assert rows[0] == E._COLUMNS, rows[0]
    for r in rows[1:]:
        assert len(r) == len(E._COLUMNS), (r, len(r))
    assert rows[1][2] == "A vs B - smoke tier", rows[1][2]
    assert rows[2][2] == "hand-written, with commas, 3 of them", rows[2][2]


def test_direction_verdict_tolerates_timer_noise() -> None:
    """A4's direction assertion must not abort a run on noise.

    Self-review fix: a live run died with "contended p95 0.1 ms < solo
    p95 0.1 ms" — two noise-scale readings. The verdict is now
    three-valued: both-below-floor = unresolved (never asserted, and
    never allowed to feed the bank policy), a within-tolerance
    difference = holds, a real inversion = violated (still raises).

    R007: verified in its failing direction — the old no-tolerance rule
    (`contended < solo`) maps a sub-resolution difference such as
    (0.1012, 0.0987) to a violation, so the pair below fails against it
    (measured: old rule -> violated, new rule -> unresolved).
    """
    from offline_lab.evaluate import _direction_verdict

    assert _direction_verdict(0.1, 0.1)[0] == "unresolved"
    assert _direction_verdict(0.1012, 0.0987)[0] == "unresolved"
    assert _direction_verdict(0.1, 0.0)[0] == "unresolved"
    assert _direction_verdict(9.0, 12.0)[0] == "holds"
    assert _direction_verdict(9.0, 8.9)[0] == "holds"      # within slack
    assert _direction_verdict(12.0, 9.0)[0] == "violated"  # inverted
    assert _direction_verdict(40.0, 10.0)[0] == "violated"


def test_seed_count_line_never_prints_nan() -> None:
    """Self-review fix: a one-seed run's report said "needs n >= nan
    paired seeds". nan is not a promise a report can make — the line now
    states the reason it is undefined.
    """
    from offline_lab.evaluate import _seed_count_line

    one_seed = _seed_count_line(float("nan"), -46717.0)
    assert "nan" not in one_seed, one_seed
    assert "undefined" in one_seed and "2 seeds" in one_seed, one_seed

    zero_mean = _seed_count_line(200.0, 0.0)
    assert "nan" not in zero_mean and "mean margin is 0" in zero_mean

    flat = _seed_count_line(0.0, 100.0)
    assert "nan" not in flat and "spread is 0" in flat

    measured = _seed_count_line(23713.83, -60817.3)
    assert "n >= 0.6" in measured, measured


def test_zero_spread_row_writes_empty_not_nan() -> None:
    """Issue #38: a run with n >= 2 pairs and ZERO spread has an undefined
    seed count — it must land as an EMPTY field, never a literal `nan`.

    Measured defect: a 20-pair run with sd 0.0 appended
    `...,160.0,160.0,nan,artifacts/.../records.jsonl`, because
    `round(nan, 1)` is `nan` and the old `""` fallback only covered
    `n < 2`. Built through the real row builder and written through the
    real writer, then parsed back.

    R007: verified in its failing direction — with the inline
    `round(_seeds_needed(...), 1)` restored, this test fails with
    "zero-spread row wrote 'nan'".
    """
    import csv
    import tempfile
    from pathlib import Path

    from offline_lab import evaluate as E

    # exactly the shape that produced the defect: 20 pairs, no spread
    s = {"n": 20, "mean": 160.0, "sd": 0.0, "ci_lo": 160.0, "ci_hi": 160.0}
    row = E._scoreboard_row(
        run_id="probe", label="zero-spread probe", tier="smoke",
        a_slug="A", b_slug="B", opp_note="--opponents override",
        n_jobs=20, n_ok=20, wins=20, losses=0, ties=0, s=s,
        out_dir=E.REPO / "artifacts" / "evaluate" / "probe")
    assert row["seeds_needed"] == "", f"zero-spread row wrote {row['seeds_needed']!r}"
    assert row["margin_sd"] == 0.0      # a real zero is a number, kept
    # and it must survive the writer as an empty field
    real = E.SCOREBOARD
    with tempfile.TemporaryDirectory() as d:
        E.SCOREBOARD = Path(d) / "scoreboard.csv"
        try:
            E._append_scoreboard(row)
        finally:
            E.SCOREBOARD = real
        raw = (Path(d) / "scoreboard.csv").open(newline="").read()
        back = list(csv.DictReader((Path(d) / "scoreboard.csv").open()))
    assert "nan" not in raw, raw
    assert back[0]["seeds_needed"] == "", back[0]
    assert back[0]["margin_mean"] == "160.0", back[0]


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
