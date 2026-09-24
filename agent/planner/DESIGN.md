# planner — rounding, repair, land (issue #13)

> **A record of the design at the time**, written on the branch that built it. The
> tense below is that moment's ("this branch", "today"). The names it cites were
> checked against the code: `Plan`, `ClassMix`, `ROW_NAMES`, `assign_tiles`,
> `violations`, `demote_to_feasible`, `plan_from_board` and `max_solves` all exist,
> and both `TODO(#12)` markers are STILL open in `columns.py` and `land.py` — so the
> two rows it declines to measure are still declined, not stale.

The layer between the master's LP (#12) and the agent's dispatcher. Issue #13
asks for three things and this branch builds the two and a half that do not need
the master to exist.

## The seam to #12

```python
Plan        # one DW column: per-day chain ids + per-day value in every row
ClassMix    # one class: N_c tiles, its plans, their λ
ROW_NAMES   # ("labour", "cash_out", "wheat_net", "fert_net", "stored")
assign_tiles(tile_keys, mixes)      # λ -> one plan per tile   (issue §1)
violations(choices, mixes, caps)    # the coupling-row check   (acceptance 1)
demote_to_feasible(...)             # walk the rows, demote    (issue §1 repair)
plan_from_board(board, i, ...)      # build a Plan from #11's output
```

`ROW_NAMES` mirrors the #12 brief's rows exactly, and a `Plan` carries the row
values the master publishes for each column — so #13 never reads the LP, and the
LP never reads the rounding. `plan_from_board()` derives the same rows from the
pricing oracle (#11), which is what makes the whole module testable today.

## Decisions

- **The simplest rounding that can work** (the #13 brief's own order): each tile
  takes the largest λ in its own class, ties to the lowest plan index. The quota
  variant (floor the counts, hand out remainders by largest fractional part) is
  the fallback for a large integrality gap — and it is deliberately *not*
  written, because R005 says the measurement that justifies complexity has to
  exist first. TODO(#12) is that measurement.
- **LOCKED tiles get nothing** (`None`), so F042's silent no-op can never be
  planned on; a class with no mix is also `None` rather than a guess.
- **Dropping is counted.** `repair_day()` returns every drop with the rule it
  enforces; a plan that quietly loses a third of its ops looks exactly like one
  that works, and the counts are the only thing that tells them apart.
- **Land stays out of the LP** (issue §3): a prefix × day enumeration with the
  master injected as a callable, capped at three evaluations per episode and
  re-checked only when the engine's own price for the next step is affordable
  (F042's table — no invented threshold).
- **The shed row is a row, not an op check**: F043's 100 is cumulative, so it
  lives in the assignment check (`violations(..., {"stored": [100]*30})`) where
  the numbers still exist, not in the repair where only ops exist.

## Not measured here (R005 — named TODOs, not guesses)

| number the issue asks for | why it is not here | where it lands |
|---|---|---|
| integrality gap `LP bound − rounded value`, target ≤ 3 % | needs the LP bound, which only #12 has | `columns.py` module docstring, TODO(#12) |
| land enumeration ≤ 400 ms at day 0 | needs a real master solve per candidate (~46 of them) | `land.py` module docstring, TODO(#12) |

What *is* enforced today is the shape of the land budget: the candidate count is
knowable before any solve, and `max_solves` refuses to run a set that cannot fit
(measured against a stand-in master would be a wrong number dressed as a right
one).

## Tests

(planned, not yet written: `tests/test_planner_integrality.py`) — rounding is a function; ties are not coin
tosses; a LOCKED tile never receives a plan; the row check bites on a tight day
and the demotion loop clears it over 20 boards; the repaired plan dispatches and
validates for 30 days; every F047 drop is counted with its rule (F031 cap, short
purse, F004, F043, F042, F032's queue order); the land prefix is the engine's
table, the enumeration keeps the best, the budget refuses to overrun, and the
cadence cap holds.