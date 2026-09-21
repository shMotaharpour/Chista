"""Integral tile→plan assignment: the master's λ, rounded (issue #13 §1).

The LP hands back a *fractional* mix: class `c` is told to spend `λ_{c,j}` of
itself on plan `j`. A tile cannot be 0.4 wheat and 0.6 melon, so this module
turns the mix into one plan per tile — deterministically, and with the tiles a
quadrant has not unlocked (F042) getting nothing at all.

**The seam to #12 is explicit, because #13 is built without it.** This module
consumes

- `Plan` — one candidate column: its per-day chain ids (so the repair can see
  the ops) and its per-day value in every **coupling row** the master publishes
  (`labour`, `cash_out`, `wheat_net`, `fert_net`, `stored` — the five rows of
  the #12 brief),
- `ClassMix` — the LP's answer for one class: how many tiles are in it (`N_c`),
  the plans, and their λ weights,

and it never solves an LP. `plan_from_board()` builds a `Plan` out of the
pricing oracle's output (#11) so the whole module is exercisable before the
master exists.

**What is NOT measured here, and why (R005 — a named TODO, not a guess):**

- the **integrality gap** `LP bound − rounded value` needs the LP bound, which
  only #12 has. TODO(#12): report it as a table over a board set and keep the
  simple rounding below unless it exceeds the issue's 3 % target.
- the **land loop's** budget needs a real master solve per candidate. TODO(#12):
  measure `planner/land.py::best_land` against a real `solve` callable and cap
  the candidate set from that measurement (the issue's 400 ms at day 0).

The rounding rule shipped here is deliberately the *simplest* one that can work
(the #13 brief: measure before reaching for anything clever): each tile takes the
plan with the largest λ among the plans of its own class, ties to the lowest
plan index. The quota variant the issue body mentions (floor the counts, hand out
the remainders by largest fractional part) is written down in `DESIGN.md` as the
fallback for when the gap measurement says the simple rule is not enough —
adding it before that measurement exists is exactly what R005 forbids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from agent.world.model import N_RESOURCE, RESOURCE_ID, RES_LABOR

# F029: 720 turns of 24 = 30 days, and the season ends with no liquidation.
DAYS = 30
LABOR_ID = RESOURCE_ID[RES_LABOR]

# The coupling rows of #12's brief, in the order they publish them.
ROW_NAMES: tuple[str, ...] = ("labour", "cash_out", "wheat_net", "fert_net",
                              "stored")
# LOCKED tiles carry the sentinel key (agent/obs.py); they are never planned.
LOCKED_KEY = -1


@dataclass(frozen=True)
class Plan:
    """One candidate column: a class's plan and what it costs in each row.

    `chains[d]` is the chain the tile runs on day `d` (from the graph's
    registry), so the repair can expand the plan into real ops; `rows[name][d]`
    is what the plan consumes of that coupling row on day `d`. Negative row
    values mean the plan *produces* the row (a net wheat or fertilizer
    producer), which is why the master's rows are sums, not per-tile caps.
    """

    chains: tuple[int, ...]
    value: float
    rows: dict[str, tuple[float, ...]]

    def row(self, name: str) -> np.ndarray:
        return np.asarray(self.rows[name], dtype=np.float64)


@dataclass(frozen=True)
class ClassMix:
    """One class's LP output: the plans available to it and their λ weights."""

    class_key: int
    count: int
    plans: tuple[Plan, ...]
    lam: tuple[float, ...]

    def check(self) -> None:
        if self.count < 0:
            raise ValueError(f"class {self.class_key}: negative tile count")
        if len(self.plans) != len(self.lam):
            raise ValueError(
                f"class {self.class_key}: {len(self.plans)} plans but "
                f"{len(self.lam)} weights")
        if not self.plans:
            raise ValueError(f"class {self.class_key}: no plans")
        if any(w < 0.0 for w in self.lam):
            raise ValueError(f"class {self.class_key}: negative λ")


@dataclass(frozen=True)
class Choice:
    """One tile's assignment: the plan it runs, by class and plan index."""

    class_key: int
    plan_index: int


def assign_tiles(tile_keys: Sequence[int],
                 mixes: dict[int, ClassMix]) -> list[Choice | None]:
    """One plan per tile: the largest λ in the tile's own class (issue #13 §1).

    `tile_keys` is the board in reading order (packed `TileState` keys, with
    `LOCKED_KEY` for the quadrants we do not own). A LOCKED tile gets `None` and
    never a plan: working one spends hours as a silent no-op (F042), and a plan
    that reaches the dispatcher for a locked quadrant is invisible in the score
    — which is why the issue asks for it as a regression test.

    A key the master has no mix for is also `None`: pricing a class we never
    solved would be a guess, and the caller can count the misses.
    Deterministic by construction: the argmax takes the first maximum, so ties
    fall to the lowest plan index.
    """
    choices: list[Choice | None] = []
    for key in tile_keys:
        if key == LOCKED_KEY:
            choices.append(None)
            continue
        mix = mixes.get(int(key))
        if mix is None or mix.count == 0:
            choices.append(None)
            continue
        weights = np.asarray(mix.lam, dtype=np.float64)
        choices.append(Choice(mix.class_key, int(np.argmax(weights))))
    return choices


def assign_by_quota(tile_keys: Sequence[int],
                    mixes: dict[int, ClassMix]) -> list[Choice | None]:
    """One plan per tile, keeping the MIX: floor the weights, then remainders.

    The simple rule above gives every tile of a class the plan with the largest
    λ. That was shipped as "the simplest one that can work", with the quota
    variant written down as the fallback for when the gap measurement says it
    is not enough. The measurement now exists, and it says so:

        LP objective (certified)   53,510.7
        rounded by argmax                0.0
        integrality gap                100 %      (#13's target: 3 %)

    Zero because the largest single weight on a day-0 board is the IDLE column
    — 19.8 tiles of 25 — so argmax idles the whole farm and throws away a mix
    of 21 plans. A rule that discards the mix discards the reason the master
    exists.

    The quota rule: plan `j` takes `floor(λ_j)` tiles, and the tiles left over
    go to the largest fractional parts. Deterministic — ties fall to the lowest
    plan index, as `assign_tiles` promises — and it reproduces the LP's mix
    exactly whenever the weights happen to be integral.
    """
    order: dict[int, list[int]] = {}
    for position, key in enumerate(tile_keys):
        if key == LOCKED_KEY:
            continue
        mix = mixes.get(int(key))
        if mix is None or mix.count == 0:
            continue
        order.setdefault(int(key), []).append(position)

    choices: list[Choice | None] = [None] * len(tile_keys)
    for key, positions in order.items():
        mix = mixes[key]
        lam = np.asarray(mix.lam, dtype=np.float64)
        seats = len(positions)
        whole = np.floor(lam).astype(np.int64)
        # A class may be offered more weight than it has tiles (the LP is
        # fractional and the board is not); the floors are trimmed from the
        # lightest plan up so the heaviest weights keep their seats.
        while int(whole.sum()) > seats:
            live = np.flatnonzero(whole > 0)
            whole[live[int(np.argmin(lam[live]))]] -= 1
        left = seats - int(whole.sum())
        if left > 0:
            frac = lam - np.floor(lam)
            # `-frac` sorts descending and `argsort` is stable, so equal
            # fractions fall to the lowest plan index.
            for j in np.argsort(-frac, kind="stable")[:left]:
                whole[int(j)] += 1
        seat = 0
        for j, n in enumerate(whole):
            for _ in range(int(n)):
                choices[positions[seat]] = Choice(key, j)
                seat += 1
    return choices


def counts(choices: Iterable[Choice | None]) -> dict[tuple[int, int], int]:
    """How many tiles took each plan, as `{(class_key, plan_index): n}`."""
    tally: dict[tuple[int, int], int] = {}
    for choice in choices:
        if choice is None:
            continue
        key = (choice.class_key, choice.plan_index)
        tally[key] = tally.get(key, 0) + 1
    return tally


def row_use(choices: Sequence[Choice | None],
            mixes: dict[int, ClassMix], row: str) -> np.ndarray:
    """The assignment's per-day use of one coupling row (issue #13's check)."""
    use = np.zeros(DAYS, dtype=np.float64)
    for choice in choices:
        if choice is None:
            continue
        mix = mixes[choice.class_key]
        use += mix.plans[choice.plan_index].row(row)
    return use


@dataclass(frozen=True)
class Violation:
    """One coupling row over its capacity on one day."""

    row: str
    day: int
    use: float
    capacity: float

    def describe(self) -> str:
        return (f"{self.row}[{self.day}]: {self.use:.3f} > "
                f"{self.capacity:.3f}")


def violations(choices: Sequence[Choice | None], mixes: dict[int, ClassMix],
               capacities: dict[str, Sequence[float]]) -> list[Violation]:
    """Every place the assignment exceeds a coupling row (the acceptance's row).

    The capacities are the master's (`H_d` hours, the day's money, the shed's
    100 — F043); this function is the check, not the source, so it can be run
    against any candidate assignment, rounded or not.
    """
    out: list[Violation] = []
    for row in ROW_NAMES:
        cap = capacities.get(row)
        if cap is None:
            continue
        use = row_use(choices, mixes, row)
        for day in range(DAYS):
            if use[day] > float(cap[day]) + 1e-9:
                out.append(Violation(row, day, float(use[day]),
                                     float(cap[day])))
    return out


def rounded_value(choices: Sequence[Choice | None],
                  mixes: dict[int, ClassMix]) -> float:
    """The assignment's objective — the number the gap is measured against."""
    total = 0.0
    for choice in choices:
        if choice is None:
            continue
        total += mixes[choice.class_key].plans[choice.plan_index].value
    return total


def demote_to_feasible(choices: Sequence[Choice | None],
                       mixes: dict[int, ClassMix],
                       capacities: dict[str, Sequence[float]], *,
                       max_steps: int = 1000):
    """Demote tiles until every coupling row fits (issue #13 §1's repair).

    Walks the violations in day order and, for the earliest one, switches the
    tile whose plan loses the least value per unit of the violated resource to
    its class's cheapest plan in that row. Returns `(choices, demotions,
    remaining)` — `remaining` is empty when the assignment came out feasible, and
    is left for the caller to report when no switch can reduce a violation (a
    silent `[]` would hide the one measurement this issue exists to make).

    Bounded: each step needs a strict row reduction, so no cycle can form.
    """
    out = list(choices)
    demotions: list[tuple[int, int, int]] = []
    for _ in range(max_steps):
        bad = violations(out, mixes, capacities)
        if not bad:
            break
        worst = bad[0]                              # earliest day, first row
        pick = None
        for index, choice in enumerate(out):
            if choice is None:
                continue
            mix = mixes[choice.class_key]
            plan = mix.plans[choice.plan_index]
            use = plan.row(worst.row)[worst.day]
            if use <= 0:
                continue
            alt = min(range(len(mix.plans)),
                      key=lambda j: (mix.plans[j].row(worst.row)[worst.day],
                                     -mix.plans[j].value))
            alt_use = mix.plans[alt].row(worst.row)[worst.day]
            if alt == choice.plan_index or alt_use >= use:
                continue
            loss = plan.value - mix.plans[alt].value
            gain = use - alt_use
            score = loss / gain if gain > 0 else float("inf")
            if pick is None or score < pick[0]:
                pick = (score, index, alt)
        if pick is None:
            break                                   # nothing can help; report
        _score, index, alt = pick
        old = out[index]
        assert old is not None
        out[index] = Choice(old.class_key, alt)
        demotions.append((index, old.plan_index, alt))
    return out, tuple(demotions), violations(out, mixes, capacities)


def plan_from_board(board, index: int, *, wages: np.ndarray | None = None,
                    prices: np.ndarray | None = None) -> Plan:
    """A `Plan` from the pricing oracle's output (#11) — the stand-in for #12.

    The row values are derived the way the master's rows are defined:

    - `labour[d]` = the plan's `LABOR_HOURS` on day `d`;
    - `cash_out[d]` = its input costs at `wages` (0 without wages: the cash row
      is the master's, and inventing a price here would be a guess — R005);
    - `wheat_net[d]` / `fert_net[d]` = consumed − produced for that resource
      (positive = a net buyer, which is how the master's `≤ buy_d` rows read);
    - `stored[d]` = the running sum of every non-labour resource the plan has
      added to the shed so far — the shed's 100 (F043) is a *cumulative* cap,
      so the row is cumulative by construction.
    """
    days = board.days
    cost = board.per_day_cost[index]           # (days, N_RESOURCE) int64
    produce = board.per_day_produce[index]
    chains = tuple(int(chain) for _d, _s, chain in board.plans[index][:days])

    labour = cost[:, LABOR_ID].astype(np.float64)
    if wages is None:
        cash = np.zeros(days, dtype=np.float64)
    else:
        # `wages` may be the flat resource vector or the master's per-day path;
        # broadcasting handles both, and the sum is per day either way.
        cash = np.sum(np.asarray(cost, dtype=np.float64)
                      * np.asarray(wages, dtype=np.float64), axis=1)
    wheat = (cost[:, RESOURCE_ID["WHEAT"]] - produce[:, RESOURCE_ID["WHEAT"]])
    fert = (cost[:, RESOURCE_ID["FERTILIZER"]]
            - produce[:, RESOURCE_ID["FERTILIZER"]])
    storable = np.ones(N_RESOURCE, dtype=bool)
    storable[LABOR_ID] = False                 # hours are spent, not stored
    stored = np.cumsum(np.asarray(cost - produce, dtype=np.float64)[:, storable]
                       .sum(axis=1) * -1.0)
    value = float(np.sum(produce * (prices if prices is not None else 0))
                  - np.sum(cost * (wages if wages is not None else 0)))
    return Plan(chains=chains, value=value,
                rows={"labour": tuple(labour),
                      "cash_out": tuple(cash),
                      "wheat_net": tuple(wheat.astype(np.float64)),
                      "fert_net": tuple(fert.astype(np.float64)),
                      "stored": tuple(stored)})
