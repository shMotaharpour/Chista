"""Every number the manager is tuned on, in ONE place. Numbers only.

This module is the single home of the agent's tunable quantities. Nothing in
`manager/`, `planner/` or `tile_dp/` may hold a cap of its own: a limit that
lives next to the code that reads it cannot be injected (the `Agent` class takes
this object), cannot be dumped for a measurement (`Config.dump`), and drifts
into a second copy of a decision.

Two kinds of number live elsewhere, on purpose:

- **Engine facts** (`world/`, cited to the engine's own lines): the season's
  length, the board, the shed's capacity, the hire ladder, the town's intervals.
  They are not tunable and the manager READS them — from `world` or from the
  run's own configuration when the observation carries one — rather than
  restating them.
- **Model structure** that a solver needs as an argument (a horizon, a class
  key) is not config either; it is passed in.

Config holds NUMBERS. It never holds a value that selects a code path.
`use_master=True` is `CHISTA_REPLAN` in a new coat, and five of those switches
hid for four days the fact that nothing was wired: the agent answered every
turn with the greedy rung and scored 2,840 against a PASS opponent's 3,000. A
number that is wrong makes the agent play worse. A switch that is wrong makes
it play something else entirely, and the difference never shows in a diff.

The test for a field belongs here: **would changing it take a different
branch?** If yes it is not config.

`Config()` is the committed default. It must be PRESENT AND LEGAL, not optimal
— the tuned one ships beside the agent as `agent/artifact/config.json` and
`Config.load()` reads it when it is there. While the numbers are still moving
that file is gitignored, so a missing file is the ordinary case and never an
error.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

#: Where the tuned numbers ship. Beside the agent, inside the uploaded repo.
ARTIFACT = Path(__file__).resolve().parent / "artifact" / "config.json"


@dataclass(frozen=True)
class Config:
    """The tuned numbers. Every field is a quantity; none is a mode.

    One field is a boolean — `never_raise` — and it is here on purpose: it
    selects no plan path (the same policy runs either way), it only decides
    whether a failure is VISIBLE. A field that changes WHICH policy runs is
    still a switch in a new coat and still does not belong here.
    """

    # --- the never-raise boundary -----------------------------------------
    #: Whether a failed turn is CAUGHT and passed, or re-raised.
    #:
    #: ON — the submission's contract: the harness is never handed an exception.
    #: The failure is recorded on `Agent.failures`, its day marked, the turn
    #: answers all-PASS, and the `A` line says so.
    #: OFF — the same record and the same `A` line, and then the exception is
    #: RE-RAISED with its traceback, so a run that reaches the boundary says WHAT
    #: reached it instead of passing the day. A silent PASS day is how the day
    #: layer's short-horizon `compile_route` refusal stayed invisible for a whole
    #: investigation.
    #:
    #: **OFF right now, by the owner's order (2026-09-23):** the hunt wants the
    #: traceback. It MUST be ON in the submission — an exception out of
    #: `agent.main` is an episode the harness cannot score.
    #:
    #: It is not a policy switch and not a fallback ladder: the plan, the market
    #: orders and the dispatcher are untouched by it. What it decides is whether
    #: a failure is VISIBLE, which is why it is allowed here where a
    #: path-choosing flag is not — and it says so next to the field.
    never_raise: bool = False

    # --- the master (the column-generation solve) -------------------------
        #: Column-generation rounds per solve, and the round count is a DECISION, not a
    #: race with the clock
#: with the clock: consulting the clock between rounds made two runs of one seed
#: disagree. The day's objective is monotone in the rounds and the wall cost is
#: linear in them, so the cap trades a measured objective for time -- a cap that is
#: too low throws the objective away, because every round prices the pool again
#: against the duals the last round produced.
    master_rounds: int = 2
    #: Solve the day as the integer program it is: `colgen.solve(integral=True)`.
    #: The LP relaxation lets a class's weight be fractional, so a plan can be
    #: half-committed and the tile counts a fraction of what they claim. False is
    #: the LP path this project has always run.
    day_integral: bool = False
    #: How many of the engine's quadrants the day may BUY (`world.rules.LAND_PRICES`,
    #: prefix order), one binary per (quadrant, day). 0 keeps every land row off, so
    #: no land is purchasable and the model sees only the quadrants it already owns.
    land_quadrants: int = 0
    #: The LP loop's own cap when a caller asks for no specific number (the
    #: library default; the manager always asks for `master_rounds`). 8 is the
    #: measured oscillation budget: the full loop's cost is the contractor's
    #: re-pricing sweep (~9.3 ms a round measured), so eight rounds is ~80 ms on
    #: the box the figures were taken on, and a board that has not settled by then
    #: is oscillating rather than converging.
    iter_cap: int = 8
    #: How many depth blocks the master prices a day's sells with. The curve is
    #: belief's (`belief.depth.sell_blocks`); the count is this LP's own modelling
    #: resolution — a maximising LP fills the rich blocks first by itself, so more
    #: blocks price the same curve more finely and cost one column block each
    #: (5 blocks x 9 goods x 30 days = 1,350 columns of the model).
    sell_blocks: int = 5
    #: How many sd of the town drain to price a sale against, on top of the
    #: forecast's own inventory (`belief.opponent.quantile_price_floor`, the same
    #: `z`): 0.0 is the mean ladder the model has always used, and any positive
    #: value prices the curve from a fuller market, which is the conservative
    #: direction. The owner's ruling: risk must be priced as a price.
    sell_risk_z: float = 0.0

    #: The cash tail's weight in the master's objective (#111). The objective
    #: becomes `(1 - kappa) * mean + kappa * low`, which is a convex combination
    #: of the mean profit and its CVaR: at three equal-weight scenarios the mean
    #: of the worst third IS the low band, so kappa = 0 is today's behaviour and
    #: kappa = 1 is the worst case. Priced in the objective, never as a row --
    #: the owner's ruling, and the reason the term is a weight rather than a
    #: constraint.
    #: The R-U weight on the worst scenario. The DEFAULT is 0 -- the mean model,
    #: bit-identical to the behaviour before this block existed -- because a
    #: default is what every caller who says nothing gets, and the suites are such
    #: callers: with a non-zero default, 18 tests that assert mean-regime facts
    #: silently changed regime, and five of them inherited it through a shared
    #: `AGENT.cfg` set by an earlier test. The regime is a RUN decision, so the
    #: working arm sets it where it runs (`risk_kappa=0.5`, no data behind the
    #: value yet) and this field stays the honest zero until the 6-seed arm has
    #: measured it (R005).
    risk_kappa: float = 0.0

    #: Weight of the EXPECTED profit across the scenario worlds, beside the
    #: pessimistic kappa term. 0.0 = today's behaviour, bit-identical: the
    #: objective is then only the CVaR penalty, which is insurance, not portfolio.
    risk_mean_weight: float = 0.0
    #: The tail CVaR averages over: `alpha` of the scenarios, so 0.10 means the
    #: worst tenth of the profit distribution. Rockafellar-Uryasev, so the LP
    #: needs `t` and one `u_s` per scenario and NO binary variable -- which is
    #: the whole reason CVaR and not VaR is the one that fits here.
    risk_alpha: float = 0.10
    #: How many rounds of pricing carry OUR OWN planned supply. Measured with 1
    #: (seed 33, 10 days, vs v3-agent): the day-value/realised-coins correlation
    #: went -0.428 -> +0.045, and the season 93,315 -> 101,925 over three seeds
    #: (33: 16,025 -> 27,155; 7: 46,789 -> 31,275; 5: 30,501 -> 43,495). Two of
    #: three seeds improved and one lost a third, so the default stays 0 and this
    #: is a swappable choice until a wider seed set settles it.
    #: path the forecast builds on its own walk (the town's drain only), which
    #: under-prices nothing and over-prices every far day: measured on the
    #: seed-33 board, MILK's day-20 quote reads 202 coins while the ladder pays
    #: far less for the units we plan to pour in that day. A positive value
    #: re-prices each day against `market inventory + our supply`, which is the
    #: price the sale would actually face.
    price_supply_rounds: int = 0
    #: The lot the day's sale is priced as, when the caller has not handed in
    #: one yet (units per good per day). 0 keeps every price at its own quote;
    #: a positive value prices the sale through the ladder for a lot that size,
    #: which is the counterweight to F035's rising path: a big lot never fetches
    #: the peak. The shed's capacity is the conservative choice.
    sell_lot_default: float = 0.0
    #: The daily discount rate on the tile DP's own cash flows (net present value).
    #: 0.0 keeps today's behaviour. The literature's tool for mixing fast- and
    #: slow-payback work is a RATE, not a truncation: the taper that zeroed prices
    #: in ten days measured worse than not tapering at all, and worse the sharper it
    #: got (over three seeds against the strong rival: 101,925 at off, 95,117 at
    #: five days, 73,692 at ten, 40,553 at fifteen), while a rate discounts every
    #: cash flow -- the costs it delays included -- and is the same NPV the
    #: capital-budgeting and cash-flow-duration frameworks use.
    discount_rate: float = 0.0
    #: Cap a sale price by what the engine really paid (agent/artifact/
    #: sell_price_caps): a drain-only forecast keeps rising past the day the
    #: season's shops have decided, so the plan is priced against a price nobody
    #: paid. The cap is the day's BEST demand bucket, so the upside of an open
    #: shop survives and only the hope above every real world goes. False keeps
    #: the shipped model exact.
    price_cap_from_archive: bool = False
    #: Price the PLAN on the bad day, not the average one: the sale path is floored
    #: at `z` standard deviations of the town's own drain (`drain_forecast`
    #: owns those two moments, exactly, without sampling). Risk as a price --
    #: the mean-CVaR objective of #111 in the form the ladder already speaks.
    #: 0.0 keeps the shipped model exact.
    price_risk_z: float = 0.0
    #: What a unit sold BEYOND the town's own appetite fetches, as a fraction of
    #: the day's price. That tier is the legacy two-tier model, used when the
    #: caller hands in no depth curve; with a curve the ladder's own blocks carry
    #: the price. Measured on the day-0 board: the first 100 MILK units average
    #: 62.0 against a 160 quote and the first 200 MELON units 132.6 against a 250
    #: quote, so one flat fraction is neither a bound nor an estimate — it is the
    #: crudest cut of a staircase, kept for the path with no forecast.
    sell_deep_factor: float = 0.5
    #: Tâtonnement damping on the price the REST of the agent reads. It may not
    #: touch the pricing step — the reduced-cost test is only a reduced cost of
    #: the LP whose duals it used. 0.5 is the measured sweep over (0.2, 0.35, 0.5,
    #: 0.7) on a fixed two-class board (pinned in `tests/test_master.py`): it
    #: reached dual-stationarity in the fewest rounds.
    alpha: float = 0.5
    #: The reduced-cost tolerance: below it a plan is within the pricer's own noise
    #: of the pool's plans and adding it fattens the master for nothing. The
    #: absolute floor is float32's epsilon, and `rc_rel_tol` carries it onto the
    #: objective's scale, because `rc` is a difference of coins: at an objective of
    #: 35,772 a 1e-6-absolute test was refusing to certify on the pricer's own
    #: rounding (measured: the loop stalled on rc 2.24e-4 and certified the SAME
    #: objective, 35,772.2194, once the tolerance was read as `1e-6 * |objective|`).
    rc_tol: float = 1e-3
    rc_rel_tol: float = 1e-3

    # --- the labour model the rows price ------------------------------------
    #: The labour row's travel/carry overhead: `H_d = 24*(1 + hands) - hands`, times
    #: this. #12's brief puts it at "start at 35 % and measure"; the day layer's own
    #: measurement (fewer hours bought fewer tiles and a LONGER route, 1.14x-1.29x)
    #: is why this stays a flat factor instead of a feedback scalar.
    hours_overhead: float = 0.35
    #: The clamp on the published labour dual. The tile DP's answer is the idle
    #: chain above ~147 coins/hour on this graph (measured sweep on the day-0 board:
    #: bare-tile value 420 at w=100, 60 at 140, 15 at 145, 0 at 147), so the clamp
    #: sits at the last live price: a wage past this edge cannot move any tile, and
    #: a diverging tâtonnement degraded to a frozen board would freeze the farm.
    labour_dead_edge: float = 145.0

    # --- the day ------------------------------------------------------------
    #: The most hands the day layer will even SCAN (`day.scan_ceiling`). The
    #: estimate it competes with is the manager's own (`planner/hands.py`), and
    #: the scan's top is `min(estimate, this)`.
    #:
    #: One, and NOT a measured preference: the table that used to justify it (a
    #: season at 1 hand against 4: 32,045/35,697 against 5,148/22,988) was
    #: taken on a manager and belief layer that have since been replaced, so it
    #: is retired as a basis rather than re-quoted. What that table was about --
    #: a committed day being re-derived every morning, so more capacity undoes
    #: more of it -- is still the open question (#79), and the value must be
    #: chosen again on a fresh measurement, or by the model itself once the
    #: hands are bought inside the MILP instead of scanned for.
    max_hands: int = 16
    #: Whether the MODEL buys the day's labour -- the `delta` columns of the
    #: labour row, priced by `world.rules.hire_cost` -- instead of the day
    #: layer scanning hand counts and paying the bill outside the matrix.
    #: OFF until a season on the frozen trees says otherwise: with it off the
    #: matrix is exactly the one that shipped (the flag reaches `solve` through
    #: `equilibrate`, and the block only exists when it is on).
    buy_hands: bool = False
    #: Master solves one `plan` may spend correcting the hours it committed.
    fit_rounds: int = 2
    #: The step the hours correction takes when the day did NOT fit: each round
    #: re-solves with `hours / applied`, where `applied` grows by at least
    #: `max(1 + hours_tolerance, the route's own overhead)`. A day that fits is
    #: never corrected back up (the search starts at the arithmetic floor, so a
    #: complete answer is not evidence of slack).
    hours_tolerance: float = 0.02
    #: How many times the day layer may re-ask its own search for a pool the day
    #: actually fits in. Fixed and small on purpose (each ask is a whole search):
    #: offering 5 hands and carrying the day with 1 must report 1, not walk down
    #: one search at a time.
    ask_rounds: int = 2
    #: Wentges dual-price smoothing (#87 follow-up sweep, 2026-09-22): the
    #: pricing step is fed `alpha*centre + (1-alpha)*LP`, where centre is the
    #: best-bound incumbent. Measured on the day-0 board with the exact
    #: pricer: alpha 0.0 = 75 rounds / 1720 ms; 0.5 = 60 / 1350; 0.7 = 55 /
    #: 1165 (-32% rounds and wall); 1.0 = 60 / 1289. Objective and bound are
    #: identical (35,772, gap 0.0000%) at every alpha — smoothing changes
    #: HOW FAST the certificate arrives, never WHAT it certifies. 0.7 is the
    #: sweep's pick. In-season (warm pool, day 3) the round count is 9-10
    #: either way, so the win is concentrated on cold/certifying solves.
    smoothing: float = 0.7

    def __post_init__(self) -> None:
        #: A typo guard, not an engine cap: the engine's own limit on a hand pool is
        #: the hire ladder (`world/rules.hire_hour`: the k-th hand of a day acts from
        #: `k//10 + 1`), and the day's work bounds it from above in any case.
        if not 1 <= self.max_hands <= 16:
            raise ValueError(f"max_hands {self.max_hands} is outside 1..16")
        if not 0.0 < self.alpha <= 1.0:
            raise ValueError(f"alpha {self.alpha} is outside (0, 1]")
        if self.master_rounds < 1 or self.iter_cap < 1:
            raise ValueError("a round cap below 1 is not a cap")
        if self.sell_blocks < 1:
            raise ValueError(f"sell_blocks {self.sell_blocks} is below 1")
        if self.rc_tol <= 0.0 or self.rc_rel_tol <= 0.0:
            raise ValueError("the reduced-cost tolerances must be positive")
        if not 0.0 <= self.hours_overhead < 1.0:
            raise ValueError(
                f"hours_overhead {self.hours_overhead} is outside [0, 1)")

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """The tuned numbers if they shipped, the defaults if they did not.

        A missing file is the ordinary case while the numbers move, so it is
        not an error. A file that is present and malformed IS: falling back to
        the defaults on a bad file is how a tuned agent plays untuned and
        nobody finds out.
        """
        target = ARTIFACT if path is None else Path(path)
        if not target.exists():
            return cls()
        data = json.loads(target.read_text())
        unknown = set(data) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(
                f"{target}: unknown config keys {sorted(unknown)} — a number "
                f"that is not read is a number that is not tuned")
        return cls(**data)

    def dump(self, path: Path | None = None) -> None:
        """Write these numbers where `load()` will find them."""
        target = ARTIFACT if path is None else Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True)
                          + "\n")
