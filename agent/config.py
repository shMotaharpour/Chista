"""Every number the agent is tuned on, in one place. Numbers only.

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

    One field is a boolean — `log_gaps` — and it is here on purpose: it selects
    no plan path (the same turns are played with it on or off), it only decides
    whether the evidence is printed. A field that changes WHICH policy runs is
    still a switch in a new coat and still does not belong here.
    """

    # --- the turn's clock --------------------------------------------------
    #: The working budget inside one turn, in ms. F046: one free second per
    #: turn, unbankable, and the harness bills ~35 ms more than measured.
    turn_budget_ms: float = 965.0
    #: Held back for compiling and dispatching, so a solve that runs to its
    #: deadline still leaves the turn a legal answer.
    reserve_ms: float = 140.0
    #: Print the wall clock between two calls of the agent, with the running
    #: mean and sd of the season so far.
    #:
    #: Off by default: 719 `G` lines belong to a run whose log we mean to read,
    #: not to every local season. Turn it on for the submission we care about.
    #: The gap is measured from the end of our previous turn to the start of
    #: this one, so on the grader — where the two seats run one after the other
    #: (F058) — it carries the engine's own overhead plus, above that floor,
    #: whatever the opponent spent thinking: the one reading of the rival's
    #: resource use a submission can take from inside.
    log_gaps: bool = False

    # --- the contractor ----------------------------------------------------
    #: Days the tile DP looks ahead when it prices the board.
    horizon_days: int = 20

    # --- the master --------------------------------------------------------
    #: Pricing rounds one `equilibrate` may spend. The certificate usually
    #: arrives well inside it (58 rounds cold on a day-0 board, 1 warm); the
    #: cap is what stops a board that will not converge from eating the turn.
    master_rounds: int = 200
    #: Damping on the price the rest of the agent reads. It may not touch the
    #: pricing step — the reduced-cost test is only a reduced cost of the LP
    #: whose duals it used.
    damping: float = 0.5

    # --- the day -----------------------------------------------------------
    #: The largest hand pool the day layer may offer. wsr caps at 16.
    #:
    #: One, measured. Three seasons at each setting, medians:
    #:
    #:     idle edges off, 1 hand   32,045      off, 4 hands    5,148
    #:     idle edges on,  1 hand   35,697      on,  4 hands   22,988
    #:
    #: More hands is worth five times the plan to the LP and less than nothing
    #: on the board, because the day the master commits is re-derived every
    #: morning and more capacity means more of it is undone. Raising this is
    #: the FIRST thing to try once a plan survives the night (#79's commitment
    #: work) — it is capped low because the churn is not fixed, not because
    #: hiring is bad.
    max_hands: int = 1
    #: Seconds the day search may spend. None lets it run to its own end,
    #: which is what an offline measurement wants and a turn does not.
    search_budget_s: float = 0.25
    #: Master solves one `plan` may spend correcting the hours it committed.
    fit_rounds: int = 2
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

    # --- the market --------------------------------------------------------
    #: Orders per turn the engine accepts (F031). A cap, not a target.
    max_orders_per_turn: int = 10

    def __post_init__(self) -> None:
        if self.turn_budget_ms <= self.reserve_ms:
            raise ValueError(
                f"a turn budget of {self.turn_budget_ms} ms leaves nothing "
                f"after the {self.reserve_ms} ms reserve")
        if not 1 <= self.max_hands <= 16:
            raise ValueError(f"max_hands {self.max_hands} is outside 1..16")
        if not 0.0 < self.damping <= 1.0:
            raise ValueError(f"damping {self.damping} is outside (0, 1]")
        if self.horizon_days < 1:
            raise ValueError(f"horizon_days {self.horizon_days} is below 1")

    @property
    def solve_budget_ms(self) -> float:
        """What one turn may spend thinking, after the reserve."""
        return self.turn_budget_ms - self.reserve_ms

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
