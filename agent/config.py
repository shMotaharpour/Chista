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
    """The tuned numbers. Every field is a quantity; none is a mode."""

    # --- the turn's clock --------------------------------------------------
    #: The working budget inside one turn, in ms. F046: one free second per
    #: turn, unbankable, and the harness bills ~35 ms more than measured.
    turn_budget_ms: float = 965.0
    #: Held back for compiling and dispatching, so a solve that runs to its
    #: deadline still leaves the turn a legal answer.
    reserve_ms: float = 140.0

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
    #: morning and more capacity means more of it is undone.
    #:
    #: It is ZERO now, and zero is a real setting: the farm runs on the farmer
    #: alone, `hire_times` is empty, and `ceiling_for` therefore answers one —
    #: the search cannot return a pool nobody offered to pay for. Nothing about
    #: hiring is disabled; the offer is simply empty, so every hiring path is
    #: exercised with the number it is given.
    max_hands: int = 0
    #: Seconds the day search may spend. None lets it run to its own end,
    #: which is what an offline measurement wants and a turn does not.
    search_budget_s: float = 0.25
    #: Master solves one `plan` may spend correcting the hours it committed.
    fit_rounds: int = 2

    #: Price only the tile the farmer is standing on, not the whole farm.
    #:
    #: The farmer spawns on a shed-access tile (F040) and that tile is one of
    #: the 25 the farm owns — it is the distance-0 class. Pricing it alone
    #: removes travel, allocation and rounding from the loop in one move, and
    #: leaves exactly belief -> DP -> the day. It is a diagnostic setting and
    #: it is meant to be turned off again.
    one_tile: bool = True

    # --- the market --------------------------------------------------------
    #: Orders per turn the engine accepts (F031). A cap, not a target.
    max_orders_per_turn: int = 10

    def __post_init__(self) -> None:
        if self.turn_budget_ms <= self.reserve_ms:
            raise ValueError(
                f"a turn budget of {self.turn_budget_ms} ms leaves nothing "
                f"after the {self.reserve_ms} ms reserve")
        if not 0 <= self.max_hands <= 16:
            raise ValueError(f"max_hands {self.max_hands} is outside 0..16")
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
