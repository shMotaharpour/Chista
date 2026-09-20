"""Every number the agent is tuned on, in one place. Numbers only.

Config holds NUMBERS. It never holds a value that selects a code path.
`use_master=True` is `CHISTA_REPLAN` in a new coat, and five of those
switches hid for four days the fact that nothing was wired: the agent
answered every turn with the greedy rung and scored 2,840 against a PASS
opponent's 3,000. A number that is wrong makes the agent play worse. A
switch that is wrong makes it play something else entirely, and the
difference never shows in a diff.

The test for a field belongs here: **would changing it take a different
branch?** If yes it is not config. `beam_width = 0` meaning "let wsr
choose" is the one place that rule bends, and it bends because `None` is
wsr's own documented default for that argument, not a mode of ours.

`Config()` is the committed default. It must be PRESENT AND LEGAL, not
optimal — the tuned one ships beside the agent as
`agent/artifact/config.json` and `Config.load()` reads it when it is
there. While the numbers are still moving that file is gitignored, so a
missing file is the ordinary case and never an error.
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

    # --- the turn's clock ------------------------------------------------
    #: The working budget inside one turn, in ms. F046: one free second per
    #: turn, unbankable, and the harness bills ~35 ms more than measured.
    turn_budget_ms: float = 965.0
    #: What one `Manager.step()` may spend. A turn holds several probes only
    #: if this is well under `turn_budget_ms`; at the default it holds one.
    probe_budget_ms: float = 700.0
    #: Held back from `probe_budget_ms` for compiling and dispatching, so a
    #: probe that runs to its deadline still leaves the turn a legal answer.
    reserve_ms: float = 120.0

    # --- the contractor ---------------------------------------------------
    #: Days the tile DP looks ahead when it prices the board.
    horizon_days: int = 20

    # --- the scarcity price of an hour ------------------------------------
    # The manager's one lever. Labour free means the DP asks for every op it
    # can justify; the price rises until the day wsr is handed actually fits.
    # These bracket the bisection, in coins per worker-hour.
    wage_floor: float = 0.0
    wage_ceiling: float = 64.0
    #: The bracket stops closing below this width: further probes buy less
    #: than the turns they cost.
    wage_tolerance: float = 0.5
    #: The first unbracketed probe, and the factor it grows by while no wage
    #: has been found that fits.
    wage_first: float = 2.0
    wage_growth: float = 4.0

    # --- the day's workforce ----------------------------------------------
    #: The largest hand pool the manager will pay for. wsr caps at 16.
    max_hands: int = 8
    #: The beam width handed to wsr. 0 asks wsr for the width its own
    #: `beam_for` implies from the day's size.
    beam_width: int = 0

    # --- the market -------------------------------------------------------
    #: Orders per turn the engine accepts (F031). A cap, not a target.
    max_orders_per_turn: int = 10
    #: Coins kept back from the day's shopping, so a plan never spends the
    #: farm down to nothing on inputs.
    cash_reserve: float = 200.0

    def __post_init__(self) -> None:
        if self.wage_floor < 0 or self.wage_ceiling <= self.wage_floor:
            raise ValueError(
                f"wage bracket is empty: floor {self.wage_floor} ceiling "
                f"{self.wage_ceiling}")
        if self.probe_budget_ms <= self.reserve_ms:
            raise ValueError(
                f"probe budget {self.probe_budget_ms} ms leaves nothing after "
                f"the {self.reserve_ms} ms reserve")
        if not 1 <= self.max_hands <= 16:
            raise ValueError(f"max_hands {self.max_hands} is outside 1..16")

    @property
    def search_budget_s(self) -> float:
        """What one probe hands wsr as `budget_s`, in seconds."""
        return (self.probe_budget_ms - self.reserve_ms) / 1000.0

    @property
    def beam(self) -> int | None:
        """The width for `wsr.beam.search`, or None to let wsr size it."""
        return int(self.beam_width) or None

    @classmethod
    def load(cls, path: Path | None = None) -> "Config":
        """The tuned numbers if they shipped, the defaults if they did not.

        A missing file is the ordinary case while the numbers are moving, so
        it is not an error. A file that is present and malformed IS an error:
        silently falling back to defaults is how a tuned agent plays untuned
        and nobody finds out.
        """
        target = ARTIFACT if path is None else Path(path)
        if not target.exists():
            return cls()
        data = json.loads(target.read_text())
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(
                f"{target}: unknown config keys {sorted(unknown)} — a number "
                f"that is not read is a number that is not tuned")
        return cls(**data)

    def dump(self, path: Path | None = None) -> None:
        """Write these numbers where `load()` will find them."""
        target = ARTIFACT if path is None else Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")
