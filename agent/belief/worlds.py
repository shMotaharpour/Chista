"""The planning worlds, read once, from the agent's own artifact.

The model plans against several futures: what their harvesting puts on the market, and what the
town wants, as one joint world with a weight. Those are the two axes the archive could measure
(the supply is the REALISED flow, from the city's inventory identity -- orders are not sales).

This is the only reader of that artifact, for the same reason `tracker` is the only reader of the
market: two readers of one thing disagree eventually, and the one that is not maintained reads
something wrong in silence.

Deliberately NOT here: prices. A world becomes a price path only through the market module's own
`_priced_paths`, so there is one definition of a price path, not two.

Absence is normal: no artifact means today's single world, and nothing changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from agent.belief.market import PRODUCTS

#: The artifact ships beside the agent (AGENTS.md: everything the entry point loads lives inside
#: `agent/`), so it is found relative to this file and never by an absolute path.
ARTIFACT = Path(__file__).resolve().parents[1] / "artifact" / "scenario_worlds.npz"


@dataclass(frozen=True)
class World:
    """One future: their supply and the town's appetite, day by day, with its share of belief."""

    weight: float
    rival_supply: np.ndarray     # (days, 9) units their harvesting really put on the market
    demand: np.ndarray           # (days, 9) units the town really wants
    phase: tuple[int, int]       # the day window this world was cut from
    support: float               # game-days behind it: the evidence for its weight

    def days(self) -> int:
        return int(self.rival_supply.shape[0])


@lru_cache(maxsize=1)
def load(path: Path | str | None = None) -> tuple[World, ...]:
    """The worlds in the artifact, or an empty tuple when it is missing.

    Empty is a supported state, not a failure: the caller keeps today's single plan, and the
    numbers stay bit-identical because nothing was added to them.
    """
    file = Path(path) if path is not None else ARTIFACT
    if not file.exists():
        return ()
    with np.load(file) as data:
        # The market reads its arrays in `PRODUCTS` order, so this artifact must be built in the
        # same order or every good lands in the wrong column -- silently, since the shapes match.
        # One definition of the order, checked where the file is opened, named in the error.
        order = tuple(str(x) for x in data["crop"])
        if order != tuple(PRODUCTS):
            raise ValueError(f"{file.name}: crop order {order} is not the market's {tuple(PRODUCTS)}")
        supply = np.asarray(data["rival_supply"], dtype=np.float64)
        demand = np.asarray(data["demand"], dtype=np.float64)
        weights = np.asarray(data["weights"], dtype=np.float64)
        p0 = np.asarray(data["phase0"], dtype=np.int64)
        p1 = np.asarray(data["phase1"], dtype=np.int64)
        support = np.asarray(data["support"], dtype=np.float64)
        n = supply.shape[0]
        if not (demand.shape == supply.shape and weights.shape == (n,) and p0.shape == (n,)):
            raise ValueError(f"{file.name}: arrays disagree on the world count ({n})")
        total = float(weights.sum())
        if abs(total - 1.0) > 1e-5:
            # The builder asserts this too; a reader that trusts it silently is how a bad weight
            # reaches a plan. Fail here, where the file is named.
            raise ValueError(f"{file.name}: weights sum to {total:.6f}, not 1")
        return tuple(World(float(w), supply[i], demand[i], (int(p0[i]), int(p1[i])), float(support[i]))
                     for i, w in enumerate(weights))


def weights_sum(path: Path | str | None = None) -> float:
    """The weights' total, for a caller that wants to state the evidence beside a plan."""
    worlds = load(path)
    return float(sum(w.weight for w in worlds))
