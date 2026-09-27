"""Guards for #87: the published labour dual must stay inside the range
where the tile DP answers with a non-idle chain, and the wage floor must
price the farmer's own hour at what the farm's pipeline produces — not at
the 1-coin marginal hand.

The defect this pins, measured: `dual_stand_in` floored the wage at
0.043/h (the marginal hire over 23 hours), the tâtonnement published
947-2131 in a season, and the DP returns identically the idle chain above
~147 — so the farm froze with 16-18 of 25 tiles idle and a purse that
never rose once in 14 days.

Epic: the third guard turns a whole season of the tatonnement, so it is marked `epic`
and left out of the default run - `pytest -m epic` runs it. The two floors above it
are cheap and structural and stay in the default run.

Run:  .venv/bin/python -m tests.test_labour_dead_zone   (also pytest)
"""

from __future__ import annotations

import numpy as np
import pytest

from offline_lab.kaggle_env import new_environment
from agent.config import Config
from agent.planner import master as M
from agent.planner.inputs import (dual_stand_in, farmer_hour_floor,
                                  load_contractor)
from agent.world.model import RESOURCE_ID

#: The measured dead edge on this graph: bare-tile value 420 at w=100,
#: 15 at 145, 0 at 147 (the #87 sweep, re-measured). The shipped number is
#: `Config.labour_dead_edge` and it must stay at the last LIVE price, not
#: inside the dead zone.
DEAD_EDGE = Config().labour_dead_edge


def test_the_floor_prices_the_farmer_not_the_first_hand() -> None:
    """`dual_stand_in`'s labour floor is the farmer's hour (~10.3/h), not
    the 1-coin marginal hand (0.043/h) that made rebuilding free."""
    obs = {"market": {"prices": {"WHEAT": 25, "FERTILIZER": 40}},
           "farms": [{"hires_today": 0, "hands": []}], "player": 0}
    _p, w = dual_stand_in(obs, days=20)
    floor = float(w[0, RESOURCE_ID["LABOR"]])
    assert floor >= farmer_hour_floor(), floor
    assert farmer_hour_floor() > 1.0, (
        "a ~0.04/h floor prices destruction at nothing (#87's replay)")


def test_the_dead_edge_is_the_last_live_price() -> None:
    """The DP answers the idle chain above ~146 on the day-0 board — the
    constant must sit at the last live price (value 15 at 145), not
    inside the dead zone."""
    env = new_environment()
    obs = env.state[0].observation
    c = load_contractor(days=20)
    p, w = dual_stand_in(obs, days=20)
    from agent.planner.master import _owned_states
    owned = _owned_states(object(), obs)[:3]
    w[:, 0] = DEAD_EDGE
    board = c.price(p, np.asarray(w), owned, travel_hours=0)
    assert float(board.tile_values[0]) > 0.0, (
        f"Config.labour_dead_edge={DEAD_EDGE} is inside the DP's dead zone: "
        "the bare tile prices at 0 there — the clamp would freeze the farm")
    w[:, 0] = DEAD_EDGE + 10.0
    board = c.price(p, np.asarray(w), owned, travel_hours=0)
    assert float(board.tile_values[0]) == 0.0, (
        "the dead zone moved UP past the edge+10: re-measure "
        "Config.labour_dead_edge from the sweep")


@pytest.mark.epic
def test_the_published_labour_dual_never_enters_the_dead_zone() -> None:
    """One full season (seed 3, the issue's fixture): every published
    w_labour cell stays at or under the dead edge, and the clamp counter
    says whether the raw loop wanted past it."""
    from agent.config import Config
    from agent.dispatch import dispatch_plan
    from agent.manager.core import Manager
    PASS = {"farmer": ["PASS"], "hands": [], "market": []}
    env = new_environment(configuration={"seed": 3})
    m = Manager(Config())
    clamped = [0]

    def me(obs, config=None):
        if int(obs["hour"]) == 0:
            m.observe(obs, config)
            day = getattr(m.day.master, "labour_clamped_cells", 0) \
                if m.day is not None else 0
            clamped[0] += day
            w = np.asarray(m.duals) if m.duals is not None else None
            if w is not None and w.size:
                assert float(np.max(w[:, 0])) <= DEAD_EDGE + 1e-9, (
                    f"a published w_labour cell entered the dead zone "
                    f"({float(np.max(w[:, 0])):.1f} > {DEAD_EDGE})")
        else:
            m.step()
        return dispatch_plan(m.best(), obs)

    env.run([me, lambda obs, config=None: dict(PASS)])
    # the counter is the honest record: a season that never clamps never
    # needed the guard, one that clamps says the raw loop diverged
    print(f"    (clamped cells this season: {clamped[0]})")


def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:                       # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} dead-zone checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
