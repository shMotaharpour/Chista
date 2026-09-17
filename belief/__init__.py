"""belief/ — what the market and the rival are doing, and what to sell.

The sell side of the agent (issue #54, parent #8). Four pieces and a contract:

- `schemas.py` — the messages the layers exchange. Nobody commands an op across a
  boundary; a message says what must be true and by when.
- `tracker.py` — phase 1: the flow residual. Exact for the seven one-way goods;
  net-only for WHEAT and FERTILIZER, because their buying and selling enter the
  same identity with the same sign.
- `opponent.py` — phase 2: their action counts, the closed-form demand forecast,
  and the inference of the order they filled their ten slots in.
- `solvers.py` — phase 3: the exact maximin mix of the slot game (an LP), its
  risk-adjusted continuous sibling (SLSQP), and the maximin season LP whose
  absorption row is what stops it selling 4,000 units into a 525-unit drain.
- `stubs.py` — what runs until each of those units exists, each naming the issue
  that retires it.

`bench/bench_market_analyzer.py` reproduces the measurements quoted in the
docstrings; `tests/test_market_analyzer.py` guards the claims that matter.
"""

from belief.opponent import (
    OpponentModel, drain_forecast, expected_price_curve, infer_rival_slot,
    quantile_price_floor,
)
from belief.schemas import (
    CarryRequirement, DaySchedule, DropRequirement, MarketState, OrderBook,
    PurchaseIntent, SellIntent, TileRequirement,
)
from belief.solvers import (
    default_schedules, maximin_mixed_lp, maximin_mixed_slsqp, round_tiles,
    season_plan_maximin, slot_game_matrix,
)
from belief.tracker import FlowRecord, MarketTracker

__all__ = [
    "CarryRequirement", "DaySchedule", "DropRequirement", "FlowRecord",
    "MarketState", "MarketTracker", "OpponentModel", "OrderBook",
    "PurchaseIntent", "SellIntent", "TileRequirement",
    "default_schedules", "drain_forecast", "expected_price_curve",
    "infer_rival_slot", "maximin_mixed_lp", "maximin_mixed_slsqp",
    "quantile_price_floor", "round_tiles", "season_plan_maximin",
    "slot_game_matrix",
]
