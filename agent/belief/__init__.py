"""agent/belief/ — what the market and the rival are doing, and what to sell.

The sell side of the agent (issue #54, parent #8). One rule: this package
ESTIMATES, it never decides. It publishes prices, probabilities and queues;
the master and the dispatcher own the choices.

## The 60-second read — what a consumer calls

    from agent.belief.market import forecast, hourly_prices
    from agent.belief.shed import market_queue
    from agent.belief.opponent import OpponentModel, drain_forecast
    import sys as _s; _s.modules.setdefault("belief_init_runs", [0]); _s.modules["belief_init_runs"][0] += 1; print("[belief.__init__] RUN #", _s.modules["belief_init_runs"][0], flush=True)
from agent.belief.ladder import sell_coins, split_days

* **What is a sale worth?** `forecast(obs, days=N)` walks the market forward
  from the observation: `fc.prices[d]` is the (9,) quote at the start of
  season day `first_day + d`, `fc.inventory_of(g, d)` the inventory behind
  it, `fc.assumptions` names every assumption (read it). The walk is one
  cumulative sum — the cadence is deterministic (F037), both seats' sales
  add supply, the town removes.
* **What is it worth AT HOUR h?** `hourly_prices(fc)` samples the same walk
  per turn: row `(d, h)` is the quote a SELL at hour h actually sees (after
  that turn's market, before its town consumption). Row 0 is the
  observation's snapshot; past hours of the current day repeat it.
* **What do n units fetch there?** `ladder.sell_coins(good, inv, n)` — the
  engine's own per-unit ladder as one cumsum (bit-identical, floor stall
  included); `ladder.split_days(good, inv, lot, drains)` is the exact best
  day split of a lot (DP, guarded against the greedy that loses).
* **What will the rival do?** `OpponentModel` (corpus-primed from 24.9M
  replays, still counting online): `policy(good, step, price)` is the
  action distribution (hierarchically smoothed toward the good's own
  marginal), `expected_sell` the units to expect. `tracker.MarketTracker`
  feeds it and reads the rival's sales EXACTLY for the seven one-way goods
  (net-only for WHEAT/FERTILIZER — the engine quotes a buy at price(I-1)).
* **What does the town eat next?** `drain_forecast(obs, steps_ahead)` —
  exact mean and sd in closed form: open shops are facts, only the future
  unlocks are draws.
* **What do I sell today?** `shed.market_queue(obs)` — the per-hour SELL
  queue (the dispatcher's shape), built on the shed guard (F043: zero
  destruction), the season-end liquidation and the exact day split.

## The pieces

- `schemas.py` — the state surface over world's names (G_IX, SHOP_BASKET,
  ...) and the messages the layers exchange. Nobody commands an op across a
  boundary; a message says what must be true and by when.
- `tracker.py` — phase 1: the flow residual (above).
- `opponent.py` — phase 2: the trained action model, the demand forecast.
- `solvers.py` — phase 3: the exact maximin mix of the slot game (an LP),
  its risk-adjusted SLSQP sibling, and the season LP whose absorption row
  stops it selling 4,000 units into a 525-unit drain.
- `ladder.py` — the price ladder as arrays (parity-tested bit-for-bit
  against the engine), and the exact day-split DP.
- `market.py` — the forecast (day rows + `hourly_prices`) and `price_paths`.
- `shed.py` — the shed projection, the sell decision, the order queue.
- `stubs.py` — what runs until each missing unit exists, naming the issue
  that retires it.

Every docstring claim is reproduced by a module under `tests/`
(`test_market_analyzer`, `test_market_ladder`, `test_market_hourly`,
`test_market_layer`) and measured by `bench/bench_market_analyzer.py` /
`bench_market_forecast.py`. The artifact builder and its evaluation live
OUTSIDE the agent (`offline_lab/build/opponent_model.py`) per the
submission-closure rule; the agent only loads `agent/artifact/
opponent_counts.npz`.
"""

from agent.belief.depth import (
    best_day_split, block_revenue, depth_blocks, depth_coins, inventory_at,
    marginal_price,
)
from agent.belief.ladder import buy_coins, plan_coins, sell_coins, split_days
from agent.belief.market import (
    MarketForecast, forecast, hourly_inventory, hourly_prices, price_paths,
)
from agent.belief.opponent import (
    OpponentModel, drain_forecast, expected_price_curve, infer_rival_slot,
    quantile_price_floor,
)
from agent.belief.schemas import (
    CarryRequirement, DaySchedule, DropRequirement, MarketState, OrderBook,
    PurchaseIntent, SellIntent, TileRequirement,
)
from agent.belief.solvers import (
    default_schedules, maximin_mixed_lp, maximin_mixed_slsqp, round_tiles,
    season_plan_maximin, slot_game_matrix,
)
from agent.belief.tracker import FlowRecord, MarketTracker

__all__ = [
    "CarryRequirement", "DaySchedule", "DropRequirement", "FlowRecord",
    "MarketForecast", "MarketState", "MarketTracker", "OpponentModel",
    "OrderBook", "PurchaseIntent", "SellIntent", "TileRequirement",
    "best_day_split", "block_revenue", "buy_coins", "default_schedules",
    "depth_blocks", "depth_coins", "drain_forecast", "expected_price_curve",
    "forecast", "hourly_inventory", "hourly_prices", "infer_rival_slot",
    "inventory_at", "marginal_price", "maximin_mixed_lp",
    "maximin_mixed_slsqp", "plan_coins", "price_paths",
    "quantile_price_floor", "round_tiles", "season_plan_maximin",
    "sell_coins", "slot_game_matrix", "split_days",
]
