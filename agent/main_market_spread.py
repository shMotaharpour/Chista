"""The arena's entry point for the market layer in `spread` mode (issues #15, #18).

Same spine as `agent.main`, with the market layer in spread mode:
the shed guard (F043), the cash rule (F038) and the forecast peak
rule (F035/F036), spread across the day's turns because the cap is
per turn (F031) and one big basket walks the price ladder down.

This module exists because the arena's episode workers run in FRESH
processes that carry no shell environment: the mode switch must be set
before `agent.runtime` is imported (its `Runtime()` reads the env once,
at construction) and this is the place that does it. The arena loads it
as `ref:agent.main_market_spread:agent`.
"""

from __future__ import annotations

import os

os.environ["CHISTA_MARKET"] = "spread"

from agent.main import agent  # noqa: E402 - the env must land first

__all__ = ["agent"]
