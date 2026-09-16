"""The arena's entry point for the market layer in `dump` mode (issues #15, #18).

Same spine as `agent.main`, with the market layer in dump mode:
every sellable item, one order, at hour 0 - the baseline the
scheduled seller has to beat (issue #15 acceptance).

This module exists because the arena's episode workers run in FRESH
processes that carry no shell environment: the mode switch must be set
before `agent.runtime` is imported (its `Runtime()` reads the env once,
at construction) and this is the place that does it. The arena loads it
as `ref:agent.main_market_dump:agent`.
"""

from __future__ import annotations

import os

os.environ["CHISTA_MARKET"] = "dump"

from agent.main import agent  # noqa: E402 - the env must land first

__all__ = ["agent"]
