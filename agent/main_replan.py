"""The coordinated entry point for the arena (issue #12 ship gate).

Same spine as `agent.main`, with the replanner rung enabled — and the
Walrasian master on, which is its own default inside the rung. This
module exists because the arena's episode workers run in FRESH
processes that carry no shell environment: the rung's opt-in switch
must be set before `agent.runtime` is imported (its `Runtime()` reads
the env once, at construction), and this module is the place that does
it. The arena loads it as `ref:agent.main_replan:agent`.
"""

from __future__ import annotations

import os

os.environ.setdefault("CHISTA_REPLAN", "1")

from agent.main import agent  # noqa: E402 - the env must land first

__all__ = ["agent"]
