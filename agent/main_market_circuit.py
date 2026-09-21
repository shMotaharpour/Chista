"""The arena's entry point for the slot circuit's sell plan (#65).

Same spine as `agent.main_market_spread`, plus the slot circuit: the day's
sell queue is built by `shed.market_queue` (the shed guard's release is the
floor), then each good's uniform hourly spread is REPLACED by the circuit's
schedule — chosen against the trained rival distribution
(`agent/artifact/opponent_counts.npz`).

The re-timing works on the SELLABLE items only (`ShedState.sellable()` —
#77's filter: the shed holds animals the market never quotes) and only on
the lots the guard's queue actually released — the circuit re-times the
plan's own decision, it never re-sizes or re-chooses it (#78's contract).
A failure in one good's schedule drops THAT good back to the spread, not
the whole day, and never silently swallows the error path: a real bug in
the circuit raises on the first day it happens, not in coin totals.

This module exists because the arena's episode workers run in FRESH
processes with no shell environment; the mode must be set before
`agent.runtime` is imported. The arena loads it as
`ref:agent.main_market_circuit:agent`.
"""

from __future__ import annotations

import os

os.environ["CHISTA_MARKET"] = "spread"

from agent.main import agent  # noqa: E402 - the env must land first


def _install_circuit() -> None:
    """Post-process `MarketLayer.plan_day`'s queue through the circuit.

    For every good the guard released (read back out of the queue, not
    out of the raw shed), `plan_day_slots` re-times the hours; the
    guard's per-hour rows for that good are replaced, up to the released
    total. A failure in one good degrades THAT good to the spread.
    """
    from agent.market_layer import MarketLayer
    from agent.belief.opponent import OpponentModel
    from agent.belief.shed import shed_state
    from agent.belief.slot_circuit import plan_day_slots

    original = MarketLayer.plan_day
    model = OpponentModel(pretrained=True)

    def with_circuit(self: MarketLayer, obs):
        queue = original(self, obs)
        # the released lots, per good, read back from the guard's own
        # queue — the raw shed holds animals and un-released stock
        released: dict[str, int] = {}
        for row in queue:
            for order in row:
                if order and order[0] == "SELL":
                    released[order[1]] = released.get(order[1], 0) + int(order[2])
        sellable = shed_state(obs).sellable()
        for item, units in released.items():
            try:
                lot = min(units, int(sellable.get(item, 0)))
                if lot <= 0:
                    continue
                sched, _rev = plan_day_slots(item, lot, obs, model)
                for h in range(len(queue)):
                    queue[h] = [o for o in queue[h]
                                if not (o and o[0] == "SELL" and o[1] == item)]
                left = lot
                for h, u in enumerate(sched):
                    u = int(round(u))
                    if u <= 0 or h >= len(queue) or left <= 0:
                        continue
                    u = min(u, left)
                    left -= u
                    if len(queue[h]) < 10:
                        queue[h].append(["SELL", item, u])
            except Exception:
                # THIS good keeps the guard's uniform spread; the day and
                # the other goods are untouched
                continue
        return queue

    MarketLayer.plan_day = with_circuit


_install_circuit()

__all__ = ["agent"]
