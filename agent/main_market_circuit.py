"""The arena's entry point for the slot circuit's sell plan (#65).

Same spine as `agent.main_market_spread`, plus the slot circuit: the day's
sell queue is built by `shed.market_queue` (the shed guard's release is the
floor), then each good's uniform hourly spread is REPLACED by the circuit's
schedule — chosen against the trained rival distribution
(`agent/artifact/opponent_counts.npz`).

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

    For every good the guard released, `plan_day_slots` re-times the hours;
    the guard's per-hour rows for that good are replaced. A failure
    degrades to the uniform spread — the layer never fails for a forecast.
    """
    from agent.market_layer import MarketLayer
    from agent.belief.opponent import OpponentModel
    from agent.belief.slot_circuit import plan_day_slots

    original = MarketLayer.plan_day
    model = OpponentModel(pretrained=True)

    def with_circuit(self: MarketLayer, obs):
        queue = original(self, obs)
        try:
            shed = dict((obs.get("private", {}) or {}).get("shed", {}) or {})
            for item, units in shed.items():
                units = int(units)
                if units <= 0:
                    continue
                sched, _rev = plan_day_slots(item, units, obs, model)
                for h in range(len(queue)):
                    queue[h] = [o for o in queue[h]
                                if not (o and o[0] == "SELL" and o[1] == item)]
                for h, u in enumerate(sched):
                    u = int(round(u))
                    if u <= 0 or h >= len(queue):
                        continue
                    u = min(u, units)
                    units -= u
                    if len(queue[h]) < 10:
                        queue[h].append(["SELL", item, u])
                    if units <= 0:
                        break
        except Exception:
            pass                                   # degrade to the uniform
        return queue

    MarketLayer.plan_day = with_circuit


_install_circuit()

__all__ = ["agent"]
