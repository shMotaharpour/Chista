# F031 — Order cap is per turn, queue walked to completion

**Summary (<=50 words):** maxMarketOrdersPerTurn = 10 — per turn, not per day; an 11th order is dropped silently and the next turn's queue is untouched. Orders are walked by index, each to completion: anything behind an empty purse is refused in silence. HIRE and BUY_LAND settle atomically before the per-unit loop.

## Finding

- `_process_market` runs every turn, not once a day.
- `maxMarketOrdersPerTurn = 10` — per turn, not per day. Orders past the tenth in one turn are dropped; the next turn's queue is untouched.
- Orders are walked by queue index, each to completion, so anything behind an empty purse is refused in silence.
- HIRE and BUY_LAND are atomic: settled at their index, before the per-unit loop, and dropped from the queue.

*Source: "Kaggriculture — the rules, as we have established them" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
