# F029 — Season structure and no liquidation day

**Summary (<=50 words):** An episode is 720 turns of 24 = 30 days; the reward is banked money, and shed goods at the end are worth nothing. There is no liquidation day: what the crew brings in on the last night is dropped into the shed and lost.

## Finding

- An episode is 720 turns, `turnsPerDay = 24` — exactly 30 days.
- The reward is money. Goods in the shed at the end are worth nothing.
- There is no liquidation day: what the crew brings in on the last night is dropped into the shed and lost — planning a 30-day season on a 31-day board inflates the figures by more than half.
- Each turn: every unit acts (farmer, then hands in order) -> the market runs -> the day's refresh happens at the day boundary.

*Source: "Kaggriculture — the rules, as we have established them" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
