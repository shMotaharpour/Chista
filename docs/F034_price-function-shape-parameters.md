# F034 — Price function shape and per-good parameters

**Summary (<=50 words):** price(inventory) pivots around I0 = 10,000: scarcity above, glut below, floored at 1. Shapes and amplitudes are per good (MARKET_PARAMS): WHEAT base 25, CARROT 35, TOMATO 60, STRAWBERRY 120.

## Finding

- `price(inventory)` around `I0 = 10,000`: scarcity above, glut below, floored at 1.
- Shapes and amplitudes are per good (`MARKET_PARAMS`): WHEAT base 25, CARROT 35, TOMATO 60, STRAWBERRY 120.

*Source: "Kaggriculture — the rules, as we have established them" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
