# F033 — Market quoting: both players, pre-commit inventory

**Summary (<=50 words):** Both players are quoted from the same pre-commit inventory at the same index, then both commit — two sellers of one good walk the price ladder down twice as fast. BUY_PRODUCT is quoted at post-buy inventory, so a buy/sell round-trip against an unchanged market nets exactly zero.

## Finding

- Both players are quoted from the same pre-commit inventory at the same index, then both commit — two sellers of one good walk the ladder down twice as fast.
- BUY_PRODUCT is quoted at *post-buy* inventory, so a buy/sell round-trip against an unchanged market nets zero.

*Source: "Kaggriculture — the rules, as we have established them" research document.*
