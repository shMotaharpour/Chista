# F030 — Action shape and unit-before-market ordering

**Summary (<=50 words):** A player's action is {"farmer": [...], "hands": [...], "market": [...]}. Market orders cost no unit action — they are a separate field. A unit acts before that turn's market, so it cannot pick up what the same turn buys.

## Finding

- A player's action is `{"farmer": [...], "hands": [...], "market": [...]}`.
- Market orders cost no unit action — they are a separate field, not something a unit does.
- A unit acts *before* that turn's market, so it cannot pick up what the same turn buys.

*Source: "Kaggriculture — the rules, as we have established them" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
