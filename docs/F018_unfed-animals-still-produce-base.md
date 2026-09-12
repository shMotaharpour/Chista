# F018 — Unfed animals still produce base

**Summary (<=50 words):** An unfed animal still produces its base unit on production nights — base = 1 sits outside the fed_today test; only the care bonus depends on feeding. Measured: a goose fed on days 0, 2, 4, 6 gained a unit on every production night.

## Finding

- An unfed animal still produces its base unit: `base = 1` sits outside the `fed_today` test; only the bonus is inside it.
- Measured: a goose fed on days 0, 2, 4 and 6 gained a unit on *every* production night.

*Source: "Crops and animals — every rule, numbered" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
