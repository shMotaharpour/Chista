# F008 — One-shot lifespan and decay

**Summary (<=50 words):** A one-shot plant's lifespan is fixed at planting: (planted_day + max_yield_day + 1) x 24. From that step it loses one unit every two steps and becomes a weed at zero. Measured weed nights for day-0 planting: wheat 5, carrot 4, melon 13.

## Finding

- Lifespan is fixed at planting: `max_lifespan_step = (planted_day + max_yield_day + 1) * 24`.
- From that step the plant loses one unit every two steps and becomes a weed at zero.
- Measured: wheat planted on day 0 is a weed by night 5, carrot by night 4, melon by night 13.

*Source: "Crops and animals — every rule, numbered" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
