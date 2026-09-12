# F027 — Ongoing harvest calendar

**Summary (<=50 words):** Ongoing ready days — TOMATO 8, 9, 10, 11; STRAWBERRY 10, 12, 14, 16. The stock cap of 4 equals four unfertilized productions; fertilized, each production is 2, so fertilizing means you must harvest every other production or lose the rest.

## Finding

- Production is at nightfall from `first_yield_day`, then every `interval` days, four times; the stock waits on the plant until it is taken.
| crop | ready on days | last day it stands |
|---|---|---|
| TOMATO | 8, 9, 10, 11 | day 12 (decaying from that morning) |
| STRAWBERRY | 10, 12, 14, 16 | day 17 |
- The stock caps at 4 — exactly four unfertilized productions. Fertilized, each production is 2, so the stock reaches the cap in two productions: **fertilize and you must harvest every other production**.

*Source: "Crops and animals — every rule, numbered" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
