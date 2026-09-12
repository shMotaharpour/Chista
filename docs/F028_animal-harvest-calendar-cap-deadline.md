# F028 — Animal harvest calendar and cap deadline

**Summary (<=50 words):** Animal harvest starts at first_yield_day (GOOSE 4 daily, COW 8 every 2 days, SHEEP 6 every 3 days); no age guard on HARVEST. The held cap is its own deadline: a goose fills in four days, a cow in twelve; day 4 is a collection day.

## Finding

- Production is at nightfall on `first_yield_day` and repeats every `interval` days for as long as the animal lives. There is no age guard on HARVEST for animals: if it holds something, it can be taken.
| animal | first ready day | then | held cap |
|---|---|---|---|
| GOOSE | day 4 | every day: 5, 6, 7, ... | 4 |
| COW | day 8 | every 2 days: 10, 12, 14, ... | 6 |
| SHEEP | day 6 | every 3 days: 9, 12, 15, ... | 6 |
- The cap is a deadline of its own: a goose left uncollected fills its four in four days and everything after that is thrown away; a cow fills six in twelve. With the care bank paying out in a lump (F021) a goose can hit its cap on the first production night — day 4 is a collection day, not a milestone.

*Source: "Crops and animals — every rule, numbered" research document.*
