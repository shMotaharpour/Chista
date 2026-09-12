# F040 — Hand spawn placement and first-hour loss

**Summary (<=50 words):** A new hand spawns on the least-occupied
shed-access tile, ties broken NWSE: (4,4), (5,4), (4,5), (5,5). A hand hired
at hour 0 first acts at hour 1 — 23 actions against the farmer's 24. Unit
spawns nest: unit u stands identically for crew u or u+3.

## Finding

- `_spawn_hand` puts a hand on the least-occupied shed-access tile, ties
  broken NWSE: (4,4), (5,4), (4,5), (5,5).
- A hand hired at hour 0 first acts at hour 1 — 23 actions, against the
  farmer's 24.
- Unit spawns nest: unit *u* stands in the same place whether the crew is
  *u* or *u+3*. Asserted every run, not assumed.

*Source: "Kaggriculture — the rules, as we have established them" research
document, section 5 (Hands).*
