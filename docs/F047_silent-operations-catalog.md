# F047 — The silent-operations catalog

**Summary (<=50 words):** The most expensive mistake class: the engine fails
silently. Purse-short orders are refused, hires and land buys no-op, LOCKED
tiles spend hours for nothing, the full shed destroys overflow, SELL and
FERTILIZE without stock refuse, over-seeded PLANT drops the crop's whole
turn, and an 11th order is dropped.

## Finding

The single most expensive class of mistake: none of these report anything —
the farm simply does less than the plan said.

| failure | effect |
|---|---|
| purse short | the order is refused, and so is everything behind it |
| `_do_hire` | no hand spawns |
| `_do_buy_land` | the quadrant stays locked, every slot on it idle |
| a LOCKED tile | the actions run and do nothing |
| the shed at 100 | the overflow is destroyed, on the nightly drop and on DROP |
| SELL with no stock | refused |
| FERTILIZE with none held | refused |
| PLANT over seeds | all plants of that crop that turn are dropped |
| an 11th order in a turn | dropped |

*Source: "Kaggriculture — the rules, as we have established them" research
document, section 12 (Things that are silent).*
