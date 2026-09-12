# F042 — Land purchases: fixed prefix, LOCKED tiles

**Summary (<=50 words):** A farm owns NW; the rest are bought in a fixed
prefix order — LAND_ORDER = [NE, SW, SE] at 1,000 / 2,000 / 4,000, never a
gap. Short-purse or all-owned buys refuse silently; working a LOCKED tile
spends hours as no-ops; shed ops resolve before the LOCKED guard.

## Finding

- A farm owns the NW quadrant. The rest are bought in a fixed order,
  `LAND_ORDER = [NE, SW, SE]` at `LAND_PRICES = [1000, 2000, 4000]` — a
  prefix, never a gap.
- `_do_buy_land` refuses in silence when short or when all are owned. Buying
  turns that quadrant's LOCKED tiles into empty ones.
- Working a LOCKED tile is not an error: the actions run, every tile
  operation no-ops, and the hours are spent on nothing.
- Shed operations resolve before the LOCKED guard, and movement onto a LOCKED
  tile is allowed — which is why three of the four shed-access tiles being
  locked does not matter.

*Source: "Kaggriculture — the rules, as we have established them" research
document, section 6 (Land).*
