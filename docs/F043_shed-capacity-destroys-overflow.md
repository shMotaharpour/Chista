# F043 — Shed capacity destroys the overflow

**Summary (<=50 words):** shedCapacity = 100 across all items together. The
nightly drop empties every unit's inventory into the shed and destroys
whatever does not fit; DROP does the same. SELL of an item the shed lacks is
refused; BUY_PRODUCT and BUY_ANIMAL are refused once the shed is full — all
silently.

## Finding

- `shedCapacity = 100`, across all items together.
- `_drop_inventories_to_shed` empties every unit's inventory each night;
  whatever does not fit is destroyed.
- DROP from a shed-access tile moves the unit's whole inventory, and
  destroys the overflow the same way.
- SELL of an item the shed does not hold is refused; BUY_PRODUCT and
  BUY_ANIMAL are refused once the shed is full.

*Source: "Kaggriculture — the rules, as we have established them" research
document, section 7 (The shed).*
