# F032 — Queue order is a design decision

**Summary (<=50 words):** Because orders execute strictly by queue index, queue order is itself a design decision. Recommended order: land first, then sales, then hires, then purchases — what an earlier order leaves in the purse decides whether later ones land.

## Finding

- Queue order is therefore a design decision: land, then sales, then hires, then purchases.
- Each order settles at its own index; what an earlier order leaves in the purse decides whether later ones land.

*Source: "Kaggriculture — the rules, as we have established them" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
