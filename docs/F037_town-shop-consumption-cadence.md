# F037 — Town shop consumption cadence

**Summary (<=50 words):** Shops consume every townShopSellInterval = 4 steps,
the town centre every townCenterSellInterval = 24. Shops unlock every
townShopUnlockInterval = 3 days, drawn with replacement — the same shop can
unlock twice, and each instance consumes separately.

## Finding

- Shops consume every `townShopSellInterval = 4` steps, the town centre every
  `townCenterSellInterval = 24`.
- Shops unlock every `townShopUnlockInterval = 3` days, drawn with
  replacement — the same shop can be unlocked twice and each instance
  consumes separately.

*Source: "Kaggriculture — the rules, as we have established them" research
document, section 3 (The town).*
