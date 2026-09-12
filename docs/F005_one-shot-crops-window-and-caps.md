# F005 — One-shot crops: window and caps

**Summary (<=50 words):** One-shot crops are born holding 1 unit. Watering adds yield only inside the age window (max_yield_day + 1) // 2 <= age <= max_yield_day: +1 per watered day, +2 if fertilized. Caps: WHEAT 6 (window 2-4), CARROT 4 (2-3), MELON 6 (6-12).

## Finding

- One-shot crops (WHEAT, CARROT, MELON) begin life holding 1 unit the moment they are planted.
- Watering adds yield only inside a window on the plant's age: `(max_yield_day + 1) // 2 <= age <= max_yield_day`. Outside it, watering keeps the plant alive and adds nothing.
| crop | seed | window (age) | per watered day | cap |
|---|---|---|---|---|
| WHEAT | 10 | 2-4 | +1, or +2 fertilized | 6 |
| CARROT | 20 | 2-3 | +1 / +2 | 4 |
| MELON | 80 | 6-12 | +1 / +2 | 6 |

*Source: "Crops and animals — every rule, numbered" research document.*
