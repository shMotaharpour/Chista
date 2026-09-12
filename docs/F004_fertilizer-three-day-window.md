# F004 — Fertilizer three-day window

**Summary (<=50 words):** FERTILIZE consumes one fertilizer from the acting unit's own inventory (a hand cannot spread the farmer's), refuses silently if none, and sets fertilized_until_day = day + 2 — coverage is exactly three days: today, tomorrow, and the day after. Re-applying inside covered days adds nothing.

## Finding

- FERTILIZE takes one FERTILIZER **from the acting unit's own inventory** — a hand cannot spread what the farmer is carrying — and returns in silence if the unit holds none.
- It sets `fertilized_until_day = day + 2`: coverage is three days — today, tomorrow and the day after.
- A second application inside already-covered days adds nothing (measured on wheat, F006).

*Source: "Crops and animals — every rule, numbered" research document.*
