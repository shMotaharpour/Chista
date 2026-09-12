# F019 — Care bank accrual and payout

**Summary (<=50 words):** CARE plus FEED on the same day banks +1 (order within the day is irrelevant — both flags are read at nightfall); CARE without FEED banks nothing. The bank pays out only on a fed production night, in full: held += 1 + bank, capped at max_held.

## Finding

- CARE + FEED on the same day adds 1 to the care bank (`pending_care_bonus`). Both flags are read at nightfall, so the order within the day does not matter — and CARE without FEED banks nothing.
- The bank is paid out only on a fed production night, and it pays out in full: `held += 1 + bank`, capped at `max_held`.

*Source: "Crops and animals — every rule, numbered" research document, project [AgriOracle](https://github.com/shMotaharpour/AgriOracle).*
