# F039 — Hire costs are Fibonacci, reset nightly

**Summary (<=50 words):** The n-th hire of a day costs fib(n): 1, 1, 2, 3, 5,
8, 13, 21, 34, 55 — five hands cost 12, a sixth 8 more. hires_today resets
nightly and the hands are cleared: re-hired every morning. One HIRE order per
hand; a short purse returns silently.

## Finding

- The n-th hire *of a day* costs fib(n): **1, 1, 2, 3, 5, 8, 13, 21, 34, 55**.
  Five hands cost 12 for the day; a sixth costs 8 more.
- `hires_today` resets nightly, and `_daily_refresh` clears the hands — they
  are re-hired every morning.
- One HIRE order per hand; it takes no count.
- `_do_hire` returns silently when money < cost.

*Source: "Kaggriculture — the rules, as we have established them" research
document, section 5 (Hands).*
