# F046 — Runtime budget: one second and a 60-second bank

**Summary (<=50 words):** The budget is 1 free second per turn plus a
60-second bank for the episode. Overrunning bills max(0, duration - 1.0);
the harness bills ~35 ms extra, so budget against 0.965 s. An exhausted bank
forfeits. A free turn buys ~8.7M Python ops; the competition machine is
~1.95x faster.

## Finding

- 1 second free per turn, plus a 60-second bank for the whole episode — not
  60 minutes.
- Overrunning draws `max(0, duration - 1.0)` from the bank; the harness
  bills about 35 ms more than you measure, so budget against 0.965 s.
- Exhausting the bank forfeits (TIMEOUT, reward None).
- `observation["remainingOverageTime"]` is live and authoritative.
- A free turn buys roughly 8.7M Python operations; the competition machine
  benchmarks about 1.95x this development laptop.

*Source: "Kaggriculture — the rules, as we have established them" research
document, section 11 (The agent's runtime budget).*
