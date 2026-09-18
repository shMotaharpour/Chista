# F048 — The season has 720 states and 719 decisions

**Summary (<=50 words):** F029's "720 turns" counts states. The agent is asked
for an action 719 times, at steps 0..718; the terminal state needs none. Day 29
therefore gets 23 decisions, ending at hour 22 — and that last decision is
processed in full, market included. Measured, not inferred.

## Finding

- `len(env.steps) == 720` — states, indices 0..719. This is what a replay store
  records, and why `n_steps` is exactly 720 for every episode.
- The agent is invoked **719** times, at `obs["step"]` 0..718. It is never
  called for state 719. 720 states, 719 transitions between them.
- `day == step // 24` and `hour == step % 24` hold for every call, with no
  exceptions.
- Days 0–28 get 24 decisions each, hours 0..23. **Day 29 gets 23**, hours 0..22.
- **The decision at step 718 is processed in full.** An order emitted there
  reaches the engine and its effect appears in state 719.

## Evidence

A PASS agent never spends, so money sits at `startingMoney` all season and any
change is exactly what was processed. Buying one wheat seed at step 718 —
while also writing the same order for steps 719 and 720, which the agent is
never given:

```
agent was called 719 times; last call at step 718
steps where it EMITTED the buy: [718]
of the targets (718, 719, 720), it was called at: [718]

 step  day hour      money  seeds WHEAT  status
  717   29   21     3000.0            0  ACTIVE
  718   29   22     3000.0            0  ACTIVE
  719   29   23     2990.0            1  DONE

len(env.steps) = 720  -> last state index = 719
final reward   = 2990.0
```

Money is still 3000 in the row for step 718 because **an observation is the
state before that turn is processed**. The ten coins and the seed appear in
state 719, which the agent never sees and which is the final reward.

Pinned by `tests/test_episode_boundary.py` on the official harness path.

## Why it is worth a file

Both halves were guessed wrong, in opposite directions, within a day.

First guess: the agent is called 720 times. The record count said 719, which
looked like a dropped log line until it was measured.

Second guess, built on the first correction: if the agent is never called at
719, then step 718's action must be too late and a final sale would be lost.
That is worse than the first error, because it sounds like a consequence.
It is false — the sale lands.

The numbers had sources; the **inferences drawn from them did not**. A chain of
reasoning over a measured number is not itself measured (R005), and here each
link was one twenty-line script away from being checked.

## What depends on it

- **End-of-season liquidation.** Shed goods are worth nothing at the end and
  there is no liquidation day (F029), so the last sale matters. It can be
  issued as late as step 718 and still count — but not later, because there is
  no later. A plan that writes orders for step 719 loses them **silently**: the
  agent is simply never asked, so nothing refuses the order (F047's class).
- **Planning budgets.** The final day has 23 action slots, not 24. Anything
  that assumes a uniform 24 over-counts the season by one unit of labour.
- **Off-by-one in any "last turn" logic.** `EPISODE_STATES - 1` is the last
  state; `EPISODE_STATES - 2` is the last decision. Both appear in the tests as
  named constants so neither is retyped.

*Source: measured on the official harness path (then `world.kaggle_env.run_episode`,
now `offline.kaggle_env.run_episode`),
kaggle-environments 1.32.7, 2026-09-15. Corrects the reading of F029.*
