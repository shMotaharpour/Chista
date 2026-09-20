# Chista — the agent's shape

**Revision 3 (2026-09-20).** What each part owns, and how the manager thinks across
24 turns instead of one. Where this document and a module disagree, the engine and
this document win; the module is the bug.

Nothing here is marked done. Revision 2 marked six migration steps `(Done.)` while
`world/model.py` was missing nine of the fourteen names it declared. Status lives in
the suite and the arena, not in this file.

---

## 1. The parts

Three are owned elsewhere and this document only states what the manager asks of
them. Two are the manager's own.

| part | owns | the manager asks it |
|---|---|---|
| `tile_dp/` | one tile, one horizon, at given prices: which chain, worth what | `best(state, w) -> chain per tile, value per tile` |
| `wsr/` | the day of the workers: feasible? how many hands? what did not fit? | `fit(tiles, state, budget_ms, warm) -> Fit` |
| `belief/` | what the market and the rival will do; what a good is worth ahead | `values(state) -> forward value per good` |
| **`manager/`** | **the decision** — it calls the three and settles the day | — |
| `runtime.py` | one turn in, one action out; the day's clock; the budget | — |

`world/` holds the vocabulary and the engine's rules, read-only. `obs.py` decodes.
`dispatch.py` slices a committed plan by hour.

### What the manager needs back

Two fields are new and both exist for §3. They are the only thing this document
asks the wsr agent to add:

The manager sends `beam.Day` and nothing else — no price, no value, no money, no
market:

```python
Day(chains     = ((cell, chain_ops, entity), ...),   # the DP's winner per tile
    available  = {good: hour},                       # when each good is in the shed
    units      = (cell, ...),                        # farmer, and hands already out
    hire_times = (hour, ...))                        # the manager's offer
```

`hire_times` is where money meets labour and it is the manager's call, not wsr's: a
hand hired for turn 0 acts from hour 1 (F040), and one the purse cannot reach until
turn 5 has 18 turns in it, not 23.

What comes back today is `Result(pool, route, complete)`. `complete=False` says the
day did not fit but not **by how much**, so the manager can only learn "no", never
how far to move the price of an hour. Two additions close that, and they are the
only thing this document asks of the wsr agent:

```python
search(day, tasks, beam=..., budget_ms=..., warm=...) -> Result
Result(pool, route, complete, hours_short, can_improve)
```

`hours_short` feeds the price loop (§2). `budget_ms` and `warm` are what make a day
thinkable in slices; without them §3 does not work.

---

## 2. The manager

The one feedback loop in the system. Everything else is a straight line.

```python
def settle(state):
    v = belief.values(state)            # forward value: the objective
    w = scarcity_prices(state)          # internal: hours, fertiliser, cash
    for _ in range(ROUNDS):
        tiles = tile_dp.best(state, v, w)
        fit   = wsr.fit(tiles, state)
        cash  = orders.afford(tiles, fit, state)
        if fit.ok and cash.ok:
            break
        w = raise_prices(w, fit.hours_short, cash.short)
    return assemble(tiles, fit, cash)
```

Three things this does that nothing does today:

1. **`hours_short` goes somewhere.** wsr says "three hours short"; the price of an
   hour rises; the DP answers with shorter chains. The tile is not dropped — it is
   re-planned. (The alternative, dropping the cheapest tile, is greedy and loses to
   this whenever one expensive tile costs more travel than two cheap ones.)
2. **Money is checked before the plan, not after.** Today `planner/repair.py` walks
   a finished plan and discards what the purse cannot pay.
3. **Value comes from ahead, scarcity from today.** `belief` says what a unit of
   wheat is worth over the horizon; the manager says what an hour is worth this
   morning. One layer doing both is what made the old master price everything at
   today's quote and never look wrong.

`orders` is the order book: the ten slots, buys and sells together. Splitting them
is what let the day plan's sells be silently discarded by a second builder.

---

## 3. Twenty-four turns, not one

`actTimeout 1` is per turn and the first second is free — F058 measured a 3 s burn
charged 2.04 s. Hours 1–23 currently replay a committed plan in microseconds, so
**23 free seconds a day are thrown away**, about 690 across a season.

The manager is therefore not a function. It is resumable:

```python
class Manager:
    def observe(self, state)     # the real board arrived
    def step(self, budget_ms)    # advance; return when the budget is gone
    def best(self)               # the best plan so far, always available
```

`runtime.py` calls `step()` every turn with what is left of the second, and `best()`
at hour 0. The manager keeps its own place; the runtime keeps the clock. Neither
knows the other's job.

**What it works on during the day.** Tomorrow's board, predicted — and the
prediction needs no simulator. The DP graph already carries the night: `edge_next`
is the tile's state after the committed chain and the night that follows it. Money
and the shed are arithmetic. The market is `belief`'s.

What cannot be predicted is the rival and the weed draw (F045: one stream across
both farms, so tonight depends on what the rival planted). So this is **speculative
work with a warm start, never a finished plan**:

- hours 1–23: refine tomorrow's plan on the predicted board, keeping the best `k`;
- hour 0: the real board arrives. Validate each of the `k` against it — a walk, not
  a search. Take the first that holds. If none does, the best of them is still the
  warm start for one short round.

The worst case is that the warm start is discarded, which is where we are today.

**Caching is the same mechanism.** `wsr`'s instance is `(cell, n_ops, carries)` per
tile plus the hand count — `TaskArray` reads `cells`, `actions`, `items`, `pred` and
never a chain id or a crop name, so a three-op wheat tile and a three-op tomato tile
are one instance. The geometry never changes, so a solved instance is reusable
across days and seeds. Exact hits will be rare; the cache's job is to return the
**nearest** solved instance as the warm start, not the answer. Whether "nearest" is
a usable idea is unmeasured, and the measurement comes before the cache.

---

## 4. Config

One rule:

> **Config holds numbers. It never holds a value that selects a code path.**

`beam_width`, `rounds`, `guard_margin` are config. `use_master` is not — that is
`CHISTA_REPLAN` in a new coat, and five of those switches hid for four days the fact
that nothing was wired.

`agent/config.py` carries the reference dataclass and every default; it is committed.
`agent/artifact/config.json` is the tuned one and **ships beside the agent**, so the
grader plays what was measured. It is gitignored only while the numbers are moving;
once they settle it is committed like anything else. The defaults therefore have to
be *present and legal*, not optimal — they are what a fresh checkout runs with.

---

## 5. No fallback ladder

Failure must be visible. A four-rung ladder answered every turn with `greedy` while
`agent/replan.py` could not even import, and the agent looked like it worked: it
scored **2,840 against a PASS opponent whose 3,000 it never beat** — worse than
doing nothing.

The entry point keeps the never-raise contract because an uncaught exception ends
the season (F058: seat 1 stopped at step 503 and was never called again). It returns
PASS and **records that it failed**. It does not quietly play a worse policy.

---

## 6. What is deleted

`replan.py` · `market_layer.py` · `greedy.py` · `main_replan.py` ·
`main_market_spread.py` · `main_market_dump.py` · `CHISTA_REPLAN` · `CHISTA_MARKET` ·
`CHISTA_MASTER` · `CHISTA_MARKET_FORECAST` · `CHISTA_TRACE` · the fallback ladder.

`planner/` dissolves: `master.py` becomes the manager's `scarcity_prices`,
`columns.py` and `land.py` move under the manager where something calls them, and
`repair.py`'s checks move into the order book where they run before the plan rather
than after it.
