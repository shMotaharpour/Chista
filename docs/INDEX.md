# INDEX

One line per file: a link to the file plus its ≤50-word summary.
Order: **Rules** (`R<NNN>_<slug>.md`) first, then **Findings**
(`F<NNN>_<slug>.md`). IDs are permanent — never renumber, never reuse.

## Rules

- [R001_competition_stack_and_commit_rules.md](R001_competition_stack_and_commit_rules.md) —
  Project stack and workflow rules: Kaggle Kaggriculture environment, CPU-only
  PyTorch, OR-Tools and scipy for optimization; every agent commit must use
  `~/.local/bin/agent-commit` so the Co-authored-by trailer separates agent
  work from the user's own commits.
- [R002_import_rules_from_kaggle_environments.md](R002_import_rules_from_kaggle_environments.md) —
  Import game constants and formulas directly from `kaggle_environments`
  instead of transcribing them: one source of truth, zero parity-harness
  maintenance, free at runtime; pinned environment version mitigates private
  API rename risk.
- [R003_simulator_wraps_real_interpreter.md](R003_simulator_wraps_real_interpreter.md) —
  Fast game simulation calls `kaggriculture.interpreter()` directly on a
  structify-cloned state instead of reimplementing rules: measured 50.6×
  faster than `env.run()` for `world/fast_sim` (the 8.2× figure in the doc is
  the earlier prototype module pair) with bit-identical rewards and a
  bit-identical agent-facing observation stream, and no second rule
  implementation to keep in sync.
- [R004_configurable_validation_dev_and_fast_modes.md](R004_configurable_validation_dev_and_fast_modes.md) —
  Every module carries a validation config: dev mode runs harness and
  validators; fast mode — the main parse-and-run path for heavy processes
  (DP, RL, MDP) — bypasses all checks via the same config so no time is
  wasted on validation when speed matters. Dev mode is deliberately stricter
  than the real harness (junk inner action shapes pass there, raise here) and
  hands out observation copies instead of live views.
- [R005_every-number-has-a-source.md](R005_every-number-has-a-source.md) —
  Every threshold, budget and constant in the repo traces to a source: an
  F-finding, the engine itself, or a measurement recorded where it was taken. A
  number without one is a defect, even when it looks reasonable. Missing values
  are measured or left as named TODOs, never guessed.

## Findings

- [F001_seeds-bypass-the-shed.md](F001_seeds-bypass-the-shed.md) — Seeds live in private["seeds"], never pass through the shed, and are exempt from its 100-unit capacity. PLANT needs an empty owned tile. The seed check is atomic: over-requesting one crop in a turn drops every PLANT of that crop, silently, not just the excess.
- [F002_planting-day-counts-as-unwatered.md](F002_planting-day-counts-as-unwatered.md) — A new plant is born with consecutive_unwatered = 1 — the planting day counts as a dry day. Two consecutive dry nights turn the tile into a WEED, not empty ground, so a plant not watered on its own planting day is gone by the next night.
- [F003_watering-once-per-day.md](F003_watering-once-per-day.md) — WATER works once per day; a second watering the same day is refused, silently. Outside its yield window (one-shot crops) watering still keeps the plant alive but adds nothing.
- [F004_fertilizer-three-day-window.md](F004_fertilizer-three-day-window.md) — FERTILIZE consumes one fertilizer from the acting unit's own inventory (a hand cannot spread the farmer's), refuses silently if none, and sets fertilized_until_day = day + 2 — coverage is exactly three days: today, tomorrow, and the day after. Re-applying inside covered days adds nothing.
- [F005_one-shot-crops-window-and-caps.md](F005_one-shot-crops-window-and-caps.md) — One-shot crops are born holding 1 unit. Watering adds yield only inside the age window (max_yield_day + 1) // 2 <= age <= max_yield_day: +1 per watered day, +2 if fertilized. Caps: WHEAT 6 (window 2-4), CARROT 4 (2-3), MELON 6 (6-12).
- [F006_one-fertilizer-worth-two-wheat.md](F006_one-fertilizer-worth-two-wheat.md) — Measured on daily-watered wheat: unfertilized 1, 1, 2, 3, 4 (weed night 5); one FERTILIZE on day 2 gives 1, 1, 3, 5, 6 — the cap. One unit of fertilizer covers the whole window and is worth +2 wheat; a second, later unit adds nothing.
- [F007_one-shot-harvest-frees-tile.md](F007_one-shot-harvest-frees-tile.md) — HARVEST on a one-shot crop takes everything and removes the plant: the tile is empty again and can be replanted the same day.
- [F008_one-shot-lifespan-decay.md](F008_one-shot-lifespan-decay.md) — A one-shot plant's lifespan is fixed at planting: (planted_day + max_yield_day + 1) x 24. From that step it loses one unit every two steps and becomes a weed at zero. Measured weed nights for day-0 planting: wheat 5, carrot 4, melon 13.
- [F009_one-shot-honest-yields.md](F009_one-shot-honest-yields.md) — Per tile, watered every day with one fertilizer: wheat yields 6 by the end of day 4, carrot 4 by day 3, melon 6 by day 10 — the honest maximums a plan may assume.
- [F010_ongoing-watering-never-adds-yield.md](F010_ongoing-watering-never-adds-yield.md) — Ongoing crops (TOMATO, STRAWBERRY) begin holding 0, and watering never adds yield — it only keeps the plant alive. All their yield comes from scheduled productions.
- [F011_ongoing-production-cadence.md](F011_ongoing-production-cadence.md) — Ongoing crops produce at nightfall from first_yield_day, repeating every interval days: +1, or +2 if that day was both watered and fertilized. TOMATO: seed 50, first day 8, daily, 4 productions. STRAWBERRY: seed 100, first day 10, every 2 days, 4 productions.
- [F012_ongoing-total-eight-with-fertilizer.md](F012_ongoing-total-eight-with-fertilizer.md) — Ongoing crops have at most four productions and a held-stock cap of 4. Harvesting between productions empties the stock, so with fertilizer on every production the season total reaches 8 collected units — double the unfertilized 4.
- [F013_ongoing-lifespan-on-last-production.md](F013_ongoing-lifespan-on-last-production.md) — On its last scheduled production an ongoing crop is assigned a lifespan of (next_day + 1) x 24 and then decays exactly like a one-shot plant, one unit every two steps, into a weed. Ongoing does not mean immortal.
- [F014_ongoing-measured-calendars.md](F014_ongoing-measured-calendars.md) — Measured, watered daily: TOMATO yields its first unit on night 7, then 8, 9, 10, and is a weed by night 12; STRAWBERRY yields on nights 9, 11, 13, 15 and dies from day 17.
- [F015_ongoing-harvest-leaves-plant.md](F015_ongoing-harvest-leaves-plant.md) — HARVEST on an ongoing crop takes the held stock and leaves the plant standing — productions continue on schedule.
- [F016_animal-placement-and-species-table.md](F016_animal-placement-and-species-table.md) — Animals need BUILD_COOP or BUILD_PASTURE first, then PLACE with the animal in the acting unit's inventory. GOOSE: 300, coop, first day 4, daily, cap 4, eggs. COW: 400, pasture, day 8, every 2 days, cap 6, milk. SHEEP: 500, pasture, day 6, every 3 days, cap 6, wool.
- [F017_two-consecutive-unfed-days-escape.md](F017_two-consecutive-unfed-days-escape.md) — Two consecutive unfed days make an animal escape, leaving the empty structure; one unfed day is free, including the placement day. Feeding every other day keeps an animal alive indefinitely.
- [F018_unfed-animals-still-produce-base.md](F018_unfed-animals-still-produce-base.md) — An unfed animal still produces its base unit on production nights — base = 1 sits outside the fed_today test; only the care bonus depends on feeding. Measured: a goose fed on days 0, 2, 4, 6 gained a unit on every production night.
- [F019_care-bank-accrual-and-payout.md](F019_care-bank-accrual-and-payout.md) — CARE plus FEED on the same day banks +1 (order within the day is irrelevant — both flags are read at nightfall); CARE without FEED banks nothing. The bank pays out only on a fed production night, in full: held += 1 + bank, capped at max_held.
- [F020_unfed-production-night-destroys-bank.md](F020_unfed-production-night-destroys-bank.md) — An unfed production night destroys the care bank: the bonus is popped only when fed, but the counter is zeroed either way — a bank of three becomes nothing. An unfed ordinary night leaves the bank untouched.
- [F021_care-bank-lumps-before-first-yield.md](F021_care-bank-lumps-before-first-yield.md) — The care bank accumulates through the wait before the first yield and pays out in one lump: fed and cared from the placement day, a goose reaches its first production night with a bank of 3 and collects 1 + 3 = 4 — exactly its max_held.
- [F022_production-past-cap-is-lost.md](F022_production-past-cap-is-lost.md) — Production past max_held is lost — held grows as min(max_held, held + base + bonus). Product already held never spoils.
- [F023_animal-fertilizer-every-night.md](F023_animal-fertilizer-every-night.md) — fertilizer_available is set true every night, for every animal, whether or not it was collected — one COLLECT_FERTILIZER per animal per day is the effective limit.
- [F024_animals-unplace-impossible-shed-loss.md](F024_animals-unplace-impossible-shed-loss.md) — A placed animal cannot be taken back: DIG returns early on a tile holding one, and only escape empties the structure. An animal still in the shed at nightfall can be destroyed by the 100-unit shed overflow like any other good.
- [F025_cow-cap-asymmetry.md](F025_cow-cap-asymmetry.md) — Table shape: GOOSE and SHEEP hold exactly as many units as the days they make you wait (4/4 and 6/6), but COW holds six against a wait of eight — the only species whose cap outpaces its cadence.
- [F026_one-shot-harvest-calendar-deadline.md](F026_one-shot-harvest-calendar-deadline.md) — One-shot crops have exactly one harvest, refused while age < first_yield_day, and it destroys the plant. Ready days: WHEAT 2, CARROT 2, MELON 10. Water before harvesting the same day — WATER's unit lands immediately. The deadline is real: a day late costs most of the tile.
- [F027_ongoing-harvest-calendar.md](F027_ongoing-harvest-calendar.md) — Ongoing ready days — TOMATO 8, 9, 10, 11; STRAWBERRY 10, 12, 14, 16. The stock cap of 4 equals four unfertilized productions; fertilized, each production is 2, so fertilizing means you must harvest every other production or lose the rest.
- [F028_animal-harvest-calendar-cap-deadline.md](F028_animal-harvest-calendar-cap-deadline.md) — Animal harvest starts at first_yield_day (GOOSE 4 daily, COW 8 every 2 days, SHEEP 6 every 3 days); no age guard on HARVEST. The held cap is its own deadline: a goose fills in four days, a cow in twelve; day 4 is a collection day.
- [F029_season-structure-no-liquidation.md](F029_season-structure-no-liquidation.md) — An episode is 720 turns of 24 = 30 days; the reward is banked money, and shed goods at the end are worth nothing. There is no liquidation day: what the crew brings in on the last night is dropped into the shed and lost.
- [F030_action-shape-unit-before-market.md](F030_action-shape-unit-before-market.md) — A player's action is {"farmer": [...], "hands": [...], "market": [...]}. Market orders cost no unit action — they are a separate field. A unit acts before that turn's market, so it cannot pick up what the same turn buys.
- [F031_order-cap-per-turn-queue-walk.md](F031_order-cap-per-turn-queue-walk.md) — maxMarketOrdersPerTurn = 10 — per turn, not per day; an 11th order is dropped silently and the next turn's queue is untouched. Orders are walked by index, each to completion: anything behind an empty purse is refused in silence. HIRE and BUY_LAND settle atomically before the per-unit loop.
- [F032_queue-order-design-decision.md](F032_queue-order-design-decision.md) — Because orders execute strictly by queue index, queue order is itself a design decision. Recommended order: land first, then sales, then hires, then purchases — what an earlier order leaves in the purse decides whether later ones land.
- [F033_market-quoting-both-players.md](F033_market-quoting-both-players.md) — Both players are quoted from the same pre-commit inventory at the same index, then both commit — two sellers of one good walk the price ladder down twice as fast. BUY_PRODUCT is quoted at post-buy inventory, so a buy/sell round-trip against an unchanged market nets exactly zero.
- [F034_price-function-shape-parameters.md](F034_price-function-shape-parameters.md) — price(inventory) pivots around I0 = 10,000: scarcity above, glut below, floored at 1. Shapes and amplitudes are per good (MARKET_PARAMS): WHEAT base 25, CARROT 35, TOMATO 60, STRAWBERRY 120.
- [F035_prices-rise-through-the-season.md](F035_prices-rise-through-the-season.md) — Prices rise through the season — the town consumes faster than one farm refills — so holding produce and selling late is worth real money.
- [F036_coarse-ladder-shallow-markets.md](F036_coarse-ladder-shallow-markets.md) — The price ladder is coarse — about 25 wheat units per coin — so a handful of extra units can tip a fifty-unit basket down a step, and one more wheat tile can be worth less than nothing. Sell pressure bites only in shallow markets.
- [F037_town-shop-consumption-cadence.md](F037_town-shop-consumption-cadence.md) — Shops consume every townShopSellInterval = 4 steps, the town centre every townCenterSellInterval = 24. Shops unlock every townShopUnlockInterval = 3 days, drawn with replacement — the same shop can unlock twice, and each instance consumes separately.
- [F038_money-binds-first-week.md](F038_money-binds-first-week.md) — startingMoney = 3,000, and every purchase refuses silently when the purse is short. A real plan ends day 1 with 13 coins and only breaks out around day 10 — the first week is where money binds.
- [F039_hire-fib-costs-nightly-reset.md](F039_hire-fib-costs-nightly-reset.md) — The n-th hire of a day costs fib(n): 1, 1, 2, 3, 5, 8, 13, 21, 34, 55 — five hands cost 12, a sixth 8 more. hires_today resets nightly and the hands are cleared: re-hired every morning. One HIRE order per hand; a short purse returns silently.
- [F040_hand-spawn-placement-first-hour.md](F040_hand-spawn-placement-first-hour.md) — A new hand spawns on the least-occupied shed-access tile, ties broken NWSE: (4,4), (5,4), (4,5), (5,5). A hand hired at hour 0 first acts at hour 1 — 23 actions against the farmer's 24. Unit spawns nest: unit u stands identically for crew u or u+3.
- [F041_order-queue-binds-before-wage.md](F041_order-queue-binds-before-wage.md) — The wage is never the constraint (42 coins buys a sixth hand for five days); the order queue is — past about six hands the day needs an eleventh order, and every unit loses an hour to the cap.
- [F042_land-purchase-prefix-locked.md](F042_land-purchase-prefix-locked.md) — A farm owns NW; the rest are bought in a fixed prefix order — LAND_ORDER = [NE, SW, SE] at 1,000 / 2,000 / 4,000, never a gap. Short-purse or all-owned buys refuse silently; working a LOCKED tile spends hours as no-ops; shed ops resolve before the LOCKED guard.
- [F043_shed-capacity-destroys-overflow.md](F043_shed-capacity-destroys-overflow.md) — shedCapacity = 100 across all items together. The nightly drop empties every unit's inventory into the shed and destroys whatever does not fit; DROP does the same. SELL of an item the shed lacks is refused; BUY_PRODUCT and BUY_ANIMAL are refused once the shed is full — all silently.
- [F044_animal-age-residue-collapse.md](F044_animal-age-residue-collapse.md) — An animal's age matters only through a residue (its position in the production interval), so per-tile animal planning collapses to a flat DP over the residue instead of the full age.
- [F045_weed-rng-couples-farms-and-market.md](F045_weed-rng-couples-farms-and-market.md) — _spawn_weeds draws rng.random() once per empty tile on both farms before the same stream picks the town's next shop (weedSpawnChance = 0.005; LOCKED tiles draw nothing). Planting a tile therefore changes tomorrow's prices, and no price path can be precomputed offline.
- [F046_runtime-budget-one-second-bank.md](F046_runtime-budget-one-second-bank.md) — The budget is 1 free second per turn plus a 60-second bank for the episode. Overrunning bills max(0, duration - 1.0); the harness bills ~35 ms extra, so budget against 0.965 s. An exhausted bank forfeits. A free turn buys ~8.7M Python ops; the competition machine is ~1.95x faster.
- [F047_silent-operations-catalog.md](F047_silent-operations-catalog.md) — The most expensive mistake class: the engine fails silently. Purse-short orders are refused, hires and land buys no-op, LOCKED tiles spend hours for nothing, the full shed destroys overflow, SELL and FERTILIZE without stock refuse, over-seeded PLANT drops the crop's whole turn, and an 11th order is dropped.
- [F048_episode-has-720-states-and-719-decisions.md](F048_episode-has-720-states-and-719-decisions.md) — F029's "720 turns" counts states. The agent is asked for an action 719 times, at steps 0..718; the terminal state needs none. Day 29 therefore gets 23 decisions, ending at hour 22 — and that last decision is processed in full, market included. Measured, not inferred.

## Tests & benchmarks

Executable checks — `.venv/bin/python -m tests.<module>` (no pytest required; they also run under pytest):

- [../tests/test_world_parity.py](../tests/test_world_parity.py) — full agent-facing parity: same seed + same actions ⇒ identical per-turn observations, money and final rewards on the harness path and `world.fast_sim`.
- [../tests/test_world_branch_purity.py](../tests/test_world_branch_purity.py) — a clone continued with a suffix equals a from-scratch replay of prefix+suffix, and exploring branches never touches the parent.
- [../tests/test_import_identity.py](../tests/test_import_identity.py) — R002 name test, no transcribed rule tables, configuration matches the shipped spec, and the pinned kaggle-environments version equals the installed one.
- [../tests/test_replay_agent.py](../tests/test_replay_agent.py) — a recorded episode replays bit-exactly on both paths, at either seat or both at once; a missing step answers PASS and the record is never mutated.
- [../tests/test_tile_dp.py](../tests/test_tile_dp.py) — tile graph contracts: day-start decode round-trip, engine calendars as truth, chain and labour cost model, no-op and dominance pruning.
- [../tests/test_opponents.py](../tests/test_opponents.py) — vendored competitor agents: every slug complete, every payload matching its recorded SHA-256 (the Apache 4(b) claim), nothing reaching outside the process, packed payloads actually decoded, and no doc pointing at a path this repo lacks.
- [../tests/test_layering.py](../tests/test_layering.py) — `opponents/` is evaluation input, never submission input: nothing outside it imports it, and the guard proves it can see the files it guards.
- [../tests/test_episode_boundary.py](../tests/test_episode_boundary.py) — F048 pinned on the official harness path: 720 states vs 719 decisions, day/hour following the step, day 29 one decision short, the step-718 action processed in money, and no call past it.
- [../bench/bench_paths.py](../bench/bench_paths.py) — reproduces the R003 timings on the current machine.
