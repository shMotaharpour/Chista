# Plan — V1.2 Rule-Based Agent v1 — APPROVED PENDING (plan only, not yet implemented)

**Goal:** Implement `agent/main.py` — a rule-based agent built on the V1.1 economics constants plus the V1.1.5 mechanics-verification corrections. Acceptance: beats starter (delta ≥ +$200, p<0.05 over 18 paired games) and enters the 19-opponent ladder to establish our position.

**Architecture:** Single file `agent/main.py` (Kaggle-submittable) — pure `agent(obs)` function, no external state between turns (env limitation). Fixed-priority decision dispatcher each turn. Light computation — target <10ms/turn.

---

## Step 1 — Agent skeleton and priority dispatcher

**Files:** `agent/main.py`

- 1.1 Structure: `agent(obs) → {"farmer": op, "hands": [ops], "market": orders}`
- 1.2 Farmer op priority (one per turn):
  1. HARVEST if any reachable crop has yield_units > 0 (decay is 1 unit/2 turns — same-day harvest critical)
  2. WATER if a plant needs it (daily mandatory; alternating allowed for ongoing crops)
  3. FEED/CARE hungry animal
  4. PLANT if seeds available and tile empty
  5. BUILD/PLACE new animal
  6. DIG weeds
  7. Move toward nearest tile needing an action (simple BFS target selection)
- 1.3 Market orders (up to 10/turn, order matters): HIRE → BUY (seeds/animals/land per budget) → SELL
- 1.4 Shared `decide_unit` function — farmer and each hand decide independently
- 1.5 Commit: `feat(agent): v1 skeleton with priority dispatcher`

## Step 2 — Economics rules (from 003 §6 + 006 corrections)

- 2.1 **Early liquidity (days 1-6):** buy + plant CARROT/WHEAT fast cycles
- 2.2 **Expansion (days 2-8):** BUY_LAND NE→SW→SE per budget (after keeping seed reserve); first GOOSE days 2-4; COW from day 5+ if money > $600
- 2.3 **Animals:** GOOSE as fertilizer factory — COLLECT_FERTILIZER daily; sell eggs freely; wheat feed line (2 permanent wheat tiles)
- 2.4 **Mid-game planting:** MELON on free tiles (budget > $300), else CARROT
- 2.5 **Fertilizer:** apply only inside bonus window; melon harvest day 8 with fert (vs day 10); FERTILIZE MELON/STRAWBERRY/TOMATO/WHEAT/CARROT per 006 deltas
- 2.6 **Selling (safe-pace rules):** WHEAT/EGG freely; MILK/STRAWBERRY/WOOL/MELON ≤ safe pace (7/7/19/51 per day); FERTILIZER consume first, sell surplus ≤ 53/day
- 2.7 **HIRE:** daily while fibonacci cost ≤ 10% of money
- 2.8 **Terminal (day 26+):** no late plantings; days 29-30 forced sell of everything (even at a loss)
- 2.9 Commit: `feat(agent): economics rules (crops, animals, land, hires, sells)`

## Step 3 — Sanity tests

- 3.1 Quick run (100 turns) without exceptions; manual validation of a few actions
- 3.2 Timing: assert average <10ms/turn
- 3.3 Commit: fixes if needed

## Step 4 — Formal evaluation

- 4.1 **Ladder:** run v1 vs each of the 19 vendored opponents (1 game each, fixed seed) → our position
- 4.2 Also vs starter: 18 paired seeds → must beat starter (acceptance)
- 4.3 If losing: analyze lost games' money-path/residue → fix → repeat (roadmap 1.3 cycle)
- 4.4 Results → `docs/research/007-agent-v1-results.md`
- 4.5 Commit: `docs: agent v1 results`

---

## Verification (definition of done)

- [ ] v1 > starter with statistical significance (18 paired seeds, p<0.05)
- [ ] <10ms per turn
- [ ] Ladder position recorded
- [ ] Single file, Kaggle-submittable (`main.py` with agent function)

## Risks / notes

- 1 farmer action/turn for the whole farm — priority order is critical; watering all tiles is impossible with 1 farmer → HIRE is vital (each hand = independent actions)
- Hands also act per turn → dispatcher must decide per hand (shared `decide_unit`)
- Movement: tiles are far apart; simple nearest-target selection is enough for v1; full VRP is V5
- Common pitfall: movement deadlock (oscillating between two targets) → rule: act on current tile if possible, else nearest need
- PICKUP lag: items bought arrive in shed AFTER the turn → PICKUP is a separate turn (verified in 006)
- End-of-day inventory drop returns hand inventories to shed — re-PICKUP needed daily
