# Plan — V1.1 Crop Economics (پلن، منتظر تایید)

**Goal:** Build the quantitative economics layer: expected profit per crop/animal, exact market price curves, and revenue-maximizing sell timing — the numbers every later agent decision references. Output: `docs/research/003-crop-economics.md` + a reusable pricing module.

**Everything computed from the real environment code (source in docs/kaggriculture-source.md), verified against live env.**

---

## Step 1 — Pricing module (`lab/prices.py`)

Port `market_price()` + `_shape()` from kaggriculture source. No guessing.

- 1.1.1 `lab/prices.py`: `MARKET_PARAMS`, `price(item, inventory)`, `quantity_to_floor(item)` (how many units sold from I0 until price hits $1)
- 1.1.2 **Verification against live env:** run a real game where a scripted agent dumps N units of wheat into the market via SELL; read back `market.prices` from observations; assert lab curve matches env prices at sampled inventories (tolerance 0)
- 1.1.3 Commit: `feat(lab): market pricing module, env-verified`

## Step 2 — Crop economics table (`lab/economics.py` + report)

- 2.1 For each crop: simulate optimal play of one tile — plant day d, water daily (bonus window math from source), fertilize variant, harvest at max_yield_day → total yield, days occupied, revenue at base price
- 2.2 Compute per crop: `profit/tile/day`, `profit/tile/season` (to day 30), `revenue/seed-cost ratio`, `days-to-first-cash` (liquidity speed)
- 2.3 Animals: steady-state — build cost amortized, wheat feed cost (1/day at market price), CARE bonus inclusion, product interval → `profit/tile/day` per animal; fertilizer byproduct revenue included
- 2.4 Land economics: value of BUY_LAND (each $1k/$2k/$4k quadrant: extra 25 tiles × best-crop profit/day vs days remaining) → breakeven day per quadrant
- 2.5 HIRE economics: a hand's marginal value = extra actions/day × best action profit — when does hiring beat not hiring (fib cost 1,1,2,3,5...)
- 2.6 Write all tables → `docs/research/003-crop-economics.md`; key formulas as code comments
- 2.7 Commit: `feat(lab): crop/animal/land economics + docs`

## Step 3 — Sell-impact curve (how much does dumping crash the price?)

- 3.1 For each product: simulate SELL of n units (1, 10, 50, 100, T, 2T) starting at I0 → price path, total revenue vs "sell all at once" naive assumption
- 3.2 Find per product: optimal daily sell volume (units/day that keeps price near base), and revenue loss % if you ignore timing entirely
- 3.3 This directly feeds V2 (predictive selling) — record `docs/research/006-sell-impact.md`
- 3.4 Commit: `docs: sell impact curves`

## Step 4 — Cross-check with real ladder games (sanity)

- 4.1 Load 3 recorded ladder games' price histories (already in ladder JSON? — if not, run 1 game of top opponent and record daily prices)
- 4.2 Compare: which products did the top opponent actually sell most of? Does its revenue match our per-crop table expectations (e.g. premium-heavy portfolio)?
- 4.3 Record observations → same doc
- 4.4 Commit: `docs: cross-check economics vs top-opponent play`

## Step 5 — Summary for agent design

- 5.1 One-page verdict in `003-crop-economics.md`: "best opening crop", "best mid-game animal", "when land purchase pays off", "how many hires are worth it"
- 5.2 These verdicts become the rule constants of agent v1 (V1.2)

---

## Verification (definition of done)

- [ ] Lab price curve == env prices exactly (Step 1.2 tolerance 0)
- [ ] Economics tables internally consistent (yield math matches README tables)
- [ ] Sell-impact: revenue numbers reproducible by running the curve by hand
- [ ] Doc readable standalone: all tables + one-page verdict
- [ ] Atomic commits per step

## Risks

- Live-env verification (1.2) needs a scripted dump-agent — small code, but must handle market order limits (10/turn)
- Fertilizer economics depend on fertilizer *availability* (animals) vs market buy — keep both variants in table
- Time: each live verification game = 720 steps ≈ fast (~10s), fine.
