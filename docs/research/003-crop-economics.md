# Crop Economics — Kaggriculture (V1.1)

> All numbers derived from the environment source and README, verified via `lab/prices.py` (exact match with the live env).

## 1. Price curve — verified against live env ✅

- `lab/prices.py` is an exact port of `market_price()`
- Verified: both branches (scarcity and glut) **exactly match env prices** (zero tolerance)
- Key detail: units sold at the $1 floor do NOT add to market inventory (floor stays responsive)

## 2. Crop economics (net profit / tile / day — at base price)

| Crop | Seed | No fert $/tile/day | Fert $/tile/day | Cycles per season | Liquidity |
|---|---|---|---|---|---|
| **MELON** | $80 | **$70.8** | **$109.2** 🏆 | 2 | Late (day 11+) |
| STRAWBERRY | $100 | $22.4 | $50.6 | 1 | Very late (day 11+) |
| WHEAT | $10 | $18.0 | $28.0 | 6 | Fast (day 3) |
| CARROT | $20 | $21.3 | $30.0 | 7 | Fast (day 3) |
| TOMATO | $50 | $15.8 | $35.8 | 2 | Late (day 9) |

**Reading the table:**
- **MELON with fertilizer is the seasonal profit king** ($109/tile/day) — but locks money until day 11 (later corrected: day 8 with fert, see docs/research/006)
- **WHEAT/CARROT are the early liquidity engine** — fast turnaround, many cycles
- **Fertilizer multiplies profit ×1.5–2.2** (within the bonus window) — fertilizer access is critical
- Strawberry is the weakest unless real (scarcity) prices rise above base

## 3. Animal economics (30 days, full care, wheat @$25)

| Animal | Cost | Product revenue | Fertilizer revenue | Feed | **Profit** | $/day |
|---|---|---|---|---|---|---|
| **COW** | $400 | $3,840 | $3,000 | $750 | **$5,690** | **$189.7** 🏆 |
| SHEEP | $500 | $3,600 | $3,000 | $750 | $5,350 | $178.3 |
| GOOSE | $300 | $2,700 | $3,000 | $750 | $4,650 | $155.0 |

**Notes:**
- **Fertilizer revenue exceeds the product itself!** ($3,000 each) — animals are fertilizer factories
- COW has the best ROI; GOOSE pays back fastest (first production day 4)
- Feed = 1 wheat/day — animals require a wheat production line

## 4. Land and labor

- **BUY_LAND almost always pays off:** NE quadrant breakeven ≈ half a working day; even SE ($4k) ≈ 1.5 days
  - Only with best-crop planting — if it locks liquidity it's bad → order: NE day 1-2, SW day 3-5, SE day 6-8
- **HIRE:** each hand = +1 action/turn = 24 actions/day. With best-action profit (~$100+ mid-game), the fibonacci cost (1,1,2,3,5,8...) is justified up to about n=6-8; beyond that not worth it

## 5. Sell impact — why timing matters

| Product | Dump T units | Loss | Dump 2T | Loss | Safe pace (90% price) |
|---|---|---|---|---|---|
| WHEAT | $20.8 avg | 17% | $20.3 | 19% | **66/day** (absorbs!) |
| CARROT | $18.7 | 47% | $11.9 | 66% | 16/day |
| TOMATO | $36.1 | 40% | $26.1 | 56% | 7/day |
| STRAWBERRY | $38.5 | 68% | $19.7 | **84%** | 7/day |
| MELON | $88.8 | 65% | $44.9 | **82%** | 51/day |
| EGG | $41.7 | 17% | $40.5 | 19% | 24/day |
| MILK | $51.0 | 68% | $26.0 | **84%** | 8/day |
| WOOL | $75.9 | 62% | $38.5 | **81%** | 19/day |
| FERTILIZER | $80.1 | 20% | $60.1 | 40% | 53/day |

**Golden rules:**
- **WHEAT and EGG are almost insensitive to dump volume** → sell freely anytime
- **MILK, STRAWBERRY, WOOL, MELON** → large dumps are a disaster (up to 84% loss) → sell drip-wise over multiple days
- MELON is the interesting exception: up to ~10 units lossless, then a hard crash (sq curve)

## 6. One-page verdict for agent v1

1. **Opening:** carrot/wheat for liquidity days 1-6 (fast cycles)
2. **Days 2-4:** BUY_LAND ×1 (NE) + first GOOSE (fastest animal, fertilizer factory)
3. **Days 5-8:** SW + COW as soon as affordable; wheat line for feed
4. **Planting on free tiles:** MELON with fertilizer wherever possible; else carrot
5. **Selling:** wheat/eggs freely; milk/strawberry/wool/melon only drip-wise (≤ safe pace)
6. **Fertilizer sales:** never above 53/day — or consume it yourself
7. **Day 26+:** no new late plantings; days 29-30 forced sell of everything (even at a loss)

## 7. Cross-check vs the #1 ladder opponent

Real game: `farming-score-v3-replay-revised` ($158.6k) vs `ecobot-v6` ($128.8k), seed 5001

**Sold (units):** STRAWBERRY 430, FERTILIZER 397, MILK 335, WHEAT 313, WOOL 179, MELON 72, CARROT 57

**Bought:** HIRE ×282 (!), COW ×6, SHEEP ×1, BUY_LAND ×2, seeds: WHEAT 22 + STRAWBERRY 13 + MELON 2

**Confirmation of our table patterns:**
- ✅ Premium portfolio (strawberry/milk/wool/melon) = the revenue backbone — exactly animals + strawberry
- ✅ Kept fertilizer off the market until it crashed late (final price $6!) — collected and sold in a controlled way
- ✅ Labor: 282 hires per season ≈ 10/day — hands are the action engine
- ✅ Land: only 2 purchases — likely preferring melon over extra space
- ⚠️ Never sold EGG — geese kept as fertilizer factories, not egg sellers
- ⚠️ Final prices WOOL $5, FERTILIZER $6 → it crashed those markets itself, but late (end of season) — cheap end-of-season selling? A note for V2
