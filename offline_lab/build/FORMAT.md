# Kaggle replay → dataset format (`kaggle_shards` format 1)

The contract between the dataset builder (`kaggle_dataset.py`), the training
code (`train_bc.py`), and the runtime manager (`agent/manager/rl_manager.py`).
Numbers in the tables are the arrays' shapes; "per sample" = one
(episode, seat, day) decision.

## The sampling contract (measured, not assumed)

The action recorded at `steps[t][seat]` is what CAUSED the transition
`obs[t-1] → obs[t]` (verified on episode 111017932: sell revenue lands in
`obs[t].money`; the hands that act are the crew of `obs[t-1]`). Therefore:

- sample = (episode, seat, day): state = the hour-0 observation
  `steps[24D][seat]`, labels = the day's actions `steps[24D+1 .. 24D+24]`;
- an op's cell = where the unit STOOD one turn before it acted — the position
  the planner itself knows at decision time.

## The one vocabulary

Every categorical value goes through ONE table, built data-driven by the
builder and shipped inside the shard:

- `vocab` — the op strings (`HIRE`, `BUY_SEED`, `SELL`, `WATER`, …) and item
  strings (`MELON`, `SHEEP`, …), one shared integer code space.

The tile side needs no vocabulary: tile keys are the graph's own state keys
(`agent/artifact/tile_graph.npz`, 703 states), LOCKED keeps its sentinel
`LOCKED_KEY`. Market ops and items live in `vocab`; that is also what makes
the market head's output space.

## Sample state (all fixed-shape, one row per sample)

| field | shape | dtype | content |
|---|---|---|---|
| `tiles_own` | (100,) | int32 | graph state key per cell, row-major |
| `tiles_opp` | (100,) | int32 | same for the rival farm |
| `money_own` / `money_opp` | scalar | float32 | **log1p(money)** — 0 stays 0 |
| `shed_own` | (12,) | int32 | 9 products + 3 animals (`SHED_ORDER`) |
| `shed_opp` | (12,) | int32 | rival shed, same order |
| `seeds` | (5,) | int32 | own seeds, CROPS order |
| `prices` | (9,) | float32 | market prices, PRODUCTS order |
| `mkt_inv` | (9,) | float32 | city inventory, PRODUCTS order |
| `shops` | (8,) | uint8 | 1 = unlocked, `sorted(SHOPS)` order |
| `day` | scalar | uint8 | 0..29 |

No distance feature: the tiles are the 10×10 board in row-major order, so a
cell's position is its index — a distance field would repeat information the
layout already carries. No hands/hires fields: both are 0 at every hour 0
(F039/F040). Seeds and shed are separate pools because the engine keeps them
separate (F001: seeds bypass the shed).

## Labels (ragged, flat arrays + per-sample pointers)

`op_ptr[i] : op_ptr[i+1]` slices sample i's ops; same for `ord_ptr`.

| field | shape | dtype | content |
|---|---|---|---|
| `op_hour` | (Σops,) | int8 | 0..23 within the day |
| `op_actor` | (Σops,) | int8 | 0 = farmer, 1..n = hand n |
| `op_cell` | (Σops,) | int16 | 0..99, the cell acted on |
| `op_code` | (Σops,) | int16 | `vocab` code of the op string |
| `op_item` | (Σops,) | int16 | `vocab` code of the item, −1 = none |
| `op_qty` | (Σops,) | int32 | the numeric argument |
| `ord_*` | (Σorders,) | same five | hour, idx, op, item, qty — all encoded |
| `op_ptr` / `ord_ptr` | (n+1,) | int64 | per-sample offsets |

Market orders are FULLY encoded (op, item, qty in the shared vocab) — the
sell model reads the same codes the runtime queue writes.

## Per-episode fields (one row per sample)

`episode_id` (int64), `seed` (int64), `seat` (int8), `agent` (str),
`reward` (int64) — the reward is the SEAT's own final score, the filter's
quality column.

## Shard files

One `.npz` per episode, `<episode_id>.npz`, compressed, written by
`kaggle_dataset.py` (one replay per process — the 1 GB box's RAM contract),
plus one `vocab.json` per build next to the shards: `{"codes": {...},
"list": [...]}` so every consumer resolves codes without re-parsing replays.

## How the model consumes this (the architecture the format serves)

- tiles: `Embedding(703+1, d)` over the 200 keys → (200, d); scalar block
  → `matmul` → d; concatenation is the per-tile view: tile embedding +
  broadcast global vector.
- tile head: score(cell, edge) = `dot(tile_emb, edge_emb)`, masked by the
  graph's own edge lists (`edges_of(state)`), softmax over the mask — the
  graph owns legality, the model owns ranking. Edge embedding = its chain
  id's embedding, learned.
- market head: small masked softmaxes per decision (sell good × qty bucket,
  buy, hire count, land) — quantities as buckets, not raw regression.
- Everything above is matmul + ReLU + masked softmax: trains in JAX, runs
  bit-comparably in pure NumPy (~50 lines forward pass, verified ≤1e-5
  against JAX at export). No recurrence, no scan, no dynamic shapes.
