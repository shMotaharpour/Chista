# tile_dp — per-tile daily state-action graphs

A tile's day, as a graph: the state is the tile at day start, an edge is one daily action
chain run by the workers, and the DP over that graph prices every tile on the board.

## Where things live

| file | role |
|---|---|
| `agent/tile_dp/tile_state.py` | the tile as a graph node, and the packed key it is stored under |
| `agent/tile_dp/chains.py` | what a chain IS, the entity codes, and the loader for the built table |
| `agent/tile_dp/graph.py` | `TileGraph` — the reader for the shipped graph, and its edges |
| `agent/tile_dp/contract.py` | the fingerprints a load is checked against |
| `agent/tile_dp/contractor.py` | the DP: one backward sweep, plans recovered forward |
| `offline_lab/build/chains.py` | the rules a chain set is generated from (build only) |
| `offline_lab/build/ledger.py` | what a chain costs and yields (build only) |
| `offline_lab/build/graph.py` | `build_graph()` — the builder |
| `agent/artifact/` | `tile_graph.npz` + `tile_graph.json`, `tile_chains.data.json` + `tile_chains.json` |

The graph is the artifact the agent loads; the chain table is its action index. They are
written by one run of the builder and stamped with one contract, so they cannot drift.

## Build

```
.venv/bin/python -m offline_lab.build.graph
```

Breadth-first over day-start states. Each node keeps the simulation that produced it and is
expanded on a clone of it, so a node's edges are measured on the tile state it really is.
Every edge is asserted: it must end within one day (`ChainSpansDays`) and land on the state it
promises (`StateMismatch` / `ChainNotRealised`). A chain the engine refuses fails the build
loudly instead of being skipped, which is what keeps the graph honest.

The weed-spawn chance is pinned to zero: otherwise a bare tile's next state would depend on
the RNG.

## The chain table

A chain is one worker-day on one tile, an ordered tuple of ops, and the empty chain
(`NO_ACTION`) is an idle day. The table is stored **packed**: one integer per chain, four bits
per op, with the op names written once beside it. Ids are positions, so the table and the
graph are read together and a mismatch is refused.

Only the chains that survive pruning are shipped. The rest are legal days nothing optimal
ever chooses.

## Cost and produce

Two separate 18-wide vectors per edge, never netted. The columns are labour, the five seeds,
the three animals and the nine products; wheat and fertilizer are the two that appear on both
sides. `cost` counts the inputs a chain spends plus one labour hour per worker op; `produce`
counts the harvested units of the tile's standing product plus the fertilizer a collection
picked up. The graph carries both, so the runtime prices from the matrices and never prices a
chain itself.

## Which chains apply to a node

`chains_for` selects by the tile's kind and then filters by the node: `CARE` without `FEED` is
a no-op and is dropped, `FERTILIZE` outside the window a dose can still reach is dropped, and
`HARVEST` is dropped when the tile has nothing to harvest or is too young for its first
yield. The crop's own nature (one-shot or ongoing) decides which chains exist at all, because
what may follow a harvest differs.

## Open

The inputs a chain spends — a seed, an animal, fertilizer, wheat — have to be bought and
carried by a day layer that does not exist yet; until then the builder realises the purchase
inline as a stand-in, and labour hours are a floor rather than the whole day.
