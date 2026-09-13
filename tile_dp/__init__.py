"""tile_dp: single-tile daily state-action graph (v1: carrot only).

Per Hossein's spec (this branch):
- State = the tile AT DAY START (hour 0): NONE | WEED | PLANT
- Edge = one daily action chain (canonical order PLANT -> FERTILIZE ->
  WATER -> HARVEST, plus DIG) executed by the workers, then idle to the
  next day start.
- Edge pruning (engine-driven, no hand tables):
    a) silent no-op sweep: chain changed nothing and consumed nothing (F047)
    b) dominance: identical next state AND >= production AND <= every
       resource use -> the dominated edge is dropped
- Edge outcomes come from FastSim execution (R003 — engine is the only
  rule source). Numeric core: numpy; strings only at the boundary.
"""

from tile_dp.tile_state import TileState, decode_tile
from tile_dp.chains import CHAIN_NAMES, CHAIN_ID_OF, chain_ops, chain_id_of
from tile_dp.graph import TileGraph, build_graph

__all__ = [
    "TileState", "decode_tile",
    "CHAIN_NAMES", "CHAIN_ID_OF", "chain_ops", "chain_id_of",
    "TileGraph", "build_graph",
]
