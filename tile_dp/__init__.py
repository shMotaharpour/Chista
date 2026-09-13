"""tile_dp: single-tile daily state-action graphs — v3 generalized.

Entities: 5 crops (WHEAT, CARROT, TOMATO, STRAWBERRY, MELON) + 3 animals
(GOOSE, COW, SHEEP). One lifecycle graph per entity; all built with
engine-verified edges (R003) and engine-driven pruning.
"""

from tile_dp.tile_state import TileState, decode_tile
from tile_dp.chains import CHAIN_NAMES, CHAIN_ID_OF, chain_ops, chain_id_of
from tile_dp.graph import TileGraph, build_graph

__all__ = [
    "TileState", "decode_tile",
    "CHAIN_NAMES", "CHAIN_ID_OF", "chain_ops", "chain_id_of",
    "TileGraph", "build_graph",
]
