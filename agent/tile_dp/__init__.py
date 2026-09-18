"""tile_dp: single-tile daily state-action graphs — v3 generalized.

Entities: 5 crops (WHEAT, CARROT, TOMATO, STRAWBERRY, MELON) + 3 animals
(GOOSE, COW, SHEEP). One lifecycle graph per entity; all built with
engine-verified edges (R003) and engine-driven pruning.
"""

from agent.tile_dp.tile_state import TileState, decode_tile
from agent.tile_dp.chains import CHAIN_NAMES, CHAIN_ID_OF, chain_ops, chain_id_of
from agent.tile_dp.graph import TileGraph
from agent.tile_dp.contractor import (HORIZON_DAYS, PricedBoard, TileContractor,
                                price_board)

__all__ = [
    "TileState", "decode_tile",
    "CHAIN_NAMES", "CHAIN_ID_OF", "chain_ops", "chain_id_of",
    "TileGraph", "build_graph",
    "TileContractor", "PricedBoard", "price_board", "HORIZON_DAYS",
]
