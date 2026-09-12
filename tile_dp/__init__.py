"""tile_dp: single-tile DP optimizers (the TileContractor).

Numeric core: numpy int arrays (numeric IDs everywhere in the hot path);
string names only at the module boundary. See DESIGN.md for the locked
decisions (selling excluded from the DP, per-day wage vector, all-int
arithmetic, RAM-first caching).
"""

from tile_dp.tile_state import TileState, pack_key, unpack_key, decode_tile
from tile_dp.chains import CHAINS, chain_ops, encode_chains
from tile_dp.graph import TileGraph, build_graph
from tile_dp.contractor import TileContractor, ContractorSolution, DailyPlan

__all__ = [
    "TileState", "pack_key", "unpack_key", "decode_tile",
    "CHAINS", "chain_ops", "encode_chains",
    "TileGraph", "build_graph",
    "TileContractor", "ContractorSolution", "DailyPlan",
]
