"""TileContractor: backward DP over the TileGraph, all-int, numpy core.

solve() reads only RAM arrays; the secretary steers via per-day integer
price/wage vectors. total_profit and all vectors are ints (money units).

Graph CSR is per (day, from_state): edge block for (day d, state s) is
edge_offsets[d, s]:edge_offsets[d, s+1] — transitions are engine-verified
per (state, day) because plant decay is anchored to the absolute planting
day.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tile_dp.chains import chain_ops
from tile_dp.graph import TileGraph


@dataclass(frozen=True)
class DailyPlan:
    day: int
    chain_id: int
    chain: tuple[str, ...]  # decoded at the boundary only


@dataclass(frozen=True)
class ContractorSolution:
    schedule: tuple[DailyPlan, ...]
    production: np.ndarray    # int32[days, 1] harvested units per day
    resource_use: np.ndarray  # int32[days, N_RESOURCE] consumed per day
    total_profit: int


class TileContractor:
    """Backward DP over one crop's TileGraph (all-int arithmetic)."""

    def __init__(self, graph: TileGraph):
        self.g = graph
        self._n_crops = graph.edge_prod.shape[0]
        self._n_res = graph.edge_use.shape[0]
        # the graph's crop prices at prices[crop_id]; its seed costs at
        # wage[1] (WHEAT) or wage[2] (CARROT)
        self._price_row = int(graph.crop_id)
        self._seed_res = 1 if graph.crop_id == 0 else 2

    def solve(self, prices, wage, start_state_id: int,
              start_day: int = 0) -> ContractorSolution:
        g = self.g
        days = g.season_days
        p = {c: np.asarray(v, dtype=np.int64) for c, v in prices.items()}
        w = {r: np.asarray(v, dtype=np.int64) for r, v in wage.items()}
        price_crop = p[self._price_row]
        wage_labor = w[0]
        wage_seed = w[self._seed_res]
        wage_fert = w[3]

        prod = g.edge_prod[0].astype(np.int64)          # [E]
        use = g.edge_use.astype(np.int64)               # [4, E]

        V = np.zeros((days + 1, g.n_states), dtype=np.int64)
        arg_edge = np.full((days, g.n_states), -1, dtype=np.int64)

        # per-(day, state) edge blocks: build a from-state id array per day
        for d in range(days - 1, -1, -1):
            off = g.edge_offsets[d]
            counts = off[1:] - off[:-1]
            if counts.sum() == 0:
                continue
            froms = np.repeat(np.arange(g.n_states, dtype=np.int32), counts)
            # edge global indices for this day
            e_idx = np.concatenate(
                [np.arange(off[s], off[s + 1]) for s in range(g.n_states)
                 if off[s + 1] > off[s]]) if counts.sum() else np.array([],
                                                                        dtype=np.int64)
            pc = int(price_crop[d])
            wl = int(wage_labor[d])
            ws = int(wage_seed[d])
            wf = int(wage_fert[d])
            val = (prod[e_idx] * pc
                   - use[0][e_idx] * wl
                   - (use[1][e_idx] + use[2][e_idx]) * ws
                   - use[3][e_idx] * wf)
            cand = val + V[d + 1][g.edge_next[e_idx]]

            best_val = np.full(g.n_states, -(1 << 62), dtype=np.int64)
            best_edge = np.full(g.n_states, -1, dtype=np.int64)
            starts = np.nonzero(np.diff(froms, prepend=-1))[0]
            ends = np.nonzero(np.diff(froms, append=len(froms)))[0]
            for k in range(len(starts)):
                lo, hi = starts[k], ends[k]
                s = froms[starts[k]]
                j = lo + int(np.argmax(cand[lo:hi]))
                best_edge[s] = e_idx[j]
                best_val[s] = cand[j]
            V[d] = best_val
            arg_edge[d] = best_edge

        # forward recovery
        schedule: list[DailyPlan] = []
        prod_total = np.zeros((days, self._n_crops), dtype=np.int32)
        use_total = np.zeros((days, self._n_res), dtype=np.int32)
        s = start_state_id
        total = 0
        for d in range(start_day, days):
            e = int(arg_edge[d][s])
            if e < 0:
                schedule.append(DailyPlan(day=d, chain_id=0,
                                          chain=chain_ops(0)))
                continue
            schedule.append(DailyPlan(day=d, chain_id=int(g.edge_chain[e]),
                                      chain=chain_ops(int(g.edge_chain[e]))))
            prod_total[d] = g.edge_prod[:, e]
            use_total[d] = g.edge_use[:, e]
            total += int((int(g.edge_prod[0][e]) * int(price_crop[d]))
                         - (int(g.edge_use[0][e]) * int(wage_labor[d]))
                         - ((int(g.edge_use[1][e]) + int(g.edge_use[2][e]))
                            * int(wage_seed[d]))
                         - (int(g.edge_use[3][e]) * int(wage_fert[d])))
            s = int(g.edge_next[e])

        return ContractorSolution(
            schedule=tuple(schedule), production=prod_total,
            resource_use=use_total, total_profit=total)
