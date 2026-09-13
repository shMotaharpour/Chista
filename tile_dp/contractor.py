"""TileContractor: DP over the lifecycle graph, all-int, numpy core.

The lifecycle graph has NO season inside it — the season layer (this DP)
walks lifecycle restarts (harvest/clear -> NONE -> replant). solve()
answers: given a start state and a season window, what is the best chained
sequence of lifecycle walks, valued at the secretary's per-day prices/wages?

The season DP is a simple backward pass over (season day, state): every
state's outgoing edges are the lifecycle graph's edges at that state's
lifecycle day (= age + first_yield_day for plants; 0 for NONE/WEED).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tile_dp.chains import chain_ops
from tile_dp.graph import TileGraph
from tile_dp.tile_state import TileState


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
    """All-int season DP chaining lifecycle walks across the window."""

    def __init__(self, graph: TileGraph):
        self.g = graph
        self._n_crops = graph.edge_prod.shape[0]
        self._n_res = graph.edge_use.shape[0]
        self._price_row = int(graph.crop_id)
        self._seed_res = 1 if graph.crop_id == 0 else 2
        g = graph
        # per (life-day, state) edge blocks: life_day -> {state: (lo, hi)}
        self._blocks: list[dict[int, tuple[int, int]]] = []
        for d in range(g.life_days):
            blocks: dict[int, tuple[int, int]] = {}
            off = g.edge_offsets[d]
            for s in range(g.n_states):
                if off[s + 1] > off[s]:
                    blocks[s] = (int(off[s]), int(off[s + 1]))
            self._blocks.append(blocks)
        self._prod = g.edge_prod[0].astype(np.int64)
        self._use = g.edge_use.astype(np.int64)
        self._states = [TileState.unpack(int(k)) for k in g.state_keys]

    def solve(self, prices, wage, start_state: TileState,
              start_day: int = 0, horizon_days: int = 30
              ) -> ContractorSolution:
        g = self.g
        H = horizon_days
        p = {c: np.asarray(v, dtype=np.int64) for c, v in prices.items()}
        w = {r: np.asarray(v, dtype=np.int64) for r, v in wage.items()}
        price_crop = p[self._price_row]
        wage_labor = w[0]
        wage_seed = w[self._seed_res]
        wage_fert = w[3]
        prod, use = self._prod, self._use
        states = self._states

        import kaggle_environments.envs.kaggriculture.kaggriculture as KK
        crop_name = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY",
                     "MELON"][int(g.crop_id)]
        fy = KK.CROPS[crop_name]["first_yield_day"]

        def life_day_of(state: TileState, t: int) -> int | None:
            """Lifecycle-day of a state at season-day t (plant day implied:
            t - (age + fy)); None if that history precedes the window."""
            if state.crop_id < 0:      # NONE / WEED: lifecycle resets
                return 0
            plant_day = t - (state.age + fy)
            if plant_day < start_day:
                return None            # planted before the window: no chain
            return state.age + fy

        # backward over season days; V[t][sid] = best profit t..H-1
        V = np.zeros((H + 1, g.n_states), dtype=np.int64)
        choice: list[dict[int, int]] = [dict() for _ in range(H)]

        for t in range(H - 1, -1, -1):
            pc = int(price_crop[t])
            wl = int(wage_labor[t])
            ws = int(wage_seed[t])
            wf = int(wage_fert[t])
            day_choice = choice[t]
            for sid, state in enumerate(states):
                # life-day of the node this state represents at season day t:
                # a state's age fixes its lifecycle day ONLY via its plant
                # day; but the forward walker carries the actual node, so
                # instead of inferring from age alone we index blocks by the
                # life-day the walker is at. To keep the DP table simple we
                # index by state and resolve the block per (t, sid) through
                # the walker's life-day. For the backward table we compute
                # V for EVERY plausible life-day by storing per state the
                # max over its edges at its OWN lifecycle day (age+fy).
                if state.crop_id < 0:
                    ld = 0
                else:
                    ld = state.age + fy
                    if not (0 <= ld < g.life_days):
                        V[t][sid] = 0
                        continue
                block = self._blocks[ld].get(sid)
                if block is None:
                    # state not alive at this lifecycle day: idle value
                    V[t][sid] = V[t + 1][sid]
                    continue
                lo, hi = block
                best_v = -(1 << 62)
                best_e = -1
                for e in range(lo, hi):
                    nid = int(g.edge_next[e])
                    nxt = states[nid]
                    immediate = (int(prod[e]) * pc
                                 - int(use[0][e]) * wl
                                 - (int(use[1][e]) + int(use[2][e])) * ws
                                 - int(use[3][e]) * wf)
                    v = immediate + V[t + 1][nid]
                    if v > best_v:
                        best_v, best_e = v, e
                V[t][sid] = best_v
                day_choice[sid] = best_e

        # forward recovery: walk the season, carrying (state, life-day)
        schedule: list[DailyPlan] = []
        prod_total = np.zeros((H, self._n_crops), dtype=np.int32)
        use_total = np.zeros((H, self._n_res), dtype=np.int32)
        s = g.state_id_of(start_state)
        state = start_state
        total = 0
        for t in range(start_day, H):
            e = choice[t].get(s, -1)
            if e < 0:
                schedule.append(DailyPlan(day=t, chain_id=0,
                                          chain=chain_ops(0)))
                # idle: nothing changes for NONE/WEED; plants decay on
                # their own only through edges, so idling a plant outside
                # its edges cannot happen in a well-formed graph.
                continue
            schedule.append(DailyPlan(day=t, chain_id=int(g.edge_chain[e]),
                                      chain=chain_ops(int(g.edge_chain[e]))))
            prod_total[t] = g.edge_prod[:, e]
            use_total[t] = g.edge_use[:, e]
            total += int((int(prod[e]) * int(price_crop[t]))
                         - (int(use[0][e]) * int(wage_labor[t]))
                         - ((int(use[1][e]) + int(use[2][e]))
                            * int(wage_seed[t]))
                         - (int(use[3][e]) * int(wage_fert[t])))
            s = int(g.edge_next[e])
            state = states[s]

        return ContractorSolution(
            schedule=tuple(schedule), production=prod_total,
            resource_use=use_total, total_profit=total)
