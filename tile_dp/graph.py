"""TileGraph: the per-crop day-transition graph, built BY THE ENGINE (R003).

Node identity = (day, TileState) — the same tile situation on different
days is a DIFFERENT node, because the engine anchors plant decay to the
absolute planting day (F008): a carrot alive at day 3 and an identical-
looking carrot alive at day 4 have different remaining lifespans. Edges
are grouped per (day, from_state) and every edge is a daily chain executed
on a FastSim clone positioned at that exact (day, state) (R003 — the
interpreter is the only rule source).

Numeric core: numpy int arrays; strings only at the boundary (DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from world.fast_sim import FastSim

from tile_dp.chains import (CHAINS, RESOURCE_ID, RESOURCE_NAMES, RES_LABOR,
                            RES_SEED_WHEAT, RES_SEED_CARROT,
                            RES_SEED_TOMATO, RES_SEED_STRAWBERRY,
                            RES_SEED_MELON, RES_FERTILIZER, chain_id_of,
                            encode_chains)
from tile_dp.tile_state import (CROP_ID, CROP_NAMES, CROP_NONE, CROP_WEED,
                                TileState, decode_tile)

N_RESOURCE = len(RESOURCE_NAMES)
N_TURNS_PER_DAY = 24
ENGINE_TAG = "kaggriculture-tile-dp-v2"


@dataclass(frozen=True)
class TileGraph:
    """One crop's day-transition graph in numpy int arrays.

    Nodes are (day, state) pairs; state_keys holds the distinct TileState
    pack keys. The edge block for (day d, state s) is
    edge_offsets[d, s]:edge_offsets[d, s+1] (empty where unreachable).
    """

    crop_id: int
    season_days: int
    n_states: int
    state_keys: np.ndarray            # int64[n_states] packed TileState keys
    edge_offsets: np.ndarray          # int64[days, states+1]
    edge_next: np.ndarray             # int32[E]  to-state id (same day+1)
    edge_chain: np.ndarray            # int16[E]
    edge_prod: np.ndarray             # int32[1, E]  (v1: the graph's crop)
    edge_use: np.ndarray              # int32[N_RESOURCE, E]
    chain_ops: tuple[tuple[str, ...], ...]
    engine_tag: str

    # ------------------------------------------------------------------ api

    def state_id_of(self, state: TileState) -> int:
        key = state.pack()
        pos = np.searchsorted(self.state_keys, key)
        if pos >= self.n_states or self.state_keys[pos] != key:
            raise KeyError(f"state {state.describe()} not in graph")
        return int(pos)

    def state_of(self, state_id: int) -> TileState:
        return TileState.unpack(int(self.state_keys[state_id]))

    def edges_of(self, day: int, state_id: int) -> tuple[int, int]:
        off = self.edge_offsets[day]
        return int(off[state_id]), int(off[state_id + 1])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            crop_id=self.crop_id, season_days=self.season_days,
            n_states=self.n_states,
            state_keys=self.state_keys, edge_offsets=self.edge_offsets,
            edge_next=self.edge_next, edge_chain=self.edge_chain,
            edge_prod=self.edge_prod, edge_use=self.edge_use,
            chain_ops=np.array(self.chain_ops, dtype=object),
            engine_tag=self.engine_tag)

    @classmethod
    def load(cls, path: Path) -> "TileGraph":
        data = np.load(Path(path), allow_pickle=True)
        tag = str(data["engine_tag"])
        if tag != ENGINE_TAG:
            raise ValueError(f"graph engine tag {tag!r} != {ENGINE_TAG!r}; "
                             "rebuild the cache")
        return cls(
            crop_id=int(data["crop_id"]),
            season_days=int(data["season_days"]),
            n_states=int(data["n_states"]),
            state_keys=data["state_keys"],
            edge_offsets=data["edge_offsets"],
            edge_next=data["edge_next"],
            edge_chain=data["edge_chain"],
            edge_prod=data["edge_prod"],
            edge_use=data["edge_use"],
            chain_ops=tuple(tuple(x) for x in data["chain_ops"]),
            engine_tag=tag)


# --------------------------------------------------------------------- build

def _new_sim(season_days: int) -> FastSim:
    return FastSim({"episodeSteps": (season_days + 3) * 24, "seed": 4242})


def _act(farmer, market=None):
    return {"farmer": farmer, "hands": [], "market": market or []}


def _exec_chain(sim: FastSim, ops: tuple[str, ...], crop_name: str
                ) -> tuple[TileState, tuple, tuple]:
    """Run one daily chain on `sim` (positioned at a day start), then PASS
    the rest of the day. Returns (next TileState, production, resource_use).

    F030 handled: market purchases land one turn BEFORE the unit op that
    needs them."""
    labor = len(ops)
    production: tuple = ()
    use: list[tuple[str, int]] = [(RES_LABOR, labor)]
    harvest_units = 0
    planted = False

    for op in ops:
        if op.startswith("PLANT_"):
            crop = op[len("PLANT_"):]
            sim.step([_act(["PASS"], [["BUY_SEED", crop, 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PLANT", crop]), _act(["PASS"])])
            use.append(({"WHEAT": RES_SEED_WHEAT, "CARROT": RES_SEED_CARROT,
                         "TOMATO": RES_SEED_TOMATO,
                         "STRAWBERRY": RES_SEED_STRAWBERRY,
                         "MELON": RES_SEED_MELON}[crop], 1))
            planted = True
        elif op == "FERTILIZE":
            sim.step([_act(["PASS"], [["BUY_PRODUCT", "FERTILIZER", 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PICKUP", "FERTILIZER", 1]), _act(["PASS"])])
            sim.step([_act(["FERTILIZE"]), _act(["PASS"])])
            use.append((RES_FERTILIZER, 1))
            labor += 2
        elif op == "HARVEST":
            obs = sim.observations()
            me = obs[0]["farms"][0]
            fx, fy = me["farmer"]
            tile = me["tiles"][fy][fx]
            harvest_units = int(tile.get("yield_units", 0)) \
                if isinstance(tile, dict) else 0
            sim.step([_act(["HARVEST"]), _act(["PASS"])])
        else:  # WATER, DIG, PASS
            sim.step([_act([op]), _act(["PASS"])])

    for _ in range(max(0, N_TURNS_PER_DAY - labor)):
        sim.step([_act(["PASS"]), _act(["PASS"])])

    if harvest_units:
        production = ((0, harvest_units),)

    obs = sim.observations()
    me = obs[0]["farms"][0]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]

    # Planting chains decode to the CANONICAL next-day state: bought seed,
    # planted and watered today → age = 1 - first_yield_day, consec=0,
    # yield=1, no fertilizer yet.
    if planted:
        spec = K.CROPS[crop_name]
        nxt = TileState(CROP_ID[crop_name], 1 - spec["first_yield_day"],
                        0, 0, 1)
        return nxt, production, tuple(use)

    nxt = decode_tile(tile, obs[0]["day"])
    return nxt, production, tuple(use)


def _chain_for_state(state: TileState, crop_name: str) -> list[tuple[str, ...]]:
    """Chains applicable AT this state (precondition pruning)."""
    out: list[tuple[str, ...]] = []
    if state.crop_id == CROP_NONE:
        out.append(("PLANT_" + crop_name, "WATER"))
        out.append(("PASS",))
        return out
    if state.crop_id == CROP_WEED:
        return [("DIG",), ("PASS",)]
    for ops in CHAINS:
        if "HARVEST" in ops and state.age < 0:
            continue  # harvest refused before first_yield_day (F026)
        out.append(ops)
    return out


def build_graph(crop_id: int, season_days: int = 30,
                progress: bool = False) -> TileGraph:
    """Reachability-forward build over (day, state) nodes. The node key is
    (day, state_key) — the same TileState on different days is a different
    node (decay is anchored to the absolute planting day, F008)."""
    crop_name = CROP_NAMES[crop_id]

    key_to_id: dict[int, int] = {}
    state_list: list[TileState] = []

    def intern(state: TileState) -> int:
        k = state.pack()
        sid = key_to_id.get(k)
        if sid is None:
            sid = len(state_list)
            key_to_id[k] = sid
            state_list.append(state)
        return sid

    none_id = intern(TileState(CROP_NONE, 0, 0, 0, 0))

    # node = (day, state_id) -> sim positioned at that day start
    reachable: list[dict[int, FastSim]] = [dict() for _ in range(season_days)]
    reachable[0][none_id] = _new_sim(season_days)

    edges: dict[tuple[int, int], list[tuple[int, int, tuple, tuple]]] = {}

    for day in range(season_days):
        sims = reachable[day]
        for sid, sim in list(sims.items()):
            state = state_list[sid]
            for ops in _chain_for_state(state, crop_name):
                branch = sim.clone()  # parent must stay at its day start
                nxt_state, prod, use = _exec_chain(branch, ops, crop_name)
                nid = intern(nxt_state)
                if day + 1 < season_days:
                    reachable[day + 1].setdefault(nid, branch)
                edges.setdefault((day, sid), []).append(
                    (nid, chain_id_of(ops), prod, use))
        if progress:
            print(f"day {day}: {len(sims)} nodes, "
                  f"total states {len(state_list)}", flush=True)

    n_states = len(state_list)
    # Canonical state ordering: sort by packed key (state_id_of uses
    # binary search) and remap edge endpoints into the sorted id space.
    order = np.argsort([s.pack() for s in state_list], kind="stable")
    old_to_new = np.empty(n_states, dtype=np.int64)
    sorted_states: list[TileState] = []
    for new_id, old_id in enumerate(order):
        old_to_new[old_id] = new_id
        sorted_states.append(state_list[old_id])
    # remap the (day, from_state) node keys into the sorted id space too
    edges = {
        (d_old, int(old_to_new[s_old])): elist
        for (d_old, s_old), elist in edges.items()
    }

    offsets = np.zeros((season_days, n_states + 1), dtype=np.int64)
    edge_next_l: list[int] = []
    edge_chain_l: list[int] = []
    edge_prod_l: list[int] = []
    edge_use_l: list[list[int]] = [[] for _ in range(N_RESOURCE)]

    for d in range(season_days):
        for s in range(n_states):
            offsets[d, s] = len(edge_next_l)
            for (nxt, ci, prod, use) in edges.get((d, s), []):
                edge_next_l.append(int(old_to_new[nxt]))
                edge_chain_l.append(ci)
                pvec = [0]
                for (c, units) in prod:
                    pvec[c] = units
                edge_prod_l.append(pvec[0])
                rvec = [0] * N_RESOURCE
                for (res, units) in use:
                    rvec[RESOURCE_ID[res]] = units
                for r in range(N_RESOURCE):
                    edge_use_l[r].append(rvec[r])
        offsets[d, n_states] = len(edge_next_l)

    edge_next = np.array(edge_next_l, dtype=np.int32)
    edge_chain = np.array(edge_chain_l, dtype=np.int16)
    edge_prod = np.array([edge_prod_l], dtype=np.int32)
    edge_use = np.array(edge_use_l, dtype=np.int32)

    return TileGraph(
        crop_id=crop_id, season_days=season_days, n_states=n_states,
        state_keys=np.array([s.pack() for s in sorted_states], dtype=np.int64),
        edge_offsets=offsets, edge_next=edge_next, edge_chain=edge_chain,
        edge_prod=edge_prod, edge_use=edge_use,
        chain_ops=encode_chains(), engine_tag=ENGINE_TAG)
