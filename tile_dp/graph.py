"""tile_dp graph build — v11 final: animal states capped at the production
cycle length (Hossein's convention: age wraps inside the positive
production range), no runaway growth. States with age beyond
(max_yield - first_yield + interval) are terminal (collapsed into the
cycle's top age).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from world.fast_sim import FastSim

from tile_dp.chains import (N_RESOURCE, RESOURCE_ID, RES_FERTILIZER,
                            RES_LABOR, RES_SEED_CARROT, RES_SEED_MELON,
                            RES_SEED_STRAWBERRY, RES_SEED_TOMATO,
                            RES_SEED_WHEAT, RES_WHEAT, chain_id_of,
                            chain_ops, chains_for)
from tile_dp.tile_state import (KIND_ANIMAL, KIND_EMPTY_STRUCTURE, KIND_NONE,
                                KIND_PLANT, KIND_WEED, TileState,
                                decode_tile)

ENGINE_TAG = "tile-dp-v11"
LIFE_DAYS = {
    "WHEAT": 7, "CARROT": 6, "TOMATO": 14, "STRAWBERRY": 19, "MELON": 15,
    "GOOSE": 10, "COW": 12, "SHEEP": 12,
}


@dataclass(frozen=True)
class TileGraph:
    entity: str
    entity_kind: str
    life_days: int
    n_states: int
    state_keys: np.ndarray
    key_index: dict[int, int]
    edge_offsets: np.ndarray
    edge_next: np.ndarray
    edge_chain: np.ndarray
    edge_prod: np.ndarray
    edge_fert_out: np.ndarray
    edge_use: np.ndarray
    engine_tag: str

    def state_id_of(self, state: TileState) -> int:
        pos = self.key_index.get(state.pack())
        if pos is None:
            raise KeyError(f"state {state.describe()} not in graph")
        return pos

    def state_of(self, state_id: int) -> TileState:
        return TileState.unpack(int(self.state_keys[state_id]))

    def edges_of(self, state_id: int) -> tuple[int, int]:
        return int(self.edge_offsets[state_id]), int(self.edge_offsets[state_id + 1])

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, entity=self.entity, entity_kind=self.entity_kind,
            life_days=self.life_days, n_states=self.n_states,
            state_keys=self.state_keys, edge_offsets=self.edge_offsets,
            edge_next=self.edge_next, edge_chain=self.edge_chain,
            edge_prod=self.edge_prod, edge_fert_out=self.edge_fert_out,
            edge_use=self.edge_use, engine_tag=self.engine_tag)

    @classmethod
    def load(cls, path: Path) -> "TileGraph":
        data = np.load(Path(path), allow_pickle=True)
        tag = str(data["engine_tag"])
        if tag != ENGINE_TAG:
            raise ValueError(f"graph engine tag {tag!r} != {ENGINE_TAG!r}; "
                             "rebuild the cache")
        return cls(
            entity=str(data["entity"]), entity_kind=str(data["entity_kind"]),
            life_days=int(data["life_days"]), n_states=int(data["n_states"]),
            state_keys=data["state_keys"],
            key_index={int(k): i for i, k in enumerate(data["state_keys"])},
            edge_offsets=data["edge_offsets"], edge_next=data["edge_next"],
            edge_chain=data["edge_chain"], edge_prod=data["edge_prod"],
            edge_fert_out=data["edge_fert_out"], edge_use=data["edge_use"],
            engine_tag=tag)


# --------------------------------------------------------------------- build

def _new_sim(life_days: int) -> FastSim:
    return FastSim({"episodeSteps": (life_days + 10) * 24, "seed": 4242})


def _act(farmer, market=None):
    return {"farmer": farmer, "hands": [], "market": market or []}


_SEED_RES = {"WHEAT": "SEED_WHEAT", "CARROT": "SEED_CARROT",
             "TOMATO": "SEED_TOMATO", "STRAWBERRY": "SEED_STRAWBERRY",
             "MELON": "SEED_MELON"}

_STRUCTURE_OF = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}

N_TURNS_PER_DAY = 24


def _use_vector(use: list[tuple[str, int]]) -> list[int]:
    vec = [0] * N_RESOURCE
    for (res, units) in use:
        vec[RESOURCE_ID[res]] = units
    return vec


def _exec_chain(sim: FastSim, ops: tuple[str, ...], entity: str,
                entity_kind: str) -> tuple[TileState, dict, list]:
    """Execute one daily chain; return (next state, outputs, resources).

    F030: market purchases land one turn BEFORE the unit op needing
    them. FEED's wheat: BUY_PRODUCT WHEAT (market) + PICKUP + FEED = 3
    turns, 1 wheat consumed."""
    labor = len(ops)
    planted = False
    placed = False
    outputs: dict = {}
    use: list[tuple[str, int]] = [(RES_LABOR, labor)]

    for op in ops:
        if op == "BUILD":
            # Buy the animal (market) + build the structure + place the
            # animal — one day's work for an empty-structure/none tile.
            structure = _STRUCTURE_OF[entity]
            sim.step([_act(["PASS"], [["BUY_ANIMAL", entity, 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PICKUP", entity, 1]), _act(["PASS"])])
            sim.step([_act([f"BUILD_{structure}"]), _act(["PASS"])])
            sim.step([_act(["PLACE", entity]), _act(["PASS"])])
            placed = True
        elif op == "PLANT":
            sim.step([_act(["PASS"], [["BUY_SEED", entity, 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PLANT", entity]), _act(["PASS"])])
            use.append((_SEED_RES[entity], 1))
            planted = True
        elif op == "FERTILIZE":
            sim.step([_act(["PASS"], [["BUY_PRODUCT", "FERTILIZER", 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PICKUP", "FERTILIZER", 1]), _act(["PASS"])])
            sim.step([_act(["FERTILIZE"]), _act(["PASS"])])
            use.append((RES_FERTILIZER, 1))
            labor += 1
        elif op == "HARVEST":
            obs = sim.observations()
            me = obs[0]["farms"][0]
            fx, fy = me["farmer"]
            tile = me["tiles"][fy][fx]
            outputs["harvest"] = int(tile.get("yield_units", 0)) \
                if isinstance(tile, dict) else 0
            sim.step([_act(["HARVEST"]), _act(["PASS"])])
        elif op == "FEED":
            sim.step([_act(["PASS"], [["BUY_PRODUCT", "WHEAT", 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PICKUP", "WHEAT", 1]), _act(["PASS"])])
            sim.step([_act(["FEED"]), _act(["PASS"])])
            use.append((RES_WHEAT, 1))
            labor += 2
        elif op == "COLLECT_FERTILIZER":
            obs = sim.observations()
            me = obs[0]["farms"][0]
            fx, fy = me["farmer"]
            tile = me["tiles"][fy][fx]
            outputs["fert_collect"] = int(tile.get("fertilizer_available", 0)) \
                if isinstance(tile, dict) \
                and "fertilizer_available" in tile else 0
            sim.step([_act(["COLLECT_FERTILIZER"]), _act(["PASS"])])
        else:  # WATER, CARE, DIG, PASS
            sim.step([_act([op]), _act(["PASS"])])

    for _ in range(max(0, N_TURNS_PER_DAY - labor)):
        sim.step([_act(["PASS"]), _act(["PASS"])])

    obs = sim.observations()
    me = obs[0]["farms"][0]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]

    if planted:
        spec = K.CROPS[entity]
        base = 0 if spec.get("ongoing") else 1
        nxt = TileState("PLANT", entity, None, None,
                        1 - spec["first_yield_day"], 0, 0, 0, 0, base)
        return nxt, outputs, use
    if placed:
        spec = K.ANIMALS[entity]
        nxt = TileState("ANIMAL", None, entity,
                        _STRUCTURE_OF[entity],
                        1 - spec["first_yield_day"], 0, 0, 0, 0, 0)
        return nxt, outputs, use

    nxt = decode_tile(tile, obs[0]["day"])
    return nxt, outputs, use


def _replay_crop(sim: FastSim, state: TileState, entity: str) -> None:
    """Replay the canonical history that lands a CROP at `state` at a
    day start: planted day 0, watered every day, fert on the day implied
    by fert_left (PICKUP + FERTILIZE that day — the nightly auto-drop
    returns items to the shed)."""
    def act(f, m=None):
        return {"farmer": f, "hands": [], "market": m or []}

    spec = K.CROPS[entity]
    today = state.age + spec["first_yield_day"]
    fert_day = (today + state.fert_left - 1) - 2 if state.fert_left > 0 \
        else None
    if fert_day is not None and fert_day < 0:
        raise ValueError("unreachable (fert predates plant)")

    sim.step([act(["PASS"], [["BUY_SEED", entity, 1],
                             ["BUY_PRODUCT", "FERTILIZER", 2]]), act(["PASS"])])
    sim.step([act(["PICKUP", "FERTILIZER", 2]), act(["PASS"])])
    fert_done = False
    for cur in range(0, today):
        if cur == 0:
            sim.step([act(["PLANT", entity]), act(["PASS"])])
            sim.step([act(["WATER"]), act(["PASS"])])
        else:
            fert_today = (not fert_done and fert_day is not None
                          and cur == fert_day)
            if fert_today:
                sim.step([act(["PICKUP", "FERTILIZER", 1]), act(["PASS"])])
                sim.step([act(["FERTILIZE"]), act(["PASS"])])
                fert_done = True
            sim.step([act(["WATER"]), act(["PASS"])])

        obs = sim.observations()
        while obs[0]["day"] == cur and not sim.done:
            sim.step([act(["PASS"]), act(["PASS"])])
            obs = sim.observations()
        if sim.done:
            break


def _replay_animal(sim: FastSim, state: TileState, entity: str) -> None:
    """Replay the canonical history that lands the ANIMAL at `state` at a
    day start: bought + placed day 0 (fed + cared), then fed and cared
    every day since (wheat bought per day)."""
    def act(f, m=None):
        return {"farmer": f, "hands": [], "market": m or []}

    today = state.age + K.ANIMALS[entity]["first_yield_day"]
    sim.step([act(["PASS"], [["BUY_ANIMAL", entity, 1],
                             ["BUY_PRODUCT", "WHEAT", 12]]), act(["PASS"])])
    sim.step([act(["PICKUP", entity, 1]), act(["PASS"])])
    sim.step([act(["PICKUP", "WHEAT", 12]), act(["PASS"])])
    structure = _STRUCTURE_OF[entity]
    sim.step([act([f"BUILD_{structure}"]), act(["PASS"])])
    sim.step([act(["PLACE", entity]), act(["PASS"])])

    for cur in range(0, today):
        sim.step([act(["PASS"], [["BUY_PRODUCT", "WHEAT", 1]]), act(["PASS"])])
        sim.step([act(["PICKUP", "WHEAT", 1]), act(["PASS"])])
        sim.step([act(["FEED"]), act(["PASS"])])
        sim.step([act(["CARE"]), act(["PASS"])])
        obs = sim.observations()
        while obs[0]["day"] == cur and not sim.done:
            sim.step([act(["PASS"]), act(["PASS"])])
            obs = sim.observations()
        if sim.done:
            break


def _replay_node(sim: FastSim, state: TileState, entity: str,
                 entity_kind: str) -> None:
    if state.kind == "PLANT":
        _replay_crop(sim, state, entity)
    elif state.kind == "ANIMAL":
        _replay_animal(sim, state, entity)
    # NONE / WEED / EMPTY_STRUCTURE: fresh sim (day 0, nothing placed)


def build_graph(entity: str, progress: bool = False) -> TileGraph:
    """Build one entity's lifecycle graph. Each node expanded exactly
    once; every edge engine-executed on a FastSim clone."""
    entity_kind = "animal" if entity in ("GOOSE", "COW", "SHEEP") else "crop"
    life_days = LIFE_DAYS[entity]

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

    start_state = TileState("NONE", None, None, None, 0, 0, 0, 0, 0, 0)
    start = intern(start_state)

    edges: dict[int, list[tuple[int, int, dict, list[int]]]] = {}
    visited: set[int] = set()
    frontier = [start]
    while frontier:
        sid = frontier.pop()
        if sid in visited:
            continue
        visited.add(sid)
        state = state_list[sid]
        sim = _new_sim(life_days)
        _replay_node(sim, state, entity, entity_kind)

        for ops in chains_for(state.kind,
                              animal_graph=(entity_kind == "animal")):
            branch = sim.clone()
            try:
                nxt_state, outputs, use = _exec_chain(
                    branch, ops, entity, entity_kind)
            except Exception:
                continue
            nid = intern(nxt_state)
            if nid not in visited:
                frontier.append(nid)
            edges.setdefault(sid, []).append(
                (nid, chain_id_of(ops), outputs, _use_vector(use)))
        if progress:
            print(f"  {entity}: visited {len(visited)}, "
                  f"states {len(state_list)}", flush=True)

    n_states = len(state_list)
    pruned: dict[int, list[tuple]] = {}
    for sid, elist in edges.items():
        kept = []
        for (nid, cid, outs, uvec) in elist:
            if nid == sid and sum(uvec) == 0 and not outs:
                continue  # no-op sweep (F047)
            kept.append((nid, cid, outs, uvec))
        final = []
        for e in kept:
            dominated = False
            for other in kept:
                if other is e or other[0] != e[0]:
                    continue
                if all(other[3][r] <= e[3][r] for r in range(N_RESOURCE)) \
                        and other[2].get("harvest", 0) >= e[2].get(
                            "harvest", 0) \
                        and (sum(other[3]) < sum(e[3])
                             or other[2].get("harvest", 0) > e[2].get(
                                 "harvest", 0)):
                    dominated = True
                    break
            if not dominated:
                final.append(e)
        pruned[sid] = final

    edge_next_l: list[int] = []
    edge_chain_l: list[int] = []
    edge_prod_l: list[int] = []
    edge_fert_l: list[int] = []
    edge_use_l: list[list[int]] = [[] for _ in range(N_RESOURCE)]
    offsets = np.zeros(n_states + 1, dtype=np.int64)
    for s in range(n_states):
        offsets[s] = len(edge_next_l)
        for (nid, cid, outs, uvec) in pruned.get(s, []):
            edge_next_l.append(nid)
            edge_chain_l.append(cid)
            edge_prod_l.append(outs.get("harvest", 0))
            edge_fert_l.append(outs.get("fert_collect", 0))
            for r in range(N_RESOURCE):
                edge_use_l[r].append(uvec[r])
    offsets[n_states] = len(edge_next_l)

    return TileGraph(
        entity=entity, entity_kind=entity_kind, life_days=life_days,
        n_states=n_states,
        state_keys=np.array([s.pack() for s in state_list], dtype=np.int64),
        key_index=key_to_id,
        edge_offsets=offsets,
        edge_next=np.array(edge_next_l, dtype=np.int32),
        edge_chain=np.array(edge_chain_l, dtype=np.int16),
        edge_prod=np.array(edge_prod_l, dtype=np.int32),
        edge_fert_out=np.array(edge_fert_l, dtype=np.int32),
        edge_use=np.array(edge_use_l, dtype=np.int32),
        engine_tag=ENGINE_TAG)
