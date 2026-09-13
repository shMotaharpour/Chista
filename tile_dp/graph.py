"""TileGraph: the carrot tile's daily state-action graph, engine-verified.

Node = the tile at a day start (TileState). Edge = one daily action chain
executed on a FastSim clone positioned at that state, then idle 24 turns.

Pruning (engine-driven — no hand tables):
  a) no-op sweep: the chain changed nothing AND consumed nothing (F047)
  b) dominance: identical next state, >= production, <= every resource use
     -> the dominated edge is removed (e.g. FERTILIZE->HARVEST is dominated
     by HARVEST: same outcome, wasted fertilizer)

Core: numpy int arrays, CSR per state; strings only at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from world.fast_sim import FastSim

from tile_dp.chains import (N_RESOURCE, RESOURCE_ID, RES_FERTILIZER,
                            RES_LABOR, RES_SEED, chain_id_of, chain_ops,
                            chains_for)
from tile_dp.tile_state import (KIND_NONE, KIND_PLANT, KIND_WEED, TileState,
                                decode_tile)

ENGINE_TAG = "tile-dp-carrot-v1"

# lifecycle length for carrot: plant day 0 .. last harvest day (age 2 = day
# 4), weed from day 5. The graph covers ages -2..2 of the plant plus the
# NONE/WEED pool states; lifecycle day == age + first_yield_day.
LIFE_DAYS = 5


@dataclass(frozen=True)
class TileGraph:
    """The carrot tile's daily state-action graph (numpy int arrays).

    Edge block for state s: edge_offsets[s]:edge_offsets[s+1]. Transitions
    are day-invariant BY DESIGN: age (not absolute day) is the time axis,
    and weed-spawn RNG is deliberately ignored (v1, Hossein-approved).
    """

    n_states: int
    state_keys: np.ndarray          # int64[n_states] packed TileState keys
    key_index: dict[int, int]       # packed key -> state id (O(1) lookup)
    edge_offsets: np.ndarray        # int64[n_states+1]
    edge_next: np.ndarray           # int32[E]
    edge_chain: np.ndarray          # int16[E]
    edge_prod: np.ndarray           # int32[E]   harvested units (carrot)
    edge_use: np.ndarray            # int32[N_RESOURCE, E]
    engine_tag: str

    # ------------------------------------------------------------------ api

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
            path,
            n_states=self.n_states, state_keys=self.state_keys,
            edge_offsets=self.edge_offsets, edge_next=self.edge_next,
            edge_chain=self.edge_chain, edge_prod=self.edge_prod,
            edge_use=self.edge_use, engine_tag=self.engine_tag)

    @classmethod
    def load(cls, path: Path) -> "TileGraph":
        data = np.load(Path(path), allow_pickle=True)
        tag = str(data["engine_tag"])
        if tag != ENGINE_TAG:
            raise ValueError(f"graph engine tag {tag!r} != {ENGINE_TAG!r}; "
                             "rebuild the cache")
        g = cls(
            n_states=int(data["n_states"]),
            state_keys=data["state_keys"],
            key_index={int(k): i for i, k in enumerate(data["state_keys"])},
            edge_offsets=data["edge_offsets"],
            edge_next=data["edge_next"],
            edge_chain=data["edge_chain"],
            edge_prod=data["edge_prod"],
            edge_use=data["edge_use"],
            engine_tag=tag)
        return g


# --------------------------------------------------------------------- build

def _new_sim() -> FastSim:
    return FastSim({"episodeSteps": (LIFE_DAYS + 3) * 24, "seed": 4242})


def _act(farmer, market=None):
    return {"farmer": farmer, "hands": [], "market": market or []}


def _exec_chain(sim: FastSim, ops: tuple[str, ...]
                ) -> tuple[TileState, int, list[tuple[str, int]]]:
    """Execute one daily chain; return (next state, production, resources).

    F030 handled: market purchases land one turn BEFORE the unit op that
    needs them (seed before PLANT, fertilizer before PICKUP)."""
    labor = len(ops)
    use: list[tuple[str, int]] = [(RES_LABOR, labor)]
    production = 0
    planted = False

    for op in ops:
        if op == "PLANT":
            sim.step([_act(["PASS"], [["BUY_SEED", "CARROT", 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PLANT", "CARROT"]), _act(["PASS"])])
            use.append((RES_SEED, 1))
            planted = True
        elif op == "FERTILIZE":
            sim.step([_act(["PASS"], [["BUY_PRODUCT", "FERTILIZER", 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PICKUP", "FERTILIZER", 1]), _act(["PASS"])])
            sim.step([_act(["FERTILIZE"]), _act(["PASS"])])
            use.append((RES_FERTILIZER, 1))
            labor += 1  # PICKUP + FERTILIZE = 2 unit turns total
        elif op == "HARVEST":
            obs = sim.observations()
            me = obs[0]["farms"][0]
            fx, fy = me["farmer"]
            tile = me["tiles"][fy][fx]
            production = int(tile.get("yield_units", 0)) \
                if isinstance(tile, dict) else 0
            sim.step([_act(["HARVEST"]), _act(["PASS"])])
        else:  # WATER, DIG
            sim.step([_act([op]), _act(["PASS"])])

    for _ in range(max(0, N_TURNS_PER_DAY - labor)):
        sim.step([_act(["PASS"]), _act(["PASS"])])

    if production:
        pass  # production already captured above

    obs = sim.observations()
    me = obs[0]["farms"][0]
    fx, fy = me["farmer"]
    tile = me["tiles"][fy][fx]

    # Planting chains decode to the CANONICAL next-day state: seed bought,
    # planted and watered today → age -1, consec=0, yield=1, no fertilizer.
    if planted:
        return TileState("PLANT", "CARROT", -1, 0, 0, 1), production, use

    nxt = decode_tile(tile, obs[0]["day"])
    return nxt, production, use


N_TURNS_PER_DAY = 24


def _use_vector(use: list[tuple[str, int]]) -> list[int]:
    vec = [0] * N_RESOURCE
    for (res, units) in use:
        vec[RESOURCE_ID[res]] = units
    return vec


def build_graph() -> TileGraph:
    """Reachability-forward build over day-start states, with no-op sweep
    and dominance pruning."""
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

    start = intern(TileState("NONE", None, 0, 0, 0, 0))

    # worklist over LIFECYCLE positions: (age, state_id) — age fixes the
    # plant's position; NONE/WEED use age = -99 (no lifecycle clock)
    node_age: dict[tuple[int, int], int] = {(0, start): -99}
    edges: dict[int, list[tuple[int, int, int, list[int]]]] = {}

    frontier = [(0, start)]
    while frontier:
        age, sid = frontier.pop()
        if (age, sid) in node_age and sid in edges:
            continue
        state = state_list[sid]
        node_age[(age, sid)] = age
        sim = _new_sim()

        # position the sim: the state encodes everything needed. For plants
        # we replay the canonical history (plant, water daily, optional
        # fert day) to reach (age, consec, fert_left, yield) at day start.
        if state.kind == KIND_PLANT:
            _replay_to(sim, state)
        # NONE/WEED sims start bare at day 0 (weed RNG ignored, v1)

        for ops in chains_for(state.kind, state.age if state.kind == KIND_PLANT else None):
            branch = sim.clone()
            nxt_state, prod, use = _exec_chain(branch, ops)
            use_vec = _use_vector(use)
            nid = intern(nxt_state)
            nxt_age = nxt_state.age if nxt_state.kind == KIND_PLANT else -99
            edges.setdefault(sid, []).append((nid, chain_id_of(ops), prod, use_vec))
            node_key = (nxt_age, nid)
            if node_key not in node_age:
                node_age[node_key] = nxt_age
                frontier.append((nxt_age, nid))

    n_states = len(state_list)
    # dominance + no-op pruning, per state
    pruned: dict[int, list[tuple[int, int, int, list[int]]]] = {}
    for sid, elist in edges.items():
        kept: list[tuple[int, int, int, list[int]]] = []
        for (nid, cid, prod, uvec) in elist:
            # (a) no-op sweep: nothing changed, nothing consumed
            total_use = sum(uvec)
            if nid == sid and total_use == 0 and prod == 0:
                continue
            kept.append((nid, cid, prod, uvec))
        # (b) dominance among kept edges with identical next state
        final: list[tuple[int, int, int, list[int]]] = []
        for e in kept:
            dominated = False
            for other in kept:
                if other is e:
                    continue
                if other[0] != e[0]:
                    continue
                if other[2] >= e[2] and all(
                        o <= u for o, u in zip(other[3], e[3])) \
                        and (other[2] > e[2] or any(
                            o < u for o, u in zip(other[3], e[3]))):
                    dominated = True
                    break
            if not dominated:
                final.append(e)
        pruned[sid] = final

    # NOTE: no sort-remap. States keep FIRST-INTERN order; lookups use the
    # key_to_id dict (O(1)). A sort+remap here was the source of a subtle
    # edge-target swap bug (pruned lists were keyed by old ids while the
    # flatten iterated new ids) — order is irrelevant for correctness.
    edge_next_l: list[int] = []
    edge_chain_l: list[int] = []
    edge_prod_l: list[int] = []
    edge_use_l: list[list[int]] = [[] for _ in range(N_RESOURCE)]
    offsets = np.zeros(n_states + 1, dtype=np.int64)
    for s in range(n_states):
        offsets[s] = len(edge_next_l)
        for (nid, cid, prod, uvec) in pruned.get(s, []):
            edge_next_l.append(nid)
            edge_chain_l.append(cid)
            edge_prod_l.append(prod)
            for r in range(N_RESOURCE):
                edge_use_l[r].append(uvec[r])
    offsets[n_states] = len(edge_next_l)

    return TileGraph(
        n_states=n_states,
        state_keys=np.array([s.pack() for s in state_list], dtype=np.int64),
        key_index=key_to_id,
        edge_offsets=offsets,
        edge_next=np.array(edge_next_l, dtype=np.int32),
        edge_chain=np.array(edge_chain_l, dtype=np.int16),
        edge_prod=np.array(edge_prod_l, dtype=np.int32),
        edge_use=np.array(edge_use_l, dtype=np.int32),
        engine_tag=ENGINE_TAG)


def _replay_to(sim: FastSim, state: TileState) -> None:
    """Replay the canonical history that lands the tile at `state` at a
    day start. Watering schedule: water EVERY past day except (today-1)
    when state.consec == 1 (that's exactly what consec encodes). Fertilize
    on the single past day that leaves fert_left coverage today."""
    def act(f, m=None):
        return {"farmer": f, "hands": [], "market": m or []}

    spec = K.CROPS["CARROT"]
    today = state.age + spec["first_yield_day"]  # lifecycle day of "now"
    days_alive = state.age + spec["first_yield_day"]
    plant_day = today - days_alive
    fert_day = (today + state.fert_left - 1) - 2 if state.fert_left > 0 \
        else None
    if state.fert_left > 0 and fert_day is not None and fert_day < plant_day:
        return None  # unreachable (fert predates plant)
    if state.consec == 1 and state.fert_left >= 2:
        # consec=1 means yesterday was dry; the fert day would have been
        # watered (it must be, to apply) → yesterday cannot be dry with
        # 2 covered days left. Unreachable — no such day-start state.
        return None

    sim.step([act(["PASS"], [["BUY_SEED", "CARROT", 1],
                             ["BUY_PRODUCT", "FERTILIZER", 2]]), act(["PASS"])])
    sim.step([act(["PICKUP", "FERTILIZER", 2]), act(["PASS"])])
    fert_done = False
    # days 0..today-1 replay their ops; after filling day today-1 the sim
    # sits at the day-`today` start = the target state.
    for cur in range(0, today):
        # --- the day's ops (each 1 turn) ---
        if cur == plant_day:
            sim.step([act(["PLANT", "CARROT"]), act(["PASS"])])
            # The planting day is ALWAYS watered in the canonical history:
            # a plant not watered on its own planting day is weed by the
            # next day (F002) — such states are not part of the graph.
            sim.step([act(["WATER"]), act(["PASS"])])
        else:
            dry_day = today - 1 if state.consec == 1 else None
            fert_today = (not fert_done and fert_day is not None
                          and cur == fert_day)
            if fert_today:
                # nightly auto-drop put the fertilizer back in the shed —
                # PICKUP it again, then FERTILIZE, then WATER (the state
                # says this day was watered and fert covers it)
                sim.step([act(["PICKUP", "FERTILIZER", 1]), act(["PASS"])])
                sim.step([act(["FERTILIZE"]), act(["PASS"])])
                fert_done = True
            if fert_today or cur != dry_day:
                sim.step([act(["WATER"]), act(["PASS"])])

        # --- fill the rest of the day until the day actually advances ---
        obs = sim.observations()
        while obs[0]["day"] == cur and not sim.done:
            sim.step([act(["PASS"]), act(["PASS"])])
            obs = sim.observations()
        if sim.done:
            break
