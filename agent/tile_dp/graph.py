"""tile_dp graph — the artifact and its reader (v16).

The graph itself: ONE tile graph covering every crop and every animal. Node = the
tile state at a day start, key = `TileState.pack()` (day-invariant: the day is a DP
dimension and prices are a DP input, so neither is in the artifact). Edge = exactly
one day, with `from`, `to`, `chain_id`, `entity_code`, `cost[]` and `produce[]`.
`cost` = inputs consumed + LABOR_HOURS; `produce` = harvest units of the entity's
product + collected fertilizer. The two are SEPARATE 18-vectors and are never netted:
wheat is both the FEED input and the WHEAT crop's product.

This module is what the runtime loads: `TileGraph.load(path)` and the queries over
it. The BUILD is not here — it drives the simulator and lives offline
(`offline_lab/build/graph.py`), because the submission is `agent/` and a builder is not
part of it. The artifact it writes is `agent/artifact/tile_graph.npz` (beside its info file).

Hossein's age conventions, enforced by the decode that labels every edge.
Crops: age 0 = START OF THE GOLDEN WINDOW for one-shot crops
((max_yield_day + 1) // 2) and max_yield_day for ongoing crops; the day the plant
starts turning into a weed (the engine's max_lifespan_step day) decodes as WEED, so
it is never a planned PLANT day. Animals: the positive age is the production phase
0..interval-1 (it wraps); the negative range 1-first_yield_day..-1 is growing up.
care_bank is capped at max_held (contract).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from agent.world.rules import TURNS_PER_DAY
from agent.tile_dp.chains import (chain_name, chain_ops)
from agent.tile_dp.contract import (contract_id, fingerprint_chains, registry_fingerprint)
from agent.tile_dp.tile_state import KEY_BITS, TileState, TileZeroCode

#: Identity of the artifact contract, COMPUTED from its inputs (chains.contract_id).
CONTRACT_ID = contract_id()
ENGINE_TAG = CONTRACT_ID


def _engine_part(tag: str) -> str:
    """The engine facts of a tag, without the build/registry fingerprints.

    The registry is checked separately and freshly (the table on disk is read again), so
    comparing the whole tag here would make a builder refuse its own output: the tag was
    stamped at import time, before this run rewrote the table.
    """
    return "+".join(part for part in tag.split("+")
                    if part.startswith(("eng=", "tpd=", "pb=")))



@dataclass(frozen=True)
class BuildSpec:
    """What a build asks for: the merged tile graph or one entity's graph."""

    entity: str | None = None
    progress: bool = False

    @property
    def entity_kind(self) -> str:
        """'tile' for the merged graph, else 'crop' / 'animal'."""
        if self.entity is None:
            return "tile"
        return "animal" if is_animal(self.entity) else "crop"


@dataclass(frozen=True)
class BuildReport:
    """Build metadata kept beside the artifact (no `life_days`: a horizon guess
    the merged graph does not have any more)."""

    spec: BuildSpec
    n_states: int
    n_edges: int
    n_expanded: int
    n_noop_edges: int
    kinds: dict[str, int]
    n_dominated_edges: int = 0      # dropped by the Pareto sweep

    def describe(self) -> str:
        return (f"{self.spec.entity or 'TILE'}: states {self.n_states}, "
                f"edges {self.n_edges}, expanded {self.n_expanded}, "
                f"no-op edges dropped {self.n_noop_edges}")


@dataclass(frozen=True)
class ChainOutcome:
    """What one executed chain produced (decision 11)."""

    next_state: TileState
    harvest: int
    fert_collect: int
    cost: list[int]
    produce: list[int]


@dataclass(frozen=True)
class Edge:
    """One CSR edge, decoded: `from_id -> to_id` by `chain_id`, for `entity_code`."""

    from_id: int
    to_id: int
    chain_id: int
    entity_code: int
    cost: tuple[int, ...]
    produce: tuple[int, ...]

    @property
    def ops(self) -> tuple[str, ...]:
        return chain_ops(self.chain_id)

    @property
    def name(self) -> str:
        return chain_name(self.ops)

    @property
    def entity(self) -> str | None:
        return entity_of_code(self.entity_code)


@dataclass(frozen=True)
class TileGraph:
    """One tile's lifecycle graph: day-invariant states, one-day edges (CSR).

    `chains` is the table this graph's chain ids index into (written by the builder, read
    by the loader); `edge_cost` / `edge_produce` are (n_edges, N_RESOURCE) int matrices and are
    never netted. State `s` owns the edge slice
    [edge_offsets[s], edge_offsets[s + 1]).
    """

    spec: BuildSpec
    report: BuildReport
    n_states: int
    state_keys: np.ndarray
    key_index: dict[int, int]
    edge_offsets: np.ndarray
    edge_next: np.ndarray
    edge_chain: np.ndarray
    edge_entity: np.ndarray
    edge_cost: np.ndarray
    edge_produce: np.ndarray
    edge_steps: np.ndarray            # engine steps the chain spends (upper
                                      # bracket on LABOR_HOURS, which counts
                                      # worker ops only - see chains.OP_STEPS)
    engine_tag: str
    chains: tuple = ()
    registry_tag: str = ""

    @property
    def entity(self) -> str | None:
        return self.spec.entity

    @property
    def entity_kind(self) -> str:
        return self.spec.entity_kind

    @property
    def n_edges(self) -> int:
        return int(self.edge_offsets[-1])

    def state_id_of(self, state: TileState) -> int:
        pos = self.key_index.get(state.pack())
        if pos is None:
            raise KeyError(f"state {state.describe()} not in graph")
        return pos

    def state_of(self, state_id: int) -> TileState:
        return TileState.unpack(TileZeroCode(int(self.state_keys[state_id])))

    def edges_of(self, state_id: int) -> tuple[int, int]:
        return (int(self.edge_offsets[state_id]),
                int(self.edge_offsets[state_id + 1]))

    def cost_of(self, edge: int, res: str) -> int:
        """Cost units of one edge, by resource name."""
        return int(self.edge_cost[edge][RESOURCE_ID[res]])

    def produce_of(self, edge: int, res: str) -> int:
        """Produced units of one edge, by resource name."""
        return int(self.edge_produce[edge][RESOURCE_ID[res]])

    def edge_at(self, state_id: int, row: int) -> Edge:
        """Decode CSR row `row` (a global edge index) of `state_id`."""
        return Edge(state_id, int(self.edge_next[row]),
                    int(self.edge_chain[row]), int(self.edge_entity[row]),
                    tuple(int(v) for v in self.edge_cost[row]),
                    tuple(int(v) for v in self.edge_produce[row]))

    def edges_from(self, state_id: int) -> Iterator[Edge]:
        lo, hi = self.edges_of(state_id)
        return (self.edge_at(state_id, row) for row in range(lo, hi))

    def save(self, path: Path) -> None:
        """Write the artifact: CSR arrays, cost/produce matrices, metadata."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, entity=self.spec.entity or "",
            merged=bool(self.spec.entity is None),
            entity_kind=self.entity_kind, n_states=self.n_states,
            state_keys=self.state_keys, edge_offsets=self.edge_offsets,
            edge_next=self.edge_next, edge_chain=self.edge_chain,
            edge_entity=self.edge_entity, edge_cost=self.edge_cost,
            edge_produce=self.edge_produce, edge_steps=self.edge_steps,
            registry=self._registry_tag(),
            engine_tag=self._tag(),
            n_expanded=self.report.n_expanded,
            n_noop_edges=self.report.n_noop_edges)

    def _registry_tag(self) -> str:
        """The stamp of the chain table THIS graph indexes into.

        The builder writes the table in the same run, so the tag must come from the chains
        the graph carries - the loader's own fingerprint is only valid for a graph read back
        from disk.
        """
        return fingerprint_chains(self.chains) if self.chains else registry_fingerprint()

    def _tag(self) -> str:
        """The engine tag with the registry part re-stamped (see `_registry_tag`)."""
        from re import sub
        return sub(r"reg=[0-9a-f]+", f"reg={self._registry_tag()}", self.engine_tag)

    @classmethod
    def load(cls, path: Path) -> "TileGraph":
        """Read an artifact written by `save` (tag and registry must match).

        An edge stores its chain as an id, i.e. as a POSITION in `CHAIN_NAMES`
        (owner's item 7), so an artifact is only readable together with the
        registry that produced it; a mismatch means the ids would decode into
        other chains and is refused instead.
        """
        data = np.load(Path(path), allow_pickle=True)
        tag = str(data["engine_tag"])
        if _engine_part(tag) != _engine_part(ENGINE_TAG):
            raise ValueError(f"graph engine tag {tag!r} != {ENGINE_TAG!r}; "
                             "rebuild the cache")
        registry = str(data["registry"])
        if registry != registry_fingerprint():
            raise ValueError(
                f"artifact built with chain registry {registry!r}, this code has "
                f"{registry_fingerprint()!r}: chain ids shifted, rebuild it")
        entity = str(data["entity"]) or None
        # Keep the stamp the artifact carries: a graph read back and saved again must stamp
        # the table it indexes into, not whatever the loader's own registry happens to be
        # (its `chains` field is empty after a load, since the table lives in its own file).

        keys = data["state_keys"]
        kinds: dict[str, int] = {}
        for key in keys:
            kind = TileState.unpack(int(key)).kind
            kinds[kind] = kinds.get(kind, 0) + 1
        spec = BuildSpec(entity=entity)
        return cls(
            spec=spec,
            registry_tag=registry,
            report=BuildReport(spec=spec, n_states=int(data["n_states"]),
                               n_edges=int(data["edge_offsets"][-1]),
                               n_expanded=int(data["n_expanded"]),
                               n_noop_edges=int(data["n_noop_edges"]),
                               kinds=kinds),
            n_states=int(data["n_states"]), state_keys=keys,
            key_index={int(k): i for i, k in enumerate(keys)},
            edge_offsets=data["edge_offsets"], edge_next=data["edge_next"],
            edge_chain=data["edge_chain"], edge_entity=data["edge_entity"],
            edge_cost=data["edge_cost"], edge_produce=data["edge_produce"],
            edge_steps=data["edge_steps"], engine_tag=tag)


