"""Build the shipped tile graph model — offline only.

Run from the repo root:

    .venv/bin/python -m offline_lab.build.graph

Writes `tile_dp/models/graph_tile_lifecycle.npz` (the merged graph) and
`agent/tile_dp/models/build_report.json`, then reloads the written file to prove the
round-trip works. The artifact is a tracked model file: it is committed, and this
builder is the only thing that writes it.

It lives outside `agent/` on purpose: it drives the simulator
(`offline_lab/fast_sim.FastSim`), the submission never imports it, and the runtime reads
the artifact with `agent.tile_dp.graph.TileGraph.load`.

Simulation inheritance (decision 6): a node is expanded with the SAME sim that
produced it (`dict[state_id, FastSim]`), so its edges are computed from the true state
of the tile that reached it; only the root (NONE, day 0) gets a fresh sim. The earlier
builder replayed a hand-written canonical history per node instead, which regularly
landed on another day-start state - measured 184/394 nodes on the current chains
(186/456 on the older build) - so every edge of such a node was computed from the
wrong state. `_replay_*` and the `replay_mismatch` script were deleted.

Assertions (decision 8): a wrong edge must fail the build, never be stored.
  * before expanding a node, its own sim must decode to the node's state;
  * after every edge the branch sim's tile is decoded and compared with the expected
    next state: a chain that changes the tile (PLANT / BUILD / PLACE / DIG) must land
    exactly on the modelled state, full TileState equality, else StateMismatch with
    both describe() strings;
  * a growth day (no kind change) must carry the tile's identity one day forward -
    same kind/crop/animal/structure, age advanced exactly one day, and the
    engine-independent day bookkeeping implied by the ops (consec / unfed /
    fert_left). The growth dims themselves (yield_units, care_bank) and the engine's
    destroy paths are the engine's own answer, read from `decode_tile`, because this
    repo never re-implements the game (R003);
  * PASS is a whole-chain op: PASS inside a multi-op chain is rejected.
"""

from __future__ import annotations

import json
import os
import resource
import time
from collections import deque
from pathlib import Path

import numpy as np

from kaggle_environments.envs.kaggriculture import kaggriculture as K

from offline_lab.fast_sim import FastSim

from agent.artifact import artifact_path, write_info
from agent.world.model import UnitAction
from agent.world.rules import ANIMAL_RULES, CROP_RULES
from offline_lab.build.chains import (chain_id_of, chain_ops, chains_for,
                                      domain_ok, registry)
from agent.tile_dp.chains import (CONSTRUCTIVE_OPS, ENTITY_CODE, ENTITY_NAMES,
                                  N_RESOURCE, OP_STEPS,
                                  RESOURCE_ID, chain_name, fingerprint_chains,
                                  chain_steps, contract_id, cost_vector,
                                  entity_code_of, entity_of_code,
                                  engine_fingerprint, is_animal, produce_vector,
                                  registry_fingerprint)
from agent.tile_dp.graph import (ENGINE_TAG, BuildReport, BuildSpec, ChainOutcome,
                                 Edge, TileGraph)
from agent.tile_dp.tile_state import (EMPTY_KIND_OF_STRUCTURE, EMPTY_KINDS,
                                      KIND_ANIMAL, KIND_EMPTY_COOP,
                                      KIND_EMPTY_PASTURE, KIND_NONE, KIND_PLANT,
                                      KIND_WEED, TURNS_PER_DAY, TileState,
                                      crop_age_origin, decode_tile)

# Which structure an entity lives in (engine data: BUILD needs it, and it is how the
# EMPTY_STRUCTURE candidates are decided). Crops are absent on purpose.
_STRUCTURE_OF = {name: K.ANIMALS[name]["structure"] for name in K.ANIMALS}

# The artifact the runtime loads: `agent/artifact/`, beside its info file (AGENTS.md:
# the submission is agent/, and the builders write their artifacts into it).
NAME = "tile_graph"
GRAPH_PATH = artifact_path(NAME, ".npz")




# ------------------------------------------------------- sim + state helpers
# ------------------------------------------------------- sim + state helpers

class ChainSpansDays(RuntimeError):
    """A daily chain needed more than one game day (cost-model bug)."""


class ChainNotRealised(RuntimeError):
    """A chain's op did not land on the tile (the engine refused it)."""


class StateMismatch(RuntimeError):
    """Decision 8: the engine's next state is not the one the chain promises."""


def _new_sim() -> FastSim:
    """Fresh tile sim (decision 7): one 30-day season, fixed seed, no weeds.

    The builder walks one day per edge, so 30 days bounds the walk and keeps a
    node's sim inside its episode. weedSpawnChance 0.0 keeps a bare tile bare: a
    random WEED would make a tile's next state a function of the RNG instead of
    the player's chain.
    """
    return FastSim({"episodeSteps": 30 * 24, "seed": 4242,
                    "weedSpawnChance": 0.0})


def _act(farmer: list, market: list | None = None) -> dict:
    """One agent's action dict; the second agent always idles."""
    return {"farmer": farmer, "hands": [], "market": market or []}


def _tile_and_day(sim: FastSim) -> tuple[dict | None, int]:
    """(tile the worker stands on, current day) - live view, read-only."""
    obs = sim.observations()
    me = obs[0]["farms"][0]
    fx, fy = me["farmer"]
    return me["tiles"][fy][fx], int(obs[0]["day"])


def _state_of(sim: FastSim) -> TileState:
    """Day-start state of the sim's own tile."""
    return decode_tile(*_tile_and_day(sim))


def verify_engine_constants() -> None:
    """Re-check the decode constants the engine tables do not state.

    TURNS_PER_DAY comes from the run configuration (kaggriculture.py:864) and
    the dry limit is a literal in the nightly loop (783, 817): a run with
    another turnsPerDay would shift every day comparison in the decoder while
    the tables still looked right. Both are checked on a scratch sim, so a
    wrong constant raises at build time instead of decoding wrongly.
    """
    sim = _new_sim()
    sim.step([_act(["PASS"], [["BUY_SEED", "MELON", 1]]), _act(["PASS"])])
    sim.step([_act(["PLANT", "MELON"]), _act(["PASS"])])
    tile, _ = _tile_and_day(sim)
    if tile is None:
        raise RuntimeError("PLANT MELON left the tile empty: cannot check the "
                           "decode constants against this engine")
    spec = K.CROPS["MELON"]
    planted = int(tile["planted_day"])
    want = (planted + int(spec["max_yield_day"]) + 1) * TURNS_PER_DAY
    got = int(tile["max_lifespan_step"])
    if got != want:
        raise RuntimeError(
            f"day length mismatch: a MELON planted on day {planted} carries "
            f"max_lifespan_step={got}, the decoder expects {want} = "
            f"(day + max_yield_day + 1) * {TURNS_PER_DAY} turns per day")
    for _ in range(TURNS_PER_DAY):
        sim.step([_act(["PASS"]), _act(["PASS"])])
    dry, _ = _tile_and_day(sim)
    if dry is None or dry.get("kind") != KIND_WEED:
        raise RuntimeError(
            f"dry limit mismatch: an unwatered MELON became {dry!r} after one "
            f"day; the decoder expects WEED after the second dry night")


def _op_probe_sequences(op: str, entity: str) -> list[list]:
    """The engine steps `_exec_chain` spends on `op`, in order.

    One shared fact in one place: the executor's per-op sequences are built
    from this table, and `verify_op_steps` replays it, so the probe and the
    executor cannot drift apart. `OP_STEPS[op]` is `len(...)` of this list.
    """
    if op == "PLANT":
        return [(["PASS"], [["BUY_SEED", entity, 1]]),
                (["PLANT", entity], [])]
    if op == "FERTILIZE":
        return [(["PASS"], [["BUY_PRODUCT", "FERTILIZER", 1]]),
                (["PICKUP", "FERTILIZER", 1], []),
                (["FERTILIZE"], [])]
    if op == "FEED":
        return [(["PASS"], [["BUY_PRODUCT", "WHEAT", 1]]),
                (["PICKUP", "WHEAT", 1], []),
                (["FEED"], [])]
    if op == "PLACE":
        return [(["PASS"], [["BUY_ANIMAL", entity, 1]]),
                (["PICKUP", entity, 1], []),
                (["PLACE", entity], [])]
    return [(["PASS"], [])]      # single-step ops: the act itself


def verify_op_steps() -> None:
    """Check `OP_STEPS` against the engine's own answer (minimality).

    Replays `_op_probe_sequences` - the same sequences `_exec_chain` runs -
    one engine step at a time, and reads the op's effect off the raw tile:
    the effect must NOT be present before the op's action step and MUST be
    present after it. If the engine ever needed one more step (a fourth for
    FERTILIZE), the effect would not have landed and this raises; if a
    sequence were over-counted, the action step would not be the last and
    the effect would appear one step early.
    """
    cases = [("PLANT", "CARROT", None), ("FERTILIZE", "CARROT", None),
             ("FEED", "COW", None), ("PLACE", "COW", None),
             ("PLACE", "GOOSE", None)]
    for op, entity, _unused in cases:
        want = OP_STEPS.get(op, 1)
        sim = _new_sim()
        if op in ("FERTILIZE",):
            # a fertiliser dose needs a plant on the tile (engine refuses on
            # a bare one): plant it first, from a scratch sim of its own
            sim.step([_act(["PASS"], [["BUY_SEED", entity, 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PLANT", entity]), _act(["PASS"])])
        elif op in ("FEED", "PLACE"):
            # feed / place need the structure the animal lives in; FEED also
            # needs the animal ON the structure (a feed on an empty one is a
            # silent refusal), so the setup places it with the same sequence
            # the PLACE probe measures
            structure = "COOP" if entity == "GOOSE" else "PASTURE"
            sim.step([_act(["PASS"], [["BUY_ANIMAL", entity, 1]]),
                      _act(["PASS"])])
            sim.step([_act([f"BUILD_{structure}"]), _act(["PASS"])])
            if op == "FEED":
                # the feed target is the animal, not the empty structure:
                # buy, carry and place it with the very sequence the PLACE
                # probe measures, then the FEED sequence lands on it
                sim.step([_act(["PICKUP", entity, 1]), _act(["PASS"])])
                sim.step([_act(["PLACE", entity]), _act(["PASS"])])
        effects = {"PLANT": ("crop", entity),
                   "FERTILIZE": ("fertilized_until_day", None),
                   "FEED": ("fed_today", True),
                   "PLACE": ("animal", entity)}
        field, value = effects[op]
        seq = _op_probe_sequences(op, entity)
        assert len(seq) == want, (op, len(seq), want)
        for _farmer, _market in seq[:-1]:
            sim.step([_act(list(_farmer), list(_market)), _act(["PASS"])])
        before, _ = _tile_and_day(sim)
        # snapshot the pre-action fields NOW: _tile_and_day returns the live
        # tile dict, and the action step mutates it in place
        # snapshot the pre-action fields NOW: _tile_and_day returns the live
        # tile dict and the action step mutates it in place
        before_fert = int(before.get("fertilized_until_day", -1)) if before else -1
        before_field = dict(before) if before else {}
        final_farmer, final_market = seq[-1]
        sim.step([_act(list(final_farmer), list(final_market)),
                  _act(["PASS"])])
        after, _ = _tile_and_day(sim)
        if after is None or not isinstance(after, dict):
            raise RuntimeError(f"op {op}: tile vanished under the probe")
        # effect present after, and (where meaningful) not before
        if op == "FERTILIZE":
            landed = int(after.get("fertilized_until_day", -1)) > before_fert
        elif op == "FEED":
            landed = bool(after.get(field)) and not before_field.get(field)
        else:
            landed = after.get(field) == value and before_field.get(field) != value
        if not landed:
            raise RuntimeError(
                f"op {op} did not produce its effect within {want} engine "
                f"steps (OP_STEPS says {want}): the step table, the probe "
                "sequences or the executor drifted")
def _next_age(state: TileState) -> int:
    """Age of the same tile one day later (crop age / animal cycle age)."""
    if state.kind == KIND_PLANT:
        return state.age + 1
    nxt = state.age + 1
    return nxt % int(K.ANIMALS[state.animal]["interval"]) if state.age >= 0 \
        else nxt


def _expected_next(state: TileState, ops: tuple[str, ...],
                   entity: str | None) -> TileState | None:
    """The state a tile-changing chain must land on, None for a growth day.

    Only the chains that change the tile are modelled here - they are exactly
    the ones where expanding the wrong parent state silently invents a node. A
    growth day's yield / care_bank bookkeeping is the engine's own answer and is
    not re-modelled (the graph is engine truth; R003).
    """
    if "PLANT" in ops:
        spec = CROP_RULES[entity]
        base = 0 if spec.get("ongoing") else 1
        # A plant made TODAY may also be fertilised today (PLANT then FERTILIZE), and
        # the dose is what the engine counts: `_successor` must say so or the successor
        # check rejects a chain the engine accepted.
        planted = ops.index("PLANT")
        return TileState(kind=KIND_PLANT, crop=entity,
                         age=1 - crop_age_origin(entity),
                         # Only a dose given AFTER this plant counts: fertilise the old
                         # crop, harvest it and replant, and the new plant is clean.
                         fert_left=2 if "FERTILIZE" in ops[planted:] else 0,
                         yield_units=base)
    if "PLACE" in ops:
        spec = K.ANIMALS[entity]
        # Placement day, then the nightly refresh: FEED keeps the animal fed,
        # CARE together with FEED banks one care day (engine
        # _daily_refresh_animals).
        unfed = 0 if "FEED" in ops else 1
        bank = 1 if ("CARE" in ops and "FEED" in ops) else 0
        return TileState(KIND_ANIMAL, None, entity, _STRUCTURE_OF[entity],
                         1 - spec["first_yield_day"], 0, unfed, 0,
                         min(bank, int(spec["max_held"])), 0)
    for op in ops:
        if op in ("BUILD_COOP", "BUILD_PASTURE"):
            structure = "COOP" if op == "BUILD_COOP" else "PASTURE"
            return TileState(EMPTY_KIND_OF_STRUCTURE[structure], None, None,
                             structure, 0, 0, 0, 0, 0, 0)
    if "DIG" in ops:
        return TileState(KIND_NONE, None, None, None, 0, 0, 0, 0, 0, 0)
    return None


def _assert_sim_at(sim: FastSim, state: TileState, where: str) -> None:
    """Decision 8, before an edge: the sim that reached a node must hold it."""
    got = _state_of(sim)
    if got.pack() != state.pack():
        raise StateMismatch(f"{where}: the sim is at {got.describe()}, "
                            f"the node is {state.describe()}")


def _growth_day_violation(state: TileState, child: TileState,
                          ops: tuple[str, ...]) -> str | None:
    """Why `child` is not the tile one growth day later, or None.

    A chain that changes no kind must carry the tile's identity forward, advance
    the age by exactly one day and apply the engine-independent day bookkeeping
    of its ops.
    """
    if child.kind != state.kind:
        # The engine's destroy paths are legal here and are its own answer: a
        # dry streak or the lifespan end turns a plant into WEED, harvesting a
        # one-shot crop clears the tile, a second unfed night frees the animal
        # (the structure stays). Anything else is a broken chain.
        if child.kind in (KIND_WEED, KIND_NONE, *EMPTY_KINDS):
            return None
        return (f"edge {chain_name(ops)} on {state.describe()}: the engine gave "
                f"{child.describe()}")
    if (child.crop, child.animal, child.structure) != \
            (state.crop, state.animal, state.structure):
        return (f"edge {chain_name(ops)} on {state.describe()}: the tile changed "
                f"identity to {child.describe()}")
    if state.kind in (KIND_PLANT, KIND_ANIMAL) and child.age != _next_age(state):
        return (f"edge {chain_name(ops)} on {state.describe()}: one day must "
                f"make the age {_next_age(state)}, the engine gave "
                f"{child.describe()}")
    if state.kind == KIND_PLANT:
        planted = ops.index("PLANT") if "PLANT" in ops else -1
        if planted >= 0:        # the tile holds a NEW plant, so the old dose is gone
            want_fert = 2 if "FERTILIZE" in ops[planted:] else 0
        else:
            want_fert = 2 if "FERTILIZE" in ops else max(0, state.fert_left - 1)
        want = (0 if "WATER" in ops else state.consec + 1, want_fert)
        if (child.consec, child.fert_left) != want:
            return (f"edge {chain_name(ops)} on {state.describe()}: expected "
                    f"consec={want[0]} fert_left={want[1]}, the engine gave "
                    f"{child.describe()}")
    elif state.kind == KIND_ANIMAL:
        want_unfed = 0 if "FEED" in ops else state.unfed + 1
        if child.unfed != want_unfed:
            return (f"edge {chain_name(ops)} on {state.describe()}: expected "
                    f"unfed={want_unfed}, the engine gave {child.describe()}")
        # The care bank only grows on a fed+cared day (engine
        # _daily_refresh_animals); the animal's production day spends it.
        cap = state.care_bank + (1 if ("CARE" in ops and "FEED" in ops) else 0)
        if child.care_bank > cap:
            return (f"edge {chain_name(ops)} on {state.describe()}: the care "
                    f"bank cannot pass {cap}, the engine gave {child.describe()}")
    return None


def _successor_violation(state: TileState, child: TileState,
                         ops: tuple[str, ...],
                         entity: str | None) -> str | None:
    """Why `child` cannot be the successor of `state` under `ops`, else None.

    The tile-changing chains are compared with `_expected_next` (full TileState
    equality); the rest with `_growth_day_violation`. Decision 8 raises on a
    violation.
    """
    expected = _expected_next(state, ops, entity)
    if expected is None:
        return _growth_day_violation(state, child, ops)
    if child.pack() == expected.pack():
        return None
    return (f"edge {chain_name(ops)} on {state.describe()}: the engine gave "
            f"{child.describe()}, the chain promises {expected.describe()}")


def _assert_successor(state: TileState, child: TileState, ops: tuple[str, ...],
                      entity: str | None) -> None:
    """Decision 8, after an edge: the decoded next state must be the promised one."""
    violation = _successor_violation(state, child, ops, entity)
    if violation is None:
        return
    if child.pack() == state.pack():
        # The engine reflected the day-start tile: the chain did nothing. A
        # species on the other structure, a build on an occupied tile and a
        # place with no structure are refused silently (kaggriculture.py:493-503,
        # probed 2026-09-14). That is not a phantom promise: the edge interns as
        # a self-loop, which the no-op sweep drops when its produce is empty.
        return
    expected = _expected_next(state, ops, entity)
    if expected is not None and child.kind != expected.kind:
        # The constructive op did not land (no stock, no money, occupied tile,
        # wrong structure ...): the engine refused it silently.
        raise ChainNotRealised(violation)
    raise StateMismatch(violation)


# ----------------------------------------------------------- chain execution

def _exec_chain(sim: FastSim, state: TileState, ops: tuple[str, ...],
                entity: str | None) -> ChainOutcome:
    """Execute one daily chain on `sim` (mutated in place) and price it.

    Every chain supplies its own prerequisites (buy + carry) as the day
    layer's stand-in, and `cost_vector` counts them. One chain is exactly one
    day: the day is filled and a chain that would bleed into day+1 raises
    ChainSpansDays. `entity` is None only for a chain that names no constructive
    op on a state that owns no entity (a bare / weed tile) - such a chain builds
    nothing, so it never needs one.
    """
    if UnitAction.PASS.value in ops:
        raise ValueError(f"PASS is not a chain op, got {ops}")
    day0 = int(sim.observations()[0]["day"])
    harvest = 0
    fert_collect = 0
    for op in ops:
        if op in frozenset():
            continue        # the market buys inside the op that needs it
        if op == UnitAction.PASS.value:
            # Nothing on this tile: the worker idles, the day still passes.
            sim.step([_act(["PASS"]), _act(["PASS"])])
        elif op in ("BUILD_COOP", "BUILD_PASTURE"):
            # BUILD = one worker action: a NONE tile becomes a structure. The
            # animal itself is bought later (by PLACE, day stand-in).
            # The OP names the structure, not the entity: a cow's chains may build a
            # COOP (that is what DIG+BUILD_COOP on a coop is), and taking the structure
            # from the entity silently built the other one.
            sim.step([_act([op]), _act(["PASS"])])
        elif op == "PLACE":
            sim.step([_act(["PASS"], [["BUY_ANIMAL", entity, 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PICKUP", entity, 1]), _act(["PASS"])])
            sim.step([_act(["PLACE", entity]), _act(["PASS"])])
        elif op == "PLANT":
            sim.step([_act(["PASS"], [["BUY_SEED", entity, 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PLANT", entity]), _act(["PASS"])])
        elif op == "FERTILIZE":
            sim.step([_act(["PASS"], [["BUY_PRODUCT", "FERTILIZER", 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PICKUP", "FERTILIZER", 1]), _act(["PASS"])])
            sim.step([_act(["FERTILIZE"]), _act(["PASS"])])
        elif op == "HARVEST":
            tile, _ = _tile_and_day(sim)
            harvest = int(tile.get("yield_units", 0)) \
                if isinstance(tile, dict) else 0
            sim.step([_act(["HARVEST"]), _act(["PASS"])])
        elif op == "FEED":
            sim.step([_act(["PASS"], [["BUY_PRODUCT", "WHEAT", 1]]),
                      _act(["PASS"])])
            sim.step([_act(["PICKUP", "WHEAT", 1]), _act(["PASS"])])
            sim.step([_act(["FEED"]), _act(["PASS"])])
        elif op == "COLLECT_FERTILIZER":
            tile, _ = _tile_and_day(sim)
            fert_collect = int(tile.get("fertilizer_available", 0)) \
                if isinstance(tile, dict) and "fertilizer_available" in tile \
                else 0
            sim.step([_act(["COLLECT_FERTILIZER"]), _act(["PASS"])])
        else:               # WATER, CARE, DIG
            sim.step([_act([op]), _act(["PASS"])])

    # ---- one chain = one day: fill the day, then verify the day boundary ----
    steps = 0
    while int(sim.observations()[0]["day"]) == day0 and not sim.done:
        sim.step([_act(["PASS"]), _act(["PASS"])])
        steps += 1
        if steps > TURNS_PER_DAY:
            raise ChainSpansDays(ops, entity)
    day = int(sim.observations()[0]["day"])
    if sim.done and day == day0:
        raise ChainSpansDays(f"{ops}: the sim ran out of episode on day {day0}")
    if day > day0 + 1:
        raise ChainSpansDays(ops, entity)

    child = _state_of(sim)
    _assert_successor(state, child, ops, entity)
    # HARVEST collects the product of the tile STANDING there, which in a
    # rotation chain (harvest the crop, then DIG/BUILD/PLACE an animal the same
    # day) is not the chain's entity: the produce side follows the parent tile.
    harvested = state.crop or state.animal or entity
    return ChainOutcome(child, harvest, fert_collect,
                        cost_vector(entity, ops),
                        produce_vector(harvested, harvest, fert_collect))


# --------------------------------------------------------------- search plan

def _own_entity(state: TileState, restrict: str | None) -> str | None:
    """The entity a state itself belongs to (decision 10).

    A chain that names no constructive op is attributed to the state's own
    entity (0 = none for a bare / weed tile). A restricted build attributes it
    to its own entity instead, which is what makes one entity's graph readable
    on its own.
    """
    if restrict is not None:
        return restrict
    if state.kind == KIND_PLANT:
        return state.crop
    if state.kind == KIND_ANIMAL:
        return state.animal
    if state.kind in EMPTY_KINDS:
        for name in ENTITY_NAMES:
            if _STRUCTURE_OF.get(name) == state.structure:
                return name
        return None
    return None                 # NONE / WEED own no entity


def _candidates(state: TileState, restrict: str | None) -> tuple[str, ...]:
    """Every entity the builder must try on a state (decision 10)."""
    if restrict is not None:
        return (restrict,)
    if state.kind in (KIND_NONE, KIND_WEED):
        return ENTITY_NAMES      # a bare tile may start any of the eight
    if state.kind == KIND_PLANT:
        # Its own crop first (the maintenance chains), then every entity again
        # for the DIG follow-up: dig the crop, then build / plant elsewhere.
        crop = state.crop
        if crop is None:
            raise ValueError(f"PLANT state without a crop: {state.describe()}")
        return (crop,) + tuple(n for n in ENTITY_NAMES if n != crop)
    if state.kind == KIND_ANIMAL:
        animal = state.animal
        if animal is None:
            raise ValueError(f"ANIMAL state without an animal: {state.describe()}")
        return (animal,)
    if state.kind in EMPTY_KINDS:
        # Own animals first (place / rebuild), then every other entity for the
        # DIG follow-up: dig this structure and build the OTHER one on the same
        # day. Engine probe 2026-09-14: BUILD_COOP, DIG, BUILD_PASTURE and PLACE
        # all run inside one day in either order (kaggriculture.py:484-503); a
        # chain that mismatches the structure is refused silently, so it lands
        # on the state it started from and `_is_noop_edge` drops it. Item 6 of
        # the v15 review: with own-structure candidates only, an empty COOP had
        # no edge to an empty PASTURE.
        own = tuple(n for n in ENTITY_NAMES
                    if _STRUCTURE_OF.get(n) == state.structure)
        return own + tuple(n for n in ENTITY_NAMES if n not in own)
    raise ValueError(f"unsupported state kind {state.kind!r}")


def _plan(state: TileState, restrict: str | None
          ) -> list[tuple[str | None, tuple[str, ...], int]]:
    """(entity to execute with, ops, entity_code) for a state's edges.

    The entity of a chain with a constructive op is the candidate entity itself;
    a chain without one is run once, with the state's own entity (decision 10).
    Chains are deduped by (entity_code, ops), so trying the extra candidates a
    PLANT node allows cannot duplicate an edge.
    """
    own = _own_entity(state, restrict)
    own_code = entity_code_of(own)
    age = state.age if state.kind in (KIND_PLANT, KIND_ANIMAL) else None
    plan: list[tuple[str | None, tuple[str, ...], int]] = []
    seen: set[tuple[int, tuple[str, ...]]] = set()
    for ent in _candidates(state, restrict):
        for ops in chains_for(state.kind, age=age,
                              yield_units=state.yield_units, entity=own):
            if not domain_ok(ops, ent, state.structure):
                continue         # domain filter (decision 9)
            if any(op in CONSTRUCTIVE_OPS for op in ops):
                run_entity, code = ent, ENTITY_CODE[ent]
            else:
                run_entity, code = own, own_code
            key = (code, ops)
            if key in seen:
                continue
            seen.add(key)
            plan.append((run_entity, ops, code))
    return plan


# ------------------------------------------------------------------- pruning

def _is_noop_edge(state_id: int, edge: Edge) -> bool:
    """A self-loop that holds no product: it lands on the node it starts from.

    Owner's rule (2026-09-14): an edge with an empty produce vector STAYS when it
    changes the tile - the state change is its product. A self-loop changes
    nothing and produces nothing, and PASS reaches the same node for free, so
    it is a no-op whether or not it spends an hour (the old rule caught only the
    zero-cost ones and let a 2-hour DIG+BUILD self-loop through).
    """
    return edge.to_id == state_id and not any(edge.produce)


def _dominates(better: Edge, worse: Edge) -> bool:
    """Dominance rule, between the two edges of one state's same target.

    `better` dominates `worse` only when it needs no more of ANY resource and
    produces no less of ANY resource: both vectors are compared component by
    component and are never netted, so a difference in a single component keeps
    both edges (1 wheat is not 1 melon, a carrot seed is not a wheat seed, one
    collected fertilizer is not nothing). One strict component is required;
    equal twins survive.
    """
    if better.to_id != worse.to_id:
        return False
    if any(better.cost[r] > worse.cost[r] for r in range(N_RESOURCE)):
        return False
    if any(better.produce[r] < worse.produce[r] for r in range(N_RESOURCE)):
        return False
    return better.cost != worse.cost or better.produce != worse.produce


def _prune(state_id: int, edges: list[Edge]) -> tuple[list[Edge], int]:
    """No-op sweep + dominance pruning (decision 9 keeps these rules)."""
    kept = [e for e in edges if not _is_noop_edge(state_id, e)]
    final = [e for e in kept
             if not any(_dominates(other, e) for other in kept
                        if other is not e)]
    # Both counts, so the report shows the Pareto sweep really ran: no-op drops and
    # dominated drops are different laws and only the first one was being reported.
    return final, len(edges) - len(kept), len(kept) - len(final)


# --------------------------------------------------------------------- build

def build_graph(entity: str | None = None, progress: bool = False) -> TileGraph:
    """Build the tile graph (decisions 1 and 12).

    `entity=None` builds the merged tile graph over all eight entities;
    `entity="CARROT"` runs the same search restricted to that entity's chains.

    Breadth-first over day-start states: every node is expanded exactly once,
    with the sim that reached it, and every edge is one real engine day on a
    clone of that sim. Assertions are on, so a chain that cannot be realised in
    one day - or that lands on a state other than the one it promises - fails
    the build loudly instead of being skipped.
    """
    verify_engine_constants()
    verify_op_steps()
    spec = BuildSpec(entity=entity, progress=progress)
    key_to_id: dict[int, int] = {}
    state_list: list[TileState] = []

    def intern(state: TileState) -> int:
        key = state.pack()
        sid = key_to_id.get(key)
        if sid is None:
            sid = len(state_list)
            key_to_id[key] = sid
            state_list.append(state)
        return sid

    root = intern(TileState(KIND_NONE, None, None, None, 0, 0, 0, 0, 0, 0))
    # state_id -> the sim that reached it (decision 6): a node is expanded with
    # the sim that produced it, so its edges see the true tile.
    sims: dict[int, FastSim] = {root: _new_sim()}
    edges: dict[int, list[Edge]] = {}
    visited: set[int] = set()
    n_edges = 0
    frontier = deque([root])
    while frontier:
        # BFS, so a node's sim sits at the node's SHORTEST day: the search can
        # never walk past the episode horizon while a shorter path exists.
        sid = frontier.popleft()
        if sid in visited:
            continue
        visited.add(sid)
        state = state_list[sid]
        # The sim is consumed here: a node is expanded exactly once, so keeping
        # its sim alive afterwards only costs memory (~0.05 MB per node, owner's
        # item 11).
        sim = sims.pop(sid)
        _assert_sim_at(sim, state, f"state {sid}")
        for run_entity, ops, code in _plan(state, entity):
            branch = sim.clone()
            outcome = _exec_chain(branch, state, ops, run_entity)
            nid = intern(outcome.next_state)
            if nid not in visited:
                sims.setdefault(nid, branch)   # first sim to reach the node wins
                frontier.append(nid)
            edges.setdefault(sid, []).append(
                Edge(sid, nid, chain_id_of(ops), code,
                     tuple(outcome.cost), tuple(outcome.produce)))
            n_edges += 1
        if progress:
            print(f"  {spec.entity or 'TILE'}: expanded {len(visited)}, "
                  f"states {len(state_list)}, edges {n_edges}", flush=True)

    n_states = len(state_list)
    edge_next: list[int] = []
    edge_chain: list[int] = []
    edge_entity: list[int] = []
    cost_rows: list[list[int]] = []
    produce_rows: list[list[int]] = []
    step_rows: list[int] = []
    offsets = np.zeros(n_states + 1, dtype=np.int64)
    n_noop = 0
    n_dominated = 0
    for sid in range(n_states):
        offsets[sid] = len(edge_next)
        kept, dropped, dominated = _prune(sid, edges.get(sid, []))
        n_noop += dropped
        n_dominated += dominated
        for edge in kept:
            edge_next.append(edge.to_id)
            edge_chain.append(edge.chain_id)
            edge_entity.append(edge.entity_code)
            cost_rows.append(list(edge.cost))
            produce_rows.append(list(edge.produce))
            step_rows.append(chain_steps(chain_ops(edge.chain_id)))
    offsets[n_states] = len(edge_next)

    # Only the chains that survived pruning: renumbered, so the artifact's table is exactly
    # the useful set with no gaps (the owner's rule: the artifact carries the USEFUL actions,
    # not every legal one).
    used = sorted({int(c) for c in edge_chain})
    remap = {old: new for new, old in enumerate(used)}
    edge_chain = [remap[int(c)] for c in edge_chain]
    used_chains = tuple(chain_ops(i) for i in used)
    print(f"chains used {len(used_chains)} of {len(registry())}", flush=True)

    kinds: dict[str, int] = {}
    for state in state_list:
        kinds[state.kind] = kinds.get(state.kind, 0) + 1
    report = BuildReport(spec=spec, n_states=n_states, n_edges=len(edge_next),
                         n_expanded=len(visited), n_noop_edges=n_noop,
                         n_dominated_edges=n_dominated,
                         kinds=kinds)
    return TileGraph(
        spec=spec, report=report, n_states=n_states,
        chains=used_chains,
        registry_tag=fingerprint_chains(used_chains),
        state_keys=np.array([s.pack() for s in state_list], dtype=np.int64),
        key_index=key_to_id, edge_offsets=offsets,
        edge_next=np.array(edge_next, dtype=np.int32),
        edge_chain=np.array(edge_chain, dtype=np.int16),
        edge_entity=np.array(edge_entity, dtype=np.int8),
        edge_cost=np.array(cost_rows, dtype=np.int32).reshape(-1, N_RESOURCE),
        edge_produce=np.array(produce_rows, dtype=np.int32).reshape(
            -1, N_RESOURCE),
        edge_steps=np.array(step_rows, dtype=np.int8),
        engine_tag=ENGINE_TAG)


def main() -> int:
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    g = build_graph()
    g.save(GRAPH_PATH)
    print("MERGED", g.report.describe(), flush=True)
    print("MERGED no-op dropped", g.report.n_noop_edges,
          "| dominated dropped", g.report.n_dominated_edges, flush=True)
    print("MERGED kinds", g.report.kinds, flush=True)
    print("MERGED bytes", os.path.getsize(GRAPH_PATH),
          "build_s", round(time.time() - t0, 1), "peak_rss_mb",
          round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
          flush=True)

    by_entity: dict[str, int] = {}
    for code in g.edge_entity:
        name = entity_of_code(int(code)) or "TILE"
        by_entity[name] = by_entity.get(name, 0) + 1
    info = write_info(NAME, kind="tile_graph", file=GRAPH_PATH.name,
                      contract=g.engine_tag,
                      engine=engine_fingerprint(),
                      registry=g._registry_tag(),
                      stats={"states": g.n_states, "edges": g.n_edges,
                             "kinds": g.report.kinds,
                             "edges_per_entity": by_entity},
                      source="offline_lab.build.graph:build_graph")
    # The chains are the graph's own action index, so they are written BY this run, under
    # the same contract: the agent loads the graph and the chains together and the two can
    # never drift apart. Nothing builds a chain artifact on its own.
    from offline_lab.build import chains as C
    table = C.write_table(g.engine_tag, g.chains)
    print("chains:", table.name, "->", C.TABLE_PATH.name)
    print("info:", info.name)
    back = TileGraph.load(GRAPH_PATH)
    print("RELOAD", back.n_states, back.n_edges, back.entity, back.engine_tag)
    try:
        shown = GRAPH_PATH.relative_to(Path.cwd()).as_posix()
    except ValueError:      # run from another cwd: print the absolute path
        shown = GRAPH_PATH.as_posix()
    print("model:", shown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
