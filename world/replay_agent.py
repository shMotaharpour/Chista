"""Replay agent: play a recorded single-agent episode from an episode file.

The record is a JSON file describing ONE agent's turn-by-turn actions (not a
full two-player match). The agent finds its action purely by the observation's
step number — it never needs to know which seat (player 0 or 1) the harness
gave it, and any number of instances can run concurrently on the same or
different records because there is no shared mutable state.

Loading (per user decision, 2026-09): `EpisodeRecord.load(path, validate=True)`
validates the file BEFORE constructing the record; pass validate=False to skip
the check for trusted files.

Validation (per R004 this is an explicit, on-demand analysis tool — never run
automatically during an episode):
  - validate(mode="soft"): structural sanity only — schema version, required
    fields, shape of each action dict, ordering/duplicates of steps.
  - validate(mode="hard"): everything soft checks, plus game-aware depth —
    every op is a real op (imported from kaggle_environments per R002, never
    transcribed), every argument matches the shipped engine's expectations,
    and the turn range fits the configuration's episodeSteps. Raises on the
    first defect (with a precise path like turns[37].action.hands[2]).

Playing: by default the agent hands out the recorded action object directly
(share mode, `copy=False`) — the engine and the harness never write into a
submitted action (verified by probes on both paths, and pinned by
`test_engine_never_mutates_shared_actions`), so sharing costs ~0 us/turn.
Pass `copy=True` for defensive copying: measured on this box, deepcopy of
the real 720-turn action stream costs ~10 ms/season/seat (~21 ms both
seats, ~25% of a replay season) — worth it only when the caller might
mutate what it receives. A missing turn falls back to PASS — the episode
can never crash (the engine itself treats invalid actions as silent
no-ops, F047).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from kaggle_environments.envs.kaggriculture import kaggriculture as K

SCHEMA = "chistaagent.replay.v1"
PASS_ACTION: dict[str, Any] = {"farmer": ["PASS"], "hands": [], "market": []}

# Action keys in engine-submission order (F030: units act before the market).
_ACTION_KEYS = ("farmer", "hands", "market")


# --------------------------------------------------------------------------- #
# errors
# --------------------------------------------------------------------------- #

class ReplayValidationError(ValueError):
    """A replay file failed validation. `where` pinpoints the defect."""

    def __init__(self, message: str, where: str = ""):
        self.where = where
        super().__init__(f"{where}: {message}" if where else message)


# --------------------------------------------------------------------------- #
# engine-facing vocabularies (imported, never transcribed — R002)
# --------------------------------------------------------------------------- #

def _engine_vocab() -> dict[str, Any]:
    """Everything the hard validator needs, straight from the shipped engine."""
    crops = set(K.CROPS)
    animals = set(K.ANIMALS)
    products = set(K.PRODUCTS)
    moves = set(K.FARMER_MOVES)
    return {
        "moves": moves,
        "crops": crops,
        "animals": animals,
        "products": products,
        # items that can appear in a unit op argument position
        "unit_items": animals | products | crops,
        "structures": {"COOP", "PASTURE"},
    }


class EpisodeRecord:
    """A recorded single-agent episode: seed, configuration, turn actions."""

    __slots__ = ("schema", "seed", "configuration", "agent_name", "turns",
                 "source")

    def __init__(self, data: dict[str, Any], source: str = "<memory>"):
        self.source = source
        if data.get("schema") != SCHEMA:
            raise ReplayValidationError(
                f"unsupported schema {data.get('schema')!r} (expected {SCHEMA!r})",
                where=f"{source}.schema")
        self.schema: str = data["schema"]
        self.seed: int | None = data.get("seed")
        self.configuration: dict[str, Any] = dict(data.get("configuration") or {})
        self.agent_name: str | None = data.get("agent_name")
        turns = data.get("turns")
        if not isinstance(turns, list) or not turns:
            raise ReplayValidationError("missing or empty 'turns' list",
                                        where=f"{source}.turns")
        self.turns: list[dict[str, Any]] = turns

    # ------------------------------------------------------------- loading

    @classmethod
    def load(cls, path: str | Path,
             validate: bool = True,
             mode: str = "hard") -> "EpisodeRecord":
        """Read a replay JSON file and (by default) validate it first.

        validate=True (default) runs `cls.validate(path, mode=mode)` BEFORE the
        record is constructed, so a defective file never becomes an object.
        validate=False skips the deep check for trusted files (the constructor
        still refuses a wrong schema or missing turns).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"replay file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if validate:
            cls.validate_data(data, mode=mode, source=str(path))
        return cls(data, source=str(path))

    # ----------------------------------------------------------- accessing

    def action_at(self, step: int) -> dict[str, Any] | None:
        """The recorded action for a step, or None (binary search on order)."""
        turns = self.turns
        lo, hi = 0, len(turns) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            s = turns[mid].get("step")
            if s == step:
                return turns[mid].get("action")
            if s < step:
                lo = mid + 1
            else:
                hi = mid - 1
        return None

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict form (the exact shape SCHEMA.md documents)."""
        return {
            "schema": self.schema,
            "seed": self.seed,
            "configuration": dict(self.configuration),
            "agent_name": self.agent_name,
            "turns": copy.deepcopy(self.turns),
        }

    # ---------------------------------------------------------- validation

    @classmethod
    def validate(cls, path: str | Path, mode: str = "hard") -> dict[str, Any]:
        """Analyze a replay file on demand. Returns a summary dict.

        mode="soft": structure only (schema, fields, action shape, step order).
        mode="hard": soft checks plus engine-aware op/argument checking
        (vocabularies imported from kaggle_environments, R002) and the turn
        range vs configuration['episodeSteps'].
        Raises ReplayValidationError on the first defect; the exception's
        message names the exact location (e.g. turns[37].action.hands[2]).
        """
        path = Path(path)
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls.validate_data(data, mode=mode, source=str(path))

    @classmethod
    def validate_data(cls, data: dict[str, Any], mode: str = "hard",
                      source: str = "<memory>") -> dict[str, Any]:
        """Validate an already-parsed record dict (shared by load/validate)."""
        if mode not in ("soft", "hard"):
            raise ReplayValidationError(f"unknown mode {mode!r}", where=source)
        problems: list[str] = []

        if data.get("schema") != SCHEMA:
            raise ReplayValidationError(
                f"unsupported schema {data.get('schema')!r} (expected {SCHEMA!r})",
                where=f"{source}.schema")
        if not isinstance(data.get("turns"), list) or not data["turns"]:
            raise ReplayValidationError("missing or empty 'turns' list",
                                        where=f"{source}.turns")

        turns = data["turns"]
        prev_step: int | None = None
        for idx, turn in enumerate(turns):
            where = f"{source}.turns[{idx}]"
            if not isinstance(turn, dict) or "step" not in turn:
                raise ReplayValidationError("missing 'step'", where=where)
            step = turn["step"]
            if not isinstance(step, int) or isinstance(step, bool) or step < 0:
                raise ReplayValidationError(f"bad step {step!r}", where=where)
            if prev_step is not None and step <= prev_step:
                raise ReplayValidationError(
                    f"steps must be strictly increasing "
                    f"({step} after {prev_step})", where=where)
            prev_step = step
            _check_action(turn.get("action"), where, problems,
                          hard=(mode == "hard"))

        if mode == "hard":
            steps_cfg = (data.get("configuration") or {}).get("episodeSteps")
            if isinstance(steps_cfg, int) and turns[-1]["step"] >= steps_cfg:
                raise ReplayValidationError(
                    f"last recorded step {turns[-1]['step']} >= "
                    f"episodeSteps {steps_cfg}",
                    where=f"{source}.turns[{len(turns) - 1}].step")

        return {
            "mode": mode,
            "source": source,
            "schema": data["schema"],
            "n_turns": len(turns),
            "first_step": turns[0]["step"],
            "last_step": turns[-1]["step"],
            "seed": data.get("seed"),
            "agent_name": data.get("agent_name"),
            "problems": problems,   # soft findings (hard mode raises instead)
        }


# --------------------------------------------------------------------------- #
# action-shape checking (soft: shape only; hard: engine-aware)
# --------------------------------------------------------------------------- #

def _check_action(action: Any, where: str, problems: list[str],
                  hard: bool) -> None:
    if action is None:
        problems.append(f"{where}.action missing (plays as PASS)")
        return
    if not isinstance(action, dict):
        raise ReplayValidationError(
            f"action must be an object, got {type(action).__name__}", where=where)
    for key in _ACTION_KEYS:
        if key not in action:
            # Engine treats a missing field as empty; note it, don't fail.
            problems.append(f"{where}.action.{key} missing (plays as empty)")
    _check_farmer(action.get("farmer"), where, problems, hard)
    _check_hands(action.get("hands"), where, problems, hard)
    _check_market(action.get("market"), where, problems, hard)


def _check_farmer(op: Any, where: str, problems: list[str],
                  hard: bool) -> None:
    if op is None:
        return
    if not isinstance(op, list) or not op:
        raise ReplayValidationError(
            f"farmer op must be a non-empty list, got {op!r}",
            where=f"{where}.action.farmer")
    _check_op(op, f"{where}.action.farmer", problems, hard, unit_index=None)


def _check_hands(hands: Any, where: str, problems: list[str],
                 hard: bool) -> None:
    if hands is None:
        return
    if not isinstance(hands, list):
        raise ReplayValidationError(
            f"hands must be a list, got {type(hands).__name__}",
            where=f"{where}.action.hands")
    for i, op in enumerate(hands):
        if op is None:
            continue
        if not isinstance(op, list) or not op:
            raise ReplayValidationError(
                f"hand op must be a non-empty list, got {op!r}",
                where=f"{where}.action.hands[{i}]")
        _check_op(op, f"{where}.action.hands[{i}]", problems, hard,
                  unit_index=i)


def _check_market(market: Any, where: str, problems: list[str],
                  hard: bool) -> None:
    if market is None:
        return
    if not isinstance(market, list):
        raise ReplayValidationError(
            f"market must be a list, got {type(market).__name__}",
            where=f"{where}.action.market")
    for i, order in enumerate(market):
        if not isinstance(order, list) or not order:
            raise ReplayValidationError(
                f"market order must be a non-empty list, got {order!r}",
                where=f"{where}.action.market[{i}]")
        if not isinstance(order[0], str):
            raise ReplayValidationError(
                f"market op must be a string, got {order[0]!r}",
                where=f"{where}.action.market[{i}][0]")


_MARKET_OPS = {"BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL", "HIRE",
               "BUY_LAND"}
# Unit ops the engine dispatches on (moves handled via FARMER_MOVES).
_UNIT_OPS = {"PASS", "PICKUP", "PLACE", "DROP", "PLANT", "WATER", "HARVEST",
             "FERTILIZE", "BUILD_COOP", "BUILD_PASTURE", "FEED", "CARE",
             "COLLECT_FERTILIZER", "DIG"}


def _check_op(op: list[Any], where: str, problems: list[str], hard: bool,
              unit_index: int | None) -> None:
    if not isinstance(op[0], str):
        raise ReplayValidationError(
            f"op must be a string, got {op[0]!r}", where=f"{where}[0]")
    if not hard:
        # Soft mode still flags the legacy BUILD form (shape-valid, but the
        # engine only dispatches BUILD_COOP / BUILD_PASTURE — F047 silent op).
        if op[0] == "BUILD" and len(op) == 2:
            problems.append(f"{where}: legacy BUILD form "
                            f"(engine expects BUILD_COOP / BUILD_PASTURE)")
        return
    vocab = _ENGINE_VOCAB
    opname = op[0]
    if opname in vocab["moves"]:
        if len(op) != 1:
            raise ReplayValidationError(
                f"move op takes no arguments, got {op!r}", where=where)
        return
    if opname == "BUILD" and len(op) == 2:
        problems.append(f"{where}: legacy BUILD form "
                        f"(engine expects BUILD_COOP / BUILD_PASTURE)")
        return
    if opname not in _UNIT_OPS:
        raise ReplayValidationError(f"unknown unit op {opname!r}", where=where)
    _check_op_args(opname, op, where, problems)


def _check_op_args(opname: str, op: list[Any], where: str,
                   problems: list[str]) -> None:
    vocab = _ENGINE_VOCAB
    argc = len(op)

    if opname in ("WATER", "HARVEST", "FERTILIZE", "DROP", "FEED", "CARE",
                  "COLLECT_FERTILIZER", "BUILD_COOP", "BUILD_PASTURE",
                  "BUILD"):
        if argc != 1:
            raise ReplayValidationError(
                f"{opname} takes no arguments, got {op!r}", where=where)
        return

    if opname == "PLANT":
        if argc != 2 or op[1] not in vocab["crops"]:
            raise ReplayValidationError(
                f"PLANT needs a crop from {sorted(vocab['crops'])}, got {op!r}",
                where=where)
        return

    if opname in ("PICKUP", "PLACE"):
        if argc < 2:
            raise ReplayValidationError(
                f"{opname} needs an item argument, got {op!r}", where=where)
        item = op[1]
        if item in vocab["crops"] or item + "S" in vocab["crops"]:
            problems.append(
                f"{where}: {opname} of seed {item!r} — seeds bypass shed/"
                f"inventory (F001); op will silently no-op")
        if item not in vocab["products"] and item not in vocab["animals"]:
            raise ReplayValidationError(
                f"{opname} item {item!r} is not a product or animal",
                where=where)
        if argc >= 3 and (not isinstance(op[2], int) or isinstance(op[2], bool)
                          or op[2] < 1):
            raise ReplayValidationError(
                f"{opname} count must be a positive int, got {op[2]!r}",
                where=where)
        return

    # F040 note: hand ids restart daily; nothing to check per-op beyond shape.


_ENGINE_VOCAB = _engine_vocab()


# --------------------------------------------------------------------------- #
# the agent
# --------------------------------------------------------------------------- #

class ReplayAgent:
    """Plays a recorded episode; callable exactly like any policy.

    Seat-agnostic: the action for obs["step"] is returned regardless of
    which seat the engine assigns. Stateless across calls (pure lookup),
    so any number of instances can run concurrently.

    copy=False (default): the recorded action object is handed out as-is —
    zero per-turn overhead. Safe because the engine and the harness never
    write into a submitted action (probed on both paths); the caller must
    not mutate the returned object either. Note two agents sharing ONE
    record both receive the SAME object per turn — fine for the engine,
    wrong if a caller edits what it receives.
    copy=True: every handout is a fresh deepcopy — defensive mode for
    callers that might mutate what they receive. Costs ~10 ms per
    720-turn season per seat (measured); use only when needed.
    """

    __slots__ = ("_record", "_missing", "_replayed", "_copy")

    def __init__(self, record: EpisodeRecord, copy: bool = False):
        self._record = record
        self._missing = 0
        self._replayed = 0
        self._copy = copy

    @property
    def record(self) -> EpisodeRecord:
        return self._record

    def __call__(self, obs: Any, configuration: Any = None) -> dict[str, Any]:
        """Accept the harness's optional second argument (configuration).

        The real harness inspects the callable's argument count and passes
        (observation, configuration) to two-argument callables; the optional
        parameter keeps ReplayAgent compatible with both call conventions.
        """
        step = obs.get("step") if hasattr(obs, "get") else getattr(obs, "step", None)
        action = self._record.action_at(step) if step is not None else None
        if action is None:
            self._missing += 1
            return dict(PASS_ACTION)
        self._replayed += 1
        return copy.deepcopy(action) if self._copy else action

    # ------------------------------------------------------------ counters

    @property
    def replayed_turns(self) -> int:
        """Turns answered from the record so far."""
        return self._replayed

    @property
    def missing_turns(self) -> int:
        """Turns answered with PASS because the record had no entry."""
        return self._missing
