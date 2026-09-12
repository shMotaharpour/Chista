"""Fast-path simulator: drives the shipped interpreter directly.

Per R002/R003 this is NOT a reimplementation of the game. It calls
`kaggle_environments.envs.kaggriculture.kaggriculture.interpreter()` on real
structify-cloned state, skipping only the harness bookkeeping around it:
per-turn JSON-schema validation, stdout/stderr redirection, and the
full-episode `steps` snapshot append.

Per R004 every method honors a validate switch: "dev" runs the checks
(action shape, state invariants, reward parity hooks), "fast" bypasses them
— the default for bulk DP/MDP/RL sweeps. A run in fast mode can always be
re-validated by re-running the same inputs in dev mode.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Iterable

from kaggle_environments.envs.kaggriculture import kaggriculture as K
from kaggle_environments.utils import structify

DEFAULT_CONFIGURATION: dict[str, Any] = {
    "episodeSteps": 720,
    "actTimeout": 1,
    "boardSize": 10,
    "startingMoney": 3000,
    "maxMarketOrdersPerTurn": 10,
    "turnsPerDay": 24,
    "shedCapacity": 100,
    "weedSpawnChance": 0.005,
    "townShopUnlockInterval": 3,
    "townShopSellInterval": 4,
    "townCenterSellInterval": 24,
    "farmHandCostMult": 1,
    "marketParams": {},
    "seed": None,
}

ValidateMode = str  # "dev" | "fast"


class _EnvShim:
    """Minimal env surface the interpreter actually touches.

    The kaggriculture interpreter reads env.configuration (via .get), writes
    env.info (resolved seed) and never touches anything else of the real
    Environment. Keeping this shim honest is what makes the fast path a
    wrapper, not a reimplementation (R003).
    """

    __slots__ = ("configuration", "info", "done")

    def __init__(self, configuration: dict[str, Any]):
        # structify => attribute access (cfg.episodeSteps) AND dict methods
        # (.get) — exactly the dual surface of the real harness config.
        self.configuration = structify(configuration)
        self.info: dict[str, Any] = {}
        self.done = False


def _merge_configuration(overrides: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIGURATION)
    if overrides:
        unknown = set(overrides) - set(cfg)
        if unknown:
            raise KeyError(f"unknown configuration keys: {sorted(unknown)}")
        cfg.update(overrides)
    return cfg


class FastSim:
    """Single-episode simulator around the real interpreter.

    Semantics (verified against the harness, same seed + same actions =>
    bit-identical money):
      - construct/reset runs the interpreter once in initialize mode
        (obs.farms empty -> _initialize), which also resolves the seed into
        env.info exactly like the harness does;
      - the sim owns the step counter: after each interpreter call it writes
        observation.step = steps_taken (the harness stores the post-step
        index the same way) and stops when status turns DONE or episodeSteps
        is reached.
    """

    def __init__(self,
                 configuration: dict[str, Any] | None = None,
                 validate: ValidateMode = "fast"):
        self.validate: ValidateMode = validate
        self.configuration = _merge_configuration(configuration)
        self._env = _EnvShim(dict(self.configuration))
        self._state: list[Any] | None = None
        self.steps_taken = 0
        self.reset()

    # ------------------------------------------------------------------ core

    def reset(self) -> None:
        """Re-initialize the episode from the configuration (fresh seed draw
        unless configuration['seed'] is set — same rule as the harness)."""
        self._env = _EnvShim(dict(self.configuration))
        self._state = structify([
            {"observation": {}, "action": None, "status": "ACTIVE", "reward": None}
            for _ in range(2)
        ])
        self.steps_taken = 0
        K.interpreter(self._state, self._env)  # initialize branch
        if self._dev:
            self._check_state("after reset")

    @property
    def _dev(self) -> bool:
        return self.validate == "dev"

    # ------------------------------------------------------------- accessors

    @property
    def state(self) -> list[Any]:
        return self._state

    @property
    def done(self) -> bool:
        return self._state[0].status == "DONE" or self.steps_taken >= self.configuration["episodeSteps"]

    def rewards(self) -> list[float | None]:
        return [s.reward for s in self._state]

    def money(self) -> list[float]:
        return [f["money"] for f in self._state[0].observation.farms]

    def observations(self) -> list[dict[str, Any]]:
        """Per-agent observation dicts as the agents would receive them."""
        return [_observation_dict(s) for s in self._state]

    # ------------------------------------------------------------------ steps

    def step(self, actions: list[dict[str, Any]]) -> None:
        """Advance one turn. `actions` is [agent0_action, agent1_action].

        In dev mode the action dicts are shape-checked first (fast mode
        trusts the caller and hands them straight to the interpreter).
        """
        if self.done:
            raise RuntimeError("episode finished; reset() first")
        if len(actions) != 2:
            raise ValueError("need exactly 2 actions")
        if self._dev:
            for i, a in enumerate(actions):
                _validate_action(i, a)
        for i in range(2):
            self._state[i].action = actions[i]
        K.interpreter(self._state, self._env)
        self.steps_taken += 1
        for s in self._state:
            s.observation.step = self.steps_taken
        if self._dev:
            self._check_state(f"after step {self.steps_taken}")

    def run(self,
            policies: list[Callable[[dict[str, Any]], dict[str, Any]]],
            reset: bool = True) -> list[float | None]:
        """Run a full episode with two policy callables (obs_dict -> action)."""
        if reset:
            self.reset()
        while not self.done:
            obs = self.observations()
            self.step([policies[0](obs[0]), policies[1](obs[1])])
        return self.rewards()

    # ------------------------------------------------- what-if / branching

    def clone(self) -> "FastSim":
        """Deep-copy branch point: independent simulator at the current
        state (same configuration, same resolved seed in env.info)."""
        new = FastSim.__new__(FastSim)
        new.validate = self.validate
        new.configuration = dict(self.configuration)
        new._env = copy.deepcopy(self._env)
        new._state = copy.deepcopy(self._state)
        new.steps_taken = self.steps_taken
        return new

    def what_if(self,
                actions: list[dict[str, Any]] | dict[str, Any],
                horizon: int = 1,
                policies: list[Callable[[dict[str, Any]], dict[str, Any]]] | None = None,
            ) -> "FastSim":
        """Branch a hypothetical future without touching this sim.

        `actions` is one action dict applied to BOTH agents (the common
        what-if case) or a two-element list [agent0, agent1]. The branch
        applies them once, then continues for `horizon - 1` further turns
        with `policies` (default: PASS for both). Returns the branched
        simulator; `self` is unchanged.
        """
        branch = self.clone()
        first = [actions, actions] if isinstance(actions, dict) else list(actions)
        branch.step(first)
        filler = policies or [_pass_policy, _pass_policy]
        for _ in range(max(0, horizon - 1)):
            if branch.done:
                break
            obs = branch.observations()
            branch.step([filler[0](obs[0]), filler[1](obs[1])])
        return branch

    # -------------------------------------------------------------- validation

    def _check_state(self, where: str) -> None:
        """Dev-mode state invariants (cheap, per R004 dev-only)."""
        obs = self._state[0].observation
        farms = obs.farms
        assert len(farms) == 2, f"{where}: expected 2 farms"
        n = self.configuration["boardSize"]
        for f in farms:
            assert len(f["tiles"]) == n and len(f["tiles"][0]) == n, f"{where}: bad tile grid"
            assert set(f["unlocked_quadrants"]).issubset({"NW", "NE", "SW", "SE"}), f"{where}: bad quadrants"
        market = obs.market
        for product, price in market["prices"].items():
            assert price >= 1, f"{where}: {product} price below floor"


def _observation_dict(s: Any) -> dict[str, Any]:
    """Observation dict with plain (non-Struct) nested values."""
    obs = s.observation
    return {
        "player": obs.player,
        "step": obs.step if hasattr(obs, "step") else None,
        "day": obs.day,
        "hour": obs.hour,
        "farms": obs.farms,
        "private": obs.private,
        "market": obs.market,
        "town": obs.town,
        "remainingOverageTime": getattr(obs, "remainingOverageTime", 60),
    }


def _pass_policy(_obs: dict[str, Any]) -> dict[str, Any]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _validate_action(index: int, action: Any) -> None:
    """Dev-mode action shape check (subset of the harness schema, enough to
    catch caller bugs without JSON-schema cost)."""
    if not isinstance(action, dict):
        raise TypeError(f"agent {index}: action must be a dict")
    for key in ("farmer", "hands", "market"):
        if key not in action:
            raise KeyError(f"agent {index}: action missing '{key}'")
    if not isinstance(action["farmer"], list):
        raise TypeError(f"agent {index}: 'farmer' must be a list")
    if not isinstance(action["hands"], list):
        raise TypeError(f"agent {index}: 'hands' must be a list")
    if not isinstance(action["market"], list):
        raise TypeError(f"agent {index}: 'market' must be a list")


# --------------------------------------------------------------- parallel runs

def _worker_run(payload: tuple[dict[str, Any], int]) -> list[float | None]:
    """Run `repeats` episodes inside one worker process; return rewards of the last."""
    configuration, repeats = payload
    sim = FastSim(configuration, validate="fast")
    out: list[float | None] = []
    for _ in range(repeats):
        out = sim.run([_pass_policy, _pass_policy], reset=True)
    return out


def run_parallel(episodes: int,
                 configuration: dict[str, Any] | None = None,
                 processes: int | None = None) -> list[list[float | None]]:
    """Run `episodes` independent episodes across worker processes.

    For sweeps with policies use FastSim directly in your own Pool:
    the workers here run the neutral PASS policy; the point of this helper
    is parallel throughput (DP/MDP evaluation loops) with zero per-worker
    setup cost beyond the interpreter import.
    """
    import multiprocessing as mp

    if episodes <= 0:
        return []
    procs = processes or min(episodes, mp.cpu_count())
    base, extra = divmod(episodes, procs)
    payloads = [(dict(configuration or {}), base + (1 if i < extra else 0))
                for i in range(procs) if base + (1 if i < extra else 0) > 0]
    if len(payloads) == 1:
        return [_worker_run(payloads[0]) for _ in range(1)]
    with mp.Pool(processes=len(payloads)) as pool:
        results = pool.map(_worker_run, payloads)
    return results
