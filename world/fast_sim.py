"""Fast-path simulator: drives the shipped interpreter directly.

Per R002/R003 this is NOT a reimplementation of the game. It calls
`kaggle_environments.envs.kaggriculture.kaggriculture.interpreter()` on real
structify-cloned state, skipping the harness bookkeeping around it.

Measured (720-step season, PASS policies, 8-core dev box, kaggle-environments
1.32.7, median of 3): 0.035 s/episode here vs 1.785 s for `env.run()` = 50.6x —
32.3x against the harness with its agent processes removed. The harness
still needs 1.09 s/episode with the agent processes removed, so the dominant
saving is its per-turn agent indirection (Pool + pickling a full observation
each turn) plus its own bookkeeping — for this environment per-turn JSON-schema
validation is a no-op (the action schema declares no typed properties) and
stdout redirection is minor. Reproduce with `python -m bench.bench_paths`.

Per R004 every method honors a validate switch: "dev" runs the checks
(action shape, state invariants, observation-copy guard), "fast" bypasses them
— the default for bulk DP/MDP/RL sweeps. A run in fast mode can always be
re-validated by re-running the same inputs in dev mode.

Observation contract: `observations()` returns detached deep copies in dev mode
and LIVE views in fast mode. Fast-mode callers must treat observations as
read-only — writing to one mutates the episode, which the harness never permits
(it gives each agent its own copy). Measured cost of the dev copies: +265 us per
turn, i.e. a 720-step season in 0.270 s instead of 0.034 s — still 6.4x faster
than the harness.
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
        # The harness's first agent-facing observation carries step=0 (the
        # framework stamps the step before the agent acts); the initialize
        # branch never writes it, so without this the first obs has step=None.
        overage = dict(self.configuration).get("remainingOverageTime", 60)
        for s in self._state:
            s.observation.step = 0
            s.reward = 0.0
            # Framework bookkeeping, mirrored so the state can be compared
            # field-for-field with the harness (tests/test_world_parity). The
            # harness decays this when an agent overruns its act timeout; this
            # simulator measures no wall clock, so it stays at the configured
            # value and observations() falls back to it.
            s.observation.remainingOverageTime = overage
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
    def seed(self) -> int | None:
        """Resolved episode seed (trainer-side only).

        resolve_episode_seed scrubs configuration['seed'] and stores the value
        in env.info, so this is the only way to recover the seed of an unseeded
        run — needed for reproducibility/replay. `_observation_dict` never
        exposes it, so the agent-facing surface stays identical to the harness
        (where the seed is deliberately hidden from agents).
        """
        return self._env.info.get("seed")

    @property
    def done(self) -> bool:
        return self._state[0].status == "DONE" or self.steps_taken >= self.configuration["episodeSteps"]

    def rewards(self) -> list[float]:
        """Per-agent rewards; non-final steps report 0.0, matching the harness.

        Core writes a reward into the state every turn (the agent-reported one,
        default 0) while the interpreter writes the final money only when the
        episode ends. Returning None here would break any trainer that
        accumulates rewards across steps (`sum(sim.rewards())`) even though the
        same code runs unmodified on the harness.
        """
        return [0.0 if s.reward is None else s.reward for s in self._state]

    def money(self) -> list[float]:
        return [f["money"] for f in self._state[0].observation.farms]

    def observations(self, copy_state: bool | None = None) -> list[dict[str, Any]]:
        """Per-agent observation dicts as the agents would receive them.

        copy_state=True returns detached deep copies (what the harness gives an
        agent: `__get_shared_state` deep-copies per agent). copy_state=False
        returns LIVE views of the episode state: mutating them mutates the
        episode, which the harness never permits. Default: True in dev mode
        (correctness, mutation guard active), False in fast mode (throughput) —
        fast-mode callers must treat the returned dicts as read-only.
        """
        if copy_state is None:
            copy_state = self._dev
        return [_observation_dict(s, copy_state=copy_state) for s in self._state]

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
            # The harness writes a reward into every state each turn (the
            # agent-reported one, default 0) while the interpreter writes the
            # final money only at episode end. Mirror that default so the state
            # is comparable field-for-field with the harness (tests/test_parity).
            if s.status != "DONE":
                s.reward = 0.0
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
            actions = [policies[0](obs[0]), policies[1](obs[1])]
            if self._dev:
                self._assert_obs_intact(obs)
            self.step(actions)
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

    def _assert_obs_intact(self, obs_dicts: list[dict[str, Any]]) -> None:
        """Dev guard: the observation handed to a policy is a copy of the live
        state; if the policy wrote into it, its bug would silently corrupt every
        fast-mode episode (where the same dicts ARE the state). Detect it here.
        """
        for i, given in enumerate(obs_dicts):
            live = _observation_dict(self._state[i], copy_state=False)
            for key, value in given.items():
                if repr(value) != repr(live[key]):
                    raise RuntimeError(
                        f"dev guard: agent {i} mutated observation field {key!r} "
                        f"(policies must treat observations as read-only)")

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


def _observation_dict(s: Any, copy_state: bool = False) -> dict[str, Any]:
    """Observation dict with plain (non-Struct) nested values.

    copy_state=True detaches the nested containers from the episode state, so a
    policy cannot write into the world through its observation.
    """
    obs = s.observation
    deep = copy.deepcopy if copy_state else (lambda v: v)
    return {
        "player": obs.player,
        "step": obs.step if hasattr(obs, "step") else None,
        "day": obs.day,
        "hour": obs.hour,
        "farms": deep(obs.farms),
        "private": deep(obs.private),
        "market": deep(obs.market),
        "town": deep(obs.town),
        "remainingOverageTime": getattr(obs, "remainingOverageTime", 60),
    }


def _pass_policy(_obs: dict[str, Any]) -> dict[str, Any]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _validate_action(index: int, action: Any) -> None:
    """Dev-mode action shape check.

    Deliberately STRICTER than the harness: the harness only requires the action
    to be an object (its schema declares no typed properties and the interpreter
    no-ops unknown ops), so a wrong inner type passes silently on the real
    environment while dev mode raises here. A dev failure therefore means
    "caller bug", not necessarily "would fail on Kaggle" — see R004.
    """
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

def _worker_episode(payload: tuple[dict[str, Any], int | None]) -> dict[str, Any]:
    """Run ONE episode in this worker and return a per-episode record."""
    configuration, seed = payload
    cfg = dict(configuration)
    if seed is not None:
        cfg["seed"] = seed
    sim = FastSim(cfg, validate="fast")
    rewards = sim.run([_pass_policy, _pass_policy], reset=True)
    return {
        "seed": sim.seed,          # the seed actually used (reproducibility)
        "rewards": rewards,
        "money": sim.money(),
        "steps": sim.steps_taken,
    }


def run_parallel(episodes: int,
                 configuration: dict[str, Any] | None = None,
                 processes: int | None = None,
                 master_seed: int | None = None) -> list[dict[str, Any]]:
    """Run `episodes` independent episodes across worker processes.

    Returns ONE RECORD PER EPISODE (length == episodes), each with the seed that
    was used, the rewards, the final money and the step count. `master_seed`
    makes the whole ensemble reproducible; `configuration["seed"]` (if set)
    overrides it and every episode is then the same episode by construction.

    For sweeps with policies use FastSim directly in your own Pool: the workers
    here run the neutral PASS policy; the point of this helper is parallel
    throughput (DP/MDP evaluation loops) with zero per-worker setup cost beyond
    the interpreter import.
    """
    import multiprocessing as mp
    import random

    if episodes <= 0:
        return []
    configuration = dict(configuration or {})
    fixed_seed = configuration.get("seed")
    rng = random.Random(master_seed)
    seeds = [fixed_seed if fixed_seed is not None else rng.randrange(2 ** 31)
             for _ in range(episodes)]
    payloads = [(dict(configuration), s) for s in seeds]
    procs = processes or min(episodes, mp.cpu_count())
    if procs <= 1:
        return [_worker_episode(p) for p in payloads]
    with mp.Pool(processes=procs) as pool:
        records = list(pool.imap_unordered(_worker_episode, payloads,
                                          chunksize=max(1, episodes // (procs * 4))))
    # imap_unordered yields in completion order; sort so the returned ensemble is
    # reproducible for a given master_seed (verified: two runs then agree).
    return sorted(records, key=lambda r: (r["seed"] is None, r["seed"]))
