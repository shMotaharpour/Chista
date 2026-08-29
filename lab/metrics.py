"""Metric extraction from kaggriculture episodes."""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict


PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]


@dataclass
class GameResult:
    agent_a: str
    agent_b: str
    seed: int | None
    episode_steps: int
    rewards: list[float]
    statuses: list[str]
    money_path_a: list[float] = field(default_factory=list)
    money_path_b: list[float] = field(default_factory=list)
    price_history: list[dict] = field(default_factory=list)  # per-day snapshot
    residue_a: dict = field(default_factory=dict)   # {item: count} unsold at end
    residue_b: dict = field(default_factory=dict)
    residue_value_a: float = 0.0                    # residue × final prices
    residue_value_b: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)


def _daily_money_path(env, player: int, turns_per_day: int = 24) -> list[float]:
    """Sample the player's bank once per in-game day."""
    path = []
    for step in range(0, len(env.steps), turns_per_day):
        obs = env.steps[step][player].observation
        farms = obs.get("farms") if hasattr(obs, "get") else getattr(obs, "farms", None)
        if farms is None:
            farms = obs.farms
        path.append(float(farms[player]["money"]))
    return path


def _price_snapshot(obs) -> dict:
    market = obs.market if hasattr(obs, "market") else obs.get("market", {})
    return {k: int(v) for k, v in dict(market["prices"]).items()}


def _residue(env, player: int) -> tuple[dict, float]:
    """Unsold shed inventory at end of episode + its value at final prices."""
    final = env.steps[-1][player].observation
    shed = dict(final.private["shed"]) if hasattr(final, "private") else dict(final.get("private")["shed"])
    residue = {k: int(v) for k, v in shed.items() if isinstance(v, (int, float)) and v > 0 and k in PRODUCTS}
    prices = _price_snapshot(final)
    value = sum(residue.get(p, 0) * prices.get(p, 0) for p in PRODUCTS)
    return residue, float(value)


def extract_result(env, agent_a: str, agent_b: str, seed: int | None, episode_steps: int) -> GameResult:
    final = env.steps[-1]
    rewards = [float(s.reward) if s.reward is not None else 0.0 for s in final]
    statuses = [s.status for s in final]
    res_a, val_a = _residue(env, 0)
    res_b, val_b = _residue(env, 1)
    # Daily price snapshots from player 0's observations.
    price_history = []
    for step in range(0, len(env.steps), 24):
        obs = env.steps[step][0].observation
        price_history.append({"step": step, "prices": _price_snapshot(obs)})
    return GameResult(
        agent_a=agent_a,
        agent_b=agent_b,
        seed=seed,
        episode_steps=episode_steps,
        rewards=rewards,
        statuses=statuses,
        money_path_a=_daily_money_path(env, 0),
        money_path_b=_daily_money_path(env, 1),
        price_history=price_history,
        residue_a=res_a,
        residue_b=res_b,
        residue_value_a=val_a,
        residue_value_b=val_b,
        meta={"turns_played": len(env.steps)},
    )
