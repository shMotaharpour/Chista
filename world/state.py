"""State — read-view over the live kaggle-environments observation.

Not a copy: lazily reads the engine observation dict and exposes typed,
planner-friendly queries. Serializable-friendly: every accessor returns
plain data (dicts/lists/scalars), so views can later be flattened for
tensors/gym without changing this class.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from world import mechanics as M
from world.rollback import snapshot  # noqa: F401  (re-export convenience)


@dataclass
class CoopView:
    x: int
    y: int
    animal: str
    placed_day: int
    fed_today: bool
    cared_today: bool
    consecutive_unfed: int
    yield_units: int
    pending_care_bonus: int
    fertilizer_available: bool


@dataclass
class PlantView:
    x: int
    y: int
    crop: str
    planted_day: int
    age: int
    watered_today: bool
    consecutive_unwatered: int
    yield_units: int
    fertilized_until_day: int


@dataclass
class State:
    """Read-view over one player's observation. Build with State.from_obs(obs, player)."""
    obs: dict
    player: int = 0

    # ---- raw access ----
    @classmethod
    def from_obs(cls, obs: dict, player: int | None = None) -> "State":
        return cls(obs=obs, player=obs.get("player", 0 if player is None else player))

    @property
    def me(self) -> dict:
        return self.obs["farms"][self.player]

    @property
    def day(self) -> int:
        return self.obs["day"]

    @property
    def hour(self) -> int:
        return self.obs.get("hour", 0)

    @property
    def turn(self) -> int:
        return self.obs.get("step", self.day * M.TURNS_PER_DAY + self.hour)

    @property
    def money(self) -> float:
        return self.me["money"]

    @property
    def farmer_xy(self) -> tuple[int, int]:
        return tuple(self.me["farmer"])

    @property
    def tiles(self) -> list[list]:
        return self.me["tiles"]

    @property
    def unlocked(self) -> list[str]:
        return list(self.me["unlocked_quadrants"])

    def unlocked_limit(self) -> int:
        """Max coordinate (exclusive) of the unlocked square. NW quadrant only
        until a land purchase: 5x5 tiles => limit 5 (x,y in 0..4)."""
        return 5 if len(self.unlocked) <= 1 else 5 * len(self.unlocked)

    @property
    def shed(self) -> dict:
        return self.obs["private"]["shed"]

    @property
    def seeds(self) -> dict:
        return self.obs["private"]["seeds"]

    @property
    def market_inventory(self) -> dict:
        return self.obs["market"]["inventory"]

    @property
    def market_prices(self) -> dict:
        return self.obs["market"]["prices"]

    @property
    def hires_today(self) -> int:
        return self.me.get("hires_today", 0)

    @property
    def hands(self) -> list:
        return list(self.me.get("hands", []))

    def inventories(self) -> list[dict]:
        return list(self.obs["private"]["inventories"])

    # ---- derived views (plain data, planner-friendly) ----

    def plants(self) -> list[PlantView]:
        out = []
        for y, row in enumerate(self.tiles):
            for x, t in enumerate(row):
                if M.is_plant(t):
                    out.append(PlantView(
                        x=x, y=y, crop=t["crop"], planted_day=t["planted_day"],
                        age=M.plant_age(t, self.day),
                        watered_today=t.get("watered_today", False),
                        consecutive_unwatered=t.get("consecutive_unwatered", 0),
                        yield_units=t.get("yield_units", 0),
                        fertilized_until_day=t.get("fertilized_until_day", -1)))
        return out

    def animal_tiles(self) -> list[CoopView]:
        out = []
        for y, row in enumerate(self.tiles):
            for x, t in enumerate(row):
                if M.is_animal_tile(t):
                    out.append(CoopView(
                        x=x, y=y, animal=t["animal"], placed_day=t.get("placed_day", self.day),
                        fed_today=bool(t.get("fed_today")),
                        cared_today=bool(t.get("cared_today")),
                        consecutive_unfed=t.get("consecutive_unfed", 0),
                        yield_units=t.get("yield_units", 0),
                        pending_care_bonus=t.get("pending_care_bonus", 0),
                        fertilizer_available=bool(t.get("fertilizer_available"))))
        return out

    def empty_unlocked_tiles(self) -> list[tuple[int, int]]:
        lim = self.unlocked_limit()
        return [(x, y) for y, row in enumerate(self.tiles)
                for x, t in enumerate(row)
                if t is None and x < lim and y < lim]

    def plants_needing_water(self) -> list[PlantView]:
        return [p for p in self.plants() if p.consecutive_unwatered < 2
                and not p.watered_today]

    def animals_needing_feed(self) -> list[CoopView]:
        return [a for a in self.animal_tiles() if not a.fed_today]

    def animals_needing_care(self) -> list[CoopView]:
        return [a for a in self.animal_tiles() if not a.cared_today]

    def harvestable_plants(self) -> list[PlantView]:
        out = []
        for p in self.plants():
            tile = self.tiles[p.y][p.x]
            if M.plant_mature(tile, self.day):
                out.append(p)
        return out

    def production_due_today(self) -> list[CoopView]:
        out = []
        for a in self.animal_tiles():
            tile = self.tiles[a.y][a.x]
            if M.animal_production_due(tile, self.day):
                out.append(a)
        return out

    def shed_eggs_milk_wool(self) -> dict:
        return {k: v for k, v in self.shed.items()
                if k in ("EGG", "MILK", "WOOL") and v}
