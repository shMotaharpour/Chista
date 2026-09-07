"""State — read-view over the live kaggle-environments observation.

Design rules (per architecture review):
- The engine's `tiles[y][x]` is ALREADY a spatial index — this class exposes
  direct O(1) lookups (tile_at/plant_at/...) instead of whole-grid scans.
  Callers that need "all X" (portfolio summaries) iterate their OWN lists.
- Not a copy: lazily reads the engine observation dict.
- Serializable-friendly: accessors return plain data, so views can later be
  flattened for tensors/gym without changing this class.
"""
from __future__ import annotations

from dataclasses import dataclass

from world import mechanics as M
from world.rollback import snapshot  # noqa: F401  (re-export convenience)


def quadrant_of(x: int, y: int) -> str:
    """Quadrant name for a tile (engine L127-129 _quadrant_of)."""
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


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
    """Read-view over one player's observation. Build with State.from_obs(obs)."""
    obs: dict
    player: int = 0

    # ---- construction ----
    @classmethod
    def from_obs(cls, obs: dict, player: int | None = None) -> "State":
        return cls(obs=obs, player=obs.get("player", 0 if player is None else player))

    # ---- raw access ----
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

    # ---- spatial O(1) lookups (tiles[y][x] IS the index — use it directly) ----

    def tile_at(self, x: int, y: int):
        return self.tiles[y][x]

    def quadrant_of(self, x: int, y: int) -> str:
        return quadrant_of(x, y)

    def is_unlocked_tile(self, x: int, y: int) -> bool:
        """Unlocked = its quadrant is in unlocked_quadrants. Land opens per
        quadrant in fixed order NE->SW->SE (engine L96, L714-721) — so the
        unlocked region is NOT a growing symmetric square."""
        return self.quadrant_of(x, y) in self.unlocked

    def plant_at(self, x: int, y: int) -> PlantView | None:
        t = self.tiles[y][x]
        if not M.is_plant(t):
            return None
        return PlantView(
            x=x, y=y, crop=t["crop"], planted_day=t["planted_day"],
            age=M.plant_age(t, self.day),
            watered_today=t.get("watered_today", False),
            consecutive_unwatered=t.get("consecutive_unwatered", 0),
            yield_units=t.get("yield_units", 0),
            fertilized_until_day=t.get("fertilized_until_day", -1))

    def animal_at(self, x: int, y: int) -> CoopView | None:
        t = self.tiles[y][x]
        if not M.is_animal_tile(t):
            return None
        return CoopView(
            x=x, y=y, animal=t["animal"], placed_day=t.get("placed_day", self.day),
            fed_today=bool(t.get("fed_today")), cared_today=bool(t.get("cared_today")),
            consecutive_unfed=t.get("consecutive_unfed", 0),
            yield_units=t.get("yield_units", 0),
            pending_care_bonus=t.get("pending_care_bonus", 0),
            fertilizer_available=bool(t.get("fertilizer_available")))

    def empty_at(self, x: int, y: int) -> bool:
        return self.tiles[y][x] is None and self.is_unlocked_tile(x, y)

    # ---- whole-grid iteration: callers own their lists; these are the ONLY
    # full scans in L0 and must be called once per turn at most ----

    def iter_plants(self):
        for y, row in enumerate(self.tiles):
            for x, t in enumerate(row):
                if M.is_plant(t):
                    yield (x, y), t

    def iter_animals(self):
        for y, row in enumerate(self.tiles):
            for x, t in enumerate(row):
                if M.is_animal_tile(t):
                    yield (x, y), t
