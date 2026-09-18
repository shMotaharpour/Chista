"""An action as a value: an op, then an item if it takes one, then a count if it takes one.

The engine takes an action as a list — `["WATER"]`, `["PLANT", "MELON"]`,
`["PICKUP", "WHEAT", 2]` (kaggriculture.py:312-375, :631-649) — and a list can hold
anything. This is the same thing with its slots pinned:

    [0]  always a `UnitAction` or a `MarketOrder`
    [1]  an item, if the op takes one: a `Crop`, an `Animal` or a `Product`
    [2]  a count, if the op takes one: an int > 0

Which ops take an item and which take a count is in `action_rules.SIGNATURE`, read from
the engine's handlers; `Action` checks itself against it, so an action that could never
be legal is refused here rather than silently ignored by the engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.world.action_rules import ITEM_OF, SIGNATURE
from agent.world.model import Animal, Crop, MarketOrder, Product, UnitAction

#: Anything an action can name: a seed's crop, a species, or a product.
Item = Crop | Animal | Product

#: Every op an action can carry, from the two vocabularies.
Op = UnitAction | MarketOrder


def _op_name(op: Any) -> str:
    """The engine's spelling of an op. `str(member)` is "UnitAction.WATER", not
    "WATER" — a str enum's `str()` is not its value, so the value is what we read."""
    return str(getattr(op, "value", op))


@dataclass(frozen=True)
class Action:
    """One action, exactly: `op`, then `item` if the op takes one, then `n` if it does."""

    op: Op
    item: Item | None = None
    n: int | None = None

    def __post_init__(self) -> None:
        takes_item, takes_n = SIGNATURE.get(_op_name(self.op), (False, False))
        if takes_item and self.item is None:
            raise ValueError(f"{self.op} takes an item")
        if not takes_item and self.item is not None:
            raise ValueError(f"{self.op} takes no item, got {self.item!r}")
        if takes_n and self.n is not None and int(self.n) <= 0:
            raise ValueError(f"{self.op} takes a count > 0, got {self.n!r}")
        if not takes_n and self.n is not None:
            raise ValueError(f"{self.op} takes no count, got {self.n!r}")
        if self.item is not None:
            allowed = ITEM_OF[_op_name(self.op)]
            if not isinstance(self.item, allowed):
                names = " or ".join(t.__name__ for t in allowed)
                raise ValueError(f"{self.op} takes a {names}, got {self.item!r}")

    @classmethod
    def parse(cls, raw: Any) -> "Action":
        """The engine's own list (or a tuple in the same shape) as an `Action`."""
        if isinstance(raw, Action):
            return raw
        if isinstance(raw, str):
            return cls(UnitAction(raw))
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ValueError(f"not an action: {raw!r}")
        op = _op_name(raw[0])
        known: dict[str, Op] = {op.value: op for op in UnitAction}
        for order in MarketOrder:
            known[order.value] = order
        if op not in known:
            raise ValueError(f"unknown action {raw[0]!r}")
        item = raw[1] if len(raw) > 1 else None
        n = raw[2] if len(raw) > 2 else None
        takes_item, takes_n = SIGNATURE.get(op, (False, False))
        if item is not None and takes_item:
            item = _as_item(str(item), op)
        if n is not None and takes_n:
            n = int(n)
        return cls(known[op], item, n)

    def as_list(self) -> list:
        """The engine's own form, ready to send."""
        out: list[Any] = [str(self.op.value)]
        if self.item is not None:
            out.append(str(getattr(self.item, "value", self.item)))
        if self.n is not None:
            out.append(int(self.n))
        return out

    def __str__(self) -> str:
        return " ".join(str(x) for x in self.as_list())


def _as_item(name: str, op: str) -> Item:
    """The named item as the member its op allows (a wrong name raises)."""
    for kind in ITEM_OF[op]:
        try:
            return kind(name)
        except ValueError:
            continue
    names = " or ".join(t.__name__ for t in ITEM_OF[op])
    raise ValueError(f"{op} takes a {names}, got {name!r}")
