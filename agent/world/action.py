"""Actions as values: a worker's, and the market's, kept apart.

The engine takes an action as a list — `["WATER"]`, `["PLANT", "MELON"]`,
`["PICKUP", "WHEAT", 2]` for a unit, `["SELL", "WOOL", 5]` for the market
(kaggriculture.py:312-375, :631-649). This is the same thing with its slots pinned:

    [0]  the op — a `UnitAction` in a `WorkerAction`, a `MarketOrder` in a `MarketAction`
    [1]  an item, if the op takes one: a `Crop`, an `Animal` or a `Product`
    [2]  a count, if the op takes one: an int > 0

They are two classes, not one with a union: a worker op and a market order are executed
by different code (a unit's action changes the tile it stands on; an order changes the
market), and a value that could be either is a value nobody can check. `parse_action`
picks the class from the op when the source is the engine's own list.

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


def item_of(name: str) -> Item:
    """The model's item for a name the engine uses: a crop, an animal or a product."""
    for kind in (Crop, Animal, Product):
        if name in kind.__members__:
            return kind(name)
    raise ValueError(f"unknown item {name!r}")


#: The worker ops, and the market's, as the two vocabularies they are.
WORKER_OPS: tuple[str, ...] = tuple(op.value for op in UnitAction)
MARKET_OPS: tuple[str, ...] = tuple(op.value for op in MarketOrder)

_BY_VALUE: dict[str, Any] = {op.value: op for op in UnitAction}
_BY_VALUE.update({op.value: op for op in MarketOrder})


def _op_name(op: Any) -> str:
    """The engine's spelling of an op. `str(member)` is "UnitAction.WATER", not
    "WATER" — a str enum's `str()` is not its value, so the value is what we read."""
    return str(getattr(op, "value", op))


@dataclass(frozen=True)
class Action:
    """The shared shape: `op`, then `item` if the op takes one, then `n` if it does."""

    op: Any
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

    def as_list(self) -> list:
        """The engine's own form, ready to send."""
        out: list[Any] = [_op_name(self.op)]
        if self.item is not None:
            out.append(str(getattr(self.item, "value", self.item)))
        if self.n is not None:
            out.append(int(self.n))
        return out

    def __str__(self) -> str:
        return " ".join(str(x) for x in self.as_list())


@dataclass(frozen=True)
class WorkerAction(Action):
    """One thing a unit does in a turn: what changes the tile it stands on."""

    op: UnitAction

    def __post_init__(self) -> None:
        if _op_name(self.op) not in WORKER_OPS:
            raise ValueError(f"{self.op!r} is not a worker op: {list(WORKER_OPS)}")
        super().__post_init__()

    @classmethod
    def parse(cls, raw: Any) -> "WorkerAction":
        """The engine's own list (or a tuple in the same shape) as a `WorkerAction`."""
        if isinstance(raw, WorkerAction):
            return raw
        op, item, n = _slots(raw)
        return cls(_BY_VALUE[op], item, n)


@dataclass(frozen=True)
class MarketAction(Action):
    """One order the market executes: the worker never runs it."""

    op: MarketOrder

    def __post_init__(self) -> None:
        if _op_name(self.op) not in MARKET_OPS:
            raise ValueError(f"{self.op!r} is not a market order: {list(MARKET_OPS)}")
        super().__post_init__()

    @classmethod
    def parse(cls, raw: Any) -> "MarketAction":
        """The engine's own list (or a tuple in the same shape) as a `MarketAction`."""
        if isinstance(raw, MarketAction):
            return raw
        op, item, n = _slots(raw)
        return cls(_BY_VALUE[op], item, n)


def parse_action(raw: Any) -> Action:
    """The class the op belongs to, for a list from the engine or a plan."""
    op, _, _ = _slots(raw)
    if op in WORKER_OPS:
        return WorkerAction.parse(raw)
    if op in MARKET_OPS:
        return MarketAction.parse(raw)
    raise ValueError(f"unknown action {raw!r}")


def _slots(raw: Any) -> tuple[str, Any, Any]:
    """`(op name, item, count)` out of the engine's list, with the item typed."""
    if isinstance(raw, Action):
        return _op_name(raw.op), raw.item, raw.n
    if isinstance(raw, str):
        return _op_name(raw), None, None
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError(f"not an action: {raw!r}")
    op = _op_name(raw[0])
    if op not in _BY_VALUE:
        raise ValueError(f"unknown action {raw[0]!r}")
    item = raw[1] if len(raw) > 1 else None
    n = raw[2] if len(raw) > 2 else None
    takes_item, takes_n = SIGNATURE.get(op, (False, False))
    if item is not None and takes_item:
        item = _as_item(str(item), op)
    if n is not None and takes_n:
        n = int(n)
    return op, item, n


def _as_item(name: str, op: str) -> Item:
    """The named item as the member its op allows (a wrong name raises)."""
    for kind in ITEM_OF[op]:
        try:
            return kind(name)
        except ValueError:
            continue
    names = " or ".join(t.__name__ for t in ITEM_OF[op])
    raise ValueError(f"{op} takes a {names}, got {name!r}")
