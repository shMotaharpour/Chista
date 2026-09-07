"""CI lock: every ENGINE RULE in world/ must cite its engine source line.

An ENGINE RULE without a `L###` citation is exactly the failure mode where
imagined logic sneaks in as "mechanics". This test fails the suite when a
function's docstring claims ENGINE RULE without citing a line, or when a
mechanics function has no docstring at all.

Run: cd /chista/Chista/ChistaAgent && . .venv/bin/activate && python -m tests.test_source_citations
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

WORLD = Path(__file__).resolve().parent.parent / "world"

_CITATION_RE = re.compile(r"\bL\d+\b")

# Module-level aliases of engine objects — they ARE the engine, no citation.
ALLOWED_UNCITED = {
    "CROPS", "ANIMALS", "PRODUCTS", "MARKET_I0", "PRICE_FLOOR",
    "LAND_PRICES", "FARM_HAND_COST_MULT", "PRODUCT_BASE", "SELLABLE",
    "SEASON_DAYS", "TURNS_PER_DAY", "BOARD", "SHED_CAP",
    "MARKET_ORDERS_PER_TURN", "WEED_CHANCE", "ONE_TIME_CROPS",
    "ONGOING_CROPS", "ANIMAL_STRUCTURE", "market_price", "sellable_items",
    "SHED_ADJACENT", "DEFAULT_SPAWN", "ONE_OP", "from_obs",
}


def test_every_world_function_documents_engine_source_or_policy():
    problems = []
    for py in sorted(WORLD.glob("*.py")):
        if py.name == "__init__.py":
            continue
        tree = ast.parse(py.read_text())
        for node in tree.body:
            if not (isinstance(node, ast.FunctionDef) and not node.name.startswith("_")):
                continue
            if node.name in ALLOWED_UNCITED:
                continue
            doc = ast.get_docstring(node) or ""
            if not doc:
                problems.append(f"{py.name}:{node.name} — no docstring")
                continue
            has_citation = bool(_CITATION_RE.search(doc))
            says_engine = "ENGINE RULE" in doc or "engine" in doc.lower()
            says_policy = "L2" in doc or "L1" in doc or "POLICY" in doc
            if not (has_citation and (says_engine or says_policy)):
                problems.append(
                    f"{py.name}:{node.name} — docstring must cite an engine "
                    f"source line (L###) as ENGINE RULE, or declare the L1/L2 "
                    f"policy it defers to")
    assert not problems, "\n".join(problems)


def test_mechanics_has_no_imagined_model_functions():
    """These once existed as imagined models (not engine guards) and must
    never return to world/mechanics."""
    import world.mechanics as M
    for banned in ("one_time_yield_at", "plant_mature", "plant_needs_water",
                   "animal_is_lost", "bfs_dist"):
        assert not hasattr(M, banned), f"{banned} must stay deleted"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}:\n{e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
