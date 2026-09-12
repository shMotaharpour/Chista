"""R002/R003 guards: rules re-imported from the package, dependency pinned.

Run:  .venv/bin/python -m tests.test_import_identity

R002 — game rules must come from the shipped package, never transcribed. Our
modules import the kaggriculture module itself, so the names we rely on must
still exist after a dependency bump, our own source must not carry copies of the
rule tables, and the configuration we simulate must match the shipped spec.
R003 — the fast path drives the environment's own interpreter through a private
API; a silent version bump can therefore change game behavior under every
simulation at once, so the installed version must equal the pinned one.
"""
from __future__ import annotations

import importlib.metadata as metadata
import re
from pathlib import Path

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as K

from world import fast_sim, kaggle_env

REPO = Path(__file__).resolve().parents[1]

# Names our code depends on inside the shipped module (R002 name test).
REQUIRED_NAMES = ("CROPS", "MARKET_PARAMS", "ANIMALS", "LAND_ORDER",
                  "interpreter", "market_price")
RULE_TABLES = ("CROPS", "MARKET_PARAMS", "ANIMALS", "LAND_ORDER")
# Shipped configuration keys fast_sim deliberately does not emulate: harness
# timeouts, not game rules.
UNMODELLED_KEYS = {"runTimeout"}


def test_required_names_still_exist() -> None:
    missing = [n for n in REQUIRED_NAMES if not hasattr(K, n)]
    assert not missing, (
        f"kaggle-environments no longer exposes {missing}: re-check world/ (R002)"
    )


def test_rules_are_imported_not_transcribed() -> None:
    """The rule objects in play must be the shipped ones, not local copies."""
    from kaggle_environments.envs.kaggriculture.kaggriculture import CROPS

    assert CROPS is K.CROPS, "rules must be re-imported, not transcribed (R002)"
    for module in (fast_sim, kaggle_env):
        copied = [n for n in RULE_TABLES if n in vars(module)]
        assert not copied, f"{module.__name__} defines its own {copied} (R002)"


def test_configuration_matches_the_shipped_spec() -> None:
    """No drift between the configuration we simulate and the published spec."""
    spec = dict(make("kaggriculture").configuration)
    for key, value in fast_sim.DEFAULT_CONFIGURATION.items():
        assert spec.get(key) == value, (
            f"configuration drift on {key!r}: fast_sim={value!r} spec={spec.get(key)!r} (R002)"
        )
    unmodelled = set(spec) - set(fast_sim.DEFAULT_CONFIGURATION)
    assert unmodelled <= UNMODELLED_KEYS, (
        f"the shipped spec gained keys we do not model: {sorted(unmodelled - UNMODELLED_KEYS)}"
    )


def test_installed_version_matches_the_requirements_pin() -> None:
    text = (REPO / "requirements.txt").read_text(encoding="utf-8")
    pins = dict(re.findall(r"^([A-Za-z0-9._-]+)==(\S+)\s*$", text, re.M))
    assert "kaggle-environments" in pins, (
        "requirements.txt must pin kaggle-environments exactly: the fast path uses "
        "the environment's private interpreter API (R002/R003)"
    )
    installed = metadata.version("kaggle-environments")
    assert installed == pins["kaggle-environments"], (
        f"installed kaggle-environments {installed} != pinned "
        f"{pins['kaggle-environments']}: re-run the parity test after bumping"
    )


def main() -> int:
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
            except Exception as exc:  # noqa: BLE001 - test runner
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
            else:
                print(f"PASS {name}")
    if failures:
        print(f"{failures} test(s) failed")
        return 1
    print("all import/version guards passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
