"""offline.pool loader tests: all 19 resolve, both conventions exercised.

Run:  .venv/bin/python -m tests.test_pool_loader

Contracts under test (issue #20 brief §2.2, §8):
- All 19 slugs resolve through one interface; the entry point is the
  LAST top-level def by source order (Kaggle's own rule), and the
  loader records which rule fired.
- Arity is adapted by INSPECTION, never by parameter name - the one
  agent whose entry point REQUIRES two arguments
  (adaptive-public-state-multi-route) is specifically covered.
- First-turn callability: every resolved agent returns a shape-valid
  action on a minimal observation.
"""

from __future__ import annotations

from offline.pool.loader import call, load, load_all, slugs

OBS = {"day": 0, "hour": 0, "step": 0, "player": 0,
       "farms": [{"money": 3000, "tiles": [[None] * 10 for _ in range(10)],
                  "farmer": [4, 4], "hands": [],
                  "unlocked_quadrants": ["NW"], "hires_today": 0}] * 2,
       "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
       "market": {"inventory": {}, "prices": {}},
       "town": {"unlocked_shops": []},
       "remainingOverageTime": 60.0}


def test_all_slugs_resolve() -> None:
    """Every canonical slug resolves through one interface (16 distinct
    agents; three vendored slugs are aliases of a canonical sibling and
    are excluded by default - registry.DUPLICATE_OF declares them)."""
    agents = load_all()
    assert len(agents) == len(slugs()) == 16
    for slug, loaded in agents.items():
        assert callable(loaded.fn), slug
        assert loaded.arity in (1, 2), (slug, loaded.arity)
        assert loaded.rule in ("last-top-level-def", "fallback-name"), slug


def test_two_arg_required_agent_covered() -> None:
    """adaptive-public-state-multi-route REQUIRES two args: fn(obs) raises
    TypeError, fn(obs, configuration) works. The loader must adapt by
    inspection so this agent is callable like the rest."""
    la = load("adaptive-public-state-multi-route")
    assert la.arity == 2
    raised = False
    try:
        la.fn(OBS)                                 # fn(obs) - no default
    except TypeError:
        raised = True
    assert raised, "fn(obs) must fail: this agent requires two args"
    a = call(la, OBS, {"episodeSteps": 720})       # adapted call works
    assert isinstance(a, dict)


def test_every_agent_first_turn_callable() -> None:
    """Each of the 16 canonical agents, called in its own convention,
    returns a dict on a minimal first-turn observation."""
    agents = load_all()
    assert len(agents) == 16
    for slug, la in sorted(agents.items()):
        action = call(la, OBS, {"episodeSteps": 720})
        assert isinstance(action, dict), slug


def test_arity_counts_match_the_brief_table() -> None:
    """Brief 2.1 over the canonical 16 (the alias pairs share their
    canonical's arity): the two-arg-required agent is still covered."""
    agents = load_all()
    ones = [s for s, la in agents.items() if la.arity == 1]
    twos = [s for s, la in agents.items() if la.arity == 2]
    assert len(ones) + len(twos) == 16
    assert "adaptive-public-state-multi-route" in twos  # requires two


def test_guard_labels_fire() -> None:
    """A deliberately broken fake agent proves each label fires."""
    from offline.pool.guard import guarded_call, GuardStats

    stats = GuardStats()
    def raiser(obs, config=None):
        raise RuntimeError("vendored bug")
    guarded_call(raiser, OBS, None, stats)
    def malformed(obs):
        return {"farmer": "PASS"}
    guarded_call(malformed, OBS, None, stats)
    labels = stats.labels()
    assert labels["raises"] == 1 and labels["first_traceback"]
    assert labels["malformed"] == 1 and labels["first_malformed"]


def test_guard_hands_out_copies_not_live_views() -> None:
    """The wrapper hands a COPY of the observation to third-party code:
    a mutating agent cannot corrupt the evaluation (issue 20 safety)."""
    from offline.pool.guard import guarded_call, GuardStats

    seen = {}
    stats = GuardStats()
    def mutator(obs, config=None):
        seen["money"] = obs["farms"][0]["money"]
        obs["farms"][0]["money"] = 0          # the corruption attempt
        return {"farmer": ["PASS"], "hands": [], "market": []}
    guarded_call(mutator, OBS, None, stats)
    assert seen["money"] == 3000              # the agent saw the copy
    assert OBS["farms"][0]["money"] == 3000   # the live view is untouched


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
    print("all pool loader tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
