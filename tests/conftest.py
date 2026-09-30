"""Suite-wide baseline: a test's regime is the SUITE's, not whatever a default says.

The suites are callers that say nothing about the config. That made a config
default a hidden input: when `risk_kappa` briefly defaulted to 0.5, eighteen tests
that assert mean-regime facts silently changed regime, and five of them inherited
it through the shared `AGENT.cfg` singleton left behind by an earlier test. Two
symptoms, one cause: nothing pinned the baseline.

This fixture pins it. `Config()`'s defaults can move freely for the runs, and a
test that wants another regime says so itself -- which is the rule, because a test
must be able to state the world it is checking.
"""
import pytest

from agent.config import Config
from agent.main import AGENT

#: The suite's own baseline. Every field a test depends on belongs here, so that
#: changing a production default can never turn a test red for the wrong reason.
BASELINE = dict(master_rounds=2, risk_kappa=0.0)


@pytest.fixture(autouse=True)
def _suite_regime(monkeypatch):
    """Every test starts from the suite's baseline, whatever ran before it."""
    monkeypatch.setattr(AGENT, "cfg", Config(**BASELINE), raising=False)
    yield
    monkeypatch.setattr(AGENT, "cfg", None, raising=False)
