"""Unit test for RLManager."""

from __future__ import annotations

import time
import pytest
from agent.manager.rl_manager import RLManager
from agent.dispatch import dispatch_plan
from offline_lab.fast_sim import FastSim


def test_rl_manager_lifecycle():
    sim = FastSim(configuration={"episodeSteps": 720, "seed": 42}, validate="fast")
    sim.reset()
    obs0 = sim.observations()[0]
    
    manager = RLManager()
    
    # Measure observe time
    t0 = time.time()
    manager.observe(obs0)
    dt = time.time() - t0
    
    # Measure warm observe time (steady state)
    t0 = time.time()
    manager.observe(obs0)
    dt_warm = time.time() - t0
    
    print(f"Cold observe: {dt*1000:.2f} ms, Warm observe: {dt_warm*1000:.2f} ms")
    assert dt_warm < 0.05, f"Warm observe took too long: {dt_warm*1000:.2f} ms"
    
    plan = manager.best()
    assert "units" in plan
    assert "market" in plan
    assert len(plan["market"]) == 24
    
    # Every market row must have <= 10 orders
    for row in plan["market"]:
        assert len(row) <= 10
        
    # Dispatch must succeed
    action = dispatch_plan(plan, obs0, manager.terms)
    assert "farmer" in action
    assert "hands" in action
    assert "market" in action
