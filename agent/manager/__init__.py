"""agent/manager/ — the decision, and the only feedback loop in the agent.

    from agent.manager.core import Manager

    m = Manager(graph)
    m.observe(obs)                  # hour 0: the board, the forward prices
    while not m.settled:            # every turn: one probe of the wage
        m.step(budget_ms=700)
    plan = m.best()                 # the cheapest wage whose day actually fits

`core.py` holds the loop; `prices.py` builds the two dual vectors the
contractor prices a board with and says where each comes from.
"""
