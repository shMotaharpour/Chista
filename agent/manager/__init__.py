"""agent/manager/ — the shell that runs the planner inside the turn's clock.

    from agent.manager.core import Manager

    m = Manager()
    m.observe(obs, config)      # hour 0: the board, and a plan for today
    m.step()                    # every turn: keep the pool warm for tomorrow
    plan = m.best()             # always a legal answer

It decides nothing. `planner/master` decides what the farm does, `planner/day`
decides whether it can be walked, `planner/market` decides what it buys — this
is the piece that gives them a clock and remembers what they found.
"""
