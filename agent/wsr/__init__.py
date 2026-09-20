"""The day: the planner's chains become the ops a worker-day is made of.

A chain per tile arrives from the DP, becomes a task array (`tasks.py`), and a beam search
(`beam.py`) decides which worker does what and when. `emit.py` turns that route into the ops the
engine reads, and `models.py` is where a chain is expanded and its order fixed.

The layer prices nothing and never touches the market: what a day costs is the planner's business,
and what it can pay for is the market's.
"""
