# F051 — the inert-opponent ceiling: a PASS seat saturates the market at ~191,812

Summary: against a PASS opponent the market saturates near 191,812 coins
and distinct competent agents land on that ceiling exactly — tables
measured against an inert opponent read the ceiling, not the agents.

Source: offline/runner.py episodes, seed 0 (measured 2026-09-15 for
issue #20's baseline). Evidence: v3-agent (345 lines) and
precomputed-schedule-policy (6,872 lines) produced DIFFERENT action
sequences (sha256 action hashes differ) and IDENTICAL rewards
(191,812.0 to the coin) — with a PASS seat the market is the binding
constraint, so any agent that extracts the available supply lands on
the same number. Consequence: baseline columns measured against an
inert opponent describe the ceiling we must beat, never the agents'
relative skill; agent-vs-agent numbers are the only ones that
discriminate.
