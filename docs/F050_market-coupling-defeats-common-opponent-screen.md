Summary: a fixed-policy reference's own score varies by sd 61,142 across
opponents — ~330× the difference between adjacent pool agents — so a
common-opponent screen cannot rank the pool: results are pair
properties, not agent scalars.

Source: offline/runner.py episodes, seed 0 (measured 2026-09-15 for
issue #20's amendment). Evidence: precomputed-schedule-policy (the one
confirmed open-loop agent) scored 191,812 vs PASS-proxy, 52,080 vs
adaptive-replay-agent, 34,486 vs adaptive-public-state-multi-route,
81,749 vs ecobot-v6 — spread 157,326 on identical own behaviour
(F033: both players are quoted from the same pre-commit inventory, so
the reference's sells hit whatever ladder the other agent left
behind). Meanwhile adaptive-replay-agent vs v13-r3 differ by 182
coins. Consequence: paired same-seed comparisons are the only
resolution available, and #18's seed count must be measured, not
carried (issue #20 §6.3).
