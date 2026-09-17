Summary: SELL reads the shed, never a unit's bag: a sale with the goods
still in the bag changes nothing. The nightly drop is therefore not a
one-day lag but a choice — a shed-adjacent unit that DROPs and SELLs in
the same turn is paid that turn (+27 coins).

Source: `bench/bench_market_forecast.py --lag` (fast_sim, seed 0, wheat
harvested on day 2 at `(4, 4)`, i.e. shed-adjacent), measured 2026-09-16.

Evidence: with 2 wheat in the farmer's bag and 0 in the shed, a `SELL
WHEAT 1` order left the bank at 2,980 coins — unchanged. On the next
turn, `DROP` (unit action) + `SELL WHEAT 1` (market order) in the SAME
turn moved the bank to 3,007: the engine runs unit actions first, then
`_process_market`, so the drop lands before the queue is quoted.

Consequence: this settles contract 2 of the epic (#8 §4), which was
still open: a day-`d` harvest is sellable on day `d+1` through the
nightly drop, **unless** a unit spends an hour dropping it at a
shed-adjacent tile that day — the same day the harvest was picked. The
master's cash row must not assume the lag is forced, and
`day/inventory.py` sells from the shed only (F043), so a harvest
sitting in a bag is not part of the sellable stock until it lands.
