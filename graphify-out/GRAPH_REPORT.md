# Graph Report - ChistaAgent  (2026-09-24)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 4188 nodes · 9063 edges · 182 communities (170 shown, 12 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 463 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d32ed49b`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- beam.py
- Config
- TileGraph
- forecast
- numpy
- sys
- pathlib
- planner/__init__.py
- economics-driven-rule-agent-ecobot-v6/agent.py
- dispatch_plan
- belief/__init__.py
- replan.py
- adaptive-replay-agent/agent.py
- test_master.py
- test_tile_dp.py
- tasks.py
- sell_coins
- TileState
- typing
- findings-from-zero-to-top-meta/agent.py
- load_contractor
- WorkerAction
- R004 as cited by the opponents docs — hand the agent a copy of the observation, never the live view
- v20-adaptive-r1-multi-route-agent/agent.py
- extract.py
- model.py
- test_rival_calendar.py
- test_opponents.py
- evaluate.py
- opponent.py
- compile
- test_market_layer.py
- test_slot_circuit.py
- test_layering.py
- Any
- test_market_wsr_check.py
- OpponentModel
- test_evaluate.py
- Crops and Animals — Every Rule, Numbered (AgriOracle research document)
- check_route
- test_tile_dp_decline.py
- market_queue
- interpreter
- corpus.py
- kaggle_probe-1.py
- test_pool_loader.py
- replay_agent.py
- test_world_parity.py
- test_mixed_day_pool.py
- FastSim
- repair.py
- docs/INDEX.md one-line-per-file index
- v13-r3-top-meta-order-safe-premium-control/agent.py
- MarketLayer
- bench_paths.py
- guarded_call
- v16-rc5-r5a-high-score-8c-4s-recovery/agent.py
- _sim
- Live observation views in fast mode
- Kaggle Probe #1 — What the Real Grading Machine Measures
- test_bounds_day.py
- test_mixed_day.py
- prices.py
- Evaluation arena — sparring opponents run in our own interpreter
- land.py
- test_plan_supply_iteration.py
- graph.py
- Action
- _process_market
- test_agent_runtime.py
- runner.py
- _FakeManager
- test_colgen.py
- test_tile_dp_contractor.py
- oracle_transitions.py
- ndarray
- 25-27-strict-future-v27-midgame-meta-reset/agent.py
- base64
- task_bn_scipy
- 44-46-strict-future-top-30-v22-price-impact/agent.py
- world/fast_sim (FastSim.run)
- emit
- v111-8c4s-economic-core-premium-lead/agent.py
- v16-rc5-high-score-8c-4s-premium-market-lead/agent.py
- _graph
- Runtime
- agent
- v3-agent/agent.py
- test_drop_empties_bag_day.py
- probe_hidden_state.py
- test_contractor_concepts.py
- agent
- generate
- planner — Rounding, Repair, Land (issue #13)
- wsr/ — The Day Layer
- _Path
- season_runner.py
- F058 — The grading platform, measured from inside a submission
- Worker distribution and scheduling problem on a 10x10 grid
- CP-SAT formulation
- kaggle_probe-2.py
- test_builder_fingerprint.py
- test_animal_drop_day.py
- test_rival_hours.py
- test_world_branch_purity.py
- ShedState
- price_board
- Agent Guide — ChistaAgent
- bench_turn_budget.py
- test_contractor_concepts_decay.py
- adaptive-public-state-multi-route/agent.py
- test_budget_day.py
- _quiet
- .solve
- Care bank (pending_care_bonus) accrual and payout
- A guard must be seen to fail
- emit
- test_planner_hands_day.py
- test_dual_bins.py
- _fingerprint
- expand_chain
- _timing_block
- ReplayAgent
- v16-rc5-high-score-8c-4s-premium-market-lead — vendored opponent agent (not our code)
- check_prune_invariance.py
- F049 — The tile optimum is a cycle, not a crop
- Ongoing ready days (TOMATO 8-11, STRAWBERRY 10-16)
- F031 — Order cap is per turn, queue walked to completion
- blas_threads
- cpu_ratio
- _patch_phase
- _column
- ndarray
- F048 — The season has 720 states and 719 decisions
- registry.py
- test_land_image_day.py
- test_spare_day.py
- solve_master
- F055 — SELL reads the shed, not the bag
- GapStats
- Chista — the System's Shape, Its One Vocabulary, and Its One Belief
- _kind_change_actions
- Held cap is its own deadline
- The silent-failure class: the engine fails silently
- classify.py
- greedy.py
- world/ — The Definitions
- F029 — Season structure and no liquidation day
- townShopSellInterval = 4 steps
- F053 — A refused op still bills the cost vector
- Wrong-tile ops refused in silence (F047)
- _apply_game_patch
- kaggle-environments==1.32.7 — pinned EXACTLY
- test_own_supply_gap.py
- Column
- spawn_assignments
- F016 — Animal placement and species table
- F054 — Town consumption is exactly modellable
- task_nn_numpy
- _chunk_pymc
- _run_a_full_day
- Graded Episode Configuration
- _make_overrun
- BuildSpec
- F038 — Money binds in the first week
- The n-th hire of a day costs fib(n): 1, 1, 2, 3, 5, 8, 13, 21, 34, 55
- Working a LOCKED tile spends hours as no-ops
- Past about six hands the order queue is the constraint
- F044 — Animal age collapses to a residue
- task_multiprocessing
- _start_burner
- world/__init__.py
- Mandatory agent-commit wrapper
- F048 mis-inference worked example
- Configuration defaults table
- task_package_census
- Import/call allowlist audit (clean)
- opponents/__init__.py
- numpy, pandas, matplotlib, seaborn — analysis and plotting
- Kaggle probes index
- Feature branch to PR to squash-merge

## God Nodes (most connected - your core abstractions)
1. `FastSim` - 98 edges
2. `forecast()` - 70 edges
3. `TileGraph` - 68 edges
4. `compile_route()` - 59 edges
5. `TaskArray` - 58 edges
6. `new_environment()` - 50 edges
7. `TileContractor` - 48 edges
8. `Config` - 45 edges
9. `check_route()` - 44 edges
10. `OpponentModel` - 41 edges

## Surprising Connections (you probably didn't know these)
- `R004 as cited by the opponents docs — hand the agent a copy of the observation, never the live view` --conceptually_related_to--> `test_observation_mutation_is_guarded_in_dev()`  [AMBIGUOUS]
  opponents/adaptive-public-state-multi-route/SOURCE.md → tests/test_world_parity.py
- `Columns are priced as rotations, not crops` --conceptually_related_to--> `Column`  [INFERRED]
  docs/F049_the-tile-optimum-is-a-cycle-not-a-crop.md → agent/planner/colgen.py
- `Results are pair properties, not agent scalars` --conceptually_related_to--> `MarketForecast`  [INFERRED]
  docs/F050_market-coupling-defeats-common-opponent-screen.md → agent/belief/market.py
- `The mean unlock policy is the default` --conceptually_related_to--> `MarketForecast`  [INFERRED]
  docs/F054_town-consumption-is-exactly-modellable.md → agent/belief/market.py
- `town_deltas(shops, step) reproduces every observed delta (6,462/6,462)` --conceptually_related_to--> `DayMarket`  [INFERRED]
  docs/F054_town-consumption-is-exactly-modellable.md → agent/planner/market.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Duplicate vendored payloads — two slugs, one agent each** — opponents_farming_score_a_mathematical_approach_source_payload, opponents_v111_8c4s_economic_core_premium_lead_source_payload, opponents_farming_score_v3_replay_revised_source_payload, opponents_v16_rc5_high_score_8c_4s_premium_market_lead_source_payload, opponents_duplicate_payloads [EXTRACTED 1.00]
- **The F-File Convention — 13 Findings Docs** — agents_findings_and_rules_files, docs_f001_seeds_bypass_the_shed_doc, docs_f002_planting_day_counts_as_unwatered_doc, docs_f003_watering_once_per_day_doc, docs_f004_fertilizer_three_day_window_doc, docs_f005_one_shot_crops_window_and_caps_doc, docs_f006_one_fertilizer_worth_two_wheat_doc, docs_f007_one_shot_harvest_frees_tile_doc, docs_f008_one_shot_lifespan_decay_doc, docs_f009_one_shot_honest_yields_doc, docs_f010_ongoing_watering_never_adds_yield_doc, docs_f011_ongoing_production_cadence_doc, docs_f012_ongoing_total_eight_with_fertilizer_doc, docs_f013_ongoing_lifespan_on_last_production_doc [EXTRACTED 1.00]
- **Repository rule set (R001-R007)** — docs_r001_competition_stack_and_commit_rules_project_stack, docs_r002_import_rules_from_kaggle_environments_reimport_decision, docs_r003_simulator_wraps_real_interpreter_wrap_not_reimplement, docs_r004_configurable_validation_dev_and_fast_modes_validation_config_switch, docs_r005_every_number_has_a_source_rule, docs_r006_non_negative_prices_and_wages_non_negative_precondition, docs_r007_a_guard_must_be_seen_to_fail_rule [EXTRACTED 1.00]
- **Silent engine failures: refused orders, no-op tiles, destroyed overflow** — docs_f047_silent_operations_catalog_silent_failure_class, docs_f038_money_binds_first_week_silent_refusal_on_short_purse, docs_f042_land_purchase_prefix_locked_locked_tile_no_op, docs_f043_shed_capacity_destroys_overflow_nightly_drop_destroys_overflow, docs_f039_hire_fib_costs_nightly_reset_do_hire_silent_refusal, docs_f053_refused_ops_still_bill_the_cost_vector_refused_but_billed [EXTRACTED 1.00]
- **Vendored opponent agents sparring in the evaluation arena** — opponents_adaptive_public_state_multi_route_source_agent, opponents_adaptive_replay_agent_source_agent, opponents_adaptive_shop_guard_source_agent, opponents_economics_driven_rule_agent_ecobot_v6_source_agent, opponents_farming_score_a_mathematical_approach_source_agent, opponents_farming_score_v3_replay_revised_source_agent, opponents_findings_from_zero_to_top_meta_source_agent, opponents_frontier_the_soil_remembers_rain_source_agent, opponents_precomputed_schedule_policy_source_agent, opponents_v111_8c4s_economic_core_premium_lead_source_agent, opponents_v13_r3_top_meta_order_safe_premium_control_source_agent, opponents_v16_rc5_high_score_8c_4s_premium_market_lead_source_agent, opponents_v16_rc5_r5a_high_score_8c_4s_recovery_source_agent, opponents_v20_adaptive_r1_multi_route_agent_source_agent, opponents_v21_r1_public_state_route_portfolio_source_agent, opponents_v3_agent_source_agent, opponents_x544_nah_i_d_win_source_agent, opponents_arena [EXTRACTED 1.00]
- **Animal production pipeline (placement, feeding, escape, base yield, fertilizer)** — docs_f016_animal_placement_and_species_table_placement_requirement, docs_f016_animal_placement_and_species_table_species_table, docs_f017_two_consecutive_unfed_days_escape_escape_rule, docs_f018_unfed_animals_still_produce_base_base_production, docs_f023_animal_fertilizer_every_night_fertilizer_availability, docs_f028_animal_harvest_calendar_cap_deadline_animal_harvest_calendar [INFERRED 0.75]
- **Silent-failure mode cluster** — docs_f060_hire_settles_after_the_units_move_silent_refusals, docs_r007_a_guard_must_be_seen_to_fail_five_false_guards, docs_r005_every_number_has_a_source_inference_is_not_measurement, docs_r006_non_negative_prices_and_wages_non_negative_precondition, opponents_readme_live_view_copy_rule [INFERRED 0.75]
- **Care bank lifecycle (accrual, zeroing, lump payout, cap)** — docs_f019_care_bank_accrual_and_payout_care_bank, docs_f020_unfed_production_night_destroys_bank_bank_zeroing, docs_f021_care_bank_lumps_before_first_yield_first_yield_lump, docs_f028_animal_harvest_calendar_cap_deadline_cap_deadline, docs_f022_production_past_cap_is_lost_cap_clamp [INFERRED 0.85]
- **Fast-simulator stack: import, wrap, validate, parity** — docs_r002_import_rules_from_kaggle_environments_reimport_decision, docs_r003_simulator_wraps_real_interpreter_wrap_not_reimplement, docs_r003_simulator_wraps_real_interpreter_world_parity_test, docs_r004_configurable_validation_dev_and_fast_modes_validation_config_switch, docs_kaggriculture_source_interpreter [INFERRED 0.85]
- **Market execution pipeline (action shape, queue walk, order cap, quoting, price function)** — docs_f030_action_shape_unit_before_market_action_shape, docs_f031_order_cap_per_turn_queue_walk_queue_walk_semantics, docs_f031_order_cap_per_turn_queue_walk_market_order_cap, docs_f033_market_quoting_both_players_pre_commit_quoting, docs_f034_price_function_shape_parameters_price_function, docs_f032_queue_order_design_decision_queue_order_decision [INFERRED 0.85]
- **The One-Shot Crop Lifecycle** — docs_f005_one_shot_crops_window_and_caps_yield_window_formula, docs_f006_one_fertilizer_worth_two_wheat_fertilizer_worth_two_wheat, docs_f007_one_shot_harvest_frees_tile_harvest_frees_tile, docs_f008_one_shot_lifespan_decay_lifespan_formula, docs_f009_one_shot_honest_yields_honest_yields [INFERRED 0.85]
- **The Ongoing Crop Lifecycle** — docs_f010_ongoing_watering_never_adds_yield_ongoing_watering_no_yield, docs_f011_ongoing_production_cadence_nightfall_production_cadence, docs_f012_ongoing_total_eight_with_fertilizer_eight_with_fertilizer, docs_f013_ongoing_lifespan_on_last_production_lifespan_on_last_production [INFERRED 0.85]
- **Opponent-coupled market measurement: pair results, the inert-opponent ceiling, paired same-seed runs** — docs_f050_market_coupling_defeats_common_opponent_screen_results_are_pair_properties, docs_f050_market_coupling_defeats_common_opponent_screen_common_opponent_screen_cannot_rank, docs_f050_market_coupling_defeats_common_opponent_screen_paired_same_seed_comparison, docs_f051_inert_opponent_market_ceiling_inert_opponent_ceiling_191812, docs_f051_inert_opponent_market_ceiling_agent_vs_agent_numbers_discriminate [INFERRED 0.85]
- **Payload audit flow: extract -> hash -> audit -> arena execution** — opponents_extract, tests_test_opponents, opponents_third_party_import_allowlist, opponents_self_bundled_module_loader, opponents_import_time_unpacking, opponents_data_tables, opponents_arena, opponents_arena_in_process_rationale, opponents_extract_never_executes [INFERRED 0.85]
- **The per-seat runtime budget: 1 s free per turn, a 60 s bank, billed overruns** — docs_f046_runtime_budget_one_second_bank_one_free_second_per_turn, docs_f046_runtime_budget_one_second_bank_sixty_second_episode_bank, docs_f046_runtime_budget_one_second_bank_overrun_billing_max0, docs_f058_grading_platform_measured_from_inside_bank_is_per_seat, docs_f058_grading_platform_measured_from_inside_only_the_offending_seat_is_stopped, docs_f058_grading_platform_measured_from_inside_graded_configuration [INFERRED 0.85]

## Communities (182 total, 12 thin omitted)

### Community 0 - "beam.py"
Cohesion: 0.03
Nodes (146): The engine's placement rule for a new unit: of the four shed-access tiles, the…, spawn_cell(), _bag(), bags_of(), beam_for(), _better(), _better_route(), _carried() (+138 more)

### Community 1 - "Config"
Cohesion: 0.02
Nodes (112): Config, Every number the agent is tuned on, in one place. Numbers only. Config holds…, What one turn may spend thinking, after the reserve., The tuned numbers if they shipped, the defaults if they did not. A missing file…, Write these numbers where `load()` will find them., The tuned numbers. Every field is a quantity; none is a mode. One field is a…, _contractor(), _load_graph() (+104 more)

### Community 2 - "TileGraph"
Cohesion: 0.03
Nodes (84): best_day_split(), depth_blocks(), depth_coins(), geometric_edges(), hourly_value(), _index(), inventory_at(), marginal_price() (+76 more)

### Community 3 - "forecast"
Cohesion: 0.03
Nodes (88): _apply_sells(), forecast(), _get(), hourly_inventory(), hourly_prices(), _hourly_rows(), mean_shop_demand(), Any (+80 more)

### Community 4 - "numpy"
Cohesion: 0.04
Nodes (78): day_envelope(), The best hour of each day per good: `(prices, hours)`, `(n_goods, days)`. The…, column_key(), Column generation: the master combines MANY plans, and stops on a certificate.…, The resource column a shed item occupies, or None if it has none. The shed's…, A plan's signature: the chain it runs on each day, and what it constructs. Two…, A plan's signature, moved `step` days forward: the day numbers renumber.…, _resource_of() (+70 more)

### Community 5 - "sys"
Cohesion: 0.03
Nodes (74): The day: the planner's chains become the ops a worker-day is made of. A chain…, collections, json, build(), load(), main(), Time the day layer over the benchmark corpus, and say what the seconds buy. The…, The layer's own `Day` and its tasks, from one corpus entry. (+66 more)

### Community 6 - "pathlib"
Cohesion: 0.04
Nodes (74): info_path(), Any, The agent's artifact store: one data file per artifact, and its info beside it.…, The info file of an artifact., Write the info file beside an artifact. An unknown key raises., Read an artifact's info, refusing a file that is not in the standard shape., read_info(), write_info() (+66 more)

### Community 7 - "planner/__init__.py"
Cohesion: 0.05
Nodes (67): assign_by_quota(), assign_tiles(), Choice, ClassMix, counts(), demote_to_feasible(), Plan, plan_from_board() (+59 more)

### Community 8 - "economics-driven-rule-agent-ecobot-v6/agent.py"
Cohesion: 0.07
Nodes (64): _add_pickup_tasks(), agent(), AgentMemory, AnimalCensus, _assign_tasks(), eligible(), bfs_step(), build_development_plan() (+56 more)

### Community 9 - "dispatch_plan"
Cohesion: 0.04
Nodes (64): dispatch_plan(), Dispatcher: a committed day plan -> one action dict for the engine. The plan…, Slice `plan` into this turn's action dict. `plan` is a dict: ``{"units": [[op,…, The compiler's output in the shape the dispatcher slices. `units[0]` is the…, to_plan(), _play(), played(), fixture (+56 more)

### Community 10 - "belief/__init__.py"
Cohesion: 0.05
Nodes (61): block_revenue(), What `blocks` pay for `units` — the evaluator side of `depth_blocks`. A…, agent/belief/ — what the market and the rival are doing, and what to sell. The…, drain_forecast(), infer_rival_slot(), Exact mean and sd of the town drain over the next `steps_ahead` turns., Infer which slot the rival's volume sat in, from the price it realised.…, default_schedules() (+53 more)

### Community 11 - "replan.py"
Cohesion: 0.05
Nodes (62): decode_farm(), _decode_one(), decode_world(), FarmView, _nearest_modelled(), PrivateView, Observation decode: harness obs -> WorldView (both farms, packed keys). Issue…, The harness observation -> WorldView (both farms, one code path).… (+54 more)

### Community 12 - "adaptive-replay-agent/agent.py"
Cohesion: 0.07
Nodes (59): F033 — Market quoting: both players, pre-commit inventory, BUY_PRODUCT quoted at post-buy inventory, Both players quoted from the same pre-commit inventory, F034 — Price function shape and per-good parameters, MARKET_PARAMS per-good shapes and amplitudes, price(inventory) pivots at I0 = 10,000, floored at 1, F036 — The price ladder is coarse; sell pressure bites only in shallow markets, Coarse price ladder: about 25 wheat units per coin (+51 more)

### Community 13 - "test_master.py"
Cohesion: 0.08
Nodes (48): CouplingSupply, The farm-level resources the LP's rows share out (per day). Sources (R005),…, The shed's capacity: the run's own override, else the world's constant. The…, The farm's day-0 coupling supply from the observation (R005). Hours: 24·(1 +…, _shed_capacity(), supply_from_obs(), skip, _bare_ids() (+40 more)

### Community 14 - "test_tile_dp.py"
Cohesion: 0.05
Nodes (54): offline_lab_build_chains, offline_lab_build_ledger, _all_edges(), _arrays_equal(), _build_with(), _carrot(), _edge(), _find() (+46 more)

### Community 15 - "tasks.py"
Cohesion: 0.05
Nodes (35): _action_code(), action_codes(), build(), chain_depth(), columns_of(), distance_matrix(), _engine_op(), _good_for() (+27 more)

### Community 16 - "sell_coins"
Cohesion: 0.07
Nodes (43): buy_coins(), good_index(), plan_coins(), ndarray, _quotes(), The market's arithmetic as arrays: the price ladder, and the exact day split.…, Units of supply a sale of `units` from `inventory` lands: the engine stalls at…, Coins `units` fetch selling from `inventory`, floor stall included. (+35 more)

### Community 17 - "TileState"
Cohesion: 0.06
Nodes (30): _bits(), decode_tile(), _field_value(), _name_code(), _name_from_code(), The DP's view of a tile: world's `TileHourZero`, plus the node key. The…, Inverse of `pack` (a key outside the layout raises)., Engine tile value at day start -> TileState (world's decode, as this class). (+22 more)

### Community 18 - "typing"
Cohesion: 0.06
Nodes (42): CarryRequirement, DaySchedule, DropRequirement, empty_state(), good_index(), MarketState, OrderBook, PurchaseIntent (+34 more)

### Community 19 - "findings-from-zero-to-top-meta/agent.py"
Cohesion: 0.10
Nodes (44): agent(), is_sell(), promotable(), _align_hands(), _base_agent(), _cash_needed(), _conditional_reorder(), _copy_action() (+36 more)

### Community 20 - "load_contractor"
Cohesion: 0.06
Nodes (42): load_contractor(), The shipped graph, cast once per process — never rebuilt at runtime., m_cfg_rounds(), main(), The Wentges smoothing sweep on the exact pricer (#87 follow-up).…, The config's master_rounds (200), read without a Manager., importlib, A column is a plan, and a plan is still a plan when the prices move. (+34 more)

### Community 21 - "WorkerAction"
Cohesion: 0.06
Nodes (32): One thing a unit does in a turn: what changes the tile it stands on., WorkerAction, in_board(), manhattan(), quadrant_of(), The board: cells, quadrants, and the shed's four doors. Numbers are in…, The engine stores the board as `tiles[y][x]`: the OUTER list is y, the inner…, Which quadrant a cell is in (kaggriculture.py:127-129). (+24 more)

### Community 22 - "R004 as cited by the opponents docs — hand the agent a copy of the observation, never the live view"
Cohesion: 0.07
Nodes (44): adaptive-public-state-multi-route — vendored opponent agent (not our code), kaggriculture-adaptive-public-state-multi-route.ipynb (source notebook), writefile packing — kaggriculture-adaptive-public-state-multi-route.ipynb extracted 387 lines / 174,502 bytes, adaptive-replay-agent — vendored opponent agent (not our code), kaggriculture-adaptive-replay-agent.ipynb (source notebook), blob packing — kaggriculture-adaptive-replay-agent.ipynb extracted 551 lines / 28,709 bytes, adaptive-shop-guard — vendored opponent agent (not our code), kaggriculture-adaptive-shop-guard.ipynb (source notebook) (+36 more)

### Community 23 - "v20-adaptive-r1-multi-route-agent/agent.py"
Cohesion: 0.15
Nodes (43): agent(), _align_hands(), _clone_distance(), _copy_action(), _demand_per_day(), _farm(), _future_sells(), _get() (+35 more)

### Community 24 - "extract.py"
Cohesion: 0.08
Nodes (40): AST, gzip, audit(), _audit_source(), _classify(), _constants(), _decode_chain(), extract() (+32 more)

### Community 25 - "model.py"
Cohesion: 0.11
Nodes (33): actions_of(), A chain's actions, each carrying its argument. The entity is what fills the…, _as_item(), item_of(), Item, Actions as values: a worker's, and the market's, kept apart. The engine takes…, The named item as the member its op allows (a wrong name raises)., The model's item for a name the engine uses: a crop, an animal or a product. (+25 more)

### Community 26 - "test_rival_calendar.py"
Cohesion: 0.10
Nodes (34): harvest_events(), ndarray, The rival's harvest calendar (#16): their board is PUBLIC — `farms[i]` tiles…, One planted rival tile's scheduled payout., The rival board's scheduled payouts, from today's decode. A planted tile's…, `(horizon, n_goods)` units/day the rival's planted tiles schedule. This is the…, RivalHarvestEvent, supply_curve() (+26 more)

### Community 27 - "test_opponents.py"
Cohesion: 0.12
Nodes (35): hashlib, agent.py payload, byte-for-byte — SHA-256 9b5e1157af3d4a64…, agent.py payload, byte-for-byte — SHA-256 3c86ae6e2afaa091…, agent.py payload, byte-for-byte — SHA-256 2fe7118964656263…, Decoded data tables (route or schedule) — no code to read, Duplicate payloads — two slugs sharing one SHA-256, agent.py payload, byte-for-byte — SHA-256 aca6dcddafd6fd6b…, agent.py payload, byte-for-byte — SHA-256 12eb55e1e2455a2b… (+27 more)

### Community 28 - "evaluate.py"
Cohesion: 0.09
Nodes (35): csv, datetime, _append_jsonl(), _append_scoreboard(), _autopsy_worker_code(), _csv_num(), _dev_split(), _fmt() (+27 more)

### Community 29 - "opponent.py"
Cohesion: 0.10
Nodes (27): quantile_price_floor(), Phase 2 — the rival's action model, the demand forecast, and the hidden order.…, The price if the drain falls `z` sd short — risk priced as a price., field_of(), Read a field from the observation whether it is a dict or a struct., committed_flow(), fib_hire_costs(), FlowRecord (+19 more)

### Community 30 - "compile"
Cohesion: 0.08
Nodes (29): compile(), priced(), queue(), A `DayPlan` -> the `{"units": [...], "market": [...]}` the dispatcher slices.…, _load_payload(), agent(), _kaggle_submission_entrypoint(), Adaptive Shop Guard for Kaggriculture. (+21 more)

### Community 31 - "test_market_layer.py"
Cohesion: 0.10
Nodes (27): _guard_margin(), plan_sales(), Slack left below the cap by the guard. The day's harvest is an ESTIMATE…, How much of which item to sell on which turn, and why. Order of the rules is…, market_at(), This turn's market orders, from a per-hour queue or a flat list. #15 made…, _forecast_stub(), Secretary B (#15): the forecast, the shed guard, the queue, the layer. Run:… (+19 more)

### Community 32 - "test_slot_circuit.py"
Cohesion: 0.12
Nodes (29): _candidate_schedules(), _center_items(), _drain_per_hour(), plan_day_hours(), plan_day_slots(), ndarray, The slot circuit: the rival's trained distribution -> one sell schedule.…, The rival's hourly placements from the trained distribution. `activity` selects… (+21 more)

### Community 33 - "test_layering.py"
Cohesion: 0.09
Nodes (32): contextlib, tempfile, _fake_repo(), _imported_modules(), _imported_roots(), main(), _module_file(), The submission never imports the competitors. Run: .venv/bin/python -m… (+24 more)

### Community 34 - "Any"
Cohesion: 0.09
Nodes (21): _EnvShim, _merge_configuration(), _observation_dict(), _pass_policy(), Any, Re-initialize the episode from the configuration (fresh seed draw unless…, Per-agent rewards; non-final steps report 0.0, matching the harness. Core…, Per-agent observation dicts as the agents would receive them. copy_state=True… (+13 more)

### Community 35 - "test_market_wsr_check.py"
Cohesion: 0.09
Nodes (29): build(), buy_orders(), DayMarket, merge(), needs(), The purchases the day needs, netted against what the farm already holds. Seeds…, Read the timetable off the merged queue. A hand hired in turn `t` acts from `t…, One queue, in the engine's settle order, capped per turn. Sells keep the head… (+21 more)

### Community 36 - "OpponentModel"
Cohesion: 0.13
Nodes (17): basket_matrix(), _load_trained(), _load_trained_qty(), OpponentModel, Any, ndarray, Empirical action counts over the rival, Laplace-smoothed. Smoothing is…, The good's aggregate action distribution (its own prior). (+9 more)

### Community 37 - "test_evaluate.py"
Cohesion: 0.07
Nodes (26): math, _paired_jobs(), (opponent, seed) jobs, interleaved so no seed strands. Seed s starts at…, bank_seconds(), F046's runtime-budget accounting over one episode's turn times. Returns…, offline.evaluate unit tests (issue #18). Run: .venv/bin/python -m…, Self-review guard: the scoreboard is the PR's DURABLE evidence, so it must…, F046: 1 free second per turn, billed per turn -> the bank draw is the SUM of… (+18 more)

### Community 38 - "Crops and Animals — Every Rule, Numbered (AgriOracle research document)"
Cohesion: 0.10
Nodes (29): Dropping Is Counted, world/rules.py — the Numbers, F002 — Planting Day Counts as Unwatered, A New Plant Starts with consecutive_unwatered = 1, Two Consecutive Dry Nights Turn the Tile into a Weed, F003 — Watering Once per Day, WATER Works Once per Day, the Second Is Refused Silently, Fertilized Coverage Is Exactly Three Days (+21 more)

### Community 39 - "check_route"
Cohesion: 0.11
Nodes (24): check_route(), Everything a compiled route promises, checked - so a caller can report instead…, _play(), played(), fixture, A good a worker uses only after its own DROP is not charged to its door load.…, The compiled day on an engine holding the two animals and the shed's wheat., _search() (+16 more)

### Community 40 - "test_tile_dp_decline.py"
Cohesion: 0.11
Nodes (24): artifact_path(), The data file of an artifact., Edge, One CSR edge, decoded: `from_id -> to_id` by `chain_id`, for `entity_code`.…, Decode CSR row `row` (a global edge index) of `state_id`., _cost(), _edge(), _graph() (+16 more)

### Community 41 - "market_queue"
Cohesion: 0.11
Nodes (26): _after_arrival(), _assert_within_cap(), _get(), market_queue(), _money(), orders_by_hour(), Any, Secretary B: the shed — what lands, what fits, what has to be sold. Issue #15.… (+18 more)

### Community 42 - "interpreter"
Cohesion: 0.09
Nodes (21): _default_spawn returns the farmer to (4,4), _end_of_day:877-882 resets the crew, ANIMALS table, _apply_unit_action(), CROPS table, _daily_refresh_animals(), _daily_refresh_plants(), _default_spawn() (+13 more)

### Community 43 - "corpus.py"
Cohesion: 0.10
Nodes (22): duckdb, importlib_util, dumps(), extract(), main(), Build the day layer's benchmark corpus: a few winning games, sampled across the…, The games our own seat won, best first., One day's real land work, or None when the dump has no such day. (+14 more)

### Community 44 - "kaggle_probe-1.py"
Cohesion: 0.09
Nodes (17): gc, _bind_seat(), _milp_problem(), mode_burn_3s(), mode_burn_the_bank(), mode_not_serialisable(), _new_seat_state(), Kaggriculture — ONE-SUBMISSION PROBE agent WHAT THIS IS A legal agent that… (+9 more)

### Community 45 - "test_pool_loader.py"
Cohesion: 0.11
Nodes (25): inspect, call(), _last_top_level_def(), load(), load_all(), LoadedAgent, Pool loader: import the 19 vendored competitors uniformly. Issue #20,…, Call a resolved agent in its own convention (arity by inspection). Never raises… (+17 more)

### Community 46 - "replay_agent.py"
Cohesion: 0.15
Nodes (21): f(), _check_action(), _check_farmer(), _check_hands(), _check_market(), _check_op(), _check_op_args(), _engine_vocab() (+13 more)

### Community 47 - "test_world_parity.py"
Cohesion: 0.12
Nodes (23): Run `episodes` independent episodes across worker processes. Returns ONE RECORD…, run_parallel(), action_for(), _clean(), config(), drive_both_paths(), harness_agent(), agent() (+15 more)

### Community 48 - "test_mixed_day_pool.py"
Cohesion: 0.09
Nodes (27): _day(), late(), fixture, The mixed day with fewer hands than it needs: what the pool decides, and where…, What the caller is handed when the day is not carried: a route, and one the…, A hand hired at hour three lands from where the field stands at hour three…, The hand-built three-hand day is legal, and the search takes it when it is…, Three hands are enough for this day - the reference proves it - so the search… (+19 more)

### Community 49 - "FastSim"
Cohesion: 0.11
Nodes (22): The town's consumption during turn `step`, exactly as the engine runs it., town_deltas(), cadence_probe(), error_table(), lag_probe(), layer_timing(), main(), overflow_suite() (+14 more)

### Community 50 - "repair.py"
Cohesion: 0.14
Nodes (20): Drop, _inventories(), land_step_price(), _money(), order_cost(), _own_tiles(), _prices(), Any (+12 more)

### Community 51 - "docs/INDEX.md one-line-per-file index"
Cohesion: 0.09
Nodes (25): docs/INDEX.md one-line-per-file index, Permanent F/R ids, F/R file naming and INDEX convention, Project stack (Kaggle Kaggriculture, CPU PyTorch, OR-Tools, scipy), Eliminated constants-parity harness, Re-import rules from kaggle_environments, Exactly one source of truth, Pinned kaggle-environments version mitigation (+17 more)

### Community 52 - "v13-r3-top-meta-order-safe-premium-control/agent.py"
Cohesion: 0.20
Nodes (25): agent(), _align_hands(), _clone_distance(), _copy_action(), _farm(), _future_base_sells(), _get(), _kaggle_submission_entrypoint() (+17 more)

### Community 53 - "MarketLayer"
Cohesion: 0.13
Nodes (20): attach(), from_env(), _market_row(), MarketLayer, Any, The market half of the day plan, riding on top of whatever rung answered (#15).…, `CHISTA_MARKET=spread|dump` -> a layer; anything else -> disabled., Attach the market half to a rung's action (never raises). Called for every… (+12 more)

### Community 54 - "bench_paths.py"
Cohesion: 0.12
Nodes (20): fast_sim(), harness_no_agents(), harness_with_agents(), main(), median_seconds(), pass_policy(), Any, Reproduce the R003 speed claim on the machine you are running on. Usage:… (+12 more)

### Community 55 - "guarded_call"
Cohesion: 0.11
Nodes (19): Any, Action-shape validation — one implementation of the truth. Promoted from…, Dev-mode action shape check (see module docstring)., validate_action(), copy_observation(), guarded_call(), GuardStats, Pool guard: wrap a vendored agent so nothing it does takes the sweep down.… (+11 more)

### Community 56 - "v16-rc5-r5a-high-score-8c-4s-recovery/agent.py"
Cohesion: 0.26
Nodes (24): _adjacent_cow_pasture_move(), agent(), _align_hands(), _copy_action(), _cow_inventory(), _cow_place_alignment(), _empty_cow_pasture(), _existing_sell() (+16 more)

### Community 57 - "_sim"
Cohesion: 0.09
Nodes (24): The named bias, asserted in its direction. Missing unlocks under-counts town…, Patch `K.market_price` and watch every forecast price move with it., `our_sells` is the farm's own price impact (F033/F036)., The two unlock policies must differ: `mean` adds the shop's demand. `none` only…, Day-start market inventories of a PASS-vs-PASS episode., Stock past the cap must come back as a queue with real orders in it. The…, The ladder's contract outranks the market layer (R007: seen to fail)., The layer's contract is a VALID dict even if the rung broke its own. (+16 more)

### Community 58 - "Live observation views in fast mode"
Cohesion: 0.10
Nodes (24): fast mode (checks bypassed), Live observation views in fast mode, Read-only observation contract, Characterisation sweep 2026-09-15 seed 0, chista-m1 baseline vs the pool, Behavioural class semantics P/S/R, 11 dev / 5 held out until M5, Declared duplicate payloads (+16 more)

### Community 59 - "Kaggle Probe #1 — What the Real Grading Machine Measures"
Cohesion: 0.10
Nodes (23): belief/ — The Sell Side, Compute Budget — 1 s Free per Turn plus a 60 s Bank, Contention Ratio 0.744 — the Opponent Costs 25.6 % of Throughput, A Dead Opponent Is Invisible in the Observation, Kaggle Probe #1 — What the Real Grading Machine Measures, Error Ladder — What the Harness Survives, Machine Drift of ±25 % across a Season, Memory Ceiling — 3.86 GiB Peak RSS of a 6.5 GiB Host (+15 more)

### Community 60 - "test_bounds_day.py"
Cohesion: 0.11
Nodes (22): _one_tile_day(), The pool a day provably cannot use: refused with a no, not with a search.…, Two corners of the board: 8 out of each nearest door, 16 in all. A worker may…, Forty tasks on the shed's tile, three hands hired at hour 20: 24, 28, 32, 36…, The planner's offer caps the pool: a hand nobody pays for is not a hand the day…, Every unit does at least one task, so a pool larger than the work is never the…, The no is an answer about the day, so it carries no route and claims no time…, A bound says the fewest a day might need. A range that reaches it is a question… (+14 more)

### Community 61 - "test_mixed_day.py"
Cohesion: 0.10
Nodes (22): _board(), played(), The day the caller keeps is the day the search priced - asserted on the…, test_the_partial_route_replays_as_the_day_it_was_priced_as(), fixture, The mixed day: three species on the shed's column, wheat across the rest of the…, The quadrant as `{(x, y): tile record}`, for tiles that hold something., The searched day, compiled and replayed once - the harness run is the expensive… (+14 more)

### Community 62 - "prices.py"
Cohesion: 0.12
Nodes (20): expected_price_curve(), Price for each candidate quantity we sell, `turns_ahead` turns out., drain_per_day(), price(), price_table(), price_vec(), prices(), ndarray (+12 more)

### Community 63 - "Evaluation arena — sparring opponents run in our own interpreter"
Cohesion: 0.26
Nodes (22): Audit finding — compile() at line 20; payload audited separately, Audit — clean: every import and call is within the allowlist, Audit finding — compile() at line 29; payload audited separately, Evaluation arena — sparring opponents run in our own interpreter, Rationale: the arena runs vendored code inside our interpreter, so its imports and calls are checked before it is ever called, Audit — clean: every import and call is within the allowlist, Rationale: the extractor parses the notebook with ast and never executes it, Audit finding — compile() at line 29; payload audited separately (+14 more)

### Community 64 - "land.py"
Cohesion: 0.14
Nodes (15): best_land(), Candidate, candidates(), LandPlanner, LandResult, prefix_cost(), The land-purchase outer loop (issue #13 §3). `BUY_LAND` is **prefix-locked**:…, When to spend a master solve on land (the issue's cadence cap). One evaluation… (+7 more)

### Community 65 - "test_plan_supply_iteration.py"
Cohesion: 0.14
Nodes (18): plan_supply_sells(), plan_supply_sells_from_pool(), ndarray, #110's own-supply half: the plan's projected sells re-enter the price path its…, The chosen mix's projected sells as `forecast(our_sells=)` reads. `produce`:…, Same, from the master's pool (columns carry `produce`; idle ones carry None and…, _produce(), ndarray (+10 more)

### Community 66 - "graph.py"
Cohesion: 0.12
Nodes (17): chain_name(), Stable name of a chain ('FERTILIZE+WATER+HARVEST', 'NO_ACTION')., BuildReport, ChainOutcome, _engine_part(), tile_dp graph — the artifact and its reader (v16). The graph itself: ONE tile…, The engine facts of a tag, without the build/registry fingerprints. The…, Build metadata kept beside the artifact (no `life_days`: a horizon guess the… (+9 more)

### Community 67 - "Action"
Cohesion: 0.15
Nodes (14): Action, MarketAction, _op_name(), parse_action(), Any, The engine's own list (or a tuple in the same shape) as a `WorkerAction`., One order the market executes: the worker never runs it., The engine's own list (or a tuple in the same shape) as a `MarketAction`. (+6 more)

### Community 68 - "_process_market"
Cohesion: 0.11
Nodes (19): _shed_access_tiles, kaggriculture.py::_spawn_hand:533, _do_buy_land(), _do_hire(), _hire_cost() and _fib(), LAND_ORDER and LAND_PRICES, MARKET_PARAMS, market_price() (+11 more)

### Community 69 - "test_agent_runtime.py"
Cohesion: 0.12
Nodes (19): io, _obs(), The spine's guards: the entry point, the dispatcher, the manager's clock, and…, The DEFAULT is a temporary owner setting, and the guard says which. It is OFF…, A failure on any turn of the day is recorded, not swallowed., Hour h gives every unit its h-th op; exhausted units PASS., More than 10 market orders never leave the agent (F031 cap)., F031: hires can fail silently, so the OBS's hand count is authoritative. (+11 more)

### Community 70 - "runner.py"
Cohesion: 0.14
Nodes (19): The canonical slugs, sorted (16 distinct agents). The vendored tree holds 19…, slugs(), offline_lab.pool sweep: the characterisation sweep driver (issue #20 part 2).…, One slug-vs-slug episode per slug (A2 pre-filter, ~30 s each)., run_selfplay(), selfplay_prefilter(), _episode_worker(), _freeze() (+11 more)

### Community 71 - "_FakeManager"
Cohesion: 0.13
Nodes (18): _FakeManager, One observe per day, one step per remaining turn - never the other way., Every step gets a positive budget no larger than the working second., The plan the manager holds is sliced by hour, not re-decided., A raising manager: PASS, a recorded failure, its day marked, a log line.…, One bad turn is one bad turn: the next turn still calls the manager., 719 turns of the spine's own loop: 30 observes, 689 steps, all legal., One reading per turn after the first: the season's own gap count. (+10 more)

### Community 72 - "test_colgen.py"
Cohesion: 0.16
Nodes (20): _idle(), _pricer(), Column generation: does it combine plans, and does it stop on a proof? The…, One class, many tiles: the answer is a MIX, not n copies of one plan., The whole point: same LP, different path, same number. Lesson 1.9's own…, `certified` may only be true when a pricing round found nothing to add., Two tiles of one state at different distances are different classes. The…, The nastiest failure in lesson 1.9 §4.2, made visible. A flipped `mu` does not… (+12 more)

### Community 73 - "test_tile_dp_contractor.py"
Cohesion: 0.13
Nodes (18): _enumerate_best(), ndarray, TileContractor tests (issue #11): the sweep, R006, the calendars, the plan.…, Every edge reward is zero, so every value is — a total indexing check. The…, The recurrence written out: one state at a time, plain Python loops.…, Bit-for-bit: the four array ops are the triple loop, exactly., Exhaustive search over every edge sequence of `days` days., Every path of a 6-day horizon, against the DP's chosen chains. (+10 more)

### Community 74 - "oracle_transitions.py"
Cohesion: 0.15
Nodes (18): is_animal(), Whether `entity` is one of the three species., tile_dp: the tile lifecycle graph, and the contractor that prices it. `graph`…, _covers(), _edges_from(), main(), _op_gated_by_canon(), op_vocab() (+10 more)

### Community 75 - "ndarray"
Cohesion: 0.15
Nodes (11): _check_non_negative(), ndarray, The graph's edge costs with `travel_hours` charged on every worked day. The…, A dual vector -> `(days, N_RESOURCE)` float32, R006-checked. Accepts either the…, Backward sweep: `(days+1, n_states)` values and per-day rewards., `EP @ p[d] - EC @ w[d]` for every day: the sweep's distance-free half. The walk…, The backward sweep for one distance, given `_base_rewards`., The day loop: every state takes the best of its own out-edges. (+3 more)

### Community 76 - "25-27-strict-future-v27-midgame-meta-reset/agent.py"
Cohesion: 0.29
Nodes (19): agent(), _align_hands(), _copy_action(), _demand_per_day(), _farm(), _get(), _impact_score(), _is_sell() (+11 more)

### Community 77 - "base64"
Cohesion: 0.17
Nodes (16): base64, copy, agent(), Exact public replay tape: episode 90558188, seat 0., _safe_terminal_action(), agent(), _apply_market_delta(), _call() (+8 more)

### Community 78 - "task_bn_scipy"
Cohesion: 0.12
Nodes (18): bn_analytic_mu(), bn_data(), bn_neg_logpost(), bn_pack(), bn_x0(), _chunk_scipy(), _data(), _lgamma() (+10 more)

### Community 79 - "44-46-strict-future-top-30-v22-price-impact/agent.py"
Cohesion: 0.27
Nodes (18): get(), One registry row (KeyError for an unknown slug - loud, R005)., agent(), _align_hands(), _copy_action(), _farm(), _get(), _impact_score() (+10 more)

### Community 80 - "world/fast_sim (FastSim.run)"
Cohesion: 0.12
Nodes (18): bench.bench_paths reproduction, world/fast_sim (FastSim.run), Direct kaggriculture.interpreter() call, structify state clone, tests/test_world_parity.py bit-identical parity, dev mode (validators run), world.FastSim(validate=dev|fast), Validation configuration switch (+10 more)

### Community 81 - "emit"
Cohesion: 0.15
Nodes (17): agent(), _arm_ladder(), _contended_steps(), _contention_verdict(), emit(), _farm_snapshot(), _opp_digest(), _opp_snapshot_steps() (+9 more)

### Community 82 - "v111-8c4s-economic-core-premium-lead/agent.py"
Cohesion: 0.31
Nodes (17): agent(), _align_hands(), _copy_action(), _existing_sell(), _farm(), _fr_state(), _front_run(), _future_quantity() (+9 more)

### Community 83 - "v16-rc5-high-score-8c-4s-premium-market-lead/agent.py"
Cohesion: 0.31
Nodes (17): agent(), _align_hands(), _copy_action(), _existing_sell(), _farm(), _fr_state(), _front_run(), _future_quantity() (+9 more)

### Community 84 - "_graph"
Cohesion: 0.14
Nodes (18): _flat(), _graph(), _owned(), F029: the season ends with no liquidation, so V[days] ≡ 0., `reduceat` prices an empty group as the element at that index., F009's honest maxima, read off the graph by the DP itself. The horizon admits…, F014's production nights and F027's ready days, from the graph alone. The tile…, No owned tile: no column, no plan, and a defined (zero) signal. (+10 more)

### Community 85 - "Runtime"
Cohesion: 0.16
Nodes (12): _day_of(), _hour_of(), _overage_of(), Any, Exception, The turn's clock, the manager, and the never-raise boundary.…, The harness's per-turn call: one legal action dict, never raising., What is left of this turn's working budget, for the manager's step. `step`… (+4 more)

### Community 86 - "agent"
Cohesion: 0.15
Nodes (17): agent(), _arm_window_summary(), _boot_env(), _burn(), _do_drain(), _drain_due(), _drain_seconds_for_day(), _predicted_cutoff() (+9 more)

### Community 87 - "v3-agent/agent.py"
Cohesion: 0.32
Nodes (16): agent(), _align_hands(), _copy_action(), _farm(), _get(), _impact_score(), _impact_slots(), _is_sell() (+8 more)

### Community 88 - "test_drop_empties_bag_day.py"
Cohesion: 0.15
Nodes (16): _board_sim(), _play(), played(), fixture, A drop hands over the WHOLE bag: the feed a worker still carries goes with it.…, The engine's own flag: an animal the route fed is fed., The compiled ops, read the way the engine reads them: a DROP empties the bag, a…, The search charges the walk back through a door, so the day it calls carried… (+8 more)

### Community 89 - "probe_hidden_state.py"
Cohesion: 0.21
Nodes (15): _continuation_diffs(), _decoder_keys(), _futures_differ(), main(), _one_day(), _print_field_split(), Hidden-state probe (issue #24, class C): is a modelled state hiding a…, Follow the successors with idle days; a difference that shows up later is still… (+7 more)

### Community 90 - "test_contractor_concepts.py"
Cohesion: 0.27
Nodes (15): chains(), duals(), one_bare_tile(), produced_per_day(), fixture, Conceptual tests: does the DP put its work where the value is? The cost side is…, `p` (what a unit of a product is worth) and `w` (what a unit costs) per day., The graph's root: the bare tile every plan starts from. (+7 more)

### Community 91 - "agent"
Cohesion: 0.21
Nodes (10): agent(), The arena's entry point for the slot circuit's sell plan (#65). Same spine as…, The arena's entry point for the market layer in `dump` mode (issues #15, #18).…, The arena's entry point for the market layer in `spread` mode (issues #15,…, Chista agent entry point. The harness calls a bare function per turn; all state…, Harness entry point: one legal action dict per turn, never raises., The coordinated entry point for the arena (issue #12 ship gate). Same spine as…, bench() (+2 more)

### Community 92 - "generate"
Cohesion: 0.14
Nodes (13): ColgenResult, generate(), lagrangian_bound(), MasterSolve, PricingDuals, NamedTuple, The loop: master over every column so far, price, add, repeat. `price(y, cash)`…, The reduced-cost tolerance for a board whose LP objective is this. One… (+5 more)

### Community 93 - "planner — Rounding, Repair, Land (issue #13)"
Cohesion: 0.14
Nodes (15): assign_tiles — λ to One Plan per Tile, ClassMix — One Class, Its Plans, Their λ, demote_to_feasible — the Row-Walk Repair, planner — Rounding, Repair, Land (issue #13), Land Stays Out of the LP, LOCKED Tiles Get Nothing, Plan — One DW Column, plan_from_board — a Plan from the Pricing Oracle (+7 more)

### Community 94 - "wsr/ — The Day Layer"
Cohesion: 0.16
Nodes (13): Beam Search — Day, Result, search(), compile_route and DayOps, wsr/ — The Day Layer, A Drop Is a Deadline, Not a Chain Op, expand_chain — a Chain as the Tasks It Is Made Of, A Hand Lands on the Least-Occupied Shed Door, Runtime Path — manager/core.py to planner/day.py to wsr/, The Rules the Search and the Compiler Share (+5 more)

### Community 95 - "_Path"
Cohesion: 0.20
Nodes (13): AgentLike, Environment, _default_html_path(), Any, Convenience: run a full episode, then write its HTML replay. Returns (env,…, Run a full episode with the real harness and return the Environment. `agents`…, Render a finished episode to an HTML file via env.render(mode="html"). Default…, Collision-free default replay name. Keyed on the seed and on a fingerprint of… (+5 more)

### Community 96 - "season_runner.py"
Cohesion: 0.14
Nodes (11): argparse, main(), _one(), rival(), Season measurement on FastSim, one subprocess per (rival, seed). Why…, Play one season in THIS process and return the result as a dict., concurrent_futures, tests/day_layer — faithful pytest-shaped port (F052) driven by tests/test_day.py (+3 more)

### Community 97 - "F058 — The grading platform, measured from inside a submission"
Cohesion: 0.24
Nodes (15): F046 — Runtime budget: one second and a 60-second bank, An exhausted bank forfeits (TIMEOUT, reward None), Budget against 0.965 s: the harness bills ~35 ms more than measured, 1 free second per turn, Overrunning draws max(0, duration - 1.0) from the bank, observation['remainingOverageTime'] is live and authoritative, 60-second bank for the whole episode, F058 — The grading platform, measured from inside a submission (+7 more)

### Community 98 - "Worker distribution and scheduling problem on a 10x10 grid"
Cohesion: 0.16
Nodes (15): Action format, Kaggriculture game overview, Observation fields, Wheat loop example agent, Ambiguities resolved by asking, never assuming, cell_state definition, Fibonacci worker cost, Least-congestion placement, NW->NE->SW->SE (+7 more)

### Community 99 - "CP-SAT formulation"
Cohesion: 0.14
Nodes (15): Brute-force cross-check independent of CP-SAT, Exact-match cache and warm start, chistawrs/ project structure and milestones, clct(s=23) global shed-return constraint, CP-SAT formulation, Deterministic precondition expansion, not open search, major_task (named chain of minors), Seven major_task recipes (+7 more)

### Community 100 - "kaggle_probe-2.py"
Cohesion: 0.18
Nodes (13): _arm_report(), _arm_schedule(), _bind_seat(), _child_env_read(), _new_seat_state(), _pct(), _pymc_child_paths(), Kaggriculture — PROBE #2: Bayesian inference, the time bank, and the sandbox… (+5 more)

### Community 101 - "test_builder_fingerprint.py"
Cohesion: 0.13
Nodes (14): offline_lab_build_fingerprint, Did the build recipe move without the artifact being rebuilt?…, A partial checkout must not produce a digest that merely looks like one.…, `stamp()` is the builder's own call, so the guard reads the same shape the…, The one that matters: stored fingerprint == the recipe's fingerprint now. A…, A bare hash ages badly: a year later nobody knows what it covered. The stamp…, The guard's whole premise, exercised: a changed recipe is a changed hash. Built…, Paths are hashed with the bytes, so the same text in a different file is a… (+6 more)

### Community 102 - "test_animal_drop_day.py"
Cohesion: 0.20
Nodes (14): _build(), played(), fixture, The animals' day: FEED, CARE and COLLECT_FERTILIZER, with the fertilizer banked…, The whole day is carried with no hands at all - and it is the portfolio that…, One more hand than the day can use alone is what banks the third drop before…, The same day written in a different order is the same day: these three ops have…, The day carried by the two hands it needs. (+6 more)

### Community 103 - "test_rival_hours.py"
Cohesion: 0.18
Nodes (13): _manager(), Guards for the manager's DATED rival supply (#16). `_rival_supply` is the…, An empty calendar gates the model out: no phantom rival supply. The model's…, What the tracker inferred is dated by the turn it happened in., No source of hours means no dated input: the daily curve is used., `never_raise` decides, exactly as it does at the turn boundary. Off (the…, The trained model covers the horizon; dropping its branch empties this., test_a_broken_tracker_spills_unless_the_run_asked_to_degrade() (+5 more)

### Community 104 - "test_world_branch_purity.py"
Cohesion: 0.25
Nodes (13): _fresh(), _pairs(), _play(), Branch purity and the RNG-from-state invariant for world.fast_sim. Run:…, The convenience wrapper must be exactly clone()+step(), and side-effect free., Apply actions in order, stopping at episode end (like FastSim.run)., Full state signature: status, reward and every observation field., A clone continued with a suffix == a from-scratch replay of prefix+suffix. This… (+5 more)

### Community 105 - "ShedState"
Cohesion: 0.14
Nodes (8): _order_of_preference(), What the market will actually quote: shed items that are PRODUCTS, with a…, The shed, the bags, and what tonight's drop will do to them., Units in the shed — the only stock a SELL can reach (F043)., Units in unit bags; the nightly drop empties these into the shed., F043's second half: at the cap, BUY_PRODUCT/BUY_ANIMAL are refused., Units the night drop will DESTROY tonight. `extra_carried` is what the day's…, ShedState

### Community 106 - "price_board"
Cohesion: 0.18
Nodes (14): price_board(), One pricing call: the sweep plus the recovered columns. Builds a…, _integer_duals(), Raising a price cannot lower any value; lowering a wage cannot raise it. Sign…, A negative price makes the pruned graph silently wrong: it must raise., Same rule on the wage side, and per-day vectors are checked too., Same graph, same duals, same plan — twice, byte-identical., The walk's edge is the argmax of `reward + V_next` at every step. (+6 more)

### Community 107 - "Agent Guide — ChistaAgent"
Cohesion: 0.26
Nodes (13): Engine Authority — kaggle_environments.envs.kaggriculture, How to Prove It — Play the Day on the Engine, Agent-Commit Trailer Rule, Agent Guide — ChistaAgent, Games Run on FastSim — Mandatory, Findings & Rules File Convention, offline_lab.fast_sim — FastSim, offline_lab.kaggle_env — Official Harness Path (+5 more)

### Community 108 - "bench_turn_budget.py"
Cohesion: 0.15
Nodes (7): bench_sweep(), _Contender, Bench the turn budget: a full 720-turn episode, per-turn timing. Run from the…, The tile-DP sweep: solo and contended, direction asserted. Only the SWEEP is…, A competing thread: an opponent deliberating in the same turn. numpy releases…, statistics, threading

### Community 109 - "test_contractor_concepts_decay.py"
Cohesion: 0.23
Nodes (10): offline_lab_build_graph, board(), chains(), days_producing(), dose_cost(), milk_price(), fixture, Scenario 3: a decaying milk price, a constant carrot price, and a dose that… (+2 more)

### Community 110 - "adaptive-public-state-multi-route/agent.py"
Cohesion: 0.27
Nodes (10): agent(), _apply_market_delta(), _call(), _emr_public_counts(), kaggle_agent(), _kaggle_submission_entrypoint(), V21-R1 public-state multi-route Kaggriculture agent., _seat() (+2 more)

### Community 111 - "test_budget_day.py"
Cohesion: 0.21
Nodes (11): _day(), _day_for(), The budget's contract: a bigger budget never places less, and a route says…, The property the branch claimed: more budget is never a worse route. Every…, A search cut short still names every hand's door, so a caller holding it can…, A route can be complete and still have crossed its deadline; it has no work…, The rule the pool loop and the halving share: carried first, then more work., test_a_bigger_budget_never_places_less() (+3 more)

### Community 112 - "_quiet"
Cohesion: 0.15
Nodes (12): Any, _quiet(), Both arms of `Config.never_raise`, on the same failing turn. ON: all-PASS, the…, The `D` line: one per day, at hour 0, carrying the manager's own numbers., A turn over the working budget gets an `A` line even when nothing raised., `Config.log_gaps`: off by default, and it prints the running stats. The quiet…, Run `fn`, returning (stdout, result) - the log is part of the contract., test_an_over_budget_turn_is_logged_as_an_anomaly() (+4 more)

### Community 113 - ".solve"
Cohesion: 0.18
Nodes (7): _extended_basis(), _lost_sale_cost(), col_sell(), col_sell_deep(), The previous basis, with the new columns nonbasic at their lower bound. A…, The restricted master over the pool, with the SHED as a stock. Variables:…, What throwing one unit of shed stock away on day `d` costs. The owner's rule:…

### Community 114 - "Care bank (pending_care_bonus) accrual and payout"
Cohesion: 0.21
Nodes (12): GOOSE (cost 300, COOP, day 4, daily, cap 4, EGG), F017 — Two consecutive unfed days escape, Two consecutive unfed days cause escape, F018 — Unfed animals still produce base, Base production outside the fed_today test, F019 — Care bank accrual and payout, Care bank (pending_care_bonus) accrual and payout, F020 — Unfed production night destroys the bank (+4 more)

### Community 115 - "A guard must be seen to fail"
Cohesion: 0.17
Nodes (12): ARCHITECTURE.md system-shape pointer, Tests and benchmarks index, CROPS identity assertion (rules.CROPS is K.CROPS), Call-site mismatch defeats reading, Five guards that passed while guarding nothing, Break it, watch it fail, restore, say so, A guard must be seen to fail, Apache License 2.0 (verbatim copy) (+4 more)

### Community 116 - "emit"
Cohesion: 0.23
Nodes (12): _chunk_jax(), cpu_ratio(), emit(), _maybe_bn_compare(), _poll_pymc_child(), pymc_worker(), One JSON line per measurement. Flushed, because the episode may end., Run fn until >= min_seconds; return (wall, cpu-seconds per wall-second, reps). (+4 more)

### Community 117 - "test_planner_hands_day.py"
Cohesion: 0.17
Nodes (7): The hands the planner offers are the hands the day layer searches with - and…, The hours are an INPUT: the hourly secretary's timetable, or the bound.…, `Day` derives nothing: the tuple IS the day's labour. One source. Its length is…, Offer 5, use 1: wsr's own number, and no second search to find it. The search…, test_a_complete_day_reports_the_hands_it_needs_not_the_offer(), test_the_day_takes_the_secretarys_hours_when_it_has_them(), test_the_hours_are_the_callers_and_the_tuple_is_the_count()

### Community 118 - "test_dual_bins.py"
Cohesion: 0.23
Nodes (10): _model(), Guards for the dual goods' buy bin (#65 follow-up): a dual good's last bin…, The artifact's (WHEAT, day=1, bucket=0, act=1) row is 99% buy counts;…, `expected_sell` and `expected_buy` must not be the same query., MILK's bins are all sales; expected_buy is 0 by construction., Unit shape: the router zeroes exactly the bins it says it does., test_a_dual_goods_buy_bin_is_not_a_sell_volume(), test_a_sell_only_good_has_no_buy_side() (+2 more)

### Community 119 - "_fingerprint"
Cohesion: 0.20
Nodes (8): _fingerprint(), fingerprint_chains(), TileChain, The registry fingerprint of a chain list: ids are positions, so this is what an…, The ONE fingerprint of a chain table, used by the builder and by the loader.…, Write the artifact: CSR arrays, cost/produce matrices, metadata., The stamp of the chain table THIS graph indexes into. The builder writes the…, The engine tag with the registry part re-stamped (see `_registry_tag`).

### Community 120 - "expand_chain"
Cohesion: 0.18
Nodes (10): expand_chain(), Expansion, MinorTask, Item, NamedTuple, One chain, expanded into the tasks the search reads. `tasks` is every op of the…, One day's chain -> the tasks the search reads, the order between them, and the…, parametrize (+2 more)

### Community 121 - "_timing_block"
Cohesion: 0.18
Nodes (11): _direction_verdict(), Serial timing run: one process, per-turn wall time for `slug`., A4's direction rule for one version's two readings (self-review fix). Returns…, The two A4 readings, plus the direction assertion. - timing_solo: vs PASS - the…, The opponent with the highest observed guard self-p95 this run., _slowest_opponent(), _timed_episode(), _timed_episode_worker_code() (+3 more)

### Community 122 - "ReplayAgent"
Cohesion: 0.18
Nodes (5): Plays a recorded episode; callable exactly like any policy. Seat-agnostic: the…, Accept the harness's optional second argument (configuration). The real harness…, Turns answered from the record so far., Turns answered with PASS because the record had no entry., ReplayAgent

### Community 123 - "v16-rc5-high-score-8c-4s-premium-market-lead — vendored opponent agent (not our code)"
Cohesion: 0.20
Nodes (11): writefile packing — economics-driven-rule-agent-ecobot-v6.ipynb extracted 1,796 lines / 69,074 bytes, writefile packing — %%writefile cell body, v111-8c4s-economic-core-premium-lead — vendored opponent agent (not our code), v111-8c4s-economic-core-premium-lead.ipynb (source notebook), writefile packing — v111-8c4s-economic-core-premium-lead.ipynb extracted 247 lines / 18,946 bytes, v16-rc5-high-score-8c-4s-premium-market-lead — vendored opponent agent (not our code), v16-rc5-high-score-8c-4s-premium-market-lead.ipynb (source notebook), writefile packing — v16-rc5-high-score-8c-4s-premium-market-lead.ipynb extracted 247 lines / 18,946 bytes (+3 more)

### Community 124 - "check_prune_invariance.py"
Cohesion: 0.27
Nodes (8): _edges(), main(), Prune-invariance check (issue #24, class B): does dominance ever discard an…, Build the graph with the dominance pass disabled (no-op sweep kept)., (to_id, cost, produce, chain_id) for every edge leaving `sid`., One random NON-NEGATIVE (days, N_RESOURCE) price vector and wage vector., _unpruned_build(), _vectors()

### Community 125 - "F049 — The tile optimum is a cycle, not a crop"
Cohesion: 0.27
Nodes (10): Atomic Seed Check — One Over-Request Drops Every PLANT, F001 — Seeds Bypass the Shed, Seeds Live in private["seeds"], Not the Shed, F007 — One-Shot Harvest Frees the Tile, HARVEST Removes the Plant and Frees the Tile the Same Day, F049 — The tile optimum is a cycle, not a crop, Columns are priced as rotations, not crops, F009's honest single-cycle maximum is not a season budget (+2 more)

### Community 126 - "Ongoing ready days (TOMATO 8-11, STRAWBERRY 10-16)"
Cohesion: 0.22
Nodes (10): F014 — Ongoing measured calendars, STRAWBERRY measured yield calendar, TOMATO measured yield calendar, F015 — Ongoing harvest leaves the plant, Ongoing HARVEST leaves the plant standing, F023 — Animal fertilizer every night, fertilizer_available set every night, one collect per animal per day, F027 — Ongoing harvest calendar (+2 more)

### Community 127 - "F031 — Order cap is per turn, queue walked to completion"
Cohesion: 0.27
Nodes (10): Turn loop: every unit acts, then the market runs, then the day refresh, F030 — Action shape and unit-before-market ordering, Action shape {farmer, hands, market}, Units act before that turn's market, F031 — Order cap is per turn, queue walked to completion, HIRE and BUY_LAND settle atomically before the per-unit loop, maxMarketOrdersPerTurn = 10, per turn, Orders walked by index, each to completion (+2 more)

### Community 128 - "blas_threads"
Cohesion: 0.22
Nodes (7): blas_threads, Context manager that limits BLAS threads, or admits it could not., _read(), task_machine_census(), task_memory(), _thread_mechanism(), object

### Community 129 - "cpu_ratio"
Cohesion: 0.33
Nodes (10): cpu_ratio(), _matmul_bench(), run(), _measure_window(), Run fn repeatedly for >= min_seconds; return (wall, cpu/wall, reps). cpu/wall…, THE question: how many cores does this container really give a single process,…, Seat 0: the reference bench, labelled by whether the opponent is busy., task_cpu_capacity() (+2 more)

### Community 130 - "_patch_phase"
Cohesion: 0.24
Nodes (10): _install_intercept(), wrapper(), _mutate_now(), _patch_phase(), _patch_target(), A live env instance, if this process has one., The only proof that matters: does OUR OWN observation show the change?, One direct attempt to change the game, plus arming the intercept. (+2 more)

### Community 131 - "_column"
Cohesion: 0.24
Nodes (10): _column(), _direct(), _plans(), price(), `mu` is an EQUALITY marginal and is free in sign; clamping breaks the test., Three plans a class may run, as (hours per day, spend, earn, revenue).…, The same LP over EVERY column, solved in one go., `cash_rows` collapses a per-(column, day) loop into two `cumsum`s. The loop it… (+2 more)

### Community 132 - "ndarray"
Cohesion: 0.22
Nodes (9): cash_relief(), cash_rows(), ndarray, The cumulative cash rows: `(days, len(pool))`, one row per day. Column `j`'s…, `rc_c` for each class: how much its best plan beats what the master pays. The…, `later[d]`: the shadow price of a coin EARNED on day d. The cash rows are…, The σ a degenerate LP cannot supply: the market path, by item. `prices` is the…, reduced_costs() (+1 more)

### Community 133 - "F048 — The season has 720 states and 719 decisions"
Cohesion: 0.36
Nodes (9): F048 — The season has 720 states and 719 decisions, 720 states, 719 decisions, at steps 0..718, Day 29 gets 23 decisions, hours 0..22, End-of-season liquidation, EPISODE_STATES - 1 is the last state, -2 the last decision, A chain of reasoning over a measured number is not itself measured, An observation is the state before that turn is processed, The decision at step 718 is processed in full (+1 more)

### Community 134 - "registry.py"
Cohesion: 0.25
Nodes (7): canonical(), canonical_slugs(), dev_and_heldout(), The pool registry: one row per competitor, loaded as data. Issue #20 brief §4.…, The canonical slug for an alias (identity for non-duplicates)., The 16 distinct slugs (aliases excluded)., The 11 dev / 5 held-out split over the canonical 16 (brief §4, recounted after…

### Community 135 - "test_land_image_day.py"
Cohesion: 0.28
Nodes (8): The land image is a view of the arrays, and the depth layer says how long a…, Every layer is a scatter of a column, so the totals have to agree with the…, A PLANT then a WATER is a chain of two, and a second WATER beside them is not a…, A chain runs on one tile, so its depth cannot be more than that tile's own task…, _tasks(), test_a_chain_runs_as_deep_as_its_precedence(), test_the_depth_never_exceeds_the_tasks_on_its_tile(), test_the_image_says_what_the_arrays_say()

### Community 136 - "test_spare_day.py"
Cohesion: 0.28
Nodes (8): _day(), The spare capacity: the turns the route leaves, and the walks come off it. A…, One task eight tiles away: the day has 24 turns, the task takes one and the…, The number a manager acts on has to move the right way when the day gets more…, The spare is counted against the hands the pool paid for, not against the offer., test_an_unhired_hand_is_not_capacity(), test_more_work_leaves_less_spare(), test_the_walk_is_not_spare()

### Community 137 - "solve_master"
Cohesion: 0.29
Nodes (6): MasterLP, The master's LP, kept across a day's rounds so its basis carries. max Σ…, One LP, cold: the master's own path is the `MasterLP` in `generate`., solve_master(), The same matrices, the same optimum, and the same three duals. `MasterLP`…, test_the_master_LP_agrees_with_scipy_linprog_and_with_a_cold_solve()

### Community 138 - "F055 — SELL reads the shed, not the bag"
Cohesion: 0.36
Nodes (8): Belief's per-hour SELL queue, or no rows if it cannot build one. Called through…, The observation with only PRODUCTS in the shed — a guard for #77.…, sell_rows(), _sellable_obs(), F055 — SELL reads the shed, not the bag, The nightly drop is a choice, not a forced one-day lag, SELL reads the shed, never a unit's bag, The engine runs unit actions first, then _process_market

### Community 139 - "GapStats"
Cohesion: 0.25
Nodes (5): GapStats, Sample sd; 0.0 before there is a second reading., Running mean and sd of the wall clock between two calls of the agent. The…, Welford's running mean and sd, against `statistics` on the same numbers.…, test_the_gap_stats_match_a_direct_computation()

### Community 140 - "Chista — the System's Shape, Its One Vocabulary, and Its One Belief"
Cohesion: 0.43
Nodes (8): world/model.py — the Names, One Vocabulary, One Belief, R004 — Validation Is a Config Switch, The Canonical Vocabulary — world/model.py, compile_chain — One Expansion, Chista — the System's Shape, Its One Vocabulary, and Its One Belief, MarketState — the One Belief, Migration — One Branch, Then One PR

### Community 141 - "_kind_change_actions"
Cohesion: 0.29
Nodes (7): delta(), _kind_change_actions(), The same tile part-way through a day: the hour, and today's facts. Everything…, One line: the day-start state, plus what has happened today., The actions consistent with one hour of change on one tile. `later` and…, What the kind change alone allows: the constructive and destructive ops. DIG…, TileInDay

### Community 142 - "Held cap is its own deadline"
Cohesion: 0.29
Nodes (8): COW (cost 400, PASTURE, day 8, every 2 days, cap 6, MILK), F021 — Care bank lumps before first yield, F026 — One-shot harvest calendar and deadline, Past max_yield_day the plant decays one unit per two turns, One-shot harvest: single yield, destroys the plant, F028 — Animal harvest calendar and cap deadline, Animal harvest calendar (no age guard on HARVEST), Held cap is its own deadline

### Community 143 - "The silent-failure class: the engine fails silently"
Cohesion: 0.43
Nodes (8): F043 — Shed capacity destroys the overflow, DROP moves the whole inventory and destroys the overflow, The nightly drop empties every inventory and destroys the overflow, SELL of an item the shed does not hold is refused, shedCapacity = 100 across all items together, F047 — The silent-operations catalog, Over-seeded PLANT drops every plant of that crop that turn, The silent-failure class: the engine fails silently

### Community 144 - "classify.py"
Cohesion: 0.29
Nodes (7): classify_from_sequences(), eq(), ClassVerdict, Class probe: P (open-loop) / S (self-reactive) / R (reactive), behaviourally.…, Pure comparison of four action sequences -> P / S / R., Exact equality of two action dicts (order-sensitive on lists)., _same_action()

### Community 145 - "greedy.py"
Cohesion: 0.38
Nodes (6): _first_plant_dict(), _greedy(), greedy_action(), M1 stub brain: a greedy per-tile policy, deliberately dumb. Exercises the spine…, The standing unit's tile, if it holds a plant., One legal action dict from the raw observation. Never raises by contract with…

### Community 146 - "world/ — The Definitions"
Cohesion: 0.29
Nodes (7): world/action.py — WorkerAction and MarketAction, world/action_rules.py — What Each Action Needs, world/board.py — Cells, Quadrants, Shed Doors, world/ — The Definitions, world/prices.py — the Price Function and the Town's Drain, world/tile.py — TileHourZero and TileInDay, world/worker.py — WorkerTrace

### Community 147 - "F029 — Season structure and no liquidation day"
Cohesion: 0.33
Nodes (7): F024 — Animals cannot be unplaced, can be lost in the shed, 100-unit shed overflow can destroy a held animal, Placed animals cannot be taken back (DIG returns early), F029 — Season structure and no liquidation day, F048 (referenced finding: final-day turn accounting), No liquidation day: end-of-season shed goods are worthless, Season structure: 720 turns, turnsPerDay 24 = 30 days

### Community 148 - "townShopSellInterval = 4 steps"
Cohesion: 0.43
Nodes (7): F035 — Prices rise through the season, Holding produce and selling late, Season price inflation: the town out-consumes a single farm, F037 — Town shop consumption cadence, townCenterSellInterval = 24 steps, townShopSellInterval = 4 steps, townShopUnlockInterval = 3 days, drawn with replacement

### Community 149 - "F053 — A refused op still bills the cost vector"
Cohesion: 0.52
Nodes (7): F053 — A refused op still bills the cost vector, chain_requirements derives cost from the chain's op list, Confirm the engine accepts each op before pricing a hand-built chain, Hand-built chains are the exposure, NOOP: refused and billed nothing is a separate number, A refused op still bills the cost vector (REFUSED_BUT_BILLED = 56), Shipped chains carry no phantom cost

### Community 150 - "Wrong-tile ops refused in silence (F047)"
Cohesion: 0.29
Nodes (7): tests/test_day_plan.py per-op instrument, HIRE settles after the turn's unit actions, day/routing.py::plan_day spawns from post-move occupancy, _process_market runs after unit actions, Wrong-tile ops refused in silence (F047), Missing step plays PASS, Hand a copy, never the live view

### Community 151 - "_apply_game_patch"
Cohesion: 0.33
Nodes (7): _apply_game_patch(), _bump_money(), _dig(), _farm_of(), _plant_melons(), Every empty tile of ours gets a MELON, using the env's own constructor. The…, Fired from the interceptor on every env step, but RESTRAINED: money once per…

### Community 152 - "kaggle-environments==1.32.7 — pinned EXACTLY"
Cohesion: 0.29
Nodes (7): Rationale: a version bump can change game behavior under every simulation at once, world/fast_sim.py drives the environment's own interpreter through private APIs, kaggle-environments==1.32.7 — pinned EXACTLY, ortools>=9.15, scipy>=1.17 — Kaggle environment and optimization core, Python 3.11 venv built with uv (uv venv .venv --python 3.11), Pin bumps ship in the same commit as tests/test_world_parity.py (R002/R003), test_installed_version_matches_the_requirements_pin()

### Community 153 - "test_own_supply_gap.py"
Cohesion: 0.38
Nodes (6): main(), _moving_ladder_coins(), The #110 own-supply finding, re-measured against the NEW master. The master's…, The engine's own answer: sell `units` one at a time, each quoted at the shared…, 21 melons at the flat path price vs the engine's moving ladder: the gap is the…, test_the_flat_price_overstates_the_moving_ladder()

### Community 154 - "Column"
Cohesion: 0.47
Nodes (6): Column, One class's plan over the horizon, as the master sees it., An index is positional: tomorrow's class 3 is not today's. A pool matched by…, test_a_carried_column_is_matched_by_class_KEY_not_by_index(), column(), price()

### Community 155 - "spawn_assignments"
Cohesion: 0.47
Nodes (6): Which tile each new unit enters through, given `(index, earliest_start)` per…, spawn_assignments(), F040 — Hand spawn placement and first-hour loss, A hand hired at hour 0 first acts at hour 1: 23 actions against 24, _spawn_hand uses the least-occupied shed-access tile, ties broken NWSE, Unit spawns nest: unit u stands identically for crew u or u+3

### Community 156 - "F016 — Animal placement and species table"
Cohesion: 0.33
Nodes (6): F016 — Animal placement and species table, Animal placement requires COOP or PASTURE, SHEEP (cost 500, PASTURE, day 6, every 3 days, cap 6, WOOL), Animal species table (GOOSE/COW/SHEEP), F025 — Cow cap asymmetry, COW cap asymmetry: cap 6 against a wait of 8

### Community 157 - "F054 — Town consumption is exactly modellable"
Cohesion: 0.53
Nodes (6): F054 — Town consumption is exactly modellable, Forward-forecast error is <=1.32 % of I0 at 10 days, The mean unlock policy is the default, Only the next shop unlock is random; the unlocked set is public, town_deltas(shops, step) reproduces every observed delta (6,462/6,462), R002 — Import rules from kaggle_environments

### Community 158 - "task_nn_numpy"
Cohesion: 0.40
Nodes (5): _nn_flops(), task_nn_jax_bench(), task_nn_numpy(), fwd(), task_nn_torch()

### Community 159 - "_chunk_pymc"
Cohesion: 0.33
Nodes (6): _chunk_pymc(), env_scan(), _is_env_like(), Is this a live environment object? The class is created dynamically…, PyMC doing inference work: the compiled model logp, evaluated flat out., _spec_present()

### Community 160 - "_run_a_full_day"
Cohesion: 0.33
Nodes (6): Fill the shed to 95 with 20 wheat in a bag, then play out the day.…, With the market layer on: nothing is destroyed, and the stock is sold., R007: the control leg, and the reason the guard exists at all., _run_a_full_day(), test_the_shed_guard_prevents_a_real_destruction_event(), test_without_the_market_layer_the_same_day_destroys_product()

### Community 161 - "Graded Episode Configuration"
Cohesion: 0.40
Nodes (5): Graded Episode Configuration, Graded Config Confirmed by Probe #2, Chista — the Avestan Deity of Wisdom, ChistaAgent, Kaggriculture — a Two-Player Farming Competition

### Community 162 - "_make_overrun"
Cohesion: 0.40
Nodes (4): _make_overrun(), _make_thread_child(), task(), One child per turn. The only thread sweep that always works. In-process…

### Community 163 - "BuildSpec"
Cohesion: 0.50
Nodes (3): BuildSpec, What a build asks for: the merged tile graph or one entity's graph., tile' for the merged graph, else 'crop' / 'animal'.

### Community 164 - "F038 — Money binds in the first week"
Cohesion: 0.83
Nodes (4): F038 — Money binds in the first week, The first week is where money binds, Every purchase refuses silently when the purse is short, startingMoney = 3,000

### Community 165 - "The n-th hire of a day costs fib(n): 1, 1, 2, 3, 5, 8, 13, 21, 34, 55"
Cohesion: 0.83
Nodes (4): F039 — Hire costs are Fibonacci, reset nightly, _do_hire returns silently when money < cost, The n-th hire of a day costs fib(n): 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, hires_today resets nightly and _daily_refresh clears the hands

### Community 166 - "Working a LOCKED tile spends hours as no-ops"
Cohesion: 0.83
Nodes (4): F042 — Land purchases: fixed prefix, LOCKED tiles, LAND_ORDER = [NE, SW, SE] at LAND_PRICES = [1000, 2000, 4000], Working a LOCKED tile spends hours as no-ops, Shed operations resolve before the LOCKED guard

### Community 167 - "Past about six hands the order queue is the constraint"
Cohesion: 1.00
Nodes (3): F041 — The order queue binds before the wage, Past about six hands the order queue is the constraint, The wage is never the constraint

### Community 168 - "F044 — Animal age collapses to a residue"
Cohesion: 1.00
Nodes (3): F044 — Animal age collapses to a residue, An animal's age matters only through a residue, The per-tile animal DP is flat over the residue

### Community 169 - "task_multiprocessing"
Cohesion: 0.67
Nodes (3): _mp_noop(), Does a second PROCESS buy throughput, or does the cgroup quota just split the…, task_multiprocessing()

## Ambiguous Edges - Review These
- `test_observation_mutation_is_guarded_in_dev()` → `R004 as cited by the opponents docs — hand the agent a copy of the observation, never the live view`  [AMBIGUOUS]
  opponents/adaptive-public-state-multi-route/SOURCE.md · relation: conceptually_related_to
- `R002 — Never Transcribe Game Rules` → `Crops and Animals — Every Rule, Numbered (AgriOracle research document)`  [AMBIGUOUS]
  AGENTS.md · relation: conceptually_related_to
- `agent.py payload, byte-for-byte — SHA-256 2fe7118964656263…` → `agent.py payload, byte-for-byte — SHA-256 12eb55e1e2455a2b…`  [AMBIGUOUS]
  opponents/farming-score-a-mathematical-approach/SOURCE.md · relation: semantically_similar_to
- `agent.py payload, byte-for-byte — SHA-256 aca6dcddafd6fd6b…` → `Import-time self-unpacking — exec_module triggers the payload unpack`  [AMBIGUOUS]
  opponents/economics-driven-rule-agent-ecobot-v6/SOURCE.md · relation: conceptually_related_to
- `agent.py payload, byte-for-byte — SHA-256 943e8c114ace4f5c…` → `Import-time self-unpacking — exec_module triggers the payload unpack`  [AMBIGUOUS]
  opponents/frontier-the-soil-remembers-rain/SOURCE.md · relation: conceptually_related_to
- `agent.py payload, byte-for-byte — SHA-256 0c3b4002c657f427…` → `Import-time self-unpacking — exec_module triggers the payload unpack`  [AMBIGUOUS]
  opponents/precomputed-schedule-policy/SOURCE.md · relation: conceptually_related_to

## Knowledge Gaps
- **124 isolated node(s):** `ClassVerdict`, `Findings & Rules File Convention`, `F017 — Two consecutive unfed days escape`, `F018 — Unfed animals still produce base`, `F019 — Care bank accrual and payout` (+119 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1696 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **12 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `test_observation_mutation_is_guarded_in_dev()` and `R004 as cited by the opponents docs — hand the agent a copy of the observation, never the live view`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `R002 — Never Transcribe Game Rules` and `Crops and Animals — Every Rule, Numbered (AgriOracle research document)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `agent.py payload, byte-for-byte — SHA-256 2fe7118964656263…` and `agent.py payload, byte-for-byte — SHA-256 12eb55e1e2455a2b…`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **What is the exact relationship between `agent.py payload, byte-for-byte — SHA-256 aca6dcddafd6fd6b…` and `Import-time self-unpacking — exec_module triggers the payload unpack`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `agent.py payload, byte-for-byte — SHA-256 943e8c114ace4f5c…` and `Import-time self-unpacking — exec_module triggers the payload unpack`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `agent.py payload, byte-for-byte — SHA-256 0c3b4002c657f427…` and `Import-time self-unpacking — exec_module triggers the payload unpack`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `FastSim` connect `FastSim` to `beam.py`, `TileGraph`, `forecast`, `numpy`, `pathlib`, `dispatch_plan`, `belief/__init__.py`, `F055 — SELL reads the shed, not the bag`, `replan.py`, `test_master.py`, `test_tile_dp.py`, `test_market_layer.py`, `test_slot_circuit.py`, `Any`, `test_market_wsr_check.py`, `check_route`, `test_world_parity.py`, `bench_paths.py`, `guarded_call`, `_sim`, `test_mixed_day.py`, `runner.py`, `test_drop_empties_bag_day.py`, `probe_hidden_state.py`, `season_runner.py`, `test_world_branch_purity.py`?**
  _High betweenness centrality (0.089) - this node is a cross-community bridge._