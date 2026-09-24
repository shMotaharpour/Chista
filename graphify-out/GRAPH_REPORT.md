# Graph Report - ChistaAgent  (2026-09-24)

## Corpus Check
- 293 files · ~333,799 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 5 file(s) not represented in the graph (top: .npz 2, (none) 1, .zip 1)

## Summary
- 4174 nodes · 9071 edges · 184 communities (170 shown, 14 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 458 edges (avg confidence: 0.86)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- colgen.py
- master.py
- Config
- equilibrate()
- rules.py
- economics-driven-rule-agent-ecobot-v6/agent.py
- belief/__init__.py
- pathlib
- chains.py
- forecast()
- planner/__init__.py
- TileState
- oracle_transitions.py
- dispatch_plan()
- replay_agent.py
- R004 as cited by the opponents docs — hand the a
- beam.py
- test_tile_dp_contractor.py
- v16-rc5-r5a-high-score-8c-4s-recovery/agent.py
- TaskArray
- findings-from-zero-to-top-meta/agent.py
- v20-adaptive-r1-multi-route-agent/agent.py
- Day
- belief/market.py
- fast_sim.py
- TileContractor
- model.py
- extract.py
- schemas.py
- test_market_layer.py
- bench_market_forecast.py
- evaluate.py
- sell_coins()
- test_slot_circuit.py
- test_layering.py
- test_evaluate.py
- Crops and Animals — Every Rule, Numbered (AgriOr
- test_market_wsr_check.py
- OpponentModel
- test_belief_depth.py
- check_route()
- interpreter()
- kaggle_probe-1.py
- test_pool_loader.py
- MarketTracker
- plan_sales()
- adaptive-replay-agent/agent.py
- repair.py
- docs/INDEX.md one-line-per-file index
- json
- Kaggle Probe #1 — What the Real Grading Machine 
- test_tile_dp.py
- v13-r3-top-meta-order-safe-premium-control/agent
- market_queue()
- MarketLayer
- bench_paths.py
- zlib
- test_agent_obs.py
- _decode_chain()
- Live observation views in fast mode
- test_world_parity.py
- test_bounds_day.py
- test_season_horizon.py
- artifact/__init__.py
- _Path
- wsr/ — The Day Layer
- Evaluation arena — sparring opponents run in our
- Action
- Cell
- expand_chain()
- _process_market()
- test_agent_runtime.py
- test_market_hourly.py
- test_tile_dp_decline.py
- runner.py
- _FakeManager
- numpy
- ndarray
- 25-27-strict-future-v27-midgame-meta-reset/agent
- Agent Guide — ChistaAgent
- probe_hidden_state.py
- test_planner_hands_day.py
- base64
- task_bn_scipy()
- FastSim
- 44-46-strict-future-top-30-v22-price-impact/agen
- TileHourZero
- test_real_days.py
- world/fast_sim (FastSim.run)
- emit()
- guarded_call()
- v111-8c4s-economic-core-premium-lead/agent.py
- v16-rc5-high-score-8c-4s-premium-market-lead/age
- test_contractor_concepts.py
- drain_forecast()
- Runtime
- WorkerAction
- agent()
- v3-agent/agent.py
- test_animal_drop_day.py
- test_drop_empties_bag_day.py
- test_mixed_second_day.py
- test_rival_hours.py
- plan_coins()
- agent()
- Worker distribution and scheduling problem on a 
- CP-SAT formulation
- kaggle_probe-2.py
- _pick_opponents()
- test_drop_door_on_route_day.py
- test_world_branch_purity.py
- test_contractor_concepts_decay.py
- ShedState
- replan.py
- world/ — The Definitions
- decode_world()
- WorldView
- F055 — SELL reads the shed, not the bag
- test_contractor_carrot_cycle.py
- bench_turn_budget.py
- adaptive-public-state-multi-route/agent.py
- test_budget_day.py
- _day()
- _quiet()
- compile()
- planner — Rounding, Repair, Land (issue #13)
- _select()
- Care bank (pending_care_bonus) accrual and payou
- A guard must be seen to fail
- emit()
- test_dual_bins.py
- _shipped()
- frontier-the-soil-remembers-rain/agent.py
- test_door_work_before_load_day.py
- Ongoing ready days (TOMATO 8-11, STRAWBERRY 10-1
- F031 — Order cap is per turn, queue walked to co
- blas_threads
- cpu_ratio()
- _patch_phase()
- test_after_drop_not_loaded_day.py
- test_seed_invariance_no_rng_in_tile_transitions(
- test_land_image_day.py
- test_spare_day.py
- GapStats
- spawn_cell()
- Held cap is its own deadline
- classify.py
- test_residual_wire.py
- _get()
- Edge
- .decode()
- F029 — Season structure and no liquidation day
- townShopSellInterval = 4 steps
- Wrong-tile ops refused in silence (F047)
- _apply_game_patch()
- search_cost.py
- kaggle-environments==1.32.7 — pinned EXACTLY
- test_own_supply_gap.py
- ._registry_tag()
- F016 — Animal placement and species table
- task_nn_numpy()
- _chunk_pymc()
- _run_autopsy()
- test_day_suite.py
- entity_of_code()
- _make_overrun()
- drain_per_day()
- Past about six hands the order queue is the cons
- F044 — Animal age collapses to a residue
- task_multiprocessing()
- _start_burner()
- world/__init__.py
- .describe()
- Mandatory agent-commit wrapper
- F048 mis-inference worked example
- Configuration defaults table
- task_package_census()
- Import/call allowlist audit (clean)
- opponents/__init__.py
- numpy, pandas, matplotlib, seaborn — analysis an
- test_the_goods_split_is_seven_one_way_two_dual()
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
8. `check_route()` - 44 edges
9. `Config` - 43 edges
10. `OpponentModel` - 41 edges

## Surprising Connections (you probably didn't know these)
- `Results are pair properties, not agent scalars` --conceptually_related_to--> `MarketForecast`  [INFERRED]
  docs/F050_market-coupling-defeats-common-opponent-screen.md → agent/belief/market.py
- `The mean unlock policy is the default` --conceptually_related_to--> `MarketForecast`  [INFERRED]
  docs/F054_town-consumption-is-exactly-modellable.md → agent/belief/market.py
- `The mean unlock policy is the default` --conceptually_related_to--> `forecast()`  [INFERRED]
  docs/F054_town-consumption-is-exactly-modellable.md → agent/belief/market.py
- `Columns are priced as rotations, not crops` --conceptually_related_to--> `Column`  [INFERRED]
  docs/F049_the-tile-optimum-is-a-cycle-not-a-crop.md → agent/planner/colgen.py
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

## Communities (184 total, 14 thin omitted)

### Community 0 - "colgen.py"
Cohesion: 0.04
Nodes (88): advance_pool(), cash_relief(), cash_rows(), ColgenResult, Column, column_key(), _extended_basis(), generate() (+80 more)

### Community 1 - "master.py"
Cohesion: 0.03
Nodes (89): classes_of(), `owned` state ids -> (class keys, counts, tile→class). A class used to be a…, dual_stand_in(), load_contractor(), Any, ndarray, What the master needs from the board, and from nothing else. These four helpers…, The shipped graph, cast once per process — never rebuilt at runtime. (+81 more)

### Community 2 - "Config"
Cohesion: 0.03
Nodes (69): Config, Every number the agent is tuned on, in one place. Numbers only. Config holds…, What one turn may spend thinking, after the reserve., The tuned numbers if they shipped, the defaults if they did not. A missing file…, The tuned numbers. Every field is a quantity; none is a mode. One field is a…, _contractor(), _load_graph(), Manager (+61 more)

### Community 3 - "equilibrate()"
Cohesion: 0.06
Nodes (66): The resource column a shed item occupies, or None if it has none. The shed's…, _resource_of(), column_cash(), CouplingSupply, equilibrate(), price(), _owned_distances(), published_duals() (+58 more)

### Community 4 - "rules.py"
Cohesion: 0.04
Nodes (63): availability(), _better(), _column_for(), day_chains(), DayFit, DayPlan, fit(), hire_bill() (+55 more)

### Community 5 - "economics-driven-rule-agent-ecobot-v6/agent.py"
Cohesion: 0.07
Nodes (64): _add_pickup_tasks(), agent(), AgentMemory, AnimalCensus, _assign_tasks(), eligible(), bfs_step(), build_development_plan() (+56 more)

### Community 6 - "belief/__init__.py"
Cohesion: 0.06
Nodes (65): agent/belief/ — what the market and the rival are doing, and what to sell. The…, expected_price_curve(), infer_rival_slot(), quantile_price_floor(), Phase 2 — the rival's action model, the demand forecast, and the hidden order.…, The price if the drain falls `z` sd short — risk priced as a price., Price for each candidate quantity we sell, `turns_ahead` turns out., Infer which slot the rival's volume sat in, from the price it realised.… (+57 more)

### Community 7 - "pathlib"
Cohesion: 0.04
Nodes (60): The day: the planner's chains become the ops a worker-day is made of. A chain…, The day as arrays: every task of the day, one per row, with its constraints as…, build(), load(), main(), Time the day layer over the benchmark corpus, and say what the seconds buy. The…, The layer's own `Day` and its tasks, from one corpus entry., search() (+52 more)

### Community 8 - "chains.py"
Cohesion: 0.04
Nodes (67): info_path(), The info file of an artifact., chain_id_of(), chain_ops(), entity_code_of(), __getattr__(), load_chains(), loaded_chains() (+59 more)

### Community 9 - "forecast()"
Cohesion: 0.05
Nodes (57): The depth curve of every good and day as LP blocks. Returns `(units, prices)`,…, sell_blocks(), forecast(), price_paths(), Forward-simulate market inventory, then price each day start. Vectorized form:…, `{item: (price, ...)}` for the horizon — what feeds the master's `p_d`. F035 is…, This turn's market forecast, or None when belief cannot build one.…, _product_price_path() (+49 more)

### Community 10 - "planner/__init__.py"
Cohesion: 0.06
Nodes (58): assign_by_quota(), assign_tiles(), Choice, ClassMix, counts(), demote_to_feasible(), Plan, plan_from_board() (+50 more)

### Community 11 - "TileState"
Cohesion: 0.06
Nodes (54): harvest_events(), ndarray, The rival's harvest calendar (#16): their board is PUBLIC — `farms[i]` tiles…, One planted rival tile's scheduled payout., The rival board's scheduled payouts, from today's decode. A planted tile's…, `(horizon, n_goods)` units/day the rival's planted tiles schedule. This is the…, RivalHarvestEvent, supply_curve() (+46 more)

### Community 12 - "oracle_transitions.py"
Cohesion: 0.05
Nodes (58): is_animal(), Whether `entity` is one of the three species., ChainOutcome, tile' for the merged graph, else 'crop' / 'animal'., What one executed chain produced (decision 11)., _covers(), _edges_from(), main() (+50 more)

### Community 13 - "dispatch_plan()"
Cohesion: 0.05
Nodes (51): dispatch_plan(), market_at(), Dispatcher: a committed day plan -> one action dict for the engine. The plan…, This turn's market orders, from a per-hour queue or a flat list. #15 made…, Slice `plan` into this turn's action dict. `plan` is a dict: ``{"units": [[op,…, The compiler's output in the shape the dispatcher slices. `units[0]` is the…, to_plan(), day_one_board() (+43 more)

### Community 14 - "replay_agent.py"
Cohesion: 0.06
Nodes (46): AgentLike, F048 — The season has 720 states and 719 decisions, 720 states, 719 decisions, at steps 0..718, Day 29 gets 23 decisions, hours 0..22, End-of-season liquidation, EPISODE_STATES - 1 is the last state, -2 the last decision, A chain of reasoning over a measured number is not itself measured, An observation is the state before that turn is processed (+38 more)

### Community 15 - "R004 as cited by the opponents docs — hand the a"
Cohesion: 0.06
Nodes (57): adaptive-public-state-multi-route — vendored opponent agent (not our code), kaggriculture-adaptive-public-state-multi-route.ipynb (source notebook), writefile packing — kaggriculture-adaptive-public-state-multi-route.ipynb extracted 387 lines / 174,502 bytes, adaptive-replay-agent — vendored opponent agent (not our code), kaggriculture-adaptive-replay-agent.ipynb (source notebook), blob packing — kaggriculture-adaptive-replay-agent.ipynb extracted 551 lines / 28,709 bytes, adaptive-shop-guard — vendored opponent agent (not our code), kaggriculture-adaptive-shop-guard.ipynb (source notebook) (+49 more)

### Community 16 - "beam.py"
Cohesion: 0.07
Nodes (53): _bag(), day_first_good(), door_load(), door_work(), first_walk_turn(), good_hours(), grow_charge(), leg_moves() (+45 more)

### Community 17 - "test_tile_dp_contractor.py"
Cohesion: 0.07
Nodes (51): price_board(), One pricing call: the sweep plus the recovered columns. Builds a…, _enumerate_best(), walk(), _flat(), _graph(), _integer_duals(), _owned() (+43 more)

### Community 18 - "v16-rc5-r5a-high-score-8c-4s-recovery/agent.py"
Cohesion: 0.10
Nodes (47): F033 — Market quoting: both players, pre-commit inventory, BUY_PRODUCT quoted at post-buy inventory, Both players quoted from the same pre-commit inventory, F034 — Price function shape and per-good parameters, MARKET_PARAMS per-good shapes and amplitudes, price(inventory) pivots at I0 = 10,000, floored at 1, F036 — The price ladder is coarse; sell pressure bites only in shallow markets, Coarse price ladder: about 25 wheat units per coin (+39 more)

### Community 19 - "TaskArray"
Cohesion: 0.06
Nodes (31): beam_for(), The width to search a day of this size with this many workers., chain_depth(), columns_of(), day_walking(), distance_matrix(), land_image(), ndarray (+23 more)

### Community 20 - "findings-from-zero-to-top-meta/agent.py"
Cohesion: 0.10
Nodes (44): agent(), is_sell(), promotable(), _align_hands(), _base_agent(), _cash_needed(), _conditional_reorder(), _copy_action() (+36 more)

### Community 21 - "v20-adaptive-r1-multi-route-agent/agent.py"
Cohesion: 0.15
Nodes (43): agent(), _align_hands(), _clone_distance(), _copy_action(), _demand_per_day(), _farm(), _future_sells(), _get() (+35 more)

### Community 22 - "Day"
Cohesion: 0.08
Nodes (40): bags_of(), _better(), _better_route(), ceiling_for(), _consistent(), Day, _fixed_point(), _hand_doors() (+32 more)

### Community 23 - "belief/market.py"
Cohesion: 0.08
Nodes (33): best_day_split(), block_revenue(), day_envelope(), depth_blocks(), depth_coins(), geometric_edges(), _index(), inventory_at() (+25 more)

### Community 24 - "fast_sim.py"
Cohesion: 0.07
Nodes (29): kaggle_environments_utils, Any, Dev-mode action shape check (see module docstring)., validate_action(), _EnvShim, _merge_configuration(), _observation_dict(), _pass_policy() (+21 more)

### Community 25 - "TileContractor"
Cohesion: 0.08
Nodes (26): _check_non_negative(), PricedBoard, ndarray, TileContractor — the pricing oracle: a backward DP over the tile graph, plus…, The graph's edge costs with `travel_hours` charged on every worked day. The…, Every state must own at least one edge (see the module docstring). The…, A dual vector -> `(days, N_RESOURCE)` float32, R006-checked. Accepts either the…, Backward sweep: `(days+1, n_states)` values and per-day rewards. (+18 more)

### Community 26 - "model.py"
Cohesion: 0.10
Nodes (35): _as_item(), item_of(), Item, Actions as values: a worker's, and the market's, kept apart. The engine takes…, The named item as the member its op allows (a wrong name raises)., The model's item for a name the engine uses: a crop, an animal or a product., ActionRule, What each action needs and what it does — the engine's own checks. Read from… (+27 more)

### Community 27 - "extract.py"
Cohesion: 0.13
Nodes (37): gzip, hashlib, agent.py payload, byte-for-byte — SHA-256 9b5e1157af3d4a64…, agent.py payload, byte-for-byte — SHA-256 3c86ae6e2afaa091…, agent.py payload, byte-for-byte — SHA-256 2fe7118964656263…, Decoded data tables (route or schedule) — no code to read, Duplicate payloads — two slugs sharing one SHA-256, agent.py payload, byte-for-byte — SHA-256 aca6dcddafd6fd6b… (+29 more)

### Community 28 - "schemas.py"
Cohesion: 0.08
Nodes (34): CarryRequirement, DaySchedule, DropRequirement, empty_state(), MarketState, OrderBook, PurchaseIntent, The turns the layers speak through. One rule, from the architecture review:… (+26 more)

### Community 29 - "test_market_layer.py"
Cohesion: 0.08
Nodes (34): Secretary B (#15): the forecast, the shed guard, the queue, the layer. Run:…, The named bias, asserted in its direction. Missing unlocks under-counts town…, ARCHITECTURE §5 step 5: the market has one reader, and this is the seam. The…, Patch `K.market_price` and watch every forecast price move with it., `our_sells` is the farm's own price impact (F033/F036)., Fill the shed to 95 with 20 wheat in a bag, then play out the day.…, With the market layer on: nothing is destroyed, and the stock is sold., R007: the control leg, and the reason the guard exists at all. (+26 more)

### Community 30 - "bench_market_forecast.py"
Cohesion: 0.08
Nodes (34): mean_shop_demand(), One consumption tick of a shop set: product -> units removed. A single-product…, The town's consumption during turn `step`, exactly as the engine runs it., Mean per-INSTANCE demand of one shop draw (the `"mean"` unlock policy). Every…, shop_demand(), town_deltas(), _first_plant_dict(), _greedy() (+26 more)

### Community 31 - "evaluate.py"
Cohesion: 0.09
Nodes (34): concurrent_futures, csv, datetime, _append_jsonl(), _append_scoreboard(), _csv_num(), _fmt(), _git_sha() (+26 more)

### Community 32 - "sell_coins()"
Cohesion: 0.09
Nodes (31): buy_coins(), good_index(), ndarray, _quotes(), The market's arithmetic as arrays: the price ladder, and the exact day split.…, Units of supply a sale of `units` from `inventory` lands: the engine stalls at…, Coins `units` fetch selling from `inventory`, floor stall included., Coins `units` cost buying down from `inventory` (quoted at I - 1). (+23 more)

### Community 33 - "test_slot_circuit.py"
Cohesion: 0.12
Nodes (29): _candidate_schedules(), _center_items(), _drain_per_hour(), plan_day_hours(), plan_day_slots(), ndarray, The slot circuit: the rival's trained distribution -> one sell schedule.…, The rival's hourly placements from the trained distribution. `activity` selects… (+21 more)

### Community 34 - "test_layering.py"
Cohesion: 0.09
Nodes (32): contextlib, tempfile, _fake_repo(), _imported_modules(), _imported_roots(), main(), _module_file(), The submission never imports the competitors. Run: .venv/bin/python -m… (+24 more)

### Community 35 - "test_evaluate.py"
Cohesion: 0.07
Nodes (29): math, _direction_verdict(), _paired_jobs(), (opponent, seed) jobs, interleaved so no seed strands. Seed s starts at…, Measured seed count replacing the provisional 16 (#20 brief 6.3): paired seeds…, A4's direction rule for one version's two readings (self-review fix). Returns…, _seeds_needed(), bank_seconds() (+21 more)

### Community 36 - "Crops and Animals — Every Rule, Numbered (AgriOr"
Cohesion: 0.09
Nodes (32): Dropping Is Counted, Atomic Seed Check — One Over-Request Drops Every PLANT, F001 — Seeds Bypass the Shed, Seeds Live in private["seeds"], Not the Shed, F002 — Planting Day Counts as Unwatered, A New Plant Starts with consecutive_unwatered = 1, Two Consecutive Dry Nights Turn the Tile into a Weed, F003 — Watering Once per Day (+24 more)

### Community 37 - "test_market_wsr_check.py"
Cohesion: 0.09
Nodes (29): build(), buy_orders(), DayMarket, merge(), needs(), The purchases the day needs, netted against what the farm already holds. Seeds…, Read the timetable off the merged queue. A hand hired in turn `t` acts from `t…, One queue, in the engine's settle order, capped per turn. Sells keep the head… (+21 more)

### Community 38 - "OpponentModel"
Cohesion: 0.13
Nodes (17): _load_trained(), _load_trained_qty(), OpponentModel, Any, ndarray, Empirical action counts over the rival, Laplace-smoothed. Smoothing is…, The good's aggregate action distribution (its own prior)., The activity key (#65 follow-up): the 3-tuple with the rival's own activity… (+9 more)

### Community 39 - "test_belief_depth.py"
Cohesion: 0.10
Nodes (25): hourly_value(), marginal_price(), What the `units`-th unit fetches (1-based): the ladder's own marginal. The…, `value(good, hour)`: what ONE MORE unit fetches at that turn. The secretary's…, _fc(), Guards for the market's DEPTH surface (`belief/depth.py`). The claim under…, Exact at the boundaries, conservative inside a block — the LP's shape., The measured shape that retires the flat deep tier (#151). On the day-0 board… (+17 more)

### Community 40 - "check_route()"
Cohesion: 0.10
Nodes (26): check_route(), Everything a compiled route promises, checked - so a caller can report instead…, _day(), _entry(), A real day whose door pass does not settle: what an unsettled pass has to hand…, The fixture has to be the day it claims: every hand offered at the same hour,…, The answer's doors are the engine's own derivation of its route, not the guess…, test_an_unsettled_door_pass_hands_back_the_doors_the_day_has() (+18 more)

### Community 41 - "interpreter()"
Cohesion: 0.09
Nodes (21): _default_spawn returns the farmer to (4,4), _end_of_day:877-882 resets the crew, ANIMALS table, _apply_unit_action(), CROPS table, _daily_refresh_animals(), _daily_refresh_plants(), _default_spawn() (+13 more)

### Community 42 - "kaggle_probe-1.py"
Cohesion: 0.09
Nodes (17): gc, _bind_seat(), _milp_problem(), mode_burn_3s(), mode_burn_the_bank(), mode_not_serialisable(), _new_seat_state(), Kaggriculture — ONE-SUBMISSION PROBE agent WHAT THIS IS A legal agent that… (+9 more)

### Community 43 - "test_pool_loader.py"
Cohesion: 0.11
Nodes (25): inspect, call(), _last_top_level_def(), load(), load_all(), LoadedAgent, Pool loader: import the 19 vendored competitors uniformly. Issue #20,…, Call a resolved agent in its own convention (arity by inspection). Never raises… (+17 more)

### Community 44 - "MarketTracker"
Cohesion: 0.10
Nodes (21): field_of(), Read a field from the observation whether it is a dict or a struct., fib_hire_costs(), land_cost(), MarketTracker, Any, ndarray, Hour-by-hour tracker for one seat. Feed it every observation, in order. (+13 more)

### Community 45 - "plan_sales()"
Cohesion: 0.10
Nodes (23): _guard_margin(), plan_sales(), Slack left below the cap by the guard. The day's harvest is an ESTIMATE…, How much of which item to sell on which turn, and why. Order of the rules is…, `units` split over `turns` orders, as evenly as integers allow., _spread(), _forecast_stub(), A price oracle, faithful to `MarketForecast`'s contract. Flat (its own peak, so… (+15 more)

### Community 46 - "adaptive-replay-agent/agent.py"
Cohesion: 0.19
Nodes (26): agent(), _align_hands(), _clone_distance(), _copy_action(), _farm(), _future_base_sells(), _get(), _kaggle_submission_entrypoint() (+18 more)

### Community 47 - "repair.py"
Cohesion: 0.14
Nodes (20): Drop, _inventories(), land_step_price(), _money(), order_cost(), _own_tiles(), _prices(), Any (+12 more)

### Community 48 - "docs/INDEX.md one-line-per-file index"
Cohesion: 0.09
Nodes (25): docs/INDEX.md one-line-per-file index, Permanent F/R ids, F/R file naming and INDEX convention, Project stack (Kaggle Kaggriculture, CPU PyTorch, OR-Tools, scipy), Eliminated constants-parity harness, Re-import rules from kaggle_environments, Exactly one source of truth, Pinned kaggle-environments version mitigation (+17 more)

### Community 49 - "json"
Cohesion: 0.12
Nodes (20): duckdb, importlib_util, json, dumps(), extract(), main(), Build the day layer's benchmark corpus: a few winning games, sampled across the…, The games our own seat won, best first. (+12 more)

### Community 50 - "Kaggle Probe #1 — What the Real Grading Machine "
Cohesion: 0.09
Nodes (26): Compute Budget — 1 s Free per Turn plus a 60 s Bank, Contention Ratio 0.744 — the Opponent Costs 25.6 % of Throughput, Kaggle Probe #1 — What the Real Grading Machine Measures, Error Ladder — What the Harness Survives, Graded Episode Configuration, Machine Drift of ±25 % across a Season, Memory Ceiling — 3.86 GiB Peak RSS of a 6.5 GiB Host, ortools Is Absent from the Grading Image (+18 more)

### Community 51 - "test_tile_dp.py"
Cohesion: 0.10
Nodes (25): offline_lab_build_ledger, _carrot(), _find(), _prod_of(), tile_dp tests: the merged tile graph. Run: .venv/bin/python -m…, A young plant is not harvested, and nothing is planted into an occupied tile: a…, The registry's own invariants (brief part 2, item 1)., Produced units of `res` on the state's `chain` edge (None if absent). (+17 more)

### Community 52 - "v13-r3-top-meta-order-safe-premium-control/agent"
Cohesion: 0.20
Nodes (25): agent(), _align_hands(), _clone_distance(), _copy_action(), _farm(), _future_base_sells(), _get(), _kaggle_submission_entrypoint() (+17 more)

### Community 53 - "market_queue()"
Cohesion: 0.12
Nodes (23): _after_arrival(), _assert_within_cap(), _get(), market_queue(), _money(), orders_by_hour(), Any, Secretary B: the shed — what lands, what fits, what has to be sold. Issue #15.… (+15 more)

### Community 54 - "MarketLayer"
Cohesion: 0.13
Nodes (20): attach(), from_env(), _market_row(), MarketLayer, Any, The market half of the day plan, riding on top of whatever rung answered (#15).…, `CHISTA_MARKET=spread|dump` -> a layer; anything else -> disabled., Attach the market half to a rung's action (never raises). Called for every… (+12 more)

### Community 55 - "bench_paths.py"
Cohesion: 0.12
Nodes (20): fast_sim(), harness_no_agents(), harness_with_agents(), main(), median_seconds(), pass_policy(), Any, Reproduce the R003 speed claim on the machine you are running on. Usage:… (+12 more)

### Community 56 - "zlib"
Cohesion: 0.11
Nodes (22): agent(), _kaggle_submission_entrypoint(), Adaptive Shop Guard for Kaggriculture., _v44_load(), _v44_package(), agent(), _kaggle_submission_entrypoint(), Adaptive Shop Guard V46: replay-calibrated market defense. (+14 more)

### Community 57 - "test_agent_obs.py"
Cohesion: 0.11
Nodes (22): _board_with_plant(), _graph(), _mini_obs(), agent obs-decode tests: both farms -> packed keys, coverage, LOCKED. Run:…, Scripted episode, weedSpawnChance = 0: every day-start key on BOTH farms is…, tile_state.py's mls TODO: plant a MELON on a live sim, read max_lifespan_step…, graph_keys=None: raw keys, unknown 0 (the caller opted out)., A key outside the graph maps to the nearest modelled state and is counted —… (+14 more)

### Community 58 - "_decode_chain()"
Cohesion: 0.10
Nodes (24): AST, audit(), _audit_source(), _classify(), _constants(), _decode_chain(), _nested_sources(), packed_payloads() (+16 more)

### Community 59 - "Live observation views in fast mode"
Cohesion: 0.10
Nodes (24): fast mode (checks bypassed), Live observation views in fast mode, Read-only observation contract, Characterisation sweep 2026-09-15 seed 0, chista-m1 baseline vs the pool, Behavioural class semantics P/S/R, 11 dev / 5 held out until M5, Declared duplicate payloads (+16 more)

### Community 60 - "test_world_parity.py"
Cohesion: 0.15
Nodes (19): random, action_for(), _clean(), config(), drive_both_paths(), harness_agent(), agent(), sim_policy() (+11 more)

### Community 61 - "test_bounds_day.py"
Cohesion: 0.11
Nodes (22): _one_tile_day(), The pool a day provably cannot use: refused with a no, not with a search.…, Two corners of the board: 8 out of each nearest door, 16 in all. A worker may…, Forty tasks on the shed's tile, three hands hired at hour 20: 24, 28, 32, 36…, The planner's offer caps the pool: a hand nobody pays for is not a hand the day…, Every unit does at least one task, so a pool larger than the work is never the…, The no is an answer about the day, so it carries no route and claims no time…, A bound says the fewest a day might need. A range that reaches it is a question… (+14 more)

### Community 62 - "test_season_horizon.py"
Cohesion: 0.11
Nodes (19): _column(), _obs_at(), Guards for the season horizon (F029) and the pool carried across days. There is…, Day 0 prices the whole season, day 29 one day, and the 30s agree. The horizon…, It shrinks to the season that is left, and honours a shorter look-ahead. The…, A forecast that does not cover the horizon degrades — it is not padded.…, A column shaped like the ones the pricing builds, with countable values., Every day-indexed field of a carried column is one day further on. `cost`,… (+11 more)

### Community 63 - "artifact/__init__.py"
Cohesion: 0.11
Nodes (20): Any, The agent's artifact store: one data file per artifact, and its info beside it.…, Write the info file beside an artifact. An unknown key raises., Read an artifact's info, refusing a file that is not in the standard shape., read_info(), write_info(), offline_lab_build_fingerprint, Did the build recipe move without the artifact being rebuilt?… (+12 more)

### Community 64 - "_Path"
Cohesion: 0.11
Nodes (17): Write these numbers where `load()` will find them., extract(), Extraction, _from_blob(), _from_inline(), _from_writefile(), _looks_like_an_agent(), NotExtractable (+9 more)

### Community 65 - "wsr/ — The Day Layer"
Cohesion: 0.14
Nodes (20): world/model.py — the Names, Beam Search — Day, Result, search(), Call Sites — planner/day.py fit() and compile(), compile_route and DayOps, wsr/ — The Day Layer, A Drop Is a Deadline, Not a Chain Op, expand_chain — a Chain as the Tasks It Is Made Of, A Hand Lands on the Least-Occupied Shed Door (+12 more)

### Community 66 - "Evaluation arena — sparring opponents run in our"
Cohesion: 0.26
Nodes (22): Audit finding — compile() at line 20; payload audited separately, Audit — clean: every import and call is within the allowlist, Audit finding — compile() at line 29; payload audited separately, Evaluation arena — sparring opponents run in our own interpreter, Rationale: the arena runs vendored code inside our interpreter, so its imports and calls are checked before it is ever called, Audit — clean: every import and call is within the allowlist, Rationale: the extractor parses the notebook with ast and never executes it, Audit finding — compile() at line 29; payload audited separately (+14 more)

### Community 67 - "Action"
Cohesion: 0.15
Nodes (14): Action, MarketAction, _op_name(), parse_action(), Any, The engine's own list (or a tuple in the same shape) as a `WorkerAction`., One order the market executes: the worker never runs it., The engine's own list (or a tuple in the same shape) as a `MarketAction`. (+6 more)

### Community 68 - "Cell"
Cohesion: 0.12
Nodes (15): in_board(), manhattan(), quadrant_of(), The board: cells, quadrants, and the shed's four doors. Numbers are in…, The engine stores the board as `tiles[y][x]`: the OUTER list is y, the inner…, Which quadrant a cell is in (kaggriculture.py:127-129)., Whether a cell is on the board at all (:326)., Turns between two cells: the board has no obstacles and a unit moves one tile a… (+7 more)

### Community 69 - "expand_chain()"
Cohesion: 0.10
Nodes (20): expand_chain(), Expansion, MinorTask, Item, NamedTuple, One chain, expanded into the tasks the search reads. `tasks` is every op of the…, One day's chain -> the tasks the search reads, the order between them, and the…, _action_code() (+12 more)

### Community 70 - "_process_market()"
Cohesion: 0.11
Nodes (19): _shed_access_tiles, kaggriculture.py::_spawn_hand:533, _do_buy_land(), _do_hire(), _hire_cost() and _fib(), LAND_ORDER and LAND_PRICES, MARKET_PARAMS, market_price() (+11 more)

### Community 71 - "test_agent_runtime.py"
Cohesion: 0.12
Nodes (19): io, _obs(), The spine's guards: the entry point, the dispatcher, the manager's clock, and…, The DEFAULT is a temporary owner setting, and the guard says which. It is OFF…, A failure on any turn of the day is recorded, not swallowed., Hour h gives every unit its h-th op; exhausted units PASS., More than 10 market orders never leave the agent (F031 cap)., F031: hires can fail silently, so the OBS's hand count is authoritative. (+11 more)

### Community 72 - "test_market_hourly.py"
Cohesion: 0.13
Nodes (19): kaggle_environments_envs_kaggriculture, Guards for the hourly price extension of `belief/market`. The claim under test:…, Hour 0 of row 0, mid-day, quotes the observed inventory exactly., The #71 guard: at hour_now = h, hour h's OWN quote is the snapshot, and the…, The rival's supply at hour 10 cheapens hour 11 onward, not hour 10.…, Naming a day's hours must not ADD to that day's calendar total., The hourly table is (days*24, 9): one row per turn, in PRODUCTS order., Every hourly quote equals K.market_price at the walk's inventory row. The walk… (+11 more)

### Community 73 - "test_tile_dp_decline.py"
Cohesion: 0.15
Nodes (20): offline_lab_build_chains, _cost(), _edge(), _graph(), _idle(), The decline edge: every node keeps its idle day (#84). Run: .venv/bin/python -m…, The law itself, on two edges of one node - no build, no artifact. A pasture…, The build's post-condition: a node that kept no free idle edge is refused. (+12 more)

### Community 74 - "runner.py"
Cohesion: 0.14
Nodes (19): The canonical slugs, sorted (16 distinct agents). The vendored tree holds 19…, slugs(), offline_lab.pool sweep: the characterisation sweep driver (issue #20 part 2).…, One slug-vs-slug episode per slug (A2 pre-filter, ~30 s each)., run_selfplay(), selfplay_prefilter(), _episode_worker(), _freeze() (+11 more)

### Community 75 - "_FakeManager"
Cohesion: 0.13
Nodes (18): _FakeManager, One observe per day, one step per remaining turn - never the other way., Every step gets a positive budget no larger than the working second., The plan the manager holds is sliced by hour, not re-decided., A raising manager: PASS, a recorded failure, its day marked, a log line.…, One bad turn is one bad turn: the next turn still calls the manager., 719 turns of the spine's own loop: 30 observes, 689 steps, all legal., One reading per turn after the first: the season's own gap count. (+10 more)

### Community 76 - "numpy"
Cohesion: 0.14
Nodes (17): plan_supply_sells(), plan_supply_sells_from_pool(), ndarray, #110's own-supply half: the plan's projected sells re-enter the price path its…, The chosen mix's projected sells as `forecast(our_sells=)` reads. `produce`:…, Same, from the master's pool (columns carry `produce`; idle ones carry None and…, numpy, A sell at hour 14 supplies the market at hour 14, not at hour 0. The walk… (+9 more)

### Community 77 - "ndarray"
Cohesion: 0.14
Nodes (20): _carried(), done_by_worker(), _door_first(), DoorLoad, _expand(), _flat(), _last_drop(), _load_ready() (+12 more)

### Community 78 - "25-27-strict-future-v27-midgame-meta-reset/agent"
Cohesion: 0.29
Nodes (19): agent(), _align_hands(), _copy_action(), _demand_per_day(), _farm(), _get(), _impact_score(), _is_sell() (+11 more)

### Community 79 - "Agent Guide — ChistaAgent"
Cohesion: 0.16
Nodes (19): The Simplest Rounding That Can Work, Engine Authority — kaggle_environments.envs.kaggriculture, How to Prove It — Play the Day on the Engine, Agent-Commit Trailer Rule, belief/ — The Sell Side, Agent Guide — ChistaAgent, Games Run on FastSim — Mandatory, Findings & Rules File Convention (+11 more)

### Community 80 - "probe_hidden_state.py"
Cohesion: 0.16
Nodes (17): tile_dp: the tile lifecycle graph, and the contractor that prices it. `graph`…, argparse, _continuation_diffs(), _decoder_keys(), _futures_differ(), main(), _one_day(), _print_field_split() (+9 more)

### Community 81 - "test_planner_hands_day.py"
Cohesion: 0.11
Nodes (14): hire_hour(), The earliest hour the k-th hand of a day (0-based) can act. A hand hired in…, preload_turns(), The hour each worker may first act. A unit already on the field acts from the…, The goods a worker may have to load at its door: every good a task of the day…, The hour each worker's WALK may begin: its own hour, the goods, and the pickups…, start_hours(), The hands the planner offers are the hands the day layer searches with - and… (+6 more)

### Community 82 - "base64"
Cohesion: 0.17
Nodes (16): base64, copy, agent(), Exact public replay tape: episode 90558188, seat 0., _safe_terminal_action(), agent(), _apply_market_delta(), _call() (+8 more)

### Community 83 - "task_bn_scipy()"
Cohesion: 0.12
Nodes (18): bn_analytic_mu(), bn_data(), bn_neg_logpost(), bn_pack(), bn_x0(), _chunk_scipy(), _data(), _lgamma() (+10 more)

### Community 84 - "FastSim"
Cohesion: 0.16
Nodes (15): FastSim, Single-episode simulator around the real interpreter. Semantics (verified…, _day_plan(), _obs(), parametrize, Each good is picked up at its own hour, and the planner's compile writes the…, One source: the market hires the tuple's own length, not a second count.…, The plan on the engine from day 0 hour 0 to `until`; the last observation.… (+7 more)

### Community 85 - "44-46-strict-future-top-30-v22-price-impact/agen"
Cohesion: 0.27
Nodes (18): get(), One registry row (KeyError for an unknown slug - loud, R005)., agent(), _align_hands(), _copy_action(), _farm(), _get(), _impact_score() (+10 more)

### Community 86 - "TileHourZero"
Cohesion: 0.11
Nodes (7): One tile as a day begins, in our terms. `kind` is a `TileKind`., A coop or pasture with no animal: PLACE can still fill it (:384-392)., What a HARVEST here yields: the crop, or the animal's product (:466-472)., The cap on `yield_units`: the crop's `max_yield`, the animal's `max_held`., Whether a FERTILIZE dose still covers today (:481)., One line, for a message or a report., TileHourZero

### Community 87 - "test_real_days.py"
Cohesion: 0.14
Nodes (17): collections, _carried_as_the_game(), _key(), _marked(), _pair(), parametrize, Real days, from the archive: the same land conversion, no more hands than the…, A task id back to the tile it works and the op it runs: `d24_place2` is the… (+9 more)

### Community 88 - "world/fast_sim (FastSim.run)"
Cohesion: 0.12
Nodes (18): bench.bench_paths reproduction, world/fast_sim (FastSim.run), Direct kaggriculture.interpreter() call, structify state clone, tests/test_world_parity.py bit-identical parity, dev mode (validators run), world.FastSim(validate=dev|fast), Validation configuration switch (+10 more)

### Community 89 - "emit()"
Cohesion: 0.15
Nodes (17): agent(), _arm_ladder(), _contended_steps(), _contention_verdict(), emit(), _farm_snapshot(), _opp_digest(), _opp_snapshot_steps() (+9 more)

### Community 90 - "guarded_call()"
Cohesion: 0.13
Nodes (14): copy_observation(), guarded_call(), GuardStats, Arity by inspection (never by parameter name): 2+ params -> pass config., A deep copy of the observation, handed to a third-party agent. Pool-analysis…, The per-agent label set the registry stores., One guarded call: contain, validate, time. Returns a legal dict. `arity` comes…, _wants_two() (+6 more)

### Community 91 - "v111-8c4s-economic-core-premium-lead/agent.py"
Cohesion: 0.31
Nodes (17): agent(), _align_hands(), _copy_action(), _existing_sell(), _farm(), _fr_state(), _front_run(), _future_quantity() (+9 more)

### Community 92 - "v16-rc5-high-score-8c-4s-premium-market-lead/age"
Cohesion: 0.31
Nodes (17): agent(), _align_hands(), _copy_action(), _existing_sell(), _farm(), _fr_state(), _front_run(), _future_quantity() (+9 more)

### Community 93 - "test_contractor_concepts.py"
Cohesion: 0.23
Nodes (17): chains(), contractor(), duals(), one_bare_tile(), produced_per_day(), fixture, Conceptual tests: does the DP put its work where the value is? The cost side is…, `p` (what a unit of a product is worth) and `w` (what a unit costs) per day. (+9 more)

### Community 94 - "drain_forecast()"
Cohesion: 0.12
Nodes (14): basket_matrix(), drain_forecast(), (n_shop_types, 9) units consumed per shop event, per shop type., Exact mean and sd of the town drain over the next `steps_ahead` turns., Any, This turn's market field, in slot order., The snapshot, from the observation alone. Inventory and prices are the…, price_table() (+6 more)

### Community 95 - "Runtime"
Cohesion: 0.16
Nodes (12): _day_of(), _hour_of(), _overage_of(), Any, Exception, The turn's clock, the manager, and the never-raise boundary.…, The harness's per-turn call: one legal action dict, never raising., What is left of this turn's working budget, for the manager's step. `step`… (+4 more)

### Community 96 - "WorkerAction"
Cohesion: 0.14
Nodes (11): actions_of(), A chain's actions, each carrying its argument. The entity is what fills the…, One thing a unit does in a turn: what changes the tile it stands on., WorkerAction, One worker's day. `hours[h]` is what it did in hour `h`, or `None`., The same trace with hour `hour` filled in (immutable, like every state)., `(hour, action)` for every hour that has been decided., The hours with nothing in them yet. (+3 more)

### Community 97 - "agent()"
Cohesion: 0.15
Nodes (17): agent(), _arm_window_summary(), _boot_env(), _burn(), _do_drain(), _drain_due(), _drain_seconds_for_day(), _predicted_cutoff() (+9 more)

### Community 98 - "v3-agent/agent.py"
Cohesion: 0.32
Nodes (16): agent(), _align_hands(), _copy_action(), _farm(), _get(), _impact_score(), _impact_slots(), _is_sell() (+8 more)

### Community 99 - "test_animal_drop_day.py"
Cohesion: 0.17
Nodes (16): _build(), played(), fixture, The animals' day: FEED, CARE and COLLECT_FERTILIZER, with the fertilizer banked…, The whole day is carried with no hands at all - and it is the portfolio that…, One more hand than the day can use alone is what banks the third drop before…, The same day written in a different order is the same day: these three ops have…, The day carried by the two hands it needs. (+8 more)

### Community 100 - "test_drop_empties_bag_day.py"
Cohesion: 0.15
Nodes (16): _board_sim(), _play(), played(), fixture, A drop hands over the WHOLE bag: the feed a worker still carries goes with it.…, The engine's own flag: an animal the route fed is fed., The compiled ops, read the way the engine reads them: a DROP empties the bag, a…, The search charges the walk back through a door, so the day it calls carried… (+8 more)

### Community 101 - "test_mixed_second_day.py"
Cohesion: 0.17
Nodes (16): _available(), _day(), _grid(), _pair(), played(), fixture, The mixed day's SECOND day: the board day 0 left, and the work that board…, Day 1 asks less of the farm than day 0 did, and the layer's own ladder says how… (+8 more)

### Community 102 - "test_rival_hours.py"
Cohesion: 0.17
Nodes (14): _manager(), Guards for the manager's DATED rival supply (#16). `_rival_supply` is the…, An empty calendar gates the model out: no phantom rival supply. The model's…, What the tracker inferred is dated by the turn it happened in., No source of hours means no dated input: the daily curve is used., `never_raise` decides, exactly as it does at the turn boundary. Off (the…, The trained model covers the horizon; dropping its branch empties this., test_a_broken_tracker_spills_unless_the_run_asked_to_degrade() (+6 more)

### Community 103 - "plan_coins()"
Cohesion: 0.19
Nodes (13): plan_coins(), Coins a SPECIFIC multi-day sell plan fetches — the manager's what-if. `plan[d]`…, _brute(), ndarray, Guards for the ladder's multi-day tools: `split_days` (the exact best split)…, The DP's optimum equals enumerating EVERY split, to the coin. The guard the old…, The property the manager prices on: the SHAPE of a sell plan changes its total…, A plan that sells more than the inventory holds clamps per day and prices the… (+5 more)

### Community 104 - "agent()"
Cohesion: 0.21
Nodes (10): agent(), The arena's entry point for the slot circuit's sell plan (#65). Same spine as…, The arena's entry point for the market layer in `dump` mode (issues #15, #18).…, The arena's entry point for the market layer in `spread` mode (issues #15,…, Chista agent entry point. The harness calls a bare function per turn; all state…, Harness entry point: one legal action dict per turn, never raises., The coordinated entry point for the arena (issue #12 ship gate). Same spine as…, bench() (+2 more)

### Community 105 - "Worker distribution and scheduling problem on a "
Cohesion: 0.16
Nodes (15): Action format, Kaggriculture game overview, Observation fields, Wheat loop example agent, Ambiguities resolved by asking, never assuming, cell_state definition, Fibonacci worker cost, Least-congestion placement, NW->NE->SW->SE (+7 more)

### Community 106 - "CP-SAT formulation"
Cohesion: 0.14
Nodes (15): Brute-force cross-check independent of CP-SAT, Exact-match cache and warm start, chistawrs/ project structure and milestones, clct(s=23) global shed-return constraint, CP-SAT formulation, Deterministic precondition expansion, not open search, major_task (named chain of minors), Seven major_task recipes (+7 more)

### Community 107 - "kaggle_probe-2.py"
Cohesion: 0.18
Nodes (13): _arm_report(), _arm_schedule(), _bind_seat(), _child_env_read(), _new_seat_state(), _pct(), _pymc_child_paths(), Kaggriculture — PROBE #2: Bayesian inference, the time bank, and the sandbox… (+5 more)

### Community 108 - "_pick_opponents()"
Cohesion: 0.14
Nodes (13): _dev_split(), _pick_opponents(), The measured dev / held-out split from #20's registry (11/5 over the canonical…, The tier's opponent list, and the sentence describing where it came from (R005:…, canonical(), canonical_slugs(), dev_and_heldout(), The pool registry: one row per competitor, loaded as data. Issue #20 brief §4.… (+5 more)

### Community 109 - "test_drop_door_on_route_day.py"
Cohesion: 0.19
Nodes (14): _board_sim(), _play(), played(), fixture, A drop is handed over at the door nearest the worker, not at the harvest's own…, A fresh engine at hour 0 with the east and south tiles ours and ripe wheat on…, The compiled day on the engine, stopped at its last hour - before the night…, The searched day, compiled and played - or the compiler's refusal, for the… (+6 more)

### Community 110 - "test_world_branch_purity.py"
Cohesion: 0.25
Nodes (13): _fresh(), _pairs(), _play(), Branch purity and the RNG-from-state invariant for world.fast_sim. Run:…, The convenience wrapper must be exactly clone()+step(), and side-effect free., Apply actions in order, stopping at episode end (like FastSim.run)., Full state signature: status, reward and every observation field., A clone continued with a suffix == a from-scratch replay of prefix+suffix. This… (+5 more)

### Community 111 - "test_contractor_concepts_decay.py"
Cohesion: 0.22
Nodes (11): artifact_path(), The data file of an artifact., board(), chains(), days_producing(), dose_cost(), milk_price(), fixture (+3 more)

### Community 112 - "ShedState"
Cohesion: 0.14
Nodes (8): _order_of_preference(), What the market will actually quote: shed items that are PRODUCTS, with a…, The shed, the bags, and what tonight's drop will do to them., Units in the shed — the only stock a SELL can reach (F043)., Units in unit bags; the nightly drop empties these into the shed., F043's second half: at the cap, BUY_PRODUCT/BUY_ANIMAL are refused., Units the night drop will DESTROY tonight. `extra_carried` is what the day's…, ShedState

### Community 113 - "replan.py"
Cohesion: 0.19
Nodes (13): dual_stand_in(), enable(), hire_count(), _poll(), Any, ndarray, Replanner: the rung the day layer replaced. **It cannot import, and nothing…, The master's duals (#12), stood in for by the engine's own quotes. Every… (+5 more)

### Community 114 - "world/ — The Definitions"
Cohesion: 0.14
Nodes (14): world/action.py — WorkerAction and MarketAction, world/action_rules.py — What Each Action Needs, world/board.py — Cells, Quadrants, Shed Doors, world/ — The Definitions, world/prices.py — the Price Function and the Town's Drain, world/rules.py — the Numbers, world/tile.py — TileHourZero and TileInDay, world/worker.py — WorkerTrace (+6 more)

### Community 115 - "decode_world()"
Cohesion: 0.23
Nodes (12): decode_farm(), _decode_one(), decode_world(), FarmView, PrivateView, Observation decode: harness obs -> WorldView (both farms, packed keys). Issue…, The harness observation -> WorldView (both farms, one code path).…, One farm: packed keys, equivalence classes, and the public numbers. (+4 more)

### Community 116 - "WorldView"
Cohesion: 0.21
Nodes (13): _nearest_modelled(), The nearest modelled key to an unmodelled one (never raises). Order, written…, WorldView, The farmer first, then the hands in `hands` order (F030)., The graph state under each unit, or None where there is nothing to price. A…, unit_positions(), unit_state_ids(), owned_tiles() (+5 more)

### Community 117 - "F055 — SELL reads the shed, not the bag"
Cohesion: 0.24
Nodes (13): Belief's per-hour SELL queue, or no rows if it cannot build one. Called through…, The observation with only PRODUCTS in the shed — a guard for #77.…, sell_rows(), _sellable_obs(), F043 — Shed capacity destroys the overflow, DROP moves the whole inventory and destroys the overflow, The nightly drop empties every inventory and destroys the overflow, SELL of an item the shed does not hold is refused (+5 more)

### Community 118 - "test_contractor_carrot_cycle.py"
Cohesion: 0.21
Nodes (11): chain_name(), Stable name of a chain ('FERTILIZE+WATER+HARVEST', 'NO_ACTION')., offline_lab_build_graph, carrot_days(), fixture, The carrot cycle, as tests: two days, three carrots, and the dose goes to the…, Scenario 3: decaying milk, a constant carrot, and a dose that gets cheap at day…, scenario() (+3 more)

### Community 119 - "bench_turn_budget.py"
Cohesion: 0.15
Nodes (7): bench_sweep(), _Contender, Bench the turn budget: a full 720-turn episode, per-turn timing. Run from the…, The tile-DP sweep: solo and contended, direction asserted. Only the SWEEP is…, A competing thread: an opponent deliberating in the same turn. numpy releases…, statistics, threading

### Community 120 - "adaptive-public-state-multi-route/agent.py"
Cohesion: 0.27
Nodes (10): agent(), _apply_market_delta(), _call(), _emr_public_counts(), kaggle_agent(), _kaggle_submission_entrypoint(), V21-R1 public-state multi-route Kaggriculture agent., _seat() (+2 more)

### Community 121 - "test_budget_day.py"
Cohesion: 0.21
Nodes (11): _day(), _day_for(), The budget's contract: a bigger budget never places less, and a route says…, The property the branch claimed: more budget is never a worse route. Every…, A search cut short still names every hand's door, so a caller holding it can…, A route can be complete and still have crossed its deadline; it has no work…, The rule the pool loop and the halving share: carried first, then more work., test_a_bigger_budget_never_places_less() (+3 more)

### Community 122 - "_day()"
Cohesion: 0.15
Nodes (13): _day(), fixture, The hand-built three-hand day is legal, and the search takes it when it is…, Three hands are enough for this day - the reference proves it - and the search…, The pool, read from the other side: a hand more is a day with more room left in…, The day as the search sees it, with the pool the caller is willing to pay for., The short-pool search: the day it was given, and the answer it gives back., The same day, one hand more - so the withheld hand is what the day was short… (+5 more)

### Community 123 - "_quiet()"
Cohesion: 0.15
Nodes (12): Any, _quiet(), Both arms of `Config.never_raise`, on the same failing turn. ON: all-PASS, the…, The `D` line: one per day, at hour 0, carrying the manager's own numbers., A turn over the working budget gets an `A` line even when nothing raised., `Config.log_gaps`: off by default, and it prints the running stats. The quiet…, Run `fn`, returning (stdout, result) - the log is part of the contract., test_an_over_budget_turn_is_logged_as_an_anomaly() (+4 more)

### Community 124 - "compile()"
Cohesion: 0.20
Nodes (9): `risk(good, hour)`: the units the RIVAL is expected to put in that turn. The…, rival_risk(), compile(), priced(), queue(), sell_rank(), A `DayPlan` -> the `{"units": [...], "market": [...]}` the dispatcher slices.…, _load_payload() (+1 more)

### Community 125 - "planner — Rounding, Repair, Land (issue #13)"
Cohesion: 0.18
Nodes (12): assign_tiles — λ to One Plan per Tile, ClassMix — One Class, Its Plans, Their λ, demote_to_feasible — the Row-Walk Repair, planner — Rounding, Repair, Land (issue #13), Land Stays Out of the LP, LOCKED Tiles Get Nothing, Plan — One DW Column, plan_from_board — a Plan from the Pricing Oracle (+4 more)

### Community 126 - "_select()"
Cohesion: 0.17
Nodes (12): _dedupe(), _empty_like(), _rank_keys(), _rankings_for(), The rankings a day is worth searching under. The extra keys chase an hour. A…, The slots each ranking keeps: the WHOLE width each, not a slice of it. Sharing…, Each ranking's sort keys, in `np.lexsort` order - the LAST key is the primary…, Keep the best `beam` children, ranked BEFORE they are built. A child is a copy… (+4 more)

### Community 127 - "Care bank (pending_care_bonus) accrual and payou"
Cohesion: 0.21
Nodes (12): GOOSE (cost 300, COOP, day 4, daily, cap 4, EGG), F017 — Two consecutive unfed days escape, Two consecutive unfed days cause escape, F018 — Unfed animals still produce base, Base production outside the fed_today test, F019 — Care bank accrual and payout, Care bank (pending_care_bonus) accrual and payout, F020 — Unfed production night destroys the bank (+4 more)

### Community 128 - "A guard must be seen to fail"
Cohesion: 0.17
Nodes (12): ARCHITECTURE.md system-shape pointer, Tests and benchmarks index, CROPS identity assertion (rules.CROPS is K.CROPS), Call-site mismatch defeats reading, Five guards that passed while guarding nothing, Break it, watch it fail, restore, say so, A guard must be seen to fail, Apache License 2.0 (verbatim copy) (+4 more)

### Community 129 - "emit()"
Cohesion: 0.23
Nodes (12): _chunk_jax(), cpu_ratio(), emit(), _maybe_bn_compare(), _poll_pymc_child(), pymc_worker(), One JSON line per measurement. Flushed, because the episode may end., Run fn until >= min_seconds; return (wall, cpu-seconds per wall-second, reps). (+4 more)

### Community 130 - "test_dual_bins.py"
Cohesion: 0.23
Nodes (10): _model(), Guards for the dual goods' buy bin (#65 follow-up): a dual good's last bin…, The artifact's (WHEAT, day=1, bucket=0, act=1) row is 99% buy counts;…, `expected_sell` and `expected_buy` must not be the same query., MILK's bins are all sales; expected_buy is 0 by construction., Unit shape: the router zeroes exactly the bins it says it does., test_a_dual_goods_buy_bin_is_not_a_sell_volume(), test_a_sell_only_good_has_no_buy_side() (+2 more)

### Community 131 - "_shipped()"
Cohesion: 0.21
Nodes (12): _all_edges(), xfail, The merged graph the agent actually ships, loaded — not rebuilt. These four…, The cost and produce vectors are separate, int, 18 entries long., The strongest witness that the vectors are never netted: one edge that spends a…, F023: one COLLECT_FERTILIZER per animal per day is real produce, so the shipped…, No edge may run CARE without FEED in the same chain (a one-hour no-op)., _shipped() (+4 more)

### Community 132 - "frontier-the-soil-remembers-rain/agent.py"
Cohesion: 0.33
Nodes (10): agent(), Fixed public Subin An policy from episode 89945750, seat 1., _SUBIN_MULTI_WEED_BASE(), _SUBIN_MULTI_WEED_SAVKO_BASE(), _SUBIN_MULTI_WEED_SHOP_BASE(), _subin_multi_weed_tile(), _subin_terminal_choice(), _SUBIN_WEED_BASE() (+2 more)

### Community 133 - "test_door_work_before_load_day.py"
Cohesion: 0.25
Nodes (10): _play(), played(), fixture, The farmer's turn 0 is not spent waiting at its door for goods bought in turn…, The compiled day on an engine holding the plants and the sheep, to the day's…, The searched day, compiled and played - or the compiler's refusal, for the…, _search(), test_the_engine_watered_every_plant_and_fed_and_cared_for_the_sheep() (+2 more)

### Community 134 - "Ongoing ready days (TOMATO 8-11, STRAWBERRY 10-1"
Cohesion: 0.22
Nodes (10): F014 — Ongoing measured calendars, STRAWBERRY measured yield calendar, TOMATO measured yield calendar, F015 — Ongoing harvest leaves the plant, Ongoing HARVEST leaves the plant standing, F023 — Animal fertilizer every night, fertilizer_available set every night, one collect per animal per day, F027 — Ongoing harvest calendar (+2 more)

### Community 135 - "F031 — Order cap is per turn, queue walked to co"
Cohesion: 0.27
Nodes (10): Turn loop: every unit acts, then the market runs, then the day refresh, F030 — Action shape and unit-before-market ordering, Action shape {farmer, hands, market}, Units act before that turn's market, F031 — Order cap is per turn, queue walked to completion, HIRE and BUY_LAND settle atomically before the per-unit loop, maxMarketOrdersPerTurn = 10, per turn, Orders walked by index, each to completion (+2 more)

### Community 136 - "blas_threads"
Cohesion: 0.22
Nodes (7): blas_threads, Context manager that limits BLAS threads, or admits it could not., _read(), task_machine_census(), task_memory(), _thread_mechanism(), object

### Community 137 - "cpu_ratio()"
Cohesion: 0.33
Nodes (10): cpu_ratio(), _matmul_bench(), run(), _measure_window(), Run fn repeatedly for >= min_seconds; return (wall, cpu/wall, reps). cpu/wall…, THE question: how many cores does this container really give a single process,…, Seat 0: the reference bench, labelled by whether the opponent is busy., task_cpu_capacity() (+2 more)

### Community 138 - "_patch_phase()"
Cohesion: 0.24
Nodes (10): _install_intercept(), wrapper(), _mutate_now(), _patch_phase(), _patch_target(), A live env instance, if this process has one., The only proof that matters: does OUR OWN observation show the change?, One direct attempt to change the game, plus arming the intercept. (+2 more)

### Community 139 - "test_after_drop_not_loaded_day.py"
Cohesion: 0.24
Nodes (6): _play(), played(), fixture, A good a worker uses only after its own DROP is not charged to its door load.…, The compiled day on an engine holding the two animals and the shed's wheat., _search()

### Community 140 - "test_seed_invariance_no_rng_in_tile_transitions("
Cohesion: 0.20
Nodes (10): _arrays_equal(), _build_with(), _graph_arrays(), The artifact's own arrays, in a fixed order, for byte comparison., A drop-in `graph._new_sim` with another seed (and optionally weeds)., Run the SHIPPED builder under another sim factory. `_new_sim` is a module-level…, Byte-compare two CSR columns, tolerating different shapes., Issue #24 class A: with `weedSpawnChance = 0.0`, NO tile transition may depend… (+2 more)

### Community 141 - "test_land_image_day.py"
Cohesion: 0.28
Nodes (8): The land image is a view of the arrays, and the depth layer says how long a…, Every layer is a scatter of a column, so the totals have to agree with the…, A PLANT then a WATER is a chain of two, and a second WATER beside them is not a…, A chain runs on one tile, so its depth cannot be more than that tile's own task…, _tasks(), test_a_chain_runs_as_deep_as_its_precedence(), test_the_depth_never_exceeds_the_tasks_on_its_tile(), test_the_image_says_what_the_arrays_say()

### Community 142 - "test_spare_day.py"
Cohesion: 0.28
Nodes (8): _day(), The spare capacity: the turns the route leaves, and the walks come off it. A…, One task eight tiles away: the day has 24 turns, the task takes one and the…, The number a manager acts on has to move the right way when the day gets more…, The spare is counted against the hands the pool paid for, not against the offer., test_an_unhired_hand_is_not_capacity(), test_more_work_leaves_less_spare(), test_the_walk_is_not_spare()

### Community 143 - "GapStats"
Cohesion: 0.25
Nodes (5): GapStats, Sample sd; 0.0 before there is a second reading., Running mean and sd of the wall clock between two calls of the agent. The…, Welford's running mean and sd, against `statistics` on the same numbers.…, test_the_gap_stats_match_a_direct_computation()

### Community 144 - "spawn_cell()"
Cohesion: 0.36
Nodes (8): Which tile each new unit enters through, given `(index, earliest_start)` per…, The engine's placement rule for a new unit: of the four shed-access tiles, the…, spawn_assignments(), spawn_cell(), F040 — Hand spawn placement and first-hour loss, A hand hired at hour 0 first acts at hour 1: 23 actions against 24, _spawn_hand uses the least-occupied shed-access tile, ties broken NWSE, Unit spawns nest: unit u stands identically for crew u or u+3

### Community 145 - "Held cap is its own deadline"
Cohesion: 0.29
Nodes (8): COW (cost 400, PASTURE, day 8, every 2 days, cap 6, MILK), F021 — Care bank lumps before first yield, F026 — One-shot harvest calendar and deadline, Past max_yield_day the plant decays one unit per two turns, One-shot harvest: single yield, destroys the plant, F028 — Animal harvest calendar and cap deadline, Animal harvest calendar (no age guard on HARVEST), Held cap is its own deadline

### Community 146 - "classify.py"
Cohesion: 0.29
Nodes (7): classify_from_sequences(), eq(), ClassVerdict, Class probe: P (open-loop) / S (self-reactive) / R (reactive), behaviourally.…, Pure comparison of four action sequences -> P / S / R., Exact equality of two action dicts (order-sensitive on lists)., _same_action()

### Community 147 - "test_residual_wire.py"
Cohesion: 0.32
Nodes (6): Guards for the forecast's residual wire (#65 step 1). The claim: feeding the…, A model that expects rival supply must price the future LOWER., On scripted self-play, model-fed residual halves the oracle gap. The rival…, _sim_to(), test_the_model_fed_path_sits_toward_the_oracle(), test_the_wire_moves_the_price_path_down()

### Community 148 - "_get()"
Cohesion: 0.29
Nodes (7): _apply_sells(), _get(), Any, The engine's own configuration lookup (`K.get`), applied to ours., The observation's absolute turn index (day * 24 + hour)., A SELL of `units` lands in the market: +1 per unit unless price is 1. Engine…, _step_of()

### Community 149 - "Edge"
Cohesion: 0.29
Nodes (6): Edge, One CSR edge, decoded: `from_id -> to_id` by `chain_id`, for `entity_code`.…, _edge(), Synthetic edge for the dominance unit tests (order = RESOURCE_ID)., A difference in ONE component keeps both edges (2026-09-14): 1 wheat is not 1…, test_dominance_compares_components()

### Community 150 - ".decode()"
Cohesion: 0.33
Nodes (5): animal_cycle_age(), Any, An animal's age: growing up (negative), or the production phase…, The engine's `tiles[y][x]` at a day start, as our tile. `day` is the day the…, The engine's `tiles[y][x]` at any hour, as our tile. Named differently from…

### Community 151 - "F029 — Season structure and no liquidation day"
Cohesion: 0.33
Nodes (7): F024 — Animals cannot be unplaced, can be lost in the shed, 100-unit shed overflow can destroy a held animal, Placed animals cannot be taken back (DIG returns early), F029 — Season structure and no liquidation day, F048 (referenced finding: final-day turn accounting), No liquidation day: end-of-season shed goods are worthless, Season structure: 720 turns, turnsPerDay 24 = 30 days

### Community 152 - "townShopSellInterval = 4 steps"
Cohesion: 0.43
Nodes (7): F035 — Prices rise through the season, Holding produce and selling late, Season price inflation: the town out-consumes a single farm, F037 — Town shop consumption cadence, townCenterSellInterval = 24 steps, townShopSellInterval = 4 steps, townShopUnlockInterval = 3 days, drawn with replacement

### Community 153 - "Wrong-tile ops refused in silence (F047)"
Cohesion: 0.29
Nodes (7): tests/test_day_plan.py per-op instrument, HIRE settles after the turn's unit actions, day/routing.py::plan_day spawns from post-move occupancy, _process_market runs after unit actions, Wrong-tile ops refused in silence (F047), Missing step plays PASS, Hand a copy, never the live view

### Community 154 - "_apply_game_patch()"
Cohesion: 0.33
Nodes (7): _apply_game_patch(), _bump_money(), _dig(), _farm_of(), _plant_melons(), Every empty tile of ours gets a MELON, using the env's own constructor. The…, Fired from the interceptor on every env step, but RESTRAINED: money once per…

### Community 155 - "search_cost.py"
Cohesion: 0.43
Nodes (6): _day(), main(), measure(), What the day search costs on the archive's own days, day by day. Not a test - a…, One day: what the search cost, and what it managed to place., report()

### Community 156 - "kaggle-environments==1.32.7 — pinned EXACTLY"
Cohesion: 0.29
Nodes (7): Rationale: a version bump can change game behavior under every simulation at once, world/fast_sim.py drives the environment's own interpreter through private APIs, kaggle-environments==1.32.7 — pinned EXACTLY, ortools>=9.15, scipy>=1.17 — Kaggle environment and optimization core, Python 3.11 venv built with uv (uv venv .venv --python 3.11), Pin bumps ship in the same commit as tests/test_world_parity.py (R002/R003), test_installed_version_matches_the_requirements_pin()

### Community 157 - "test_own_supply_gap.py"
Cohesion: 0.38
Nodes (6): main(), _moving_ladder_coins(), The #110 own-supply finding, re-measured against the NEW master. The master's…, The engine's own answer: sell `units` one at a time, each quoted at the shared…, 21 melons at the flat path price vs the engine's moving ladder: the gap is the…, test_the_flat_price_overstates_the_moving_ladder()

### Community 158 - "._registry_tag()"
Cohesion: 0.40
Nodes (3): Write the artifact: CSR arrays, cost/produce matrices, metadata., The stamp of the chain table THIS graph indexes into. The builder writes the…, The engine tag with the registry part re-stamped (see `_registry_tag`).

### Community 159 - "F016 — Animal placement and species table"
Cohesion: 0.33
Nodes (6): F016 — Animal placement and species table, Animal placement requires COOP or PASTURE, SHEEP (cost 500, PASTURE, day 6, every 3 days, cap 6, WOOL), Animal species table (GOOSE/COW/SHEEP), F025 — Cow cap asymmetry, COW cap asymmetry: cap 6 against a wait of 8

### Community 160 - "task_nn_numpy()"
Cohesion: 0.40
Nodes (5): _nn_flops(), task_nn_jax_bench(), task_nn_numpy(), fwd(), task_nn_torch()

### Community 161 - "_chunk_pymc()"
Cohesion: 0.33
Nodes (6): _chunk_pymc(), env_scan(), _is_env_like(), Is this a live environment object? The class is created dynamically…, PyMC doing inference work: the compiled model logp, evaluated flat out., _spec_present()

### Community 162 - "_run_autopsy()"
Cohesion: 0.33
Nodes (5): _autopsy_worker_code(), Re-run one paired episode with full per-turn capture and dump JSON. The autopsy…, _run_autopsy(), _full(), Resolved episode seed (trainer-side only). resolve_episode_seed scrubs…

### Community 163 - "test_day_suite.py"
Cohesion: 0.33
Nodes (4): tests/day_layer — faithful pytest-shaped port (F052) driven by tests/test_day.py, pytest>=8 — part of the test path, subprocess, The day layer, run under Chista's one-command convention. Run: .venv/bin/python…

### Community 164 - "entity_of_code()"
Cohesion: 0.40
Nodes (4): _entities(), What each day's chosen edge constructs on this tile, by name., entity_of_code(), Inverse of `entity_code_of` (0 = none; an out-of-range code raises).

### Community 165 - "_make_overrun()"
Cohesion: 0.40
Nodes (4): _make_overrun(), _make_thread_child(), task(), One child per turn. The only thread sweep that always works. In-process…

### Community 166 - "drain_per_day()"
Cohesion: 0.50
Nodes (4): drain_per_day(), What the town takes out of the market on one turn (kaggriculture.py:728-749).…, The same drain over one whole day, for a plan that thinks in days., town_drain()

### Community 167 - "Past about six hands the order queue is the cons"
Cohesion: 1.00
Nodes (3): F041 — The order queue binds before the wage, Past about six hands the order queue is the constraint, The wage is never the constraint

### Community 168 - "F044 — Animal age collapses to a residue"
Cohesion: 1.00
Nodes (3): F044 — Animal age collapses to a residue, An animal's age matters only through a residue, The per-tile animal DP is flat over the residue

### Community 169 - "task_multiprocessing()"
Cohesion: 0.67
Nodes (3): _mp_noop(), Does a second PROCESS buy throughput, or does the cgroup quota just split the…, task_multiprocessing()

## Ambiguous Edges - Review These
- `test_observation_mutation_is_guarded_in_dev()` → `R004 as cited by the opponents docs — hand the agent a copy of the observation, never the live view`  [AMBIGUOUS]
  opponents/adaptive-public-state-multi-route/SOURCE.md · relation: conceptually_related_to
- `R002 — Never Transcribe Game Rules` → `Crops and Animals — Every Rule, Numbered (AgriOracle research document)`  [AMBIGUOUS]
  AGENTS.md · relation: conceptually_related_to
- `Import-time self-unpacking — exec_module triggers the payload unpack` → `agent.py payload, byte-for-byte — SHA-256 aca6dcddafd6fd6b…`  [AMBIGUOUS]
  opponents/economics-driven-rule-agent-ecobot-v6/SOURCE.md · relation: conceptually_related_to
- `Import-time self-unpacking — exec_module triggers the payload unpack` → `agent.py payload, byte-for-byte — SHA-256 943e8c114ace4f5c…`  [AMBIGUOUS]
  opponents/frontier-the-soil-remembers-rain/SOURCE.md · relation: conceptually_related_to
- `Import-time self-unpacking — exec_module triggers the payload unpack` → `agent.py payload, byte-for-byte — SHA-256 0c3b4002c657f427…`  [AMBIGUOUS]
  opponents/precomputed-schedule-policy/SOURCE.md · relation: conceptually_related_to
- `agent.py payload, byte-for-byte — SHA-256 2fe7118964656263…` → `agent.py payload, byte-for-byte — SHA-256 12eb55e1e2455a2b…`  [AMBIGUOUS]
  opponents/farming-score-a-mathematical-approach/SOURCE.md · relation: semantically_similar_to

## Knowledge Gaps
- **124 isolated node(s):** `ClassVerdict`, `Findings & Rules File Convention`, `OpenBLAS Thread Collapse at 3-4 Runtime Threads`, `Machine Drift of ±25 % across a Season`, `Error Ladder — What the Harness Survives` (+119 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 1691 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **14 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `test_observation_mutation_is_guarded_in_dev()` and `R004 as cited by the opponents docs — hand the agent a copy of the observation, never the live view`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `R002 — Never Transcribe Game Rules` and `Crops and Animals — Every Rule, Numbered (AgriOracle research document)`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Import-time self-unpacking — exec_module triggers the payload unpack` and `agent.py payload, byte-for-byte — SHA-256 aca6dcddafd6fd6b…`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Import-time self-unpacking — exec_module triggers the payload unpack` and `agent.py payload, byte-for-byte — SHA-256 943e8c114ace4f5c…`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Import-time self-unpacking — exec_module triggers the payload unpack` and `agent.py payload, byte-for-byte — SHA-256 0c3b4002c657f427…`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `agent.py payload, byte-for-byte — SHA-256 2fe7118964656263…` and `agent.py payload, byte-for-byte — SHA-256 12eb55e1e2455a2b…`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **Why does `FastSim` connect `FastSim` to `Config`, `equilibrate()`, `test_door_work_before_load_day.py`, `belief/__init__.py`, `pathlib`, `chains.py`, `forecast()`, `test_after_drop_not_loaded_day.py`, `test_seed_invariance_no_rng_in_tile_transitions(`, `dispatch_plan()`, `R004 as cited by the opponents docs — hand the a`, `test_residual_wire.py`, `fast_sim.py`, `test_market_layer.py`, `bench_market_forecast.py`, `test_slot_circuit.py`, `_run_autopsy()`, `test_market_wsr_check.py`, `test_belief_depth.py`, `MarketTracker`, `test_tile_dp.py`, `bench_paths.py`, `test_agent_obs.py`, `test_world_parity.py`, `test_season_horizon.py`, `test_market_hourly.py`, `runner.py`, `probe_hidden_state.py`, `guarded_call()`, `test_drop_empties_bag_day.py`, `test_drop_door_on_route_day.py`, `test_world_branch_purity.py`, `F055 — SELL reads the shed, not the bag`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._