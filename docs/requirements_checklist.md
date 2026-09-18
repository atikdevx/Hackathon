# Official Requirements Checklist

Audit of the implementation against both official documents.
**PASS** means verified by an automated test or a recorded manual run (evidence given).
**PARTIAL** means implemented, but it depends on an external or manual action not yet done, or it is verified only in part.
**N/A** means not applicable to this service.

PS = Problem Statement, PG = Participant Guide & Evaluation Rubric.

## API contract (PS §6, §7, §10)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 1 | `GET /health` returns 200 and `{"status":"ok"}` | PASS | `tests/test_api.py::test_health`; `curl` against local and Docker runs |
| 2 | `POST /optimize-energy` has the exact endpoint name | PASS | `app/main.py`; all API tests |
| 3 | Request fields scenario_id, operator_notes[1..3], hours[24], battery with 5 fields | PASS | `app/models/schemas.py`; `test_missing_*`, `test_invalid_operator_notes_is_400` |
| 4 | Hours are unique integers 0..23, one per hour | PASS | `test_duplicate_hours_is_400`, `test_missing_hour_is_400`, `test_out_of_range_hour_is_400` |
| 5 | Notes are non-empty strings | PASS | `test_invalid_operator_notes_is_400` (blank, empty, non-string) |
| 6 | Numbers are finite JSON numbers (no strings, bools, NaN) | PASS | `test_invalid_numeric_types_are_400`, `test_nan_is_400` |
| 7 | 400 for malformed JSON or structurally invalid requests | PASS | `test_malformed_json_is_400`, `test_non_object_json_is_400` |
| 8 | 422 (optional) for semantically invalid requests | PASS | `test_semantically_invalid_values_are_422`, `test_infeasible_directives_return_422` |
| 9 | 500 only for controlled internal errors, without secrets or stack traces | PASS | `tests/test_reliability.py` (`*_does_not_leak`, `*_not_logged`) |
| 10 | Response has all 7 top-level fields; scenario_id is echoed | PASS | `test_valid_request_returns_exact_schema`; `scripts/judge.py::check_schema` on every sample |
| 11 | directive_interpretation entries have the 5 fields, in note_index order | PASS | stage-2 validator; judge schema check |
| 12 | hourly_plan has 24 entries with the 6 fields, sorted 0..23 | PASS | `app/services/schedule.py`; `test_hours_out_of_order_are_accepted_and_plan_sorted` |
| 13 | battery_action is charge, discharge or idle; battery_kwh ≥ 0 and 0 when idle | PASS | schedule netting; replay and judge checks |
| 14 | JSON in and JSON out, including errors and 404 | PASS | `test_unknown_route_is_json_404`; error handlers in `app/main.py` |

## LLM interpretation (PS §2, §4, §5.1, §11.1, §11.4; PG §4)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 15 | A language-capable generative model interprets every note on the path to the optimizer | PASS | `app/llm/interpreter.py` → `pipeline.run_pipeline`; live runs with a real model (README, Verification results) |
| 16 | No phrase matching as sole interpreter; no hard-coded sample phrases, IDs or values | PASS | no keyword rules exist; failures raise controlled errors (`test_provider_down_is_controlled`) |
| 17 | Exactly one entry per note, in order 0..N-1 | PASS | both guardrail stages; `test_invalid_outputs_rejected[duplicate/missing/out of order]` |
| 18 | Only the six directive types | PASS | JSON-schema enum plus `test_unsupported_directive_never_applied` |
| 19 | no_op uses applies=false and adjustment=null; other types use applies=true | PASS | stage 1 derives it; stage 2 enforces it (`no_op applies true`, `directive applies false` tests) |
| 20 | Exact structured_adjustment shapes | PASS | `ADJUSTMENT_KEYS`; `test_each_directive_shape_accepted`, extra/missing-key rejection |
| 21 | Hours are start-inclusive, end-exclusive (1–3 PM → [13,14]) | PASS | `tests/test_llm_output.py::test_expand_window`; live `PV production ... 13:00 and 15:00` |
| 22 | 12-hour and 24-hour time expressions | PASS | live semantic cases (7B local model); prompt rules |
| 23 | factor is the remaining fraction (80% reduction → 0.2) | PASS | live SAMPLE-09 and `Expect an 80% reduction ...` |
| 24 | Percentages and equivalent numeric wording (share of capacity, one-fifth, MWh) | PASS | live `75% full` → 180, `0.2 MWh` → 200, SAMPLE-03 (50% → 100) |
| 25 | Paraphrase robustness | PARTIAL | 29/30 live paraphrase cases with the local 7B model; not yet run against the production model `gpt-5.6-terra` because no OpenAI key was available (run `pytest -m live`) |
| 26 | LLM must not invent demand, tariff, solar or battery values | PASS | the model never receives or returns those fields; extra keys rejected (`test_extra_or_missing_fields_rejected`, `extra adjustment key`) |
| 27 | Structured-output / JSON-schema invocation | PASS | OpenAI `response_format.json_schema` with `strict: true` (`test_openai_reasoning_model_request_shape`); Claude `output_config.format` (`test_request_shape_and_text_extraction`) |
| 28 | Provider/model configurable through env; no hard-coded credentials | PASS | `app/config.py`, `.env.example`; `test_api_key_not_in_settings_repr` |
| 29 | Model/provider documented | PASS | README table and Configuration section |

## Guardrails (PS §8)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 30 | Allowed types only | PASS | `unknown type` rejection tests |
| 31 | Note mapping: existing index, each once | PASS | `duplicate note mapping`, `missing note mapping`, `note index out of range` |
| 32 | Hours are unique integers 0..23, ascending | PASS | `duplicate hours`, `unsorted hours`, `hour 24`, `negative hour`, `float hour`, `bool hour` |
| 33 | Solar factor in [0,1] | PASS | `factor < 0`, `factor > 1`, `factor nan` |
| 34 | Reserve is finite, ≥ 0 and ≤ capacity | PASS | `reserve > capacity`, `reserve negative`, `reserve inf` |
| 35 | Grid cap is finite and ≥ 0 | PASS | `test_negative_grid_cap_rejected` |
| 36 | applies semantics | PASS | see #19 |
| 37 | No invention | PASS | see #26 |
| 38 | Safe failure: malformed or unsupported output gives a controlled error, no crash, no invented directive | PASS | `test_malformed_json_twice_is_controlled_failure`, `test_malformed_llm_output_over_api` |
| 39 | Final replay verifies every extracted directive | PASS | `app/validation/replay.py`; `test_replay_failure_blocks_response` |

## Directive application and energy rules (PS §5.3, §9, §11.2, §11.3)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 40 | effective_solar = solar × factor | PASS | `test_solar_reduction_changes_effective_solar`; judge replay |
| 41 | Reserve: E_after ≥ max(base, directive) | PASS | `test_minimum_reserve_window` |
| 42 | No-charge hours have charge = 0 | PASS | `test_no_charge_window` |
| 43 | No-discharge hours have discharge = 0 | PASS | `test_no_discharge_window` |
| 44 | grid ≤ max_grid_kwh in listed hours | PASS | `test_grid_cap_window` |
| 45 | Multiple directives compose | PASS | `test_multiple_simultaneous_directives`, `test_overlapping_directives_compose_conservatively` |
| 46 | Battery transitions charge/discharge/idle | PASS | replay and judge in every optimizer and sample test |
| 47 | minimum ≤ E_after ≤ capacity | PASS | `test_battery_minimum_respected`; replay |
| 48 | Charge and discharge rate limits | PASS | `test_rate_limits_respected` |
| 49 | 0 ≤ solar_used ≤ effective solar; curtailment; no export | PASS | `test_high_solar_is_curtailed_not_exported` |
| 50 | Hourly energy balance | PASS | LP equality; replay; judge |
| 51 | End-of-day neutrality | PASS | `test_end_of_day_neutrality_exact` (exact to 1e-9) |
| 52 | All numeric outputs finite and non-negative | PASS | replay and judge checks |
| 53 | Totals and peak recalculated from hourly_plan | PASS | `compute_totals`; replay and judge totals checks |
| 54 | Numeric tolerance 0.01 | PASS | outputs exact to 1e-6; internal replay tolerance 1e-4 |

## Optimization (PS §5.2; PG §7)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 55 | Minimize Σ grid × tariff after all directives | PASS | `app/optimizer/lp.py` (HiGHS LP, exact optimum) |
| 56 | Validity before cost | PASS | infeasible → 422; replay gate before responding |
| 57 | Optimal cost on public samples | PASS | 10/10 equal to the reference cost (difference 0.00 BDT) |
| 58 | Equivalent optimal schedules accepted | PASS | judge compares cost and validity, not the hour sequence |
| 59 | LLM not used for the math | PASS | the LLM only produces directives |

## Performance and reliability (PG §3, §8)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 60 | /health ready within 60 s of start | PASS | about 1.8 s in Docker (measured) |
| 61 | Each request finishes within 30 s | PASS | total LLM budget capped at 25 s (`TOTAL_LLM_BUDGET_SECONDS`); `test_timeout_is_controlled_and_bounded`; the LP takes a few ms |
| 62 | p95 ≤ 5 s | PARTIAL | pipeline overhead is under 20 ms and cache hits take 0.01 s; p95 with `gpt-5.6-terra` is not measured without a key. The local 7B model on a laptop gave p95 of about 24 s. If it is too slow, set `LLM_EFFORT=none` or `LLM_MODEL=gpt-5.6-luna`. |
| 63 | Stable across repeated requests | PASS | `test_repeated_api_requests_are_stable` (15×); 25+ live requests without errors |
| 64 | Provider timeout, rate limit, outage, malformed output handled | PASS | `tests/test_reliability.py`, `tests/test_openai_provider.py`, `tests/test_anthropic_provider.py` |
| 65 | Optimizer failure and infeasibility handled | PASS | `test_optimizer_failure_is_controlled`, `test_infeasible_directives_return_422` |
| 66 | No secrets in repo, logs or responses | PASS | secret-leak tests; repo grep; `.gitignore` / `.dockerignore` |

## Deployment and Docker (PG §2, §3, §7)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 67 | Dockerfile builds; binds 0.0.0.0; exposes the documented port | PASS | built and run locally; `EXPOSE 8000`; `--host 0.0.0.0` |
| 68 | No secrets baked into the image | PASS | `docker history` checked; key is supplied at run time |
| 69 | /health works in the container; full pipeline works in the container | PASS | 10/10 public samples through the container with a real LLM |
| 70 | Pullable registry image with an exact tag or digest | PARTIAL | **Manual:** push `ghcr.io/<user>/gridwise-llm:1.0.0` (commands in README) and make the package public |
| 71 | Public endpoint reachable, no auth, alive during judging | PARTIAL | **Manual:** deploy (`render.yaml` provided) and set `LLM_API_KEY` as a host secret |
| 72 | Both endpoints tested from outside the dev environment | PARTIAL | **Manual:** after deploying, run `scripts/run_public_samples.py --base-url <public URL>` |
| 73 | LLM available during judging (keys, quota) | PARTIAL | **Manual:** a funded OpenAI key (with enough rate limit) must be configured on the host |

## Documentation and repository (PG §2, §4, §5, §7)

| # | Requirement | Status | Evidence |
|---|---|---|---|
| 74 | README: setup, env names, model/provider, LLM role, guardrails, solver, run command, curl examples, sample test, dependencies, limitations, secret handling | PASS | `README.md` |
| 75 | Copy-paste quickstart from a clean environment | PASS | README Local quickstart, run verbatim on a clean copy of the repo (fresh venv, `pip install -r requirements-dev.txt`, `pytest`, `python -m app`, `/health`, sample runner) |
| 76 | Docker pull/run fallback instructions | PASS | README Docker section |
| 77 | External tools and libraries credited | PASS | README Dependencies and attribution |
| 78 | `.env.example` with names only | PASS | `.env.example` |
| 79 | Repo created after question reveal, private during the event, public after the deadline | PARTIAL | **Manual:** visibility was not changed automatically. Make it public after the deadline. |
| 80 | 3-minute video | PARTIAL | script in `docs/video_script.md`; **manual:** record and upload (≤ 3:00) |
| 81 | Only synthetic challenge data | PASS | no other data is used |
