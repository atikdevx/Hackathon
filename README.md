# GridWise: LLM-Assisted Smart Campus Energy Optimizer

BUP CSE Fest 2026 Hackathon, Online Preliminary.

GridWise is one HTTP service. It takes a 24-hour campus energy scenario (demand, solar, tariff, battery) plus 1–3 natural-language operator notes. It then:

1. uses a **language model (OpenAI GPT)** to turn each note into exactly one structured directive, or `no_op`;
2. checks that output with **two stages of deterministic guardrails** before any of it is used;
3. applies the accepted directives as hard constraints in a **linear program** that minimizes grid cost;
4. **replays** the finished schedule hour by hour and returns it only if every rule holds.

| Item | Value |
|---|---|
| Endpoints | `GET /health`, `POST /optimize-energy` |
| Language / framework | Python 3.12, FastAPI, Pydantic v2 |
| LLM provider / model | OpenAI, `gpt-5.6-terra` by default, via Chat Completions with Structured Outputs (configurable, see [Configuration](#configuration)) |
| Solver | SciPy `linprog` with the HiGHS LP solver (exact, deterministic) |
| Port | `8000` (override with `PORT`) |
| Docker image | `ghcr.io/atikdevx/gridwise-llm:1.0.0` (see [Docker](#docker)) |

---

## Contents

- [Architecture](#architecture)
- [The LLM's role](#the-llms-role)
- [Supported directives](#supported-directives)
- [Guardrails](#guardrails)
- [Verification results](#verification-results)
- [Optimizer and energy rules](#optimizer-and-energy-rules)
- [Requirements](#requirements)
- [Configuration](#configuration)
- [Local quickstart](#local-quickstart)
- [API examples](#api-examples)
- [Testing](#testing)
- [Docker](#docker)
- [Deployment](#deployment)
- [Project layout](#project-layout)
- [Dependencies and attribution](#dependencies-and-attribution)
- [Security and secret handling](#security-and-secret-handling)
- [Known limitations](#known-limitations)

---

## Architecture

```
HTTP request
  │
  ▼
Request validation ── 400 malformed/structural, 422 physically impossible values
  │
  ▼
LLM interpreter (OpenAI GPT, one call for all notes, JSON-schema-constrained output)
  │   untrusted JSON: type, stated time windows, value, explanation per note
  ▼
Guardrail stage 1: strict raw-output check + window expansion (start-inclusive, end-exclusive)
  ▼
Guardrail stage 2: official-contract check (types, applies, exact shape, hours, ranges)
  │   reject at either stage? → one repair turn → still bad? → controlled 500, nothing applied
  │   validated directives
  ▼
Directive application (effective solar, active reserve, charge/discharge limits, grid caps per hour)
  │
  ▼
LP optimizer (HiGHS): minimize Σ grid·tariff, then minimize battery cycling at that same cost
  │
  ▼
Schedule assembly (net charge/discharge, exact neutrality) + totals recalculated from hourly_plan
  │
  ▼
Final replay validator ── any violation → controlled 500, schedule is not returned
  │
  ▼
JSON response
```

Each stage is its own module (see [Project layout](#project-layout)). Business rules live in one place each. Directive math is in `app/directives/model.py`. Guardrails are in `app/guardrails/validator.py`. The replay validator re-checks every directive directly against the plan, so a bug in directive composition or in the optimizer cannot slip through.

## The LLM's role

The LLM sits on the interpretation path, and its output is what the optimizer uses. It is not decoration.

- **One model call per request** interprets all 1–3 notes (`app/llm/interpreter.py`). This keeps latency low.
- **Structured outputs:** the request carries a strict JSON Schema (`app/llm/prompt.py`). For each note the model returns exactly: `explanation` (written first, which works as a short reasoning step), `note_index`, `directive_type` (an enum of the six types), `time_windows` (the start and end of each period exactly as the note states it, on a 24-hour clock), and the single value its type needs (`factor`, `minimum_energy_kwh` or `max_grid_kwh`, with the others `null`).
- **The model reads; code counts.** The model makes every language decision: relevance, directive type, AM/PM, numeric values and percentage conversion. Deterministic code (`app/guardrails/llm_output.py`) expands each stated window into hours using the spec's rule: start included, end excluded, wrapping past midnight. The model never enumerates hours itself. This removed the off-by-one errors we measured when models listed hours directly (see [Verification results](#verification-results)). It is the deterministic post-processing for normalization that the Participant Guide allows.
- **What the model sees:** the notes, plus the battery `capacity_kwh` and base minimum. It needs these to turn "keep 50% of capacity" into kWh. It never sees demand, solar or tariff data. There is no output field for them, so it cannot change them.
- **Prompt rules:** window boundaries (`1 PM to 3 PM → start 13, end 15 → hours [13, 14]`), 12- and 24-hour clocks, midnight wrap-around, battery charge vs. discharge direction, `factor` as the fraction of solar that remains (`80% reduction → 0.2`), percentage-of-capacity reserves, unit conversion (MWh→kWh, kW over 1 h = kWh), and when a note is a `no_op` distractor.
- **Provider abstraction:** `app/llm/base.py` defines a small `LLMProvider` protocol. The default `OpenAICompatibleProvider` calls OpenAI Chat Completions with `response_format: json_schema` (`strict: true`). For reasoning models (GPT-5.x/6, o-series) it sends `reasoning_effort` and leaves out `temperature`, which those models reject. The same class works with any OpenAI-compatible server (Ollama, vLLM, Docker Model Runner, Groq, OpenRouter). `AnthropicProvider` (Claude) is an alternative. Adding another provider means writing one class.
- **No phrase-matching fallback.** If the model is unavailable or its output fails the guardrails, the request fails with a controlled error. It never quietly falls back to hard-coded rules.
- **Safe cache:** identical `(notes, capacity, base minimum, model, prompt version)` inputs reuse a validated interpretation in memory. Repeated requests then skip the model call. The cache never changes scenario numbers, because the schedule is always re-optimized.

## Supported directives

| `directive_type` | `structured_adjustment` | Effect on the optimizer |
|---|---|---|
| `solar_reduction` | `{"hours": [...], "factor": f}` | `effective_solar[h] = solar[h] × f` |
| `minimum_battery_reserve` | `{"hours": [...], "minimum_energy_kwh": x}` | `E_after[h] ≥ max(base minimum, x)` |
| `no_charge_window` | `{"hours": [...]}` | charge = 0 |
| `no_discharge_window` | `{"hours": [...]}` | discharge = 0 |
| `max_grid_window` | `{"hours": [...], "max_grid_kwh": g}` | `grid[h] ≤ g` |
| `no_op` | `null` (with `applies: false`) | no change |

If several directives cover the same hour, they are combined conservatively so that each one still holds: solar factors multiply, reserves take the maximum, and grid caps take the minimum.

## Guardrails

Model output is untrusted. Two deterministic stages **reject** bad output and never guess.

**Stage 1: raw extraction** (`app/guardrails/llm_output.py`)

- Exactly N entries with exactly the seven raw fields. `note_index` must equal the entry's position.
- `no_op` must have no windows and all values `null`. An applying type needs at least one window, and exactly its own value field must be a finite number while the others are `null`.
- Each window has integer `start_hour` in 0..23 and `end_hour` in 1..24, and it may not be empty. Windows are expanded (start inclusive, end exclusive, wrapping past midnight) and merged into a sorted `hours` list.

**Stage 2: official contract** (`app/guardrails/validator.py`), applied to the assembled entries

- Top level must be exactly `{"interpretations": [...]}`, with exactly N entries for N notes.
- Each entry has exactly the five required fields. `note_index` must equal its position, which rules out missing, duplicate or out-of-order mappings.
- `directive_type` must be one of the six allowed values.
- `no_op` ⇔ `applies = false` and `structured_adjustment = null`. Every other type needs `applies = true` and **exactly** the required keys, with no extras.
- `hours` must be a non-empty list of unique integers from 0 to 23, in ascending order (booleans and floats are rejected).
- `factor` must be finite and in [0, 1]. `minimum_energy_kwh` must be finite and in [0, battery capacity]. `max_grid_kwh` must be finite and ≥ 0.
- No invention: extra keys (for example, an attempt to set `demand_kwh` or a tariff) are rejected.

When validation fails, the model gets **one repair turn** that lists the violations. This happens only while the latency budget allows. If the second answer also fails, the API returns `500 {"error": {"code": "llm_output_rejected", ...}}` and no directive is applied.

## Verification results

These are the results recorded while building the service. Re-run them with the commands in [Testing](#testing).

| Check | Result |
|---|---|
| Offline suite (`python -m pytest`) | **all passed** (live tests skipped without a model) |
| Public samples through the full pipeline, with the reference interpretation fed in as the model output | 10/10, cost equal to the reference optimum (difference 0.00 BDT) |
| **Real LLM in the loop.** Local `ai/qwen2.5:7B-Q4_K_M` through Docker Model Runner, using the same OpenAI client code | `pytest -m live`: **29/30**. All 10 public samples pass end to end over HTTP, both from `python -m app` and from the Docker image. |
| Same small model, first design (model listed hours itself) | 21/30. The failures were mostly off-by-one hour lists, which led to the stage-1 window design. |
| `gpt-5.6-terra` (default production model) | Needs an OpenAI key: run `pytest -m live` and `scripts/run_public_samples.py` after setting `LLM_API_KEY`. The request format (structured outputs, `reasoning_effort`, no `temperature`) and error handling are covered offline by `tests/test_openai_provider.py`. |
| Docker image | Builds; runs as non-root; `/health` ready in about 2 s; `HEALTHCHECK` healthy; no secrets in the image |

The one remaining live miss with the 7B model is the paraphrase "Nobody may draw power from the storage bank", which it read as a charging ban. A frontier model is expected to handle this, but that is not verified here because no key was available.

## Optimizer and energy rules

`app/optimizer/lp.py` builds the LP. Each hour `h` has 5 variables: grid `g`, solar used `s`, charge `c`, discharge `d`, and energy after `E`.

```
minimize    Σ tariff_h · g_h
subject to  g_h + s_h + d_h = demand_h + c_h                 (energy balance)
            E_h = E_{h-1} + c_h − d_h,  E_{-1} = initial     (battery transition)
            E_23 = initial                                   (end-of-day neutrality)
            0 ≤ g_h ≤ grid_cap_h                             (no export; directive caps)
            0 ≤ s_h ≤ effective_solar_h                      (curtailment allowed)
            0 ≤ c_h ≤ max_charge_h   (0 in no-charge hours)
            0 ≤ d_h ≤ max_discharge_h (0 in no-discharge hours)
            active_min_h ≤ E_h ≤ capacity
```

- **Two-phase solve:** phase 1 finds the minimum cost. Phase 2 keeps that cost and minimizes total charge + discharge, so equally cheap schedules come out without pointless cycling. The total LP time is a few milliseconds.
- **Clean output:** charge and discharge in the same hour are netted into one `battery_action`. Netting leaves the balance and transition unchanged and can never exceed a rate limit. `battery_kwh` is always a non-negative magnitude, and it is `0` when the action is `idle`. Values are rounded to 6 decimals, and end-of-day energy is set to exactly the initial energy.
- **Totals** (`total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh`) are always recalculated from the returned `hourly_plan`.
- **Infeasible** scenarios (for example, a grid cap of 0 with discharge also forbidden) return `422 infeasible_scenario`, never an invalid plan.
- **Replay validator** (`app/validation/replay.py`) checks every condition from Problem Statement §9/§11 with a 1e-4 tolerance, 100× stricter than the judge's 0.01. It covers balance, effective solar, transitions, capacity, base minimum, reserve, rate limits, no-charge/no-discharge windows, grid caps, action consistency, neutrality, non-negativity, finiteness and totals.

## Requirements

- **Local run:** Python **3.12 or newer** and `pip` (the pinned NumPy/SciPy need 3.12+). If your system Python is older, use the `uv` variant below, which downloads Python 3.12 for you.
- **Docker run:** Docker 20+.
- **An LLM API key:** an OpenAI API key for the default configuration (or an Anthropic key with `LLM_PROVIDER=anthropic`).

## Configuration

All configuration comes from environment variables. `.env.example` lists every name with no values. Copy it to `.env` and add your key. `.env` is git-ignored and docker-ignored.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `openai` | `openai` (alias `openai_compatible`) or `anthropic` |
| `LLM_MODEL` | `gpt-5.6-terra` | Model id. Cheaper/faster: `gpt-5.6-luna`; stronger: `gpt-5.6-sol`. With `anthropic` the default is `claude-opus-5`. |
| `LLM_API_KEY` | none | **Required** for hosted providers. `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY` for `anthropic`) is also read. Optional for a local OpenAI-compatible server. |
| `LLM_BASE_URL` | `https://api.openai.com/v1` | Leave empty for OpenAI; set it for a proxy or another OpenAI-compatible server |
| `LLM_TIMEOUT_SECONDS` | `12` | Per-attempt model timeout |
| `LLM_MAX_ATTEMPTS` | `2` | Model attempts per request (attempt 2 is the guardrail repair turn); total LLM time is capped at 25 s |
| `LLM_EFFORT` | `low` | OpenAI `reasoning_effort` for reasoning models (`none`/`low`/`medium`/`high`), or Claude `output_config.effort` |
| `LLM_ENABLE_FALLBACKS` | `true` | Anthropic only: Claude Opus 5 server-side refusal fallbacks |
| `LLM_CACHE_SIZE` | `256` | In-memory interpretation cache size (`0` disables it) |
| `PORT` | `8000` | HTTP port |
| `LOG_LEVEL` | `INFO` | Python log level |
| `WEB_CONCURRENCY` | `2` | Uvicorn worker processes (Docker image only) |

**Model/provider used for this submission:** OpenAI `gpt-5.6-terra` through the OpenAI Chat Completions API, with Structured Outputs (`json_schema`, `strict: true`) and `reasoning_effort=low`.

To use Claude instead: `LLM_PROVIDER=anthropic LLM_API_KEY=<anthropic key>` (default model `claude-opus-5`).

**Keyless local model** (useful for offline reproduction). Any OpenAI-compatible local server works, for example Docker Model Runner or Ollama. No key is needed when `LLM_BASE_URL` is set:

```bash
docker desktop enable model-runner --tcp=12434 && docker model pull ai/qwen2.5:7B-Q4_K_M
LLM_PROVIDER=openai LLM_BASE_URL=http://localhost:12434/engines/v1 \
LLM_MODEL=ai/qwen2.5:7B-Q4_K_M LLM_TIMEOUT_SECONDS=25 python -m app
# inside Docker, use LLM_BASE_URL=http://host.docker.internal:12434/engines/v1
```

A 7B model on a laptop takes about 7–24 s per request, so use it only for reproduction, not for judging.

To run with a different OpenAI-compatible model:

```bash
LLM_PROVIDER=openai LLM_BASE_URL=https://api.groq.com/openai/v1 \
LLM_MODEL=<model-id> LLM_API_KEY=<key> python -m app
```

## Local quickstart

Copy and paste from a clean machine:

```bash
git clone https://github.com/atikdevx/Hackathon.git gridwise && cd gridwise
python3 --version             # must be 3.12+; otherwise use the uv variant below
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env          # then edit .env and set LLM_API_KEY=<your key>
set -a && source .env && set +a
python -m app                 # serves on http://0.0.0.0:8000
```

No Python 3.12 available? Use [uv](https://docs.astral.sh/uv/) (`pip install uv` or `brew install uv`) to create the environment:

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt
```

You can also start the server directly with uvicorn: `uvicorn app.main:app --host 0.0.0.0 --port 8000 --env-file .env`.

In a second terminal:

```bash
curl -s http://localhost:8000/health                  # {"status":"ok"}
python scripts/run_public_samples.py                  # posts all 10 public samples, judges each one
```

## API examples

### `GET /health`

```bash
curl -s http://localhost:8000/health
```
```json
{"status": "ok"}
```

### `POST /optimize-energy`

Extract one public sample and send it:

```bash
python3 -c "import json; print(json.dumps(json.load(open('BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json'))['cases'][0]['input']))" > /tmp/sample1.json
curl -s -X POST http://localhost:8000/optimize-energy \
     -H 'Content-Type: application/json' -d @/tmp/sample1.json
```

Response for public sample `SAMPLE-01` (hourly plan shortened to 3 of 24 entries; `explanation` wording varies between model runs):

```json
{
  "scenario_id": "SAMPLE-01",
  "directive_interpretation": [
    {"note_index": 0, "applies": true, "directive_type": "solar_reduction",
     "structured_adjustment": {"hours": [12, 13], "factor": 0.25},
     "explanation": "Panel washing leaves about 25% of forecast solar from 12:00 to 14:00."},
    {"note_index": 1, "applies": false, "directive_type": "no_op",
     "structured_adjustment": null,
     "explanation": "A registration deadline does not affect the energy schedule."}
  ],
  "hourly_plan": [
    {"hour": 0, "grid_kwh": 90.0, "solar_used_kwh": 0.0, "battery_action": "idle", "battery_kwh": 0.0, "battery_energy_after_kwh": 110.0},
    {"hour": 1, "grid_kwh": 45.0, "solar_used_kwh": 0.0, "battery_action": "discharge", "battery_kwh": 40.0, "battery_energy_after_kwh": 70.0},
    {"hour": 2, "grid_kwh": 130.0, "solar_used_kwh": 0.0, "battery_action": "charge", "battery_kwh": 50.0, "battery_energy_after_kwh": 120.0}
  ],
  "total_grid_kwh": 2692.5,
  "total_cost_bdt": 38365.0,
  "peak_grid_kwh": 175.0,
  "plan_summary": "Applied solar limited to 25% of forecast during 12:00-14:00. Ignored 1 note(s) that do not affect today's energy schedule. Battery charges 335 kWh in cheaper hours (avg 8.54 BDT/kWh) and discharges 335 kWh in pricier hours (avg 20.57 BDT/kWh), ending at its initial 110 kWh. Grid import 2692.5 kWh, cost 38365.00 BDT, peak 175 kWh."
}
```

### Error responses

Every error is JSON in the form `{"error": {"code", "message", "details?"}}`. Stack traces, prompts and secrets are never included.

| Status | `code` | When |
|---|---|---|
| 400 | `invalid_request` | Malformed JSON, a non-object body, missing fields, wrong types (strings/booleans/NaN as numbers), not 1–3 non-empty notes, hours not exactly 0..23 once each |
| 422 | `semantically_invalid_request` | Negative demand/solar/battery values, minimum > capacity, initial energy outside [minimum, capacity] |
| 422 | `infeasible_scenario` | No schedule satisfies the rules plus the interpreted directives |
| 500 | `llm_unavailable` | No key configured, provider timeout, rate limit, or outage |
| 500 | `llm_output_rejected` | Model output failed the guardrails (after the repair turn); nothing was applied |
| 500 | `optimizer_failure` / `schedule_validation_failed` / `internal_error` | Controlled internal failures |

## Testing

```bash
pip install -r requirements-dev.txt

# Full offline suite (API, guardrails, optimizer, replay, reliability, all 10 public samples).
# The LLM is replaced by a scripted stand-in, so no key or network is needed.
python -m pytest

# Live semantic tests against the REAL model: 19 paraphrase/percentage/12h-24h/no_op notes,
# a 3-note call, and all 10 public samples end to end. Needs LLM_API_KEY.
set -a && source .env && set +a
python -m pytest -m live -v

# Public samples against a running server (local, Docker, or deployed):
python scripts/run_public_samples.py --base-url http://localhost:8000
python scripts/run_public_samples.py --base-url https://<your-deployment> --show
```

**Expected result:** every public case prints `PASS` with a cost equal to the reference optimum, then a summary line like `10/10 cases passed | latency median=…s p95=…s`.

`scripts/judge.py` is an **independent** re-implementation of the organizer checks. It uses only the standard library and shares no code with `app/`. For each case it checks:

- the response schema and the order of interpretation entries;
- semantic equality of each interpretation with the reference (type, applies, hours, values within 0.01; explanation text is ignored);
- a full replay of the plan under the **reference** directives;
- totals recalculated from the plan;
- cost equal to the reference optimum within 0.01 BDT. Different hour-by-hour schedules with the same optimal cost are accepted.

Test coverage by area:

| File | What it covers |
|---|---|
| `tests/test_api.py` | health, valid request and exact schema, malformed JSON, missing fields, note count, duplicate/missing/out-of-range hours, invalid numeric types, NaN, 422 semantics, unconfigured LLM |
| `tests/test_llm_output.py` | guardrail stage 1: window expansion (end-exclusive, midnight, wrap-around, all day), union of multiple windows, rejection of empty or out-of-range windows, missing/stray values, unsupported types, extra fields, wrong note mapping |
| `tests/test_openai_provider.py` | OpenAI request shape (endpoint, auth, strict `json_schema`, `reasoning_effort`, no `temperature` for reasoning models; `temperature=0` and no auth for local models), 400/401/403/404/429/5xx, timeouts, refusal/truncation/empty/non-JSON bodies, default-provider config |
| `tests/test_anthropic_provider.py` | Claude request shape (model, schema, effort, fallbacks beta), Haiku differences, 401/429/5xx/connection errors, refusal/truncation/empty handling, all through a mocked HTTP transport |
| `tests/test_guardrails.py` | unknown type, duplicate/unsorted/out-of-range/float/empty hours, factor < 0 or > 1, reserve > capacity, negative grid cap, wrong applies/no_op combinations, malformed adjustments, extra keys, duplicate/missing/out-of-order note mapping |
| `tests/test_optimizer.py` | tariff shifting (exact expected saving), no solar, high solar with curtailment, battery minimum, rate limits, no-charge, no-discharge, reserve, grid cap, 5 simultaneous directives, overlapping directives, exact neutrality, infeasibility, zero-capacity battery. Every plan is also replayed by the independent judge. |
| `tests/test_reliability.py` | guardrail repair retry, malformed LLM JSON, unsupported directive, provider error then recovery, timeout within budget, cache, 15 repeated requests, optimizer failure, replay failure blocks the response, no secrets in responses or logs |
| `tests/test_public_samples.py` | all 10 public samples through the HTTP pipeline, judged independently |
| `tests/test_llm_live.py` | real-model semantics (marked `live`) |

## Docker

The image is based on `python:3.12-slim`. It uses pinned dependencies, runs as a non-root user, has a built-in `HEALTHCHECK`, binds to `0.0.0.0:$PORT` (default 8000), and contains **no secrets**.

```bash
# Build locally
docker build -t gridwise-llm:1.0.0 .

# Run (pass the key at run time)
docker run --rm -p 8000:8000 -e LLM_API_KEY=<your-key> gridwise-llm:1.0.0
# or: docker run --rm -p 8000:8000 --env-file .env gridwise-llm:1.0.0

curl -s http://localhost:8000/health
python scripts/run_public_samples.py --base-url http://localhost:8000
```

Pull the published fallback image instead of building:

```bash
docker pull ghcr.io/atikdevx/gridwise-llm:1.0.0
docker run --rm -p 8000:8000 -e LLM_API_KEY=<your-key> ghcr.io/atikdevx/gridwise-llm:1.0.0
```

To publish the image (maintainers):

```bash
# GitHub Container Registry (needs a PAT with write:packages)
echo $GITHUB_TOKEN | docker login ghcr.io -u <github-user> --password-stdin
docker buildx build --platform linux/amd64,linux/arm64 -t ghcr.io/<github-user>/gridwise-llm:1.0.0 --push .
# Then set the package to Public in GitHub → Packages so judges can pull without logging in.

# Docker Hub alternative
docker login -u <dockerhub-user>
docker buildx build --platform linux/amd64,linux/arm64 -t <dockerhub-user>/gridwise-llm:1.0.0 --push .
```

Record the pushed digest (`docker buildx imagetools inspect <image>:1.0.0`) in the submission form.

## Deployment

Any host that runs a container and sets `PORT` will work. `render.yaml` is a ready-made blueprint for [Render](https://render.com):

1. Push this repository to GitHub.
2. In Render, choose **New → Blueprint** and select the repo. Render builds the `Dockerfile`.
3. Set the secret environment variable `LLM_API_KEY` in the dashboard. It is never stored in the repo.
4. After the deploy, check both endpoints from outside your network:
   `curl https://<service>.onrender.com/health` and
   `python scripts/run_public_samples.py --base-url https://<service>.onrender.com`.

Keep the service on an always-on plan during judging, because free instances sleep and cold starts are slow. The same image also runs unchanged on Railway, Fly.io, Google Cloud Run, Azure Container Apps and similar hosts: set `LLM_API_KEY` and expose the port.

## Project layout

```
app/
  main.py                    FastAPI app, routes, error handlers
  config.py                  environment-driven Settings
  errors.py                  controlled error types -> JSON bodies
  models/schemas.py          request/response models, directive enum, exact adjustment keys
  llm/prompt.py              system prompt, JSON schema, repair message
  llm/base.py                LLMProvider protocol
  llm/openai_compatible_provider.py  OpenAI Chat Completions + Structured Outputs (default)
  llm/anthropic_provider.py  Claude alternative (structured outputs, effort, refusal fallbacks)
  llm/interpreter.py         single call, guardrails, repair turn, time budget, cache
  llm/factory.py             builds the provider/interpreter from settings
  guardrails/llm_output.py   stage 1: raw model output check + window expansion
  guardrails/validator.py    stage 2: official-contract validation
  directives/model.py        validated directives -> per-hour limits
  optimizer/lp.py            two-phase HiGHS linear program
  services/schedule.py       hourly_plan assembly and totals
  services/request_checks.py semantic (422) request checks
  services/summary.py        deterministic plan_summary
  services/pipeline.py       end-to-end orchestration
  validation/replay.py       final independent replay validator
scripts/judge.py             independent organizer-style checker (stdlib only)
scripts/run_public_samples.py  runs the public samples against any base URL
tests/                       pytest suite
docs/                        requirements checklist, video script
Dockerfile, .dockerignore, render.yaml, .env.example
requirements.in / requirements.txt (fully pinned lock) / requirements-dev.txt
```

## Dependencies and attribution

| Library | Use | License |
|---|---|---|
| [FastAPI](https://fastapi.tiangolo.com) / Starlette | HTTP API | MIT / BSD-3 |
| [Uvicorn](https://www.uvicorn.org) | ASGI server | BSD-3 |
| [Pydantic v2](https://docs.pydantic.dev) | typed validation | MIT |
| [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python) | optional Claude provider | MIT |
| [HTTPX](https://www.python-httpx.org) | OpenAI API client (Chat Completions over HTTPS), test client | BSD-3 |
| [SciPy](https://scipy.org) (with [HiGHS](https://highs.dev)) / [NumPy](https://numpy.org) | linear programming | BSD-3 / MIT |
| [pytest](https://pytest.org) | tests | MIT |

External services: the OpenAI API (model `gpt-5.6-terra`). AI coding assistance (Claude Code) was used during development. The architecture, guardrails and optimization design are the team's own. All scenario data is the organizers' synthetic data.

## Security and secret handling

- Keys come only from environment variables at run time. `.env` is git-ignored and docker-ignored, and `.env.example` contains names only. The Docker image contains no credentials.
- The key is held in memory only. It is excluded from `Settings.__repr__` and never logged.
- API errors are fixed, hand-written messages. They never include stack traces, prompts, provider error bodies or input values.
- Unexpected exceptions are logged by type and stack frames only, without the exception message, so request content cannot leak into logs (covered by tests).
- Deployment hosts should store `LLM_API_KEY` as a secret variable, never in `render.yaml` or other committed files.

## Known limitations

- **Hosted-LLM dependency.** Interpretation needs the configured provider to be reachable. Outages or quota exhaustion produce a controlled `500 llm_unavailable`, not a guessed schedule, because falling back to phrase matching is not allowed. Keep quota available during judging.
- **Latency depends on the model.** The LP takes a few milliseconds, so almost all request time is the single model call. `gpt-5.6-terra` at `reasoning_effort=low` is the default. If p95 latency is above 5 s, try `LLM_EFFORT=none` or `LLM_MODEL=gpt-5.6-luna` (one env var each) and re-run `pytest -m live` to confirm accuracy holds.
- **One directive per note.** This follows the spec. A note that states two constraints at once (for example "battery fully offline") would be mapped to the single best-fitting type.
- **Conservative composition.** If two notes reduce solar in the same hour, the factors multiply. The spec does not define this case, and multiplying keeps the plan valid under either reading.
- **Cache per worker.** The interpretation cache lives in each worker's memory and is not shared between replicas.
- **Inconsistent battery inputs.** If `initial_energy_kwh` lies outside [minimum, capacity], the request is rejected with 422, because end-of-day neutrality would make every schedule invalid.