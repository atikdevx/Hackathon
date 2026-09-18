# 3-Minute Video Script: GridWise LLM

Target length is about 2:50. Timings are cumulative. **[Screen]** lines say what to show.

---

### 0:00 – 0:20 · The problem

**[Screen]** Title slide, then the request JSON from `BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json`.

> "GridWise plans a campus's next 24 hours of electricity: grid, rooftop solar and a battery. Tariffs change hour to hour, so the goal is the cheapest valid schedule. The twist is that operators also send free-text notes, like *'Panel washing from one until three will leave roughly one-fifth of normal solar'*. Some notes change the rules; others are distractors. We have to understand them, enforce them exactly, and still minimize cost."

### 0:20 – 0:45 · Architecture

**[Screen]** The architecture diagram from the README.

> "It's one FastAPI service with two endpoints, `/health` and `/optimize-energy`. The pipeline is: request validation, then the LLM interpreter, then deterministic guardrails, then directive application, then a linear-program optimizer, then an independent replay validator, then the JSON response. Nothing from the LLM reaches the math until code has validated it. Nothing reaches the client until the schedule has been replayed hour by hour."

### 0:45 – 1:15 · LLM interpretation

**[Screen]** `app/llm/prompt.py`: the schema, then the time/percentage rules.

> "Claude Opus 5 interprets all notes in a single call, using structured outputs. A JSON Schema forces one of our six directive types and exactly the fields each note needs. For every note it decides relevance, the type, the value, and the time window as stated, like 'start 13, end 15'. Then deterministic code expands that window with the spec's start-inclusive, end-exclusive rule into [13, 14]. The model reads, the code counts. In our tests this removed the off-by-one hour errors models make when they list hours themselves. The prompt also covers the other conventions: factor as the solar that *remains*, so an 80% reduction is 0.2; 12- and 24-hour clocks; percent-of-capacity reserves; and `no_op` for distractors. The model only sees the notes and the battery size. It never sees demand or tariffs, so it can't invent or change them."

### 1:15 – 1:40 · Deterministic guardrails

**[Screen]** `app/guardrails/validator.py`, then `tests/test_guardrails.py` passing.

> "The model's output is treated as untrusted and passes two deterministic stages. The first checks the raw extraction and expands the windows. The second checks the official contract: exactly one entry per note, in order. It checks allowed types, that `no_op` means applies false with a null adjustment, and exact keys. Hours must be unique integers from 0 to 23 in ascending order. The factor must be in [0, 1], the reserve can't exceed capacity, and the grid cap must be non-negative. It rejects, it never repairs. On a rejection, the model gets one repair turn listing the violations. If that also fails, we return a controlled error. We never guess with phrase matching."

### 1:40 – 2:10 · Optimization

**[Screen]** The LP formulation in the README, then `app/optimizer/lp.py`.

> "Validated directives become per-hour bounds: effective solar, a raised minimum energy, charge or discharge forced to zero, and grid caps. SciPy's HiGHS solver minimizes the sum of grid times tariff. The constraints are hourly energy balance, battery transitions, capacity, rate limits, curtailable solar, no export, and end-of-day neutrality. A second pass keeps the optimal cost but minimizes battery throughput, so the plan has no pointless cycling. We net charge and discharge into one action, recalculate totals from the plan, then replay every rule and every directive again before responding."

### 2:10 – 2:35 · Testing

**[Screen]** Terminal: `python -m pytest`, then `python scripts/run_public_samples.py`.

> "The offline suite covers the API contract, every guardrail rejection, optimizer scenarios, and failure handling: timeouts, malformed model JSON, and no secrets in responses or logs. `scripts/judge.py` is an independent checker that shares no code with the service. It replays each public sample under the reference directives. All ten samples pass, at exactly the reference optimal cost, including with a real LLM in the loop over HTTP and inside Docker. Live tests run paraphrased notes through the model."

### 2:35 – 2:50 · Run and deploy

**[Screen]** README quickstart; `docker run`; `curl /health`.

> "To run it: copy `.env.example`, set `LLM_API_KEY`, then `python -m app`, or use `docker run -p 8000:8000 -e LLM_API_KEY=… ghcr.io/atikdevx/gridwise-llm:1.0.0`. The image runs as non-root, has no baked-in secrets, and includes a health check. Thanks!"
