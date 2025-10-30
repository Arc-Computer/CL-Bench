# Arc CRM Benchmark Sandbox

This repository provides a synthetic CRM environment and evaluation harness for testing state-modifying workflows. It mirrors a real CRM schema (contacts, clients, opportunities, quotes, contracts, documents, notes), enforces foreign-key relationships, and ships with a golden-case suite plus tooling to reproduce our baseline runs with leaderboard models such as Claude 4.5 Sonnet and GPT‑4.1.

## What's Inside

### Schema-backed models
Pydantic entity definitions (with UUID defaults, enum validation, and assignment checks) covering every table in `data/fake_crm_tables_schema.json`.

### Mock CRM API
`MockCrmApi` offers in-memory storage with production-style guards: duplicate-email rejection, non-negative quote amounts, relationship validation, and human-readable error messaging.

### Golden cases
71 scripted scenarios for the five highest-impact CRM tasks. They include both happy paths and negative edge cases (enum casing, invalid IDs, missing required fields, probability/date bounds, malformed document uploads) so you can surface real failure modes.
Recent updates expanded the negative coverage to include whitespace/mixed-case enums, past or malformed close dates, numeric boundary violations (0/100 probabilities, non-numeric amounts), type mismatches, forbidden extra fields, unsafe filenames/extensions, cross-entity ID mix-ups, and edits to closed opportunities.

### Evaluation harness
Prompt → tool → validator runner that executes the golden cases against an agent, captures tool calls, diffs CRM state, writes JSONL logs, and can optionally send the suite to a GPT-based “judge” for coverage feedback.

## Repository Structure

- `src/`
  - `crm_sandbox.py` – entity models and `MockCrmApi` implementation.
  - `golden_cases.py` – scenario definitions and seeding helpers.
  - `validators.py` – deterministic state checks used by tests and the harness.
  - `harness.py` – baseline harness (Claude/GPT integrations, logging, result types).
  - `crm_env.py` – Gymnasium-compatible environment exposing the sandbox for RL loops.
- `data/`
  - `fake_crm_tables_schema.json` – authoritative schema.
  - `Agent tasks.csv` – task-frequency signals used to prioritize scenarios.
- `tests/` – pytest suite covering models, API tools, validators, golden cases, and the harness.
- `artifacts/` (generated after runs) – JSONL baselines and judge feedback.

## Quickstart

### Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Start the Postgres CRM backend
1. Copy the defaults and adjust as needed:
   ```bash
   cp .env.example .env
   ```
2. Launch the services (Docker Desktop must be running):
   ```bash
   docker compose up -d
   ```
   The Postgres container exposes port `${DB_PORT:-5432}` and automatically runs the schema + seed scripts found in `sql/`.
3. Verify the database is ready:
   ```bash
   docker compose ps
   ```
   Once the `db` service is `healthy`, you can connect via `psql`, your IDE, or `pgAdmin` at `http://localhost:${PGADMIN_PORT:-8080}`.
4. Re-run the seed data at any time (idempotent):
   ```bash
   ./scripts/db_seed.sh
   ```

Default credentials (safe for local dev) live in `.env.example`:

| Variable | Value |
| --- | --- |
| `DB_HOST` | `localhost` |
| `DB_PORT` | `5432` |
| `DB_NAME` | `crm_sandbox` |
| `DB_USER` | `crm_app` |
| `DB_PASSWORD` | `crm_password` |

> This Dockerized backend delivers the first milestone in Phase A of `CONTINUAL_LEARNING_PLAN.md`, giving the upcoming Atlas integrations a realistic datastore to target. Issue #9 will wire the runtime tools to this database while keeping the existing `MockCrmApi` for fast unit tests.

### Run tests
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

### Execute the baseline harness
1. Export `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` (or place them in `.env`).
2. Example usage:
   ```python
   from pathlib import Path
   from src.harness import BaselineHarness, ClaudeAgent

   harness = BaselineHarness(agent=ClaudeAgent(), log_path=Path("artifacts/baseline_claude.jsonl"))
   harness.run()
   ```
   The harness will replay all golden cases, diff CRM state via validators, and write one JSONL record per scenario containing the tool call, validator outcome, and expectations.

### Programmatic sandbox usage
```python
from src.crm_sandbox import MockCrmApi

api = MockCrmApi()
client = api.create_new_client(
    name="Acme Inc.",
    email="ops@acme.example",
    status="Active",
)
opportunity = api.create_new_opportunity(
    name="Acme – FY26 Renewal",
    client_id=client.client_id,
    amount=125_000.0,
    stage="Negotiation",
)
api.create_quote(
    opportunity_id=opportunity.opportunity_id,
    amount=125_000.0,
    status="Draft",
)
```
All relationship and enum rules are enforced automatically; invalid calls raise `ValueError` with clear descriptions.

## Reinforcement Learning Environment

- `from src.crm_env import CrmEnv`: wraps `MockCrmApi` + validators as a Gymnasium environment.
- Observations include task metadata, last tool outcome, and compact CRM state summaries. Set `expose_reference=False` to hide ground-truth arguments during training and rely on natural-language reasoning instead.
- Actions combine a discrete tool index and JSON arguments; `CrmEnv.step` also accepts structured dictionaries for convenience.
- Optional prompt aids: pass `include_tool_hints=True` to surface signature-style descriptions for each tool.
- Default reward is binary 0/1; optional shaping hooks are exposed via `RewardConfig`.
- See `docs/crm_env.md` for the full schema plus rollout examples (`python examples/run_crm_env.py`). For a live model demo (OpenAI, Anthropic, or mock), run `python examples/run_crm_env_with_llm.py --provider openai --model gpt-4.1 --episodes 5` and inspect the logged telemetry.

## Contributing & Next Steps

- Update `data/fake_crm_tables_schema.json` and `data/Agent tasks.csv` first when new schema fields or task priorities emerge.
- Extend `golden_cases.py` with additional scenarios (mixed-case enums, alternate date formats, extreme numeric values) to cover new failure modes.
- Use the JSONL baselines as inputs to your own analytics or continual-learning pipelines.

Issues and pull requests are welcome—especially if you spot missing edge cases or want to contribute additional evaluation tooling.
