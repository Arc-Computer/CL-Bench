# Arc CRM Benchmark Sandbox

Arc is building infrastructure for continual learning in AI agents. This repository contains Workstream 1 of a joint benchmark with a CRM customer whose goal is to raise agent reliability on high-impact, state-modifying workflows from ~80 % to at least 95 % before their worldwide retreat in mid-November 2025.

## Engagement Context

**Our proposal (Arc)**
- Build a synthetic CRM sandbox that mirrors the customer’s production schema so we can run agents safely offline.
- Use the Salesforce CRM LLM Leaderboard (Oct 8 2025; Claude 4 Sonnet currently leads) as a baseline reference.
- Apply Arc’s continual learning stack (reward traces, teacher/student adapters, drift tracking) to demonstrate how reliability climbs toward the 95 % threshold.
- Instrument the evaluation to capture:  
  • *What* the agent learned (reward breakdowns, learning strings)  
  • *How* it learned (ordered adapter events, guidance)  
  • *Why* it adapted (explicit reasons, drift notes)

**Customer response**
- Delivered `fake_crm_tables_schema.json` plus entity relationships to serve as our schema ground truth.
- Shared `Agent tasks.csv`, a frequency-ranked list of real agent intents. The top five state-changing tasks define initial tool coverage.
- Still anonymizing historical failure data (≈ 15 k LLM-as-a-judge interactions). In the interim we must synthesize failure modes ourselves.
- Success is judged via deterministic checks against CRM state (e.g., verifying a contact was actually created).
- Established a single point of contact who will provide updates and review findings.

## Benchmark Objectives

1. **Workstream 1 (this repo):** Build and validate the in-memory CRM sandbox that encodes schema fidelity and relationship rules.
2. **Workstream 2:** Author deterministic validators, generate golden task suites, and run a baseline agent (Claude 4 Sonnet) to capture failure modes.
3. **Workstream 3:** Share insights with the customer, unblock access to anonymized failures, and align on observed pain points.

This repository concentrates exclusively on Workstream 1 while laying groundwork for Workstream 2.

## Source of Truth

- `data/fake_crm_tables_schema.json` – Defines fields, types, and enums for all CRM entities.
- `data/Agent tasks.csv` – Observed task frequencies driving tool prioritization.
- Entity references (customer-provided):  
  Contact → Client, Opportunity → Client, Quote → Opportunity, Contract → Client & Opportunity, Document/Note → Generic entity + ID.

All code changes must stay aligned with these assets; update the files first if the customer shares revisions.

## Repository Layout

- `src/crm_sandbox.py` – Pydantic data models plus the `MockCrmApi` implementation of the top-five tasks.
- `src/__init__.py` – Makes the sandbox importable as `src.crm_sandbox`.
- `data/` – Schema and task frequency inputs.
- `tests/` – Pytest suite (in progress) validating happy paths, relationship enforcement, and schema-level guards.

## Requirements & Setup

- Python 3.11+
- [Pydantic](https://docs.pydantic.dev/)
- [pytest](https://docs.pytest.org/) (for the validation suite)

```bash
python -m venv .venv
source .venv/bin/activate
pip install pydantic pytest
```

## Testing

Third-party pytest plugins installed globally can interfere with the sandbox tests. Disable auto-loading and run the suite from the project root:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

## Usage

```python
from src.crm_sandbox import MockCrmApi

api = MockCrmApi()
client = api.create_new_client(
    name="Acme Inc.",
    email="ops@acme.example",
    status="Active",
)
opportunity = api.create_new_opportunity(
    name="Acme - FY26 Renewal",
    client_id=client.client_id,
    amount=125_000.0,
    stage="Negotiation",
)
```

Relationship rules are enforced automatically. For example, creating a quote without a valid opportunity raises a human-readable `ValueError`, mirroring production expectations.

## Next Steps

1. Finish the pytest suite to lock in sandbox behavior (Workstream 1 validation).
2. Build deterministic state-check helpers and golden task cases (Workstream 2).
3. Share synthetic failure findings with the customer to accelerate real data access (Workstream 3).
