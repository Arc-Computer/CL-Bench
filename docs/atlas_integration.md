# Atlas SDK Integration Runbook

This guide walks through the complete workflow for running the CRM benchmark inside Atlas so the teacher can judge end-to-end outcomes. Follow the steps in order before attempting a full benchmark run.

## 1. Prerequisites

- **Python**: 3.10 or newer (Atlas SDK is validated on 3.13).
- **Postgres**: Two logical databases running on the same server or container:
  - `crm_sandbox` – used by the CRM harness to store case state.
  - `atlas` – used by Atlas telemetry, rewards, and learning.
- **Environment variables**:
  - `OPENAI_API_KEY` (student + teacher + judge via LiteLLM/OpenAI adapter).
  - `GEMINI_API_KEY` (small/large judges plus learning synthesizer).
  - CRM DB connection: `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME=crm_sandbox`.
  - Atlas telemetry DB: `STORAGE__DATABASE_URL=postgresql://atlas:atlas@<host>:<port>/atlas`.
  - Optional: `ATLAS_OFFLINE_MODE=1` during local dry-runs (unset for real LLM validation).
- **System packages**: `psycopg` is already listed in repo requirements; ensure Postgres CLI tools are available if you plan to inspect databases manually.

> Copy `configs/atlas/.env.example` to `.env` (or your shell profile) and adjust credentials. The repo’s `.env` already contains the necessary API keys—duplicate the values there so Atlas picks them up.

## 2. Install / Update Atlas SDK

Atlas is vendored under `external/atlas-sdk`, but you must install it in editable mode inside your Python environment so `atlas` CLI commands resolve:

```bash
pip install -e external/atlas-sdk[dev]
```

Notes:

- This brings in `litellm>=1.77.7`. If you also need packages that pin older `litellm` (e.g., `bespokelabs-curator==1.61.3`), use a separate virtualenv.
- The editable install lets you modify the SDK adapters inside this repo (e.g., the CRM harness adapter) without re-installing.
- After pulling new commits that touch `external/atlas-sdk`, rerun the same command to ensure your env stays in sync.

Reference: `external/atlas-sdk/README.md` (Quick Start and Storage sections) and `.env.example`.

## 3. Seed and Verify the CRM Postgres Backend

The CRM harness talks to Postgres via `ConversationHarness._create_backend()` (see `src/evaluation/conversation_harness.py:596-603`). To guarantee each case starts clean:

1. Start the Postgres service (local Docker or managed instance) with both databases created (`crm_sandbox`, `atlas`).
2. Run the existing seeding flow (if available) or execute the CRM schema migrations.
3. Validate connectivity by running a quick harness smoke test:
   ```bash
   python -m src.evaluation.run_harness \
     --dataset data/conversations/dev.jsonl \
     --backend postgres \
     --sample 1
   ```
   During initialization the harness will:
   - Call `PostgresCrmBackend.begin_session(reset=True)` (see `src/crm_backend.py:90-134`).
   - Seed initial entities via `_seed_postgres_backend`.
   - Roll back the session after the run (so each task starts from the same snapshot).
4. If the command fails, double-check the DB_* env vars and that the `crm_sandbox` schema exists.

## 4. Dataset Selection & Revision Tracking

Atlas needs to know which dataset version produced each run. Use `_compute_dataset_revision` (`src/integration/atlas_integration.py:90-138`) to record either:

- The current Git commit SHA (preferred).
- Or the modification timestamp of the dataset file when Git metadata is unavailable.

Workflow:

1. Place the final CRM conversations JSONL in `data/conversations/<name>.jsonl`.
2. Record the intended subset (full benchmark vs. sample).
3. When generating task payloads (next section), pass the dataset path so the script embeds the revision string in every task payload and, later, in `artifacts/baselines/<ts>/atlas/metrics.json`.

## 5. Prepare Task Payloads & Run Atlas

### 5.1 Generate Atlas Task JSONL

Create (or run) the CLI that wraps `conversation_to_payload` (`src/integration/atlas_common.py:19-43`). It should:

1. Load the dataset JSONL into Conversation objects.
2. For each conversation, build a payload:
   ```json
   {
     "task_id": "<conversation_id>::<run_id>",
     "run_id": "<timestamp>-<suffix>",
     "conversation": { ...serialized Conversation... },
     "dataset_revision": "<git sha or timestamp>",
     "backend": "postgres",
     "use_llm_judge": true
   }
   ```
3. Write the list to `artifacts/baselines/<timestamp>/atlas/tasks.jsonl`.

### 5.2 Configure Atlas

1. Copy `configs/atlas/.env.example` to your env and export the values.
2. Use `configs/atlas/crm_harness.yaml` (or `.dev.yaml`) as the runtime config:
   - `agent.type: crm_harness`
   - Student model: GPT‑4.1‑mini
   - Teacher model: GPT‑4.1 (low temperature)
   - `rim.small_model`: Gemini 2.5 Flash
   - `rim.large_model`: Gemini 2.5 Pro
   - `learning.llm`: Gemini 2.5 Flash
   - `orchestration.forced_mode: paired` (capability probe disabled)
   - `storage.database_url`: Atlas telemetry DB

### 5.3 Execute the Run

```bash
atlas run \
  --config configs/atlas/crm_harness.yaml \
  --task-file artifacts/baselines/<timestamp>/atlas/tasks.jsonl \
  --output-dir artifacts/baselines/<timestamp>/atlas
```

Flags of note:

- `--task-file` injects the serialized CRM conversations directly; the CRM harness adapter consumes `task_payload`.
- `--output-dir` mirrors Atlas’ own `.atlas/runs/...` artifacts into our repo-standard location for easier handoff.
- Ensure `ATLAS_OFFLINE_MODE` is unset (or `0`) so real LLMs execute.

## 6. Artifacts & Telemetry

After the run, you should see:

- `artifacts/baselines/<timestamp>/atlas/tasks.jsonl` – input payloads.
- `artifacts/baselines/<timestamp>/atlas/sessions.jsonl` – copy of Atlas session traces (`atlas/cli/jsonl_writer.py:330-488`).
- `artifacts/baselines/<timestamp>/atlas/metrics.json` – aggregated stats including:
  - `execution_mode`, `adaptive_summary` (will read “paired” throughout because the probe is disabled).
  - `reward_stats`, `session_reward`, token usage (`prompt_tokens`, `completion_tokens`, `calls`).
  - CRM verification results: `overall_success`, `first_failed_turn`, etc.
- `artifacts/baselines/<timestamp>/atlas/README.md` – human-readable summary (timestamp, config path, dataset, success counts).

Use these files to compare Atlas rewards vs. baseline harness metrics and to prepare reports for downstream training.

## 7. Smoke-Test Checklist

Before launching the full benchmark:

1. **Single conversation run**:
   ```bash
   atlas run --config configs/atlas/crm_harness.yaml \
     --task-file artifacts/baselines/smoke/tasks.jsonl \
     --limit 1
   ```
2. Ensure `ATLAS_OFFLINE_MODE=0` (unset).
3. Inspect the single session record:
   - `metadata.execution_mode` must be `paired`.
   - `session_reward.score > 0` (or judge rationale shows a meaningful verdict).
   - CRM Postgres tables reflect the tool calls during execution, and `PostgresCrmBackend.rollback_session()` returned the DB to the initial state afterward.
4. Verify `artifacts/.../sessions.jsonl` and `metrics.json` include token usage and adaptive summaries.
5. If everything looks good, proceed with the full dataset run using the same config/runbook.

Document every smoke run (timestamp, dataset, config, reward score) so we can show a clear progression from baseline to Atlas-graded runs.

