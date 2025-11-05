# Arc CRM Benchmark

A synthetic CRM environment for evaluating continual learning in production LLM agents. This benchmark provides a production-realistic testbed for measuring agent adaptation through the [Atlas](https://arc.computer) continual learning framework, enabling systematic evaluation of runtime adaptation ([atlas-sdk](https://github.com/Arc-Computer/atlas-sdk)) and offline training improvements ([ATLAS Core](https://github.com/Arc-Computer/ATLAS)).

## Purpose

Arc CRM Benchmark tests whether agents can improve from ~80% to ≥95% reliability on state-modifying CRM workflows through continual learning. It provides:

- **Production-realistic CRM environment** with full schema (contacts, clients, opportunities, quotes, contracts, documents, notes), enforcing foreign-key relationships, enum validation, and business logic constraints
- **1,500+ synthetic scenarios** generated via Curator with GPT-5-mini, covering 28 CRM tasks with controlled success/failure ratios
- **Atlas SDK integration** for runtime adaptive learning with dual-agent supervision (Student + Teacher)
- **Evaluation harness** measuring agent performance, reward signals, and learning metrics
- **Baseline comparisons** for Claude 4.5 Sonnet and GPT-4.1

This environment serves as the evaluation layer in the Atlas continual learning loop:

```
Runtime (atlas-sdk):  Agent executes CRM tasks → Adaptive supervision → Telemetry
      ↓
Benchmark (this repo): Scenario harness → Reward evaluation → Learning metrics
      ↓
Training (ATLAS Core): GRPO on exported traces → New teacher checkpoints
```

## How It Fits Into Atlas

**Atlas SDK** ([runtime](https://github.com/Arc-Computer/atlas-sdk)) wraps any agent with an adaptive dual-agent reasoning loop. The Student executes tasks while the Teacher provides supervision based on a capability probe. Tasks route into supervision lanes (`auto`, `paired`, `coach`) depending on confidence, allowing agents to stay fast on familiar work while receiving guidance on novel challenges.

**Arc CRM Benchmark** (this repo) provides the synthetic CRM environment and evaluation harness to measure adaptation. Scenarios test realistic state-modifying workflows (create client, update opportunity, generate quote) with production-style constraints.

**ATLAS Core** ([training](https://github.com/Arc-Computer/ATLAS)) consumes exported runtime traces and uses GRPO (Group Relative Policy Optimization) to train improved teacher checkpoints from production data.

## Key Features

### Synthetic Dataset (1,500+ Scenarios)
Generated using [Curator](https://github.com/bespokelabsai/curator) with GPT-5-mini, producing diverse test cases with:
- Natural language task descriptions derived from production CRM patterns
- Realistic entity relationships (companies, contacts, opportunities)
- Contextual failures (not random mutations): wrong enums, missing FKs, workflow violations
- Controlled 60/40 success/failure ratio for balanced evaluation

### Production-Realistic CRM Schema
Pydantic models with strict validation:
- UUID defaults, enum constraints (Literal types), foreign-key enforcement
- Production-style guards: duplicate email rejection, non-negative amounts, relationship validation
- Human-readable error messages matching real CRM APIs

### Evaluation Harness
Executes scenarios against agents with:
- Tool call capture and CRM state diffing
- Structured JSONL logging compatible with Atlas telemetry
- Verifier scoring (deterministic validators + optional LLM judge)
- Support for multiple agent providers (Claude, OpenAI, mock)
- Both in-memory (`mock`) and Postgres backends

### Gymnasium-Compatible RL Environment
For custom reinforcement learning workflows:
- `CrmEnv` wraps CRM backend as standard Gymnasium environment
- Configurable observations, actions, and reward shaping
- Automatic state reset and transaction isolation

## Repository Structure

```
src/
  crm_sandbox.py              # Entity models and MockCrmApi
  scenario_generator.py       # Curator-based synthetic generation
  scenario_harness.py         # Scenario execution and validation
  crm_argument_schemas.py     # Pydantic schemas with Literal enum types
  run_baseline.py             # CLI for baseline evaluations
  harness.py                  # Agent integrations (Claude/OpenAI)
  crm_env.py                  # Gymnasium RL environment
  validators.py               # Deterministic state checks
data/
  fake_crm_tables_schema.json # Canonical CRM schema
  Agent tasks.csv             # Task frequency weights for generation
tests/                        # Pytest suite
artifacts/
  generated_scenarios/        # Curator-generated scenarios (JSONL)
  baseline_*.jsonl            # Baseline evaluation results
```

## Quick Start

### Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Set Up Postgres Backend (Optional)

For testing with a real database:

```bash
cp .env.example .env
docker compose up -d
./scripts/db_seed.sh
```

Default credentials are in `.env.example` (safe for local development).

### Run Baseline Evaluation

```bash
# Set API keys in .env
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key

# Run evaluation
python -m src.run_baseline \
    --mode scenario \
    --scenarios artifacts/generated_scenarios/scenarios.jsonl \
    --agent claude \
    --backend mock \
    --sample 10 \
    --output artifacts/baseline_results.jsonl
```

**Available agents:**
- `claude` - Claude 4.5 Sonnet
- `gpt4.1` - GPT-4.1
- `gpt4` - GPT-4
- `mock` - Ground truth (for validation)

**Available backends:**
- `mock` - In-memory (fast, deterministic)
- `postgres` - Real database (production-realistic)

### Programmatic Usage

```python
from src.crm_sandbox import MockCrmApi

api = MockCrmApi()

# Create entities with automatic validation
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

quote = api.create_quote(
    opportunity_id=opportunity.opportunity_id,
    amount=125_000.0,
    status="Draft",
)
```

All relationship and enum constraints are enforced with clear error messages.

## Data Generation Pipeline

The benchmark’s multi-turn conversations are built in layers so every turn stays grounded in validated data while remaining easy to reproduce and scale:

1. **Validated single-turn library.** We begin with `artifacts/scenarios_500/scenarios_clean.jsonl`, a curated 60/40 success/failure set. `ScenarioRepository` indexes these records by tool/outcome and enriches them with entity metadata so downstream turns can share clients, opportunities, quotes, etc.

2. **Workflow templates and chains.** `src/conversation_templates.py` defines deterministically structured workflows (e.g., onboarding, quote generation). `WorkflowChain` objects stitch templates into multi-segment journeys with explicit success/failure patterns and handoff rules, encoding the 60/40 mix at the segment level.

3. **Curator-guided sampling.** Two Bespoke Curator models drive generation:
   - `ScenarioSelector` receives turn metadata and selects compatible single-turn scenarios from the validated pool.
   - `ChainUtteranceGenerator` writes natural-language user turns that reference the selected arguments and prior context.
   Both emit structured Pydantic responses (`ScenarioSelectionResponse`, `TurnUtteranceResponse`) so the pipeline remains reproducible. Setting `CURATOR_SIMPLE_DATASET=1` switches to deterministic offline stubs for tests and CI.

4. **Mock CRM simulation.** `src/generation/chain_conversation_generator.py` resolves cross-turn references, seeds the mock CRM (`MockCrmApi`) with required entities, executes each tool call, and captures reference payloads. Per-segment summaries (expected/actual outcome, entities created/referenced) and per-turn annotations (scenario IDs, persona hints, handoff traces) are recorded for analytics and continual-learning signals.

5. **Harness validation.** `ConversationHarness` replays each conversation against a fresh CRM instance, failing fast if a success segment fails (or vice versa). The CLI (`scripts/generate_conversations.py --mode chain`) wraps this flow—smoke tests print expected vs. actual segment outcomes, while full runs validate every generated conversation before writing `chains.jsonl`.

6. **Reproducible artifacts.** The chained generator exports a manifest (`artifacts/chains/manifest.json`), analytics report (`artifacts/reports/chains_baseline.md`), and a no-fallback verification pass (`scripts/verify_no_fallbacks.py`) whose output lives next to each run log (e.g., `verification_report.json`, `quality_checks.md`). The README and docs capture the exact commands, seeds, and model names used so datasets can be regenerated or scaled (e.g., new workflow chains, alternative Curator backends).

To scale the pipeline, define additional workflow templates/chains, run the generator with your preferred Curator model (Gemini 2.5 Flash or GPT‑5‑mini via LiteLLM), and regenerate the manifest/analytics/verification artifacts. Because every step is validated against the CRM schema (and the no-fallback audit), the resulting dataset remains production-quality without manual clean-up.

### Current Artifact Snapshot

- **Single-turn scenarios** (validated 60/40 mix): `artifacts/scenarios_500/scenarios_clean.jsonl` (494 records)
- **Chained conversations** (200 conversations, 40 % expected failures): `artifacts/conversations_chains/chains.jsonl`
- **Manifest**: `artifacts/chains/manifest.json` (includes failure ratio/tolerance flags)
- **Analytics report**: `artifacts/reports/chains_baseline.md`
- **Verification log**: `artifacts/conversations_chains/20251105T033709Z/full/verification_report.json`
- **Quality summary**: `artifacts/conversations_chains/20251105T033709Z/full/quality_checks.md`

## Integration with Atlas SDK

### Runtime Evaluation

```python
from atlas.sdk import Agent, Scenario
from src.harness import BaselineHarness
from src.crm_backend import DatabaseConfig
from src.scenario_harness import load_scenarios_from_jsonl

# Load curated scenarios
scenarios = load_scenarios_from_jsonl("artifacts/generated_scenarios/scenarios.jsonl")

# Configure agent via Atlas SDK (Teacher + Student)
agent = Agent.from_config("configs/atlas_teacher_student.yaml")

harness = BaselineHarness(
    agent=agent,
    backend="postgres",
    db_config=DatabaseConfig.from_env(),
    log_path="artifacts/baseline_claude_postgres.jsonl",
)

# Run evaluation
result = harness.evaluate(scenarios)
print(f"Reliability: {result.success_rate:.3f}")

# Telemetry is automatically logged for Atlas ingest
```

### Continual Learning Loop

```python
from atlas.sdk import atlas_run
from src.harness import ClaudeAgent
from src.scenario_harness import load_scenarios_from_jsonl

agent = ClaudeAgent(model_name="claude-sonnet-4.5")
scenarios = load_scenarios_from_jsonl("artifacts/generated_scenarios/scenarios.jsonl")

result = atlas_run(
    agent=agent,
    scenarios=scenarios[:10],
    storage_url="postgresql://atlas:atlas@localhost:5433/atlas",
)

# Export telemetry for training
# arc-atlas --database-url postgresql://... --output traces.jsonl
```

### Training with ATLAS Core

After collecting runtime traces, train improved teacher checkpoints:

```bash
# Export runtime traces from Atlas SDK
arc-atlas \
  --database-url postgresql://atlas:atlas@localhost:5433/atlas \
  --output crm_traces.jsonl \
  --include-status approved

# Train new teacher checkpoint with ATLAS Core
cd /path/to/ATLAS
python scripts/run_offline_pipeline.py \
  --export-path crm_traces.jsonl \
  output_dir=results/crm-teacher-grpo
```

The CRM benchmark scenarios provide realistic failure modes for the training loop to learn from, enabling systematic improvement on state-modifying workflows.

## Reinforcement Learning

For custom RL workflows (Atlas SDK provides adaptive learning out-of-the-box):

```python
from src.crm_env import CrmEnv

env = CrmEnv(
    backend="mock",
    reset_database_each_episode=True,
    expose_reference=False,  # Hide ground truth for training
)

observation, info = env.reset()
action = {
    "tool": "opportunity_search",
    "arguments": {"stage": "Prospecting"}
}
observation, reward, terminated, truncated, info = env.step(action)
```

The environment provides binary or shaped rewards (configurable via `RewardConfig`).

## Running Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest
```

Tests cover entity models, API methods, validators, scenario generation, and the evaluation harness.

## Learning Metrics

The benchmark tracks five core metrics demonstrating continual learning (aligned with Atlas SDK telemetry):

| Metric | Definition | Target |
|--------|-----------|--------|
| **Cue Hit Rate** | % of episodes triggering playbook retrieval | ≥60% |
| **Adoption Rate** | % of retrieved playbooks injected into prompts | ≥90% |
| **Reward Delta** | Δ(reward) baseline vs. guided episodes | ≥+15pp |
| **Token Delta** | Δ(tokens) baseline vs. guided episodes | ≥-40% |
| **Transfer Success** | Accuracy improvement on held-out tasks | ≥+40% |

These metrics validate that agents are learning production-transferable patterns, not memorizing synthetic data.

## Scenario Generation

Scenarios are generated using Curator with strict schema validation:

```bash
python -m src.generate_scenarios \
    --method curator \
    --count 1500 \
    --success-ratio 0.6 \
    --seed 42
```

Each scenario includes:
- `task` - Natural language description
- `setup_entities` - Pre-populated CRM state
- `expected_args` - Ground truth API arguments with validated enums
- `verification_mode` - Success criteria
- `expect_success` - True for happy paths, False for edge cases

## Contributing

Contributions are welcome to expand benchmark coverage:

- **New scenarios**: Add task types or edge cases to `data/Agent tasks.csv`
- **Additional validators**: Extend `validators.py` with new state checks
- **Backend integrations**: Add support for other CRM APIs or databases
- **Evaluation metrics**: Propose new reward functions or success criteria

Please open an issue to discuss major changes before submitting a PR.

## Research & Resources

- **Atlas Documentation**: [docs.arc.computer](https://docs.arc.computer)
- **Atlas SDK**: [github.com/Arc-Computer/atlas-sdk](https://github.com/Arc-Computer/atlas-sdk)
- **ATLAS Core**: [github.com/Arc-Computer/ATLAS](https://github.com/Arc-Computer/ATLAS)
- **Curator**: [github.com/bespokelabsai/curator](https://github.com/bespokelabsai/curator)

## License

MIT License - see LICENSE file for details.

## Citation

If you use this benchmark in your research:

```bibtex
@software{arc_crm_benchmark,
  title = {Arc CRM Benchmark: A Synthetic Environment for Continual Learning Evaluation},
  author = {Arc Computer},
  year = {2025},
  url = {https://github.com/arc-ai/arc-crm-benchmark}
}
```
