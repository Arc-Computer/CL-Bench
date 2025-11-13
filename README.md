# Arc Continual Learning Benchmark

A comprehensive benchmark for evaluating LLM agents on continual learning in stateful environments. This framework tests agent reliability, adaptation, and performance on multi-turn workflows involving state mutations, cross-entity relationships, and production-style constraints.

## Overview

This benchmark framework evaluates whether LLM agents can learn and adapt in complex stateful environments where actions modify persistent state, entities have cross-references, and workflows span multiple turns. Unlike analytics or search tasks, stateful environments require agents to:

- **Track state mutations** across multiple operations (create, modify, delete)
- **Maintain entity relationships** with referential integrity
- **Execute multi-turn workflows** with cross-turn dependencies
- **Adapt through continual learning** to improve consistency and reliability

**Current implementation:** CRM workflows serve as the first comprehensive testbed, providing production-realistic constraints and a diverse task distribution. Additional stateful environments (financial services, telco, healthcare, manufacturing) will be added to expand the benchmark's scope.

### What This Benchmark Provides

- **Production-realistic stateful environment** with full schema and validation constraints
- **1,200+ multi-turn conversations** covering diverse workflows with varying complexity
- **Comprehensive evaluation harness** measuring tool execution, response quality, and task completion
- **Flexible agent integrations** supporting Claude, GPT-4.1, GPT-4.1-mini, and custom agents
- **Optional Atlas SDK integration** for runtime adaptive learning with dual-agent supervision
- **Extensible framework** designed for additional stateful environment implementations

## Why Stateful Environments Matter

Most LLM benchmarks focus on search, question-answering, or single-turn tasks. Stateful environments present unique challenges:

1. **State Persistence**: Actions modify persistent state that affects subsequent operations
2. **Cross-Entity Dependencies**: Operations must respect foreign key relationships and referential integrity
3. **Multi-Turn Workflows**: Complex tasks require maintaining context across 7-10+ turns with cross-turn references
4. **Production Constraints**: Real-world validation rules (enums, business logic, duplicate detection) must be enforced
5. **Continual Learning Opportunity**: Repeated workflows provide learning signal for adaptation

These challenges mirror production systems where LLM agents must reliably interact with databases, APIs, and enterprise applications—making this benchmark essential for evaluating real-world agent deployments.

## Key Features

### Production-Realistic Stateful Environment (CRM)

The current implementation uses a comprehensive CRM environment:

- **Full entity model**: Clients, contacts, opportunities, quotes, contracts, documents, notes
- **Strict validation**: Foreign-key relationships, enum constraints, business logic guards
- **Realistic constraints**: Duplicate email rejection, non-negative amounts, relationship validation
- **Human-readable errors**: Error messages matching real CRM API patterns
- **Deterministic reproducibility**: Every conversation can be regenerated from seed data
- **Schema provenance**: The CRM schema is a renamed and lightly simplified version of the sanitized reference Reply shared with us. Only the structure (tables, fields, enums, relations) is retained; all field names were changed and no production data or metadata left Reply systems. See `data/fake_crm_tables_schema.json` for the exact artifact included in this repository.

### Comprehensive Dataset

- **1,200+ conversations** with varying complexity:
  - **Simple** (1-3 turns): Single-entity operations
  - **Medium** (4-6 turns): Cross-entity workflows
  - **Complex** (7-10 turns): Multi-step processes with state mutations
- **Schema-grounded**: All conversations respect production constraints
- **Standardized evaluation subset** (400 conversations, seed=42): Maintains complexity distribution for consistent baseline comparisons
- **Evaluation usage**: The 400-conversation subset (`artifacts/datasets/evaluation_400.jsonl`) is reused across baselines and Atlas continual-learning runs. Baseline agents use it purely for scoring (no learning). Atlas operates in an online-learning regime on the same stream; we document this so performance comparisons are grounded in identical inputs even though no separate hold-out slice is currently published. If you need a hold-out test run, we can generate one from the remaining 800 conversations on request.

Every conversation is generated deterministically from the sanitized schema and seed entities. No production Reply data or identifiers are present in any artifact shipped with the benchmark.

### Evaluation Harness

- **Tool execution validation**: Verifies correct tool calls and arguments
- **Response quality assessment**: Evaluates natural language responses via LLM judge
- **Multiple backends**: In-memory (`mock`) and PostgreSQL (`postgres`) options
- **Structured logging**: JSONL output compatible with analysis pipelines
- **Token usage tracking**: Comprehensive metrics for cost analysis
- **Multi-granularity metrics**: Conversation-level, turn-level, and operational metrics

### Baseline Agent Configuration

All baseline agents share the same prompting and runtime configuration so model comparisons isolate capability differences:

- **Prompt inputs** (`src/evaluation/agents.py`):
  - System message contains (a) a summary of the sanitized CRM schema, (b) the top Reply-derived workflow frequencies from `data/Agent_tasks.csv`, and (c) a catalog of the 38 available tools pulled from `MockCrmApi`.
  - User message per turn is a JSON payload containing the natural-language request, prior tool outputs (so IDs can be reused), expected success flags, and the dataset’s `expected_response` structure (text template, acceptable answers, evaluation rubric) when it exists. This gives the models clear response-format guidance while they still have to choose tools/arguments on their own. No raw database dumps, few-shot exemplars, or hidden instructions are injected.
- **Models & parameters**: Claude Sonnet 4.5 (`--agent claude`), GPT-4.1 (`--agent gpt4.1 --model gpt-4.1`), and GPT-4.1 Mini (`--model gpt-4.1-mini`), all run with `temperature=0.0`, `max_output_tokens=800`, and the Postgres backend.
- **Judge configuration**: GPT-4.1 judge with 70% goal / 30% process weighting and a 0.8 pass threshold (`src/evaluation/llm_judge.py`). The judge only activates if exact argument matching fails but the tool executed successfully.

This setup ensures the baseline is as strong and reproducible as possible while keeping all schema knowledge limited to the sanitized artifact already included in the repo.
### Optional Atlas Integration

The benchmark can integrate with [Atlas SDK](https://github.com/Arc-Computer/atlas-sdk) for runtime adaptive learning:

- **Student/Teacher loop**: Dual-agent supervision for continual learning
- **Learning synthesis**: Automatic guidance generation from teacher interventions
- **Telemetry persistence**: PostgreSQL-backed learning state management

See `docs/atlas_integration.md` for Atlas setup instructions.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Arc-Computer/arc-crm-benchmark.git
cd arc-crm-benchmark

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

**Lightweight installation** (for evaluation only, without dataset generation):

```bash
pip install -r requirements-judge.txt
```

### Set Up PostgreSQL Backend (Optional)

For realistic database interactions:

```bash
# Copy environment template
cp .env.example .env

# Start PostgreSQL via Docker
docker compose up -d

# Seed the database
./scripts/db_seed.sh
```

Default credentials are in `.env.example` (safe for local development).

### Run Your First Evaluation

```bash
# Load environment variables
set -a; source .env; set +a

# Run baseline evaluation (5 conversations, quick test)
python3 -m src.evaluation.run_baseline \
    --conversations artifacts/datasets/final_1200.jsonl \
    --agent claude \
    --model claude-sonnet-4-5-20250929 \
    --backend postgres \
    --sample 5 \
    --output artifacts/evaluation/my_first_run.jsonl \
    --temperature 0.0 \
    --max-output-tokens 800
```

**Available agents:**
- `claude` – Claude Sonnet 4.5 (or override with `--model`)
- `gpt4.1` – OpenAI GPT-4.1 (or override with `--model`)
- `mock` – Deterministic replay (useful for harness validation)

**Common options:**
- `--sample N` – Evaluate a subset of conversations
- `--seed 42` – Reproducible random sampling
- `--no-judge` – Disable LLM judge (tool execution only)
- `--backend mock` – Use in-memory backend (no database required)

## Repository Structure

```
arc-crm-benchmark/
├── src/
│   ├── evaluation/          # Evaluation harness and baseline runners
│   ├── integration/         # Atlas SDK integration (optional)
│   └── crm_sandbox.py       # CRM entity models and API
├── scripts/                 # Utility scripts for evaluation and analysis
├── configs/                 # Configuration files (Atlas, agents)
├── data/                    # Schema definitions and task templates
├── docs/                    # Documentation and guides
├── artifacts/
│   ├── datasets/            # Pre-generated conversation datasets
│   └── evaluation/          # Evaluation results and outputs
└── tests/                   # Test suite
```

## Dataset

The benchmark includes a pre-generated dataset of 1,200 multi-turn conversations:

- **Location**: `artifacts/datasets/final_1200.jsonl`
- **Format**: JSONL (one conversation per line)
- **Complexity distribution**:
  - Simple: ~23% (1-3 turns)
  - Medium: ~52% (4-6 turns)
  - Complex: ~25% (7-10 turns)

**Standardized evaluation subset** (400 conversations, seed=42):
- **Location**: `artifacts/datasets/evaluation_400.jsonl`
- Maintains full dataset's complexity distribution
- Use for consistent baseline comparisons

Each conversation includes:
- Natural language user turns
- Expected tool calls and arguments
- Expected responses (for LLM judge evaluation)
- Initial CRM state (entities to seed)
- Cross-turn entity references

## Evaluation Metrics

The harness tracks comprehensive metrics at multiple granularities:

### Conversation-Level Metrics
- **Strict Success**: All turns in conversation succeeded
- **Tool Success**: At least one tool call succeeded
- **Response Success**: At least one response met quality standards

### Turn-Level Metrics
- **Tool execution accuracy**: Did the tool execute correctly?
- **Response quality**: Did the agent's response meet expectations?
- **Verification method**: Exact match vs. LLM judge evaluation

### Operational Metrics
- **Token usage**: Prompt tokens, completion tokens, total tokens
- **Cost estimates**: Based on model pricing
- **Execution time**: Wall-clock time per conversation

## Programmatic Usage

### Basic CRM Operations

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

### Running Evaluations Programmatically

```python
from pathlib import Path
from src.evaluation.run_baseline import run_baseline

results = run_baseline(
    conversations_path=Path("artifacts/datasets/final_1200.jsonl"),
    agent="claude",
    model="claude-sonnet-4-5-20250929",
    backend="postgres",
    output_path=Path("artifacts/evaluation/results.jsonl"),
    sample=100,
    seed=42,
)
```

## Atlas SDK Integration (Optional)

For runtime adaptive learning, integrate with [Atlas SDK](https://github.com/Arc-Computer/atlas-sdk):

### Setup

1. **Install Atlas SDK**:
   ```bash
   pip install -e external/atlas-sdk[dev]
   ```

2. **Configure environment** (see `docs/SETUP_GUIDE.md`):
   - Set up PostgreSQL databases (`crm_sandbox` and `atlas`)
   - Configure `.env` with API keys and database credentials
   - Apply required Atlas SDK modifications (if needed)

3. **Run Atlas evaluation**:
   ```bash
   python scripts/run_atlas_evaluation.py \
       --conversations artifacts/datasets/evaluation_400.jsonl \
       --config configs/atlas/crm_harness.yaml \
       --output-dir artifacts/evaluation/atlas_run \
       --sample 400 \
       --seed 42
   ```

See `docs/atlas_integration.md` for complete setup instructions and `docs/evaluation_execution_commands.md` for command reference.

## Running Tests

```bash
# Run test suite
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest

# Run specific test file
pytest tests/test_crm_sandbox.py
```

Tests cover entity models, API methods, validators, scenario generation, and the evaluation harness.

## Extending the Benchmark

This framework is designed for extensibility to additional stateful environments:

### Adding New Environments

Future environments under consideration:
- **Financial Services**: Trading systems, portfolio management, transaction processing
- **Telecommunications**: Network provisioning, customer service workflows, billing systems
- **Healthcare**: Patient records, appointment scheduling, clinical workflows
- **Manufacturing**: Inventory management, production scheduling, quality control

To add a new environment:
1. Define schema and entity models (following `src/crm_sandbox.py` patterns)
2. Create task definitions and workflow templates
3. Generate conversations using `schema_pipeline/`
4. Integrate with evaluation harness

### Contributing New Use Cases

We welcome contributions of additional stateful environments! Areas for contribution:

- **New environment implementations**: Define schemas, workflows, and datasets for new domains
- **Extended scenarios**: Add task types or edge cases to existing environments
- **Evaluation metrics**: Propose new reward functions or success criteria specific to domain constraints
- **Backend integrations**: Add support for real APIs or databases beyond PostgreSQL
- **Documentation**: Improve guides, examples, or API documentation

**Getting started:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

Please open an issue to discuss major changes before submitting a PR.

## Documentation

- **[Setup Guide](docs/SETUP_GUIDE.md)**: Complete environment setup instructions
- **[Evaluation Commands](docs/evaluation_execution_commands.md)**: Command reference for running evaluations
- **[Atlas Integration](docs/atlas_integration.md)**: Atlas SDK integration guide

## Research & Resources

- **Atlas SDK**: [github.com/Arc-Computer/atlas-sdk](https://github.com/Arc-Computer/atlas-sdk)
- **ATLAS Core**: [github.com/Arc-Computer/ATLAS](https://github.com/Arc-Computer/ATLAS)
- **Atlas Documentation**: [docs.arc.computer](https://docs.arc.computer)

## Citation

If you use this benchmark in your research:

```bibtex
@software{arc_continual_learning_benchmark,
  title = {Arc Continual Learning Benchmark: Evaluating LLM Agents on Stateful Environments},
  author = {Arc Computer},
  year = {2025},
  url = {https://github.com/Arc-Computer/arc-crm-benchmark}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

This benchmark framework was developed to provide a production-realistic testbed for evaluating LLM agents on continual learning in stateful environments. The CRM implementation demonstrates the framework with enterprise-grade constraints, enabling researchers and practitioners to evaluate agent reliability, efficiency, and adaptation capabilities in scenarios that mirror real-world deployments.
