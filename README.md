# Arc CRM Benchmark

A production-realistic synthetic CRM environment for evaluating LLM agents on state-modifying workflows. This benchmark provides a comprehensive testbed for measuring agent performance, reliability, and adaptation through continual learning frameworks.

## Overview

Arc CRM Benchmark tests whether agents can reliably execute complex, multi-turn CRM workflows involving state mutations, cross-entity relationships, and production-style validation constraints. It provides:

- **Production-realistic CRM environment** with full schema (contacts, clients, opportunities, quotes, contracts, documents, notes)
- **1,200+ multi-turn conversations** covering diverse CRM workflows with varying complexity
- **Comprehensive evaluation harness** measuring tool execution, response quality, and task completion
- **Flexible agent integrations** supporting Claude, GPT-4.1, GPT-4.1-mini, and custom agents
- **Optional Atlas SDK integration** for runtime adaptive learning with dual-agent supervision

## Key Features

### Production-Realistic CRM Schema

- **Full entity model**: Clients, contacts, opportunities, quotes, contracts, documents, notes
- **Strict validation**: Foreign-key relationships, enum constraints, business logic guards
- **Realistic constraints**: Duplicate email rejection, non-negative amounts, relationship validation
- **Human-readable errors**: Error messages matching real CRM API patterns

### Comprehensive Dataset

- **1,200+ conversations** with varying complexity:
  - **Simple** (1-3 turns): Single-entity operations
  - **Medium** (4-6 turns): Cross-entity workflows
  - **Complex** (7-10 turns): Multi-step processes with state mutations
- **Deterministic and reproducible**: Every conversation can be regenerated from seed data
- **Schema-grounded**: All conversations respect production CRM constraints

### Evaluation Harness

- **Tool execution validation**: Verifies correct tool calls and arguments
- **Response quality assessment**: Evaluates natural language responses via LLM judge
- **Multiple backends**: In-memory (`mock`) and PostgreSQL (`postgres`) options
- **Structured logging**: JSONL output compatible with analysis pipelines
- **Token usage tracking**: Comprehensive metrics for cost analysis

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

## Contributing

Contributions are welcome! Areas for contribution:

- **New scenarios**: Add task types or edge cases to expand coverage
- **Additional validators**: Extend validation logic for new failure modes
- **Backend integrations**: Add support for other CRM APIs or databases
- **Evaluation metrics**: Propose new reward functions or success criteria
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
@software{arc_crm_benchmark,
  title = {Arc CRM Benchmark: A Synthetic Environment for LLM Agent Evaluation},
  author = {Arc Computer},
  year = {2025},
  url = {https://github.com/Arc-Computer/arc-crm-benchmark}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

This benchmark was developed to provide a production-realistic testbed for evaluating LLM agents on state-modifying workflows. The CRM schema and workflows are designed to mirror real-world enterprise CRM systems, enabling researchers and practitioners to evaluate agent reliability, efficiency, and adaptation capabilities in realistic scenarios.
