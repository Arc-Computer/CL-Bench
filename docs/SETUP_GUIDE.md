# Complete Setup Guide for Evaluation Runs

This guide provides step-by-step instructions for setting up the environment to run baseline and Atlas evaluations. Follow these steps in order before attempting any evaluation runs.

## Prerequisites

### System Requirements
- **Python**: 3.10 or newer (Atlas SDK is validated on 3.13)
- **PostgreSQL**: Version 12+ (for CRM backend and Atlas telemetry)
- **Docker**: Optional, for running PostgreSQL via docker-compose
- **Git**: For cloning and managing the repository

### Required API Keys
- `OPENAI_API_KEY`: For GPT-4.1, GPT-4.1-mini, and LLM judge
- `ANTHROPIC_API_KEY`: For Claude 4.5 Sonnet
- `GEMINI_API_KEY`: For Atlas judges and learning synthesizer

## Step 1: Clone and Navigate to Repository

```bash
git clone <repository-url>
cd arc-crm-benchmark
git checkout evaluation-run-20251111  # or your evaluation branch
```

## Step 2: Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**Important**: Always activate the virtual environment before running any commands.

## Step 3: Install Core Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs:
- Core CRM sandbox dependencies (Pydantic, NumPy)
- Agent integrations (Anthropic, OpenAI, LiteLLM)
- PostgreSQL driver (psycopg)
- Testing tools (pytest)

## Step 4: Install Atlas SDK (Required for Atlas Evaluations)

```bash
pip install -e external/atlas-sdk[dev]
```

**Note**: This brings in `litellm>=1.77.7`. If you also need packages that pin older `litellm` (e.g., `bespokelabs-curator==1.61.3`), use a separate virtualenv.

### Required Atlas SDK Modification

**Critical**: After installing the Atlas SDK, you must apply a local modification to support environment variable override for the storage database URL.

**File**: `external/atlas-sdk/atlas/config/models.py`

**Location**: Add a `model_validator` to the `StorageConfig` class (around line 533, after the `apply_schema_on_connect` field):

```python
@model_validator(mode="before")
@classmethod
def _override_with_env_var(cls, data: Any) -> Any:
    """Override database_url with STORAGE__DATABASE_URL if set."""
    import os
    if isinstance(data, dict):
        env_url = os.getenv("STORAGE__DATABASE_URL")
        if env_url:
            data = {**data, "database_url": env_url}
    return data
```

**Verification**: After making this change, verify it works:

```bash
python3 << 'EOF'
import os
from pathlib import Path
from atlas.config.loader import load_config

# Load .env
env_file = Path(".env")
if env_file.exists():
    with env_file.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

config = load_config("configs/atlas/crm_harness.yaml")
if config.storage:
    print(f"✅ Storage database_url: {config.storage.database_url}")
    storage_url_env = os.getenv("STORAGE__DATABASE_URL")
    if storage_url_env and config.storage.database_url == storage_url_env:
        print("✅ Environment variable override working correctly")
    else:
        print("⚠️  Config value differs from .env - check your modification")
EOF
```

## Step 5: Set Up PostgreSQL Databases

You need **two PostgreSQL databases**:
1. `crm_sandbox` - Used by the CRM harness to store case state
2. `atlas` - Used by Atlas telemetry, rewards, and learning

### Option A: Using Docker Compose (Recommended)

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your database credentials and API keys
# Ensure these are set:
# - DB_HOST=localhost
# - DB_PORT=5432
# - DB_NAME=crm_sandbox
# - DB_USER=crm_user
# - DB_PASSWORD=crm_password
# - STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas

# Start PostgreSQL containers
docker compose up -d

# Seed the CRM database
./scripts/db_seed.sh
```

### Option B: Using Existing PostgreSQL Instance

1. Create two databases:
   ```sql
   CREATE DATABASE crm_sandbox;
   CREATE DATABASE atlas;
   ```

2. Update `.env` with your connection details:
   ```bash
   DB_HOST=your-host
   DB_PORT=5432
   DB_NAME=crm_sandbox
   DB_USER=your-user
   DB_PASSWORD=your-password
   STORAGE__DATABASE_URL=postgresql://atlas:atlas@your-host:5433/atlas
   ```

3. Run schema migrations:
   ```bash
   psql -h your-host -U your-user -d crm_sandbox -f sql/01_schema.sql
   psql -h your-host -U your-user -d crm_sandbox -f sql/02_seed_data.sql
   ```

## Step 6: Configure Environment Variables

Create or update `.env` file in the repository root:

```bash
# LLM API Keys (REQUIRED)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Postgres CRM Backend (REQUIRED)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=crm_sandbox
DB_USER=crm_user
DB_PASSWORD=crm_password

# Atlas Storage Database (REQUIRED for Atlas evaluations)
STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas

# Optional: Atlas offline mode (set to 1 for dry-runs without LLM calls)
# ATLAS_OFFLINE_MODE=0
```

**Important**: Never commit `.env` to git. It contains sensitive credentials.

## Step 7: Verify Dataset Location

The evaluation uses the final clean dataset:

```bash
# Verify dataset exists
ls -lh artifacts/deterministic/final_conversations_final_clean.jsonl

# Check dataset size (should be ~1,200 conversations)
wc -l artifacts/deterministic/final_conversations_final_clean.jsonl
```

**Expected**: ~1,200 conversations (one per line in JSONL format)

## Step 8: Pre-Flight Checks

Before running any evaluation, verify everything is set up correctly:

### 8.1 Verify Python Environment

```bash
python3 --version  # Should be 3.10+
which python3      # Should point to venv/bin/python3
pip list | grep -E "pydantic|litellm|atlas"  # Check key packages
```

### 8.2 Verify Database Connectivity

```bash
# Test CRM database connection
python3 << 'EOF'
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
from src.crm_backend import PostgresCrmBackend, DatabaseConfig

config = DatabaseConfig(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", "5432")),
    database=os.getenv("DB_NAME", "crm_sandbox"),
    user=os.getenv("DB_USER", "crm_user"),
    password=os.getenv("DB_PASSWORD", "crm_password"),
)

backend = PostgresCrmBackend(config)
print("✅ CRM database connection successful")
EOF

# Test Atlas database connection
python3 << 'EOF'
import os
from pathlib import Path
from atlas.config.loader import load_config
from atlas.runtime.storage.database import Database
import asyncio

# Load .env
env_file = Path(".env")
if env_file.exists():
    with env_file.open() as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()

config = load_config("configs/atlas/crm_harness.yaml")
if config.storage:
    database = Database(config.storage)
    async def test():
        await database.connect()
        print("✅ Atlas database connection successful")
        await database.disconnect()
    asyncio.run(test())
EOF
```

### 8.3 Verify API Keys

```bash
# Test OpenAI API key
python3 << 'EOF'
import os
from dotenv import load_dotenv
load_dotenv()
import openai
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
models = client.models.list()
print("✅ OpenAI API key valid")
EOF

# Test Anthropic API key
python3 << 'EOF'
import os
from dotenv import load_dotenv
load_dotenv()
import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
print("✅ Anthropic API key valid")
EOF
```

### 8.4 Verify Dataset Format

```bash
# Check first conversation structure
python3 << 'EOF'
import json
with open("artifacts/deterministic/final_conversations_final_clean.jsonl", "r") as f:
    first_line = f.readline()
    conv = json.loads(first_line)
    print("✅ Dataset format valid")
    print(f"   Conversation ID: {conv.get('conversation_id', 'N/A')}")
    print(f"   Turns: {len(conv.get('turns', []))}")
    print(f"   Complexity: {conv.get('complexity_level', 'N/A')}")
EOF
```

## Step 9: Run Smoke Tests

Before running full evaluations, always run smoke tests to verify everything works:

### 9.1 Baseline Smoke Test

```bash
# Load environment
set -a
source .env
set +a

# Run Claude smoke test (5 conversations)
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent claude \
  --model claude-sonnet-4-5-20250929 \
  --backend postgres \
  --sample 5 \
  --output artifacts/evaluation/baseline_smoke_claude.jsonl \
  --temperature 0.0 \
  --max-output-tokens 800
```

**Expected**: 5 conversations executed, progress logging visible, output file created.

### 9.2 Atlas Smoke Test

```bash
# Load environment
set -a
source .env
set +a

# Run Atlas smoke test (5 scenarios)
python3 scripts/evaluate_atlas_learning_loop.py
```

**Expected**: 5 scenarios executed, learning state grows, database verification passes.

## Step 10: Ready for Full Evaluation

Once smoke tests pass, you're ready to run full evaluations. See `docs/evaluation_execution_commands.md` for complete command reference.

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'atlas'"

**Solution**: Make sure you've installed Atlas SDK:
```bash
pip install -e external/atlas-sdk[dev]
```

### Issue: "Database connection failed"

**Solution**: 
1. Verify PostgreSQL is running: `docker ps` or `pg_isready`
2. Check `.env` credentials match your database
3. Verify databases exist: `psql -l | grep -E "crm_sandbox|atlas"`

### Issue: "STORAGE__DATABASE_URL not found"

**Solution**:
1. Verify `.env` has `STORAGE__DATABASE_URL` set
2. Verify Atlas SDK modification was applied (Step 4)
3. Reload environment: `set -a; source .env; set +a`

### Issue: "Dataset file not found"

**Solution**:
1. Verify dataset exists: `ls artifacts/deterministic/final_conversations_final_clean.jsonl`
2. Check you're in the repository root directory
3. Verify branch has the dataset: `git log --oneline --all -- artifacts/deterministic/`

### Issue: "API key invalid"

**Solution**:
1. Verify API keys in `.env` are correct
2. Check API key has sufficient credits/quota
3. Test API key directly (see Step 8.3)

## Multiple Evaluation Runs

To run multiple evaluation attempts in parallel or sequentially:

1. **Use separate output directories**:
   ```bash
   --output artifacts/evaluation/run_20251111_001/baseline_claude.jsonl
   --output artifacts/evaluation/run_20251111_002/baseline_claude.jsonl
   ```

2. **Use separate Atlas output directories**:
   ```bash
   --output-dir artifacts/evaluation/run_20251111_001/atlas_full
   --output-dir artifacts/evaluation/run_20251111_002/atlas_full
   ```

3. **Tag results with run identifiers**:
   - Include timestamp in directory names
   - Document run parameters in a README per run directory
   - Keep separate analysis reports per run

## Next Steps

- Review `docs/evaluation_execution_commands.md` for complete command reference
- Review `docs/atlas_integration.md` for Atlas-specific details
- Review `docs/reply-case-study.md` for evaluation context and objectives

