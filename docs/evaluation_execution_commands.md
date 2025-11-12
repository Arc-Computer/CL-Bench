# Evaluation Execution Commands

Complete command reference for running baseline and Atlas evaluations against the final clean dataset (1,200 conversations).

## Prerequisites

**Before running any evaluation, ensure you have completed the setup:**

1. ✅ Followed `docs/SETUP_GUIDE.md` completely
2. ✅ Virtual environment activated
3. ✅ All dependencies installed (including Atlas SDK with modification)
4. ✅ PostgreSQL databases running (`crm_sandbox` and `atlas`)
5. ✅ `.env` file configured with all API keys and database credentials
6. ✅ Smoke tests passed (see Phase 1 below)

**Quick Setup Check:**
```bash
# Verify environment
source venv/bin/activate
set -a; source .env; set +a

# Verify dataset exists
ls -lh artifacts/deterministic/final_conversations_final_clean.jsonl

# Run smoke test (5 conversations)
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent claude \
  --model claude-sonnet-4-5-20250929 \
  --backend postgres \
  --sample 5 \
  --output artifacts/evaluation/smoke_test.jsonl
```

## Environment Setup

Ensure `.env` contains all required credentials:

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

# Atlas Storage (REQUIRED for Atlas evaluations)
STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas
```

**Load environment before every evaluation run:**
```bash
set -a
source .env
set +a
```

**Note**: If you haven't completed setup, see `docs/SETUP_GUIDE.md` for complete instructions.

## Phase 1: Smoke Tests (Verification)

### 1.1 Baseline Smoke Test - LLM Judge Verification

**Purpose:** Verify LLM judge evaluates task completion (goal achievement) not process matching.

**Command:**
```bash
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

**Verification Steps:**
1. Check output JSONL for `judge_used`, `judge_pass`, `judge_score`, `judge_rationale` fields
2. Verify judge rationale focuses on goal achievement, not exact tool matching
3. Confirm `overall_success` reflects task completion, not process adherence
4. Review sample judge evaluations manually
5. Verify progress logging shows every conversation with running success rate

**Expected Output:**
- 5 conversations executed
- Judge used when exact match fails but execution succeeds
- Judge scores reflect goal achievement
- Success rate calculated from task completion
- Progress logged for each conversation with running success rate

**Progress Logging:**
Each conversation will be logged with format:
```
[1/5] Conversation: SKEL-... | Success: ✓ | Running Success Rate: 100.0% (1/1) | ETA: 00:02:30
```

### 1.2 Atlas Smoke Test - Learning Loop Verification

**Purpose:** Verify Atlas learning loop is active, learning persists, and judge evaluates task completion.

**Command:**
```bash
python3 scripts/evaluate_atlas_learning_loop.py
```

**Note:** This script runs 5 scenarios. Verify:
1. Learning state grows across scenarios
2. Learning persists to Postgres database
3. Learning re-injects into subsequent sessions
4. Judge evaluates based on task completion
5. Session rewards track correctly
6. Progress logged for each scenario with running success rate

**Expected Output:**
- 5 scenarios executed sequentially
- Learning state increases across scenarios
- Database verification passes
- Learning re-injection verified
- Progress logged for each scenario with running success rate

**Progress Logging:**
Each scenario will display:
```
SCENARIO 2/5: SKEL-...
Running Success Rate: 50.0% (1/1)
...
Running Success Rate: 50.0% (1/2)
```

## Phase 2: Full Baseline Evaluation

### 2.1 Claude 4.5 Sonnet Baseline

**Command:**
```bash
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent claude \
  --model claude-sonnet-4-5-20250929 \
  --backend postgres \
  --output artifacts/evaluation/baseline_claude_sonnet_4_5.jsonl \
  --temperature 0.0 \
  --max-output-tokens 800
```

**Progress Tracking:**
- Enhanced logging: Every conversation logged with running success rate
- Real-time success rate calculation: `successes/total * 100`
- Monitor judge usage: Count `judge_used: true` in results
- ETA calculated based on current rate

**Expected Duration:** ~4-6 hours for 1,200 conversations

**Output Format:**
```
[1/1200] Conversation: SKEL-... | Success: ✓ | Running Success Rate: 100.0% (1/1) | ETA: 04:30:00
[2/1200] Conversation: SKEL-... | Success: ✗ | Running Success Rate: 50.0% (1/2) | ETA: 04:28:30
...
```

### 2.2 GPT-4.1 Baseline

**Command:**
```bash
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent gpt4.1 \
  --model gpt-4.1 \
  --backend postgres \
  --output artifacts/evaluation/baseline_gpt4_1.jsonl \
  --temperature 0.0 \
  --max-output-tokens 800
```

**Expected Duration:** ~4-6 hours for 1,200 conversations

### 2.3 GPT-4.1 Mini Baseline

**Command:**
```bash
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent gpt4.1 \
  --model gpt-4.1-mini \
  --backend postgres \
  --output artifacts/evaluation/baseline_gpt4_1_mini.jsonl \
  --temperature 0.0 \
  --max-output-tokens 800
```

**Expected Duration:** ~4-6 hours for 1,200 conversations

## Phase 3: Atlas Evaluation

### 3.1 Full Atlas Run

**Command:**
```bash
python3 scripts/run_atlas_evaluation.py \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --config configs/atlas/crm_harness.yaml \
  --output-dir artifacts/evaluation/atlas_full
```

**Note:** The wrapper script `scripts/run_atlas_evaluation.py` provides a CLI interface for the `run_atlas_baseline` function.

**Progress Tracking:**
- Enhanced logging: Every conversation logged with running success rate
- Monitor learning state growth across sessions
- Track session rewards and success rates
- Real-time success rate calculation displayed

**Expected Duration:** ~6-8 hours for 1,200 conversations (with learning overhead)

**Output Files:**
- `artifacts/evaluation/atlas_full/sessions.jsonl` - All Atlas session results
- `artifacts/evaluation/atlas_full/metrics.json` - Aggregated metrics
- `artifacts/evaluation/atlas_full/tasks.jsonl` - Task payloads

## Phase 4: Results Analysis

### 4.1 Analysis Command

**Note:** The current evaluation focuses on Baseline and Atlas Runtime phases. The "Atlas + GKD" phase mentioned in the case study (distillation with Atlas Core) is a future enhancement and not included in this evaluation run.

**Command:**
```bash
python3 scripts/analyze_evaluation_results.py \
  --baseline-claude artifacts/evaluation/baseline_claude_sonnet_4_5.jsonl \
  --baseline-gpt4 artifacts/evaluation/baseline_gpt4_1.jsonl \
  --baseline-gpt4mini artifacts/evaluation/baseline_gpt4_1_mini.jsonl \
  --atlas-sessions artifacts/evaluation/atlas_full/sessions.jsonl \
  --output-report artifacts/evaluation/evaluation_report.md \
  --output-json artifacts/evaluation/evaluation_summary.json
```

**Output Files:**
- **Console:** Formatted summary printed to stdout
- `artifacts/evaluation/evaluation_report.md`: Detailed markdown report with tables and analysis
- `artifacts/evaluation/evaluation_summary.json`: JSON summary data for further processing

**Metrics Calculated:**
- Task success rate (conversation-level)
- Turn-level success rate
- Judge usage statistics
- Token usage and cost estimates
- Atlas-specific: learning growth, reward trends, cue hits, action adoptions, token usage

## Optional: Real-time Progress Monitoring

### Monitor Baseline Evaluation

**Terminal 1:** Run evaluation
```bash
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent claude \
  --model claude-sonnet-4-5-20250929 \
  --backend postgres \
  --output artifacts/evaluation/baseline_claude_sonnet_4_5.jsonl
```

**Terminal 2:** Monitor progress (optional)
```bash
python3 scripts/monitor_evaluation_progress.py \
  --input artifacts/evaluation/baseline_claude_sonnet_4_5.jsonl \
  --update-interval 5
```

## Success Criteria

### Smoke Tests
- ✅ Judge evaluates goal achievement, not process matching
- ✅ Atlas learning loop active and persisting
- ✅ Learning re-injects into subsequent sessions
- ✅ Progress logging shows every conversation with running success rate

### Full Evaluation
- ✅ All 1,200 conversations executed for each baseline
- ✅ Atlas completes full dataset with learning accumulation
- ✅ Results analysis generates comprehensive comparison report
- ✅ Task success rates calculated correctly
- ✅ Token usage and cost estimates provided
- ✅ Progress logged for every conversation with running success rate

## Timeline Estimate

- Smoke tests: ~15 minutes (5 conversations × 3 baselines + Atlas 5 scenarios)
- Full baseline evaluation: ~12-18 hours (1,200 conversations × 3 baselines)
- Atlas evaluation: ~6-8 hours (1,200 conversations with learning overhead)
- Results analysis: ~30 minutes
- **Total: ~18-26 hours**

## Crash Recovery & Resume Support

**Both baseline and Atlas evaluations support automatic resume functionality:**

- **Incremental Writing**: Results are written immediately after each conversation completes
- **Automatic Resume**: If a run crashes or is interrupted, simply re-run the same command - it will automatically detect existing results and skip already-processed conversations
- **Progress Preservation**: Running success rates and ETAs account for previously completed conversations

**How It Works:**
1. On startup, the evaluation checks if the output file already exists
2. If it exists, loads existing results and identifies already-processed conversation IDs
3. Filters out completed conversations from the remaining work
4. Continues processing only the remaining conversations
5. Appends new results to the existing file

**Example Resume Scenario:**
```bash
# First run processes 500 conversations, then crashes
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent claude \
  --model claude-sonnet-4-5-20250929 \
  --backend postgres \
  --output artifacts/evaluation/baseline_claude_sonnet_4_5.jsonl

# Re-run the same command - it will automatically resume from conversation 501
# Logs will show: "Found 500 existing results, will resume from remaining conversations"
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent claude \
  --model claude-sonnet-4-5-20250929 \
  --backend postgres \
  --output artifacts/evaluation/baseline_claude_sonnet_4_5.jsonl
```

**Important Notes:**
- Do NOT delete or modify the output file while a run is in progress
- If you want to start fresh, delete the output file before running
- Individual conversation failures are caught and logged, but don't stop the entire run
- Results are flushed to disk after each conversation for maximum crash safety

## Troubleshooting

### Setup Issues

**"ModuleNotFoundError: No module named 'atlas'"**
- Solution: Install Atlas SDK: `pip install -e external/atlas-sdk[dev]`
- Verify: `python3 -c "import atlas; print(atlas.__version__)"`

**"Database connection failed"**
- Solution: Verify PostgreSQL is running (`docker ps` or `pg_isready`)
- Check `.env` credentials match your database
- Verify databases exist: `psql -l | grep -E "crm_sandbox|atlas"`

**"STORAGE__DATABASE_URL not found"**
- Solution: Verify `.env` has `STORAGE__DATABASE_URL` set
- Verify Atlas SDK modification was applied (see `docs/SETUP_GUIDE.md` Step 4)
- Reload environment: `set -a; source .env; set +a`

**"Dataset file not found"**
- Solution: Verify dataset exists: `ls artifacts/deterministic/final_conversations_final_clean.jsonl`
- Check you're in repository root directory
- Verify branch has dataset: `git log --oneline --all -- artifacts/deterministic/`

### Runtime Issues

**Judge Not Evaluating Correctly**
- Check `OPENAI_API_KEY` is set and valid
- Verify judge is enabled: `use_llm_judge=True` (default)
- Review judge rationale in output JSONL
- Check API quota/credits available

**Atlas Learning Not Persisting**
- Verify `STORAGE__DATABASE_URL` is correct in `.env`
- Check Atlas database connection (see `docs/SETUP_GUIDE.md` Step 8.2)
- Verify Atlas SDK modification applied (see `docs/SETUP_GUIDE.md` Step 4)
- Review learning state queries in logs
- Check database schema initialized: `psql -d atlas -c "\dt"`

**Progress Logging Not Showing**
- Ensure latest code with enhanced logging is used
- Check log level is INFO or DEBUG
- Verify output file is being written
- Check terminal supports Unicode (for ✓/✗ symbols)

**Evaluation Running Slowly**
- Check API rate limits (OpenAI, Anthropic, Gemini)
- Verify database connection pooling
- Monitor system resources (CPU, memory, disk I/O)
- Consider running evaluations in parallel on separate machines

**UUID Serialization Errors**
- Verify UUID serialization fix is in place (should be in latest code)
- Check `src/evaluation/llm_judge.py` has `_serialize_for_json` method
- Ensure all UUIDs are converted to strings before JSON serialization

### Multiple Run Management

**Running Multiple Evaluations in Parallel**
- Use separate output directories for each run
- Use separate PostgreSQL databases or ensure transaction isolation
- Monitor API rate limits across parallel runs
- Tag each run with unique identifier (timestamp, run number)

**Organizing Multiple Evaluation Runs**
```bash
# Create run-specific directories
mkdir -p artifacts/evaluation/run_$(date +%Y%m%d_%H%M%S)

# Use run-specific outputs
--output artifacts/evaluation/run_20251111_001/baseline_claude.jsonl
--output-dir artifacts/evaluation/run_20251111_001/atlas_full

# Document each run
echo "Run parameters: ..." > artifacts/evaluation/run_20251111_001/README.md
```

## Additional Resources

- **Complete Setup Guide**: `docs/SETUP_GUIDE.md` - Step-by-step setup instructions
- **Atlas Integration Details**: `docs/atlas_integration.md` - Atlas-specific configuration
- **Case Study Context**: `docs/reply-case-study.md` - Evaluation objectives and methodology
- **Repository README**: `README.md` - General repository information

