# Evaluation Execution Commands

Complete command reference for running baseline and Atlas evaluations against the final clean dataset (1,200 conversations).

## Environment Setup

Ensure `.env` contains all required credentials:

```bash
# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GEMINI_API_KEY=...

# Postgres CRM Backend
DB_HOST=localhost
DB_PORT=5432
DB_NAME=crm_benchmark
DB_USER=crm_user
DB_PASSWORD=crm_password

# Atlas Storage
STORAGE__DATABASE_URL=postgresql://atlas:atlas@localhost:5433/atlas
```

Load environment:
```bash
set -a
source .env
set +a
```

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
- Atlas-specific: learning growth, reward trends, playbook entries

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

## Troubleshooting

### Judge Not Evaluating Correctly
- Check `OPENAI_API_KEY` is set
- Verify judge is enabled: `use_llm_judge=True` (default)
- Review judge rationale in output JSONL

### Atlas Learning Not Persisting
- Verify `STORAGE__DATABASE_URL` is correct
- Check Atlas database connection
- Review learning state queries in logs

### Progress Logging Not Showing
- Ensure latest code with enhanced logging is used
- Check log level is INFO or DEBUG
- Verify output file is being written

