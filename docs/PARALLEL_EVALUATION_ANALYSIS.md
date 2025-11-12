# Parallel Evaluation Analysis

## Question: Can baseline and Atlas evaluations run in parallel?

## Database Architecture

### Current Setup
- **CRM Database**: Single `crm_sandbox` database used by both baseline and Atlas
- **Atlas Telemetry Database**: Separate `atlas` database (isolated, no conflicts)
- **Transaction Model**: Each conversation runs in its own transaction:
  - `BEGIN` transaction
  - `TRUNCATE` tables (if `reset=True`)
  - Seed initial entities
  - Execute conversation turns
  - `ROLLBACK` transaction (no persistence)

### Potential Conflicts

1. **TRUNCATE Operations**
   - Both evaluations call `TRUNCATE` on the same tables simultaneously
   - PostgreSQL `TRUNCATE` requires `ACCESS EXCLUSIVE` lock
   - Parallel `TRUNCATE` operations will block each other
   - **Risk**: Deadlocks or serialization delays

2. **Connection Pooling**
   - Each `PostgresCrmBackend` creates its own connection
   - Both evaluations connect to the same database
   - **Risk**: Connection exhaustion under high load

3. **Transaction Isolation**
   - While transactions are isolated, `TRUNCATE` is not fully transactional
   - `TRUNCATE` commits immediately (can't be rolled back)
   - **Risk**: One evaluation's `TRUNCATE` affects the other

## API Rate Limits

### Current API Usage
- **Baseline**: 3 agents × 1,200 conversations = 3,600 API calls
- **Atlas**: 1,200 conversations (with teacher/student) = ~2,400+ API calls
- **Total**: ~6,000+ API calls

### Parallel Execution Impact
- **Doubles API call rate**: ~6,000 calls in parallel vs sequential
- **Risk**: Hitting rate limits faster
- **Impact**: Failed requests, retries, increased latency

## Recommendation: **Sequential Execution**

### Why Sequential is Better

1. **Database Safety**
   - No `TRUNCATE` conflicts
   - No transaction isolation issues
   - Cleaner database state
   - Easier debugging

2. **Rate Limit Management**
   - Spreads API calls over time
   - Reduces risk of hitting limits
   - More predictable costs

3. **Monitoring & Debugging**
   - Clearer logs (not interleaved)
   - Easier to track progress
   - Simpler error diagnosis

4. **Resource Management**
   - Lower peak memory usage
   - More predictable CPU usage
   - Better for long-running overnight runs

### Sequential Execution Plan

**Phase 1: Baseline Evaluations (Sequential)**
```bash
# Run one baseline at a time
python3 -m src.evaluation.run_baseline --agent claude --model claude-sonnet-4-5-20250929 ...
python3 -m src.evaluation.run_baseline --agent gpt4.1 --model gpt-4.1 ...
python3 -m src.evaluation.run_baseline --agent gpt4.1 --model gpt-4.1-mini ...
```

**Phase 2: Atlas Evaluation (After baselines complete)**
```bash
python3 scripts/run_atlas_evaluation.py ...
```

**Total Time**: ~18-24 hours sequential vs ~12-16 hours parallel (but with risks)

## Alternative: Parallel Baseline Agents

### Safe Parallel Option

**You CAN run the 3 baseline agents in parallel** because:
- Each uses separate output files
- Each uses the same database but transactions are isolated
- `TRUNCATE` happens per-conversation, not globally
- Different API keys/models (no shared rate limits)

**Command:**
```bash
# Terminal 1
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent claude \
  --model claude-sonnet-4-5-20250929 \
  --backend postgres \
  --output artifacts/evaluation/baseline_claude_sonnet_4_5.jsonl &

# Terminal 2
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent gpt4.1 \
  --model gpt-4.1 \
  --backend postgres \
  --output artifacts/evaluation/baseline_gpt4_1.jsonl &

# Terminal 3
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent gpt4.1 \
  --model gpt-4.1-mini \
  --backend postgres \
  --output artifacts/evaluation/baseline_gpt4_1_mini.jsonl &

wait  # Wait for all to complete
```

**Benefits:**
- 3x faster baseline evaluation (~4-6 hours instead of ~12-18 hours)
- No database conflicts (each conversation is isolated)
- Different API endpoints (Anthropic vs OpenAI)

**Risks:**
- Higher API rate (but different providers, so manageable)
- More complex monitoring (need to watch 3 processes)

## Final Recommendation

### Option 1: Fully Sequential (Safest)
- Run baselines sequentially, then Atlas
- **Time**: ~18-24 hours
- **Risk**: Lowest
- **Best for**: First run, debugging, stability

### Option 2: Parallel Baselines + Sequential Atlas (Balanced)
- Run 3 baseline agents in parallel, then Atlas sequentially
- **Time**: ~10-14 hours total
- **Risk**: Low (baselines are isolated)
- **Best for**: Faster results while maintaining safety

### Option 3: Fully Parallel (Not Recommended)
- Run baselines and Atlas all in parallel
- **Time**: ~8-12 hours
- **Risk**: High (database conflicts, rate limits)
- **Best for**: Only if you have separate databases

## Implementation

If choosing Option 2 (parallel baselines), create a script:

```bash
#!/bin/bash
# scripts/run_all_baselines_parallel.sh

set -e
set -a
source .env
set +a

echo "Starting parallel baseline evaluations..."

# Start all three baselines in background
python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent claude \
  --model claude-sonnet-4-5-20250929 \
  --backend postgres \
  --output artifacts/evaluation/baseline_claude_sonnet_4_5.jsonl \
  --temperature 0.0 \
  --max-output-tokens 800 > artifacts/evaluation/baseline_claude.log 2>&1 &
CLAUDE_PID=$!

python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent gpt4.1 \
  --model gpt-4.1 \
  --backend postgres \
  --output artifacts/evaluation/baseline_gpt4_1.jsonl \
  --temperature 0.0 \
  --max-output-tokens 800 > artifacts/evaluation/baseline_gpt4_1.log 2>&1 &
GPT4_PID=$!

python3 -m src.evaluation.run_baseline \
  --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
  --agent gpt4.1 \
  --model gpt-4.1-mini \
  --backend postgres \
  --output artifacts/evaluation/baseline_gpt4_1_mini.jsonl \
  --temperature 0.0 \
  --max-output-tokens 800 > artifacts/evaluation/baseline_gpt4_1_mini.log 2>&1 &
GPT4MINI_PID=$!

echo "Started 3 baseline evaluations:"
echo "  Claude: PID $CLAUDE_PID"
echo "  GPT-4.1: PID $GPT4_PID"
echo "  GPT-4.1 Mini: PID $GPT4MINI_PID"
echo ""
echo "Waiting for all to complete..."

wait $CLAUDE_PID
CLAUDE_EXIT=$?
wait $GPT4_PID
GPT4_EXIT=$?
wait $GPT4MINI_PID
GPT4MINI_EXIT=$?

echo ""
echo "All baselines completed:"
echo "  Claude: Exit code $CLAUDE_EXIT"
echo "  GPT-4.1: Exit code $GPT4_EXIT"
echo "  GPT-4.1 Mini: Exit code $GPT4MINI_EXIT"

if [ $CLAUDE_EXIT -eq 0 ] && [ $GPT4_EXIT -eq 0 ] && [ $GPT4MINI_EXIT -eq 0 ]; then
    echo "✅ All baselines completed successfully!"
    exit 0
else
    echo "❌ Some baselines failed. Check logs."
    exit 1
fi
```

