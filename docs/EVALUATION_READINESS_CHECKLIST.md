# Evaluation Readiness Checklist

Complete checklist to ensure all components are ready for the Reply case study evaluation.

## ✅ Dataset Readiness

- [x] **Dataset exists**: `artifacts/deterministic/final_conversations_final_clean.jsonl`
- [x] **Dataset size**: Exactly 1,200 conversations
- [x] **Dataset composition**: 
  - 280 simple conversations (1-3 turns)
  - 625 medium conversations (4-6 turns)
  - 295 complex conversations (7-10 turns)
  - 9 unique workflow categories
- [x] **Dataset format**: Valid JSONL, all conversations executable
- [x] **No placeholders**: All template tokens resolved

## ✅ Environment Setup

- [x] **Virtual environment**: Created and activated
- [x] **Core dependencies**: `requirements.txt` installed
- [x] **Atlas SDK**: Installed with `pip install -e external/atlas-sdk[dev]`
- [x] **Atlas SDK modification**: Applied (environment variable override)
- [x] **PostgreSQL databases**: 
  - `crm_sandbox` database exists and seeded
  - `atlas` database exists and schema initialized
- [x] **Environment variables**: `.env` configured with:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`
  - `GEMINI_API_KEY`
  - `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`
  - `STORAGE__DATABASE_URL`

## ✅ Scripts & Tools

- [x] **Baseline runner**: `src/evaluation/run_baseline.py` - Working
- [x] **Atlas evaluation**: `scripts/run_atlas_evaluation.py` - Working
- [x] **Atlas smoke test**: `scripts/evaluate_atlas_learning_loop.py` - Working
- [x] **Results analysis**: `scripts/analyze_evaluation_results.py` - Working
- [x] **Progress monitor**: `scripts/monitor_evaluation_progress.py` - Available
- [x] **Crash recovery**: Implemented and verified

## ✅ Configuration Files

- [x] **Atlas config**: `configs/atlas/crm_harness.yaml` - Present and configured
- [x] **Config validation**: All required fields present
- [x] **Model configurations**: Correct models specified

## ✅ Documentation

- [x] **Setup guide**: `docs/SETUP_GUIDE.md` - Complete
- [x] **Execution commands**: `docs/evaluation_execution_commands.md` - Complete
- [x] **Atlas integration**: `docs/atlas_integration.md` - Complete
- [x] **Case study**: `docs/reply-case-study.md` - Framework ready

## ✅ Metrics & Analysis

- [x] **Baseline metrics**: Task success rate, turn success rate, judge usage, token usage
- [x] **Atlas metrics**: Task success rate, learning growth, reward trends
- [x] **Cost estimation**: Token usage → cost calculation
- [x] **Analysis script**: Generates console, markdown, and JSON outputs

## ⚠️ Known Limitations / Future Work

### Atlas + GKD Phase (Not Yet Implemented)

The case study mentions an "Atlas + GKD" phase where Atlas Core distills teacher interventions. This phase is **not yet implemented** in the current codebase. The evaluation will focus on:

1. **Baseline** (3 agents: Claude 4.5 Sonnet, GPT-4.1, GPT-4.1 mini)
2. **Atlas Runtime** (student/teacher loop with learning)

The GKD distillation phase would require:
- Exporting Atlas sessions from database
- Running Atlas Core training pipeline
- Re-evaluating with distilled checkpoints

**Status**: This is documented as a future phase. Current evaluation focuses on runtime learning improvements.

### Runtime Efficiency Metrics

The case study mentions capturing "wall-clock, tool latency" metrics. Currently:
- **Wall-clock time**: Not explicitly captured per conversation (but can be calculated from logs)
- **Tool latency**: Not explicitly captured (but could be added to harness)

**Status**: These are nice-to-have metrics. Core success rate metrics are fully captured.

### Cue Hits & Action Adoptions

The case study mentions tracking "cue hits, action adoptions" from Atlas. Currently:
- These metrics are extracted from Atlas metadata when available
- May not be present in all Atlas sessions depending on learning configuration

**Status**: Captured when available in Atlas telemetry.

## ✅ Pre-Evaluation Verification

Before starting evaluation, verify:

1. **Smoke tests pass**:
   ```bash
   # Baseline smoke test (5 conversations)
   python3 -m src.evaluation.run_baseline \
     --conversations artifacts/deterministic/final_conversations_final_clean.jsonl \
     --agent claude \
     --model claude-sonnet-4-5-20250929 \
     --backend postgres \
     --sample 5 \
     --output artifacts/evaluation/smoke_baseline.jsonl
   
   # Atlas smoke test (5 scenarios)
   python3 scripts/evaluate_atlas_learning_loop.py
   ```

2. **Database connectivity**:
   ```bash
   # Test CRM database
   psql -h localhost -U crm_user -d crm_sandbox -c "SELECT COUNT(*) FROM clients;"
   
   # Test Atlas database
   psql -h localhost -U atlas -d atlas -c "\dt"
   ```

3. **API keys valid**:
   ```bash
   # Test OpenAI
   python3 -c "import openai; import os; client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY')); print('✅ OpenAI OK')"
   
   # Test Anthropic
   python3 -c "import anthropic; import os; client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY')); print('✅ Anthropic OK')"
   ```

4. **Dataset accessible**:
   ```bash
   ls -lh artifacts/deterministic/final_conversations_final_clean.jsonl
   wc -l artifacts/deterministic/final_conversations_final_clean.jsonl  # Should be 1200
   ```

## ✅ Evaluation Execution Plan

### Phase 1: Smoke Tests (15 minutes)
- [ ] Baseline Claude smoke test (5 conversations)
- [ ] Baseline GPT-4.1 smoke test (5 conversations)
- [ ] Baseline GPT-4.1 mini smoke test (5 conversations)
- [ ] Atlas smoke test (5 scenarios)

### Phase 2: Full Baseline Evaluation (~12-18 hours)
- [ ] Claude 4.5 Sonnet baseline (1,200 conversations)
- [ ] GPT-4.1 baseline (1,200 conversations)
- [ ] GPT-4.1 mini baseline (1,200 conversations)

### Phase 3: Atlas Evaluation (~6-8 hours)
- [ ] Full Atlas run (1,200 conversations)

### Phase 4: Results Analysis (~30 minutes)
- [ ] Run analysis script
- [ ] Review markdown report
- [ ] Verify JSON summary

## ✅ Post-Evaluation Deliverables

After evaluation completes:

1. **Results files**:
   - `artifacts/evaluation/baseline_claude_sonnet_4_5.jsonl`
   - `artifacts/evaluation/baseline_gpt4_1.jsonl`
   - `artifacts/evaluation/baseline_gpt4_1_mini.jsonl`
   - `artifacts/evaluation/atlas_full/sessions.jsonl`
   - `artifacts/evaluation/atlas_full/metrics.json`

2. **Analysis outputs**:
   - `artifacts/evaluation/evaluation_report.md`
   - `artifacts/evaluation/evaluation_summary.json`

3. **Case study updates**:
   - Fill in Key Findings table (Section 1.4)
   - Document Key Learnings (Section 1.5)
   - Complete Conclusion (Section 1.6)
   - Add Results & Analysis (Section 2.3)

## Notes

- **Crash recovery**: All evaluations support automatic resume
- **Progress logging**: Real-time success rates and ETAs
- **Multiple runs**: Use separate output directories for parallel runs
- **Documentation**: All commands documented in `docs/evaluation_execution_commands.md`

