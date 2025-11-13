# Artifacts Directory

This directory contains the final evaluation datasets, baseline results, and Atlas results.

## Structure

### `datasets/`
- **`final_1200.jsonl`** - Full 1,200 conversation dataset
- **`evaluation_400.jsonl`** - Standardized 400-conversation subset used for evaluation

### `baselines/`
Standardized baseline results (exactly 400 conversations each):
- **`claude_4_5_sonnet.jsonl`** - Claude 4.5 Sonnet baseline results ✅
- **`gpt4_1.jsonl`** - GPT-4.1 baseline results ⏳ (will be finalized once GPT-4.1 reaches 400)
- **`gpt4_1_mini.jsonl`** - GPT-4.1 Mini baseline results ✅

### `atlas/`
- **`atlas_400_results/`** - Atlas evaluation results (400 conversations with learning accumulation)

## Notes

- All baseline results are standardized to exactly 400 conversations to ensure fair comparison
- The evaluation dataset (`evaluation_400.jsonl`) matches the conversations used in all baseline and Atlas evaluations
- Baseline results include per-turn metrics, tool execution success, response success, and LLM judge evaluations
- Atlas results include learning state, playbook entries, and session telemetry

## Usage

To analyze results:
```bash
python3 scripts/analyze_evaluation_results.py \
  --baseline-claude artifacts/baselines/claude_4_5_sonnet.jsonl \
  --baseline-gpt4 artifacts/baselines/gpt4_1.jsonl \
  --baseline-gpt4mini artifacts/baselines/gpt4_1_mini.jsonl \
  --atlas-sessions artifacts/atlas/atlas_400_results/atlas/sessions.jsonl \
  --output-report artifacts/evaluation_report.md
```
