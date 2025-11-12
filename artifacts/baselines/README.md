# Baseline Results

This directory contains standardized baseline evaluation results for all three models.

## Files

- **`claude_4_5_sonnet.jsonl`** - Claude 4.5 Sonnet baseline (400 conversations)
- **`gpt4_1.jsonl`** - GPT-4.1 baseline (400 conversations) - *Will be finalized once GPT-4.1 reaches 400*
- **`gpt4_1_mini.jsonl`** - GPT-4.1 Mini baseline (400 conversations)

## Standardization

All baseline results are standardized to exactly 400 conversations to ensure fair comparison with:
- The evaluation dataset (`artifacts/datasets/evaluation_400.jsonl`)
- Atlas evaluation results (`artifacts/atlas/atlas_400_results/`)

## Finalizing GPT-4.1

Once GPT-4.1 baseline reaches 400 conversations, run:
```bash
./scripts/finalize_gpt4_1_baseline.sh
```

This will standardize the GPT-4.1 results and move them to this directory.
