# Multi-Turn Generation Drop – 20251107T134304Z

Cleaned dataset composed of previously generated LLM conversations with refreshed
expected responses and success-only splits.

- Eval split: 600 conversations (chains_eval.jsonl)
- Holdout split: 400 conversations (chains_holdout.jsonl)
- Stress split: 200 conversations (chains_stress.jsonl)
- Extra reserve: 800 conversations (chains_extra.jsonl)

All files live under `full/`. Use `scripts/export_dataset_stats.py` to compute
aggregate manifests for any subset.
