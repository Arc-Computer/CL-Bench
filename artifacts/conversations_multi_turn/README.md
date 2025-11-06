# Multi-Turn Conversation Datasets

This directory hosts the validated chained benchmark conversations.

- `chains.jsonl` – canonical multi-turn dataset (1,500 conversations; 900 simple / 450 medium / 150 complex; 40% expected failures).
- `20251106T152518Z/` – source run directory containing per-bucket generation logs plus `full/` with manifest, analytics report, lint summary, verification output, and the merged dataset for promotion.

Use the commands recorded in `20251106T152518Z/full/quality_checks.md` to regenerate the dataset with a fresh timestamp if needed. Keep prior timestamps only when auditing historical runs; otherwise, replace them with the latest validated drop before handoff.
