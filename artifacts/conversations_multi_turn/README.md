# Multi-Turn Conversation Datasets

This directory hosts the validated chained benchmark conversations.

- `chains.jsonl` – canonical multi-turn dataset (1,500 conversations; 900 simple / 450 medium / 150 complex; 40% expected failures).
- `20251106T152518Z/` – source run directory containing per-bucket generation logs plus `full/` with manifest, analytics report, lint summary, verification output, normalization overrides, and the merged dataset for promotion.

Use the commands recorded in `20251106T152518Z/full/quality_checks.md` to regenerate the dataset with a fresh timestamp if needed. Keep prior timestamps only when auditing historical runs; otherwise, replace them with the latest validated drop before handoff.

Metadata alignment artifacts:
- Single-turn overrides: `artifacts/scenarios_single_turn/20251106T141945Z/seed_overrides.json`
- Conversation-level canonical overrides: `artifacts/conversations_multi_turn/20251106T152518Z/full/entity_overrides.json`
