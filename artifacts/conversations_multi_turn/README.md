# Multi-Turn Conversation Datasets

This directory hosts the validated chained benchmark conversations.

- `chains.jsonl` – canonical multi-turn dataset (1,500 conversations; 900 simple / 450 medium / 150 complex; 40% expected failures).
- `20251105T144453Z/full/` – source run directory containing the dataset plus manifest, analytics report, verification, lint, and quality logs used for handoff.

Older intermediate runs have been pruned to keep the handoff footprint clean. If additional generations are required, replicate the commands listed in `full/quality_checks.md` using new timestamps and move the resulting artifacts into this folder.
