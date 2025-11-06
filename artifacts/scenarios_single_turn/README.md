# Single-Turn Scenario Library

Validated single-turn scenarios used as building blocks for chained conversations.

- `scenarios_clean.jsonl` – curated 60/40 success/failure pool (505 records).
- `scenarios.jsonl` – raw scenario export prior to cleaning (kept for reference).
- `validation_report.json` – verification summary produced during curation.

All chained generation flows should reference `scenarios_clean.jsonl` via `ScenarioRepository`.
