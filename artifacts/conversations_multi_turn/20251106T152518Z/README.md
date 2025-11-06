# Multi-Turn Generation Drop – 20251106T152518Z

This directory contains the Phase 2 multi-turn regeneration outputs. The dataset follows the 60/40 success mix and 40 % conversation failure distribution required for Stage 1 baselines.

- `simple/` – 900 conversations spanning `client_management_chain` and `contact_document_note` (seed 42). Includes `chains.jsonl` and curator run log.
- `medium/` – 450 conversations across `client_opp_quote`, `search_quote_review`, `quote_remediation`, `summary_contract`, and `clone_expansion` chains (seed 43). Includes `chains.jsonl` and run log.
- `complex/` – 150 conversations covering `onboarding_pipeline_contract` and `onboarding_opp_deal` chains (seed 44). Includes `chains.jsonl` and run log.
- `full/` – merged dataset (`chains.jsonl`) plus manifest, analytics report, lint summary, verification results, concatenated run log, and `quality_checks.md` documenting the exact commands.

Promotion summary:

- Conversations: 1,500 total (900 success / 600 failure | 60 % success).
- Every turn includes `expected_tool`, `expected_args`, `expect_success`, `expected_error_substring`, and `expected_response` for tool vs. response metrics.
- Analytics (`manifest.json`, `report.md`) and QA (`lint_report.json`, `verification_report.json`) executed post-merge; see `full/quality_checks.md` for provenance.
