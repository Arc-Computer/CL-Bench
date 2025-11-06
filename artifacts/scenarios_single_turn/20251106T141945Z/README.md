# Single-Turn Scenarios (20251106T141945Z)

- Source: augmented from `artifacts/scenarios_single_turn/scenarios_clean.jsonl` using `scripts/augment_expected_responses.py`
- Validated scenarios: 340 total
  - Success: 204
  - Failure: 136 (40.0%)
- Every scenario includes `expected_response` metadata (structured evaluation only)
- Imported legacy success coverage for `modify_client`, `modify_quote`, and `compare_quotes` to unblock multi-turn chaining; appended matching failure scenarios to preserve the 60/40 ratio (see `generate_modify_client_success.log` for curator runs).
- To regenerate:
  1. `PYTHONPATH=. python scripts/augment_expected_responses.py --destination artifacts/scenarios_single_turn/<TIMESTAMP>/scenarios_final.jsonl`
  2. Promote the generated file to `artifacts/scenarios_single_turn/scenarios_clean.jsonl`

Validation run: `PYTHONPATH=. python scripts/augment_expected_responses.py --source artifacts/scenarios_single_turn/20251106T141945Z/scenarios_final.jsonl --destination .../scenarios_validated.jsonl`
