# Quality Checks – Chained Dataset (20251105T033709Z)

- **Generator/Harness Audit**
  - Verified `scripts/generate_conversations.py`, `src/generation/chain_conversation_generator.py`, and `src/evaluation/conversation_harness.py` contain no fallback execution paths; the generator now raises if Curator fails to provide deterministic selections.
  - Offline mode (`CURATOR_SIMPLE_DATASET=1`) remains deterministic and uses validated scenarios only.
- **Automated Validation**
  - `pytest tests/test_chain_conversation_generator.py`
  - `PYTHONPATH=. python scripts/verify_no_fallbacks.py --artifacts-dir artifacts/conversations_chains/20251105T033709Z --output artifacts/conversations_chains/20251105T033709Z/full/verification_report.json`
- **Dataset Integrity**
  - 200 conversations (120 success / 80 expected failures) – failure ratio 40.0%, within ±2% tolerance.
  - Every `_failure` chain variant (CHAIN-00xB) contains a failing segment; success-only variants have zero failures.
  - `artifacts/chains/manifest.json` and `artifacts/reports/chains_baseline.md` updated to reflect the new dataset.
- **Artifacts**
  - Raw conversations: `artifacts/conversations_chains/chains.jsonl`
  - Manifest: `artifacts/chains/manifest.json`
  - Report: `artifacts/reports/chains_baseline.md`
  - Verification log: `artifacts/conversations_chains/20251105T033709Z/full/verification_report.json`

This run satisfies the Phase 5 no-fallback requirement and is ready for scale-out generation.
