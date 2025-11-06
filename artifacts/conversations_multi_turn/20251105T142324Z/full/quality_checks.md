# Quality Checks – Chained Dataset (20251105T142324Z)

- **Generation Summary**
  - Ran `CURATOR_SIMPLE_DATASET=1 PYTHONPATH=. python scripts/generate_conversations.py --mode chain --count 1200 --seed 42 --model-name gpt-4.1-mini --output artifacts/conversations_multi_turn/20251105T142324Z/full`.
  - Failure ratio 480/1200 (40.0%) within ±2.0% guard; each CHAIN-00xB variant produced the expected failure segment.
  - Output stored in `artifacts/conversations_multi_turn/20251105T142324Z/full/chains.jsonl`.
- **Analytics**
  - `PYTHONPATH=. python analysis/chains_manifest.py --dataset artifacts/conversations_multi_turn/20251105T142324Z/full/chains.jsonl --output artifacts/conversations_multi_turn/20251105T142324Z/full/manifest.json --seed 42 --model-name gpt-4.1-mini`
  - `PYTHONPATH=. python analysis/generate_chains_report.py --dataset artifacts/conversations_multi_turn/20251105T142324Z/full/chains.jsonl --output artifacts/conversations_multi_turn/20251105T142324Z/full/report.md --seed 42 --model-name gpt-4.1-mini --baseline artifacts/conversations_multi_turn/20251105T142324Z/run.log`
- **Verification**
  - `PYTHONPATH=. python scripts/verify_no_fallbacks.py --artifacts-dir artifacts/conversations_multi_turn/20251105T142324Z/full --output artifacts/conversations_multi_turn/20251105T142324Z/full/verification_report.json` → 0 issues detected, no fallback utterances.
- **Harness Regression Tests**
  - `CURATOR_SIMPLE_DATASET=1 PYTHONPATH=. pytest tests/test_chain_conversation_generator.py`
- **Artifacts for promotion**
  - Conversations: `artifacts/conversations_multi_turn/20251105T142324Z/full/chains.jsonl`
  - Manifest: `artifacts/conversations_multi_turn/20251105T142324Z/full/manifest.json`
  - Report: `artifacts/conversations_multi_turn/20251105T142324Z/full/report.md`
  - Verification: `artifacts/conversations_multi_turn/20251105T142324Z/full/verification_report.json`
