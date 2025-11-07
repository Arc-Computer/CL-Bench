# LLM Judge QA

- Conversations evaluated: 50
- Turns evaluated: 276
- Pass rate: 58.70%
- Conversation-level failures: 45 (45 unique conversations)

## How to Reproduce
```bash
PYTHONPATH=. python analysis/dataset_judge.py \
    --dataset artifacts/conversations_multi_turn/20251107T134304Z/full/chains_eval_enriched.jsonl \
    --model gpt-4.1-mini \
    --output-dir artifacts/qa/<timestamp>/
```

## Flagged Conversations
- CHAIN-EVAL-20251107T134304Z-0000: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0001: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0002: 3 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0003: 4 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0004: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0005: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0006: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0007: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0008: 4 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0009: 3 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0010: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0011: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0012: 4 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0013: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0014: 4 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0015: 4 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0016: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0018: 4 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0019: 5 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0021: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0022: 5 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0023: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0024: 3 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0025: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0026: 4 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0027: 4 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0029: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0030: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0031: 3 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0032: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0033: 3 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0034: 5 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0035: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0036: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0037: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0039: 4 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0040: 4 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0041: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0042: 8 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0043: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0045: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0046: 3 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0047: 1 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0048: 3 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0049: 3 failed turn(s)
