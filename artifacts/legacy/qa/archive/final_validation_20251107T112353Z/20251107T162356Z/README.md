# LLM Judge QA

- Conversations evaluated: 20
- Turns evaluated: 108
- Pass rate: 0.00%
- Conversation-level failures: 20 (20 unique conversations)

## How to Reproduce
```bash
PYTHONPATH=. python analysis/dataset_judge.py \
    --dataset artifacts/conversations_multi_turn/20251107T134304Z/full/chains_eval.jsonl \
    --model gpt-4.1-mini \
    --output-dir artifacts/qa/<timestamp>/
```

## Flagged Conversations
- CHAIN-EVAL-20251107T134304Z-0000: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0001: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0002: 6 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0003: 8 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0004: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0005: 8 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0006: 7 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0007: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0008: 8 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0009: 6 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0010: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0011: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0012: 8 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0013: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0014: 8 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0015: 8 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0016: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0017: 2 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0018: 8 failed turn(s)
- CHAIN-EVAL-20251107T134304Z-0019: 15 failed turn(s)
