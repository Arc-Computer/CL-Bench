# LLM Judge QA

- Conversations evaluated: 10
- Turns evaluated: 20
- Pass rate: 50.00%
- Conversation-level failures: 10 (10 unique conversations)

## How to Reproduce
```bash
PYTHONPATH=. python analysis/dataset_judge.py \
    --dataset artifacts/conversations_multi_turn/20251106T152518Z/full/chains.jsonl \
    --model gpt-4.1-mini \
    --output-dir artifacts/qa/<timestamp>/
```

## Flagged Conversations
- CHAIN-006A-0000: 1 failed turn(s)
- CHAIN-006A-0001: 1 failed turn(s)
- CHAIN-006A-0002: 1 failed turn(s)
- CHAIN-006A-0003: 1 failed turn(s)
- CHAIN-006A-0004: 1 failed turn(s)
- CHAIN-006A-0005: 1 failed turn(s)
- CHAIN-006A-0006: 1 failed turn(s)
- CHAIN-006A-0007: 1 failed turn(s)
- CHAIN-006A-0008: 1 failed turn(s)
- CHAIN-006A-0009: 1 failed turn(s)
