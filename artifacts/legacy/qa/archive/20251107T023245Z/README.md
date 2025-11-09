# LLM Judge QA

- Conversations evaluated: 5
- Turns evaluated: 10
- Pass rate: 50.00%
- Conversation-level failures: 5 (5 unique conversations)

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
